"""FastAPI web demo (SCOPE.md, Web). One server-rendered page; drafts only, never sends.

Run with `client-inquiry-web` (main() below): listens on $PORT, which Render sets, else 8000.

create_app() takes every dependency as an argument so tests pass fakes;
build_default_app() wires the real Gemini client from GEMINI_API_KEY / CLIENT_AGENT_MODEL
(environment only, never .env), with the web retry rule: one retry after 10 s for
503/504/dropped connections, never for 429.

- Knowledge is always public_brand/ (BrandSource). The web demo never uses the Second Brain.
- One request = one Gemini call, run on a worker thread. No job queue: the call is short, and
  a queue would mean holding the draft (and so the message's subject) on the server.
- Privacy: the message is never stored or logged. Logs carry only the category, the timing
  and the guard flags, or an error kind. The uvicorn access log is off (no IPs or paths).
  The rate limiter holds visitor addresses and timestamps only.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from client_inquiry_agent.agent import (
    BrandSource,
    InquiryError,
    KnowledgeSource,
    ReplyFormatError,
    SourceError,
    answer_inquiry,
    validate_message,
)
from client_inquiry_agent.config import MAX_MESSAGE_CHARS, WEB_RETRY_WAITS_S, ConfigError, load_config
from client_inquiry_agent.gemini_client import GeminiLLM, failure_kind, make_gemini_call
from client_inquiry_agent.pacing import Pacer
from client_inquiry_agent.ports import LLM, LLMError
from client_inquiry_agent.web.limits import Limiter, Limits

log = logging.getLogger("client_inquiry_agent.web")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))  # autoescape on

# A 2,000-character message, URL-encoded, is at most 2,000 x 4 UTF-8 bytes x 3 = 24,000 bytes.
MAX_FORM_BYTES = 32 * 1024

QUOTA_MESSAGE = "The demo has reached today's limit. Please try again tomorrow."
BUSY_MESSAGE = "Gemini is busy right now. Please try again in a minute."
FAILED_MESSAGE = "Something went wrong while drafting the reply. Please try again."
FORMAT_MESSAGE = "The AI's reply could not be read. Please try again."
NOT_CONFIGURED_MESSAGE = "The demo is not configured right now. Please try again later."
UNREADABLE_MESSAGE = "The message could not be read. Please try again."
TOO_LARGE_MESSAGE = f"The message is too long; the limit is {MAX_MESSAGE_CHARS:,} characters."

CATEGORY_LABELS = {
    "new_project": "New project",
    "pricing": "Pricing",
    "timeline": "Timeline",
    "policy": "Policy",
    "support": "Support",
    "other": "Other",
}

# Made-up example messages for the example buttons.
EXAMPLES = (
    ("Weekend question", "Hi Grace, I'm busy with work during the week. Could we have our project calls on Saturdays instead?"),
    ("Price question", "Hello! How much would a five-page website for my bakery cost?"),
    ("New project", "Hi, I'd like a web app where my customers can book appointments and pay online. How do we get started?"),
)

_LLM_FAILURES = {
    "quota": (429, QUOTA_MESSAGE),
    "busy": (503, BUSY_MESSAGE),
    "other": (502, FAILED_MESSAGE),
}


@dataclass(frozen=True)
class WebSettings:
    retry_waits_s: tuple[float, ...] = WEB_RETRY_WAITS_S  # one retry after 10 s
    limits: Limits = field(default_factory=Limits)


def visitor_id(request: Request) -> str:
    """First IP in X-Forwarded-For (Render's proxy), else the direct peer."""
    first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return first or (request.client.host if request.client else "unknown")


def create_app(
    *,
    llm: LLM | None,
    limiter: Limiter,
    source: KnowledgeSource | None = None,
    settings: WebSettings = WebSettings(),
) -> FastAPI:
    app = FastAPI(title="Client Inquiry Agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.source = source or BrandSource()  # public_brand/ only; never the Second Brain

    def page(request: Request, status: int = 200, headers: dict | None = None, **context) -> Response:
        context = {
            "limits": settings.limits,
            "max_chars": MAX_MESSAGE_CHARS,
            "examples": EXAMPLES,
            "category_labels": CATEGORY_LABELS,
            "message": "",
            **context,
        }
        return TEMPLATES.TemplateResponse(request, "index.html", context, status_code=status, headers=headers)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return page(request)

    @app.get("/draft")
    def draft_get() -> Response:  # a refresh after POST, or a typed URL
        return RedirectResponse("/", status_code=303)

    @app.post("/draft", response_class=HTMLResponse)
    async def draft(request: Request) -> Response:
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return page(request, 411, error=UNREADABLE_MESSAGE)
        if int(length) > MAX_FORM_BYTES:
            log.info("draft rejected=too_large")
            return page(request, 413, error=TOO_LARGE_MESSAGE)
        try:
            form = await request.form(max_files=0, max_fields=5)
        except Exception:  # malformed body
            log.info("draft rejected=bad_form")
            return page(request, 400, error=UNREADABLE_MESSAGE)
        raw = form.get("message")
        # Browsers send textarea line breaks as CRLF; count them as one character, like the counter.
        message = raw.replace("\r\n", "\n") if isinstance(raw, str) else ""

        try:  # before the limiter, so an empty or too-long message does not use up the quota
            text = validate_message(message)
        except InquiryError as exc:
            log.info("draft rejected=bad_message")
            return page(request, 400, error=str(exc), message=message)

        if llm is None:
            log.info("draft rejected=not_configured")
            return page(request, 503, error=NOT_CONFIGURED_MESSAGE, message=message)

        refusal = limiter.try_start(visitor_id(request))
        if refusal is not None:
            log.info("draft refused=limit")
            return page(request, 429, {"Retry-After": str(refusal.retry_after_s)},
                        error=refusal.message, message=message)

        started = time.monotonic()
        try:
            result = await run_in_threadpool(answer_inquiry, text, app.state.source, llm)
        except LLMError as exc:
            kind = failure_kind(exc)
            log.info("draft error=%s seconds=%.1f", kind, time.monotonic() - started)
            status, public = _LLM_FAILURES[kind]
            return page(request, status, error=public, message=message)
        except ReplyFormatError:
            log.info("draft error=format seconds=%.1f", time.monotonic() - started)
            return page(request, 502, error=FORMAT_MESSAGE, message=message)
        except SourceError:
            log.info("draft error=source seconds=%.1f", time.monotonic() - started)
            return page(request, 502, error=FAILED_MESSAGE, message=message)

        log.info(
            "draft category=%s seconds=%.1f price_guard=%s number_check=%s",
            result.category,
            time.monotonic() - started,
            result.price_guard_applied,
            result.number_check_passed,
        )
        return page(request, result=result, message=message)

    @app.api_route("/health", methods=["GET", "HEAD"])  # UptimeRobot's HTTP monitors send HEAD
    def health() -> Response:
        """For Render and UptimeRobot: answers at once, no Gemini call."""
        return PlainTextResponse("ok")

    return app


def build_default_app() -> FastAPI:
    """Real wiring. GEMINI_API_KEY (this project's own key) is a Render secret; never .env."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    log.setLevel(logging.INFO)  # our lines carry only category, timing and flags

    settings = WebSettings()
    llm = None
    try:
        config = load_config()
    except ConfigError:
        log.warning("GEMINI_API_KEY is not set: drafts are unavailable")
    else:
        llm = GeminiLLM(
            config.model,
            make_gemini_call(config.model, config.gemini_api_key),
            Pacer(),  # one shared pacer for the whole service (thread-safe)
            retry_waits=settings.retry_waits_s,
        )
    return create_app(llm=llm, limiter=Limiter(settings.limits), settings=settings)


DEFAULT_PORT = 8000


def port_from_env(env: dict[str, str] | None = None) -> int:
    """Render provides the port in PORT; locally fall back to 8000. A bad value fails loudly."""
    value = (os.environ if env is None else env).get("PORT", "").strip()
    return int(value) if value else DEFAULT_PORT


def main() -> None:
    """Entry point `client-inquiry-web` (used by the Dockerfile)."""
    import uvicorn  # only in the `web` extra

    uvicorn.run(
        "client_inquiry_agent.web.app:build_default_app",
        factory=True,
        host="0.0.0.0",
        port=port_from_env(),
        access_log=False,  # no client IPs or paths in the logs
    )
