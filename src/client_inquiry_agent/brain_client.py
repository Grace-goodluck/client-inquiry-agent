"""HTTP client for the Second Brain's POST /ask (used with --brain). Same as job_application_agent.

Contract (second_brain/src/second_brain/api/app.py):
  request  {"question": str}
  200      {"status": "answered", "answer": str, "sources": [str]}
  200      {"status": "not_found", "note": str}
  502      {"status": "invalid", "reason": str}
  503      {"status": "error", "detail": str}

Only answered and not_found pass through. Everything else (invalid, 503, 422, any other status,
a malformed body, a timeout, a refused connection) becomes BrainReply(status="error").
This client never raises.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from client_inquiry_agent.config import BRAIN_TIMEOUT_S
from client_inquiry_agent.pacing import Pacer

_BODY_PREVIEW = 200


@dataclass(frozen=True)
class BrainReply:
    """One /ask result, already normalised."""

    status: Literal["answered", "not_found", "error"]
    answer: str = ""
    sources: tuple[str, ...] = ()
    note: str = ""  # the brain's not_found note
    error: str | None = None


def _error(message: str) -> BrainReply:
    return BrainReply(status="error", error=message)


def parse_ask_response(status_code: int, body: str) -> BrainReply:
    """Turn one HTTP response from /ask into a BrainReply."""
    try:
        payload = json.loads(body)
    except ValueError:
        return _error(f"brain returned HTTP {status_code} with a non-JSON body: {body[:_BODY_PREVIEW]!r}")
    if not isinstance(payload, dict):
        return _error(f"brain returned HTTP {status_code} with an unexpected body: {body[:_BODY_PREVIEW]!r}")

    status = payload.get("status")

    if status_code == 200 and status == "answered":
        answer = payload.get("answer")
        sources = payload.get("sources")
        if not isinstance(answer, str) or not answer.strip():
            return _error("brain said 'answered' but the answer was empty or missing")
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            return _error("brain said 'answered' but 'sources' was not a list of strings")
        return BrainReply(
            status="answered",
            answer=answer.strip(),
            sources=tuple(s.strip() for s in sources if s.strip()),
        )

    if status_code == 200 and status == "not_found":
        note = payload.get("note")
        return BrainReply(status="not_found", note=note.strip() if isinstance(note, str) else "")

    if status_code == 502 and status == "invalid":
        return _error(f"brain reply was invalid (502): {payload.get('reason', '')}")

    if status_code == 503 and status == "error":
        return _error(f"brain unavailable (503): {payload.get('detail', '')}")

    return _error(f"unexpected brain reply (HTTP {status_code}): {body[:_BODY_PREVIEW]!r}")


class HttpBrain:
    def __init__(
        self,
        base_url: str,
        timeout: float = BRAIN_TIMEOUT_S,
        opener: Callable[..., object] = urllib.request.urlopen,
        pacer: Pacer | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/ask"
        self.timeout = timeout
        self._open = opener
        self._pacer = pacer  # share the run's Pacer: each /ask spends Gemini quota too

    def ask(self, question: str) -> BrainReply:
        if self._pacer is not None:
            self._pacer.wait_turn()
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"question": question}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._open(request, timeout=self.timeout) as response:
                return parse_ask_response(response.status, response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            # 4xx/5xx still carry the brain's JSON body.
            body = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else ""
            return parse_ask_response(exc.code, body)
        except TimeoutError:
            return _error(f"brain timed out after {self.timeout:g} s")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                return _error(f"brain timed out after {self.timeout:g} s")
            return _error(f"could not reach the brain at {self.url}: {exc.reason}")
        except (OSError, ValueError) as exc:
            return _error(f"brain call failed: {exc}")
