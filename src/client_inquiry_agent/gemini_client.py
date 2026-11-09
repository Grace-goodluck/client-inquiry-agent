"""Gemini as an LLM: one paced call per message, with timeout and retries (SCOPE.md, Engine).

Same design as data_analysis_agent, without the reply cache (client messages are not stored):
- Pacer: every attempt, retries included, waits its turn on the shared pacer.
- Timeout: each attempt gets 60 s. No reply in time raises TimeoutError, which counts as a
  dropped connection. The SDK's own HTTP timeout is set to the same 60 s.
- Retry: a 503, a 504, a dropped connection or a timeout is retried (command line: 30 s, then
  60 s; web: once after 10 s). A 429 quota error is never retried.
- make_gemini_call: the only place that touches the google-genai SDK. The API key is passed in
  by the caller, which reads it from the GEMINI_API_KEY environment variable.
"""

import threading
import time
from collections.abc import Callable, Sequence

import httpx

from client_inquiry_agent.config import CLI_RETRY_WAITS_S, GEMINI_TIMEOUT_S
from client_inquiry_agent.pacing import Pacer
from client_inquiry_agent.ports import LLMError


def _status_code(exc: BaseException) -> int | None:
    """The SDK's APIError carries an int .code; otherwise fall back to its message,
    which starts with the status code ("503 UNAVAILABLE...")."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    head = str(exc).lstrip()[:3]
    return int(head) if head.isdigit() else None


def is_unavailable(exc: BaseException) -> bool:
    """A Gemini 503."""
    return _status_code(exc) == 503


def is_deadline_exceeded(exc: BaseException) -> bool:
    """A Gemini 504 DEADLINE_EXCEEDED: the server enforcing the 60 s timeout."""
    return _status_code(exc) == 504


def is_quota_exceeded(exc: BaseException) -> bool:
    """A Gemini 429. Never retried."""
    return _status_code(exc) == 429


# A dropped connection or timeout as it comes out of the SDK (which calls httpx),
# plus the built-in equivalents.
_DROPPED = (
    ConnectionError,
    TimeoutError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def is_dropped_connection(exc: BaseException) -> bool:
    return isinstance(exc, _DROPPED)


def is_retryable(exc: BaseException) -> bool:
    """503, 504, dropped connection or timeout. Never a 429."""
    if is_quota_exceeded(exc):
        return False
    return is_unavailable(exc) or is_deadline_exceeded(exc) or is_dropped_connection(exc)


def _event_wait(event: threading.Event, seconds: float) -> None:
    event.wait(seconds)


def call_with_timeout(
    call: Callable[[str], str],
    prompt: str,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[threading.Event, float], None] = _event_wait,
) -> str:
    """Run call(prompt) on a worker thread; raise TimeoutError if it has not answered by the deadline.

    Tests inject a fake clock and wait so a call that never answers times out without real
    waiting. A timed-out call's thread is a daemon and is abandoned (the SDK's HTTP timeout
    ends it soon after).
    """
    done = threading.Event()
    outcome: dict[str, object] = {}

    def work() -> None:
        try:
            outcome["text"] = call(prompt)
        except BaseException as exc:  # handed back to the caller below
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=work, name="gemini-call", daemon=True).start()
    deadline = clock() + timeout_s
    while not done.is_set():
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(f"no reply from Gemini within {timeout_s:g} s")
        wait(done, remaining)
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["text"]  # type: ignore[return-value]


class GeminiLLM:
    def __init__(
        self,
        model: str,
        call: Callable[[str], str],
        pacer: Pacer,
        retry_waits: Sequence[float] = CLI_RETRY_WAITS_S,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: float = GEMINI_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[threading.Event, float], None] = _event_wait,
    ) -> None:
        self.model = model
        self._call = call
        self._pacer = pacer
        self._retry_waits = tuple(retry_waits)
        self._sleep = sleep
        self._timeout_s = timeout_s
        self._clock = clock
        self._wait = wait

    def generate(self, prompt: str) -> str:
        waits = iter(self._retry_waits)
        while True:
            self._pacer.wait_turn()  # every attempt, retries included, goes through the pacer
            try:
                text = call_with_timeout(self._call, prompt, self._timeout_s, self._clock, self._wait)
                break
            except Exception as exc:  # SDK raises many types; all mean "this call failed"
                wait = next(waits, None) if is_retryable(exc) else None
                if wait is None:  # not retryable, or retries used up
                    raise LLMError(f"Gemini call failed: {exc}") from exc
                self._sleep(wait)
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Gemini returned an empty reply")
        return text


def make_gemini_call(model: str, api_key: str, timeout_s: float = GEMINI_TIMEOUT_S) -> Callable[[str], str]:
    from google import genai  # imported here so tests never need the SDK
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))  # ms
    config = types.GenerateContentConfig(response_mime_type="application/json")

    def call(prompt: str) -> str:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        return response.text or ""

    return call
