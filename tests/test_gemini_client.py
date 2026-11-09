"""Gemini client: pacing, timeout and retry rules (SCOPE.md, Engine). No real API calls."""

import httpx
import pytest

from client_inquiry_agent.config import (
    CLI_RETRY_WAITS_S,
    DEFAULT_MODEL,
    GEMINI_TIMEOUT_S,
    WEB_RETRY_WAITS_S,
    ConfigError,
    load_config,
)
from client_inquiry_agent.gemini_client import GeminiLLM, call_with_timeout, is_retryable
from client_inquiry_agent.pacing import Pacer
from client_inquiry_agent.ports import LLMError
from fakes import FakeAPIError, FakeClock


class ScriptedCall:
    """Raises or returns scripted items in order; records prompts."""

    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _llm(call, retry_waits=CLI_RETRY_WAITS_S, clock=None):
    clock = clock or FakeClock()
    pacer = Pacer(13.0, clock.clock, clock.sleep)
    llm = GeminiLLM("gemini-test", call, pacer, retry_waits, sleep=clock.sleep, clock=clock.clock, wait=clock.wait)
    return llm, clock


# --- config -------------------------------------------------------------------------------


def test_config_defaults_and_values_match_scope():
    config = load_config({"GEMINI_API_KEY": "key"})
    assert config.model == DEFAULT_MODEL == "gemini-3.6-flash"
    assert GEMINI_TIMEOUT_S == 60.0
    assert CLI_RETRY_WAITS_S == (30.0, 60.0)
    assert WEB_RETRY_WAITS_S == (10.0,)


def test_model_from_environment():
    assert load_config({"GEMINI_API_KEY": "k", "CLIENT_AGENT_MODEL": "gemini-x"}).model == "gemini-x"


def test_missing_key_is_a_config_error():
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_config({"GEMINI_API_KEY": "  "})


def test_package_never_loads_dotenv():
    from pathlib import Path

    src = Path(__file__).parents[1] / "src" / "client_inquiry_agent"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "dotenv" not in text, path.name


# --- retries ------------------------------------------------------------------------------


def test_success_first_time():
    call = ScriptedCall("ok")
    llm, clock = _llm(call)
    assert llm.generate("p") == "ok"
    assert call.prompts == ["p"]
    assert clock.sleeps == []


def test_429_is_never_retried():
    call = ScriptedCall(FakeAPIError(429, "RESOURCE_EXHAUSTED"), "never reached")
    llm, clock = _llm(call)
    with pytest.raises(LLMError, match="429"):
        llm.generate("p")
    assert len(call.prompts) == 1
    assert clock.sleeps == []


def test_429_in_message_only_is_never_retried():
    call = ScriptedCall(Exception("429 RESOURCE_EXHAUSTED"), "never reached")
    llm, _ = _llm(call)
    with pytest.raises(LLMError):
        llm.generate("p")
    assert len(call.prompts) == 1


def test_503_is_retried_after_30_s():
    call = ScriptedCall(FakeAPIError(503, "UNAVAILABLE"), "ok")
    llm, clock = _llm(call)
    assert llm.generate("p") == "ok"
    assert len(call.prompts) == 2
    assert clock.sleeps == [30.0]  # 30 s is longer than the 13 s pacing, so no extra pacer sleep


def test_503_twice_then_success_waits_30_then_60():
    call = ScriptedCall(FakeAPIError(503), FakeAPIError(503), "ok")
    llm, clock = _llm(call)
    assert llm.generate("p") == "ok"
    assert clock.sleeps == [30.0, 60.0]


def test_command_line_gives_up_after_2_retries():
    call = ScriptedCall(FakeAPIError(503), FakeAPIError(503), FakeAPIError(503), "never reached")
    llm, clock = _llm(call)
    with pytest.raises(LLMError, match="503"):
        llm.generate("p")
    assert len(call.prompts) == 3
    assert clock.sleeps == [30.0, 60.0]


def test_web_retries_once_after_10_s():
    call = ScriptedCall(FakeAPIError(503), FakeAPIError(503), "never reached")
    llm, clock = _llm(call, retry_waits=WEB_RETRY_WAITS_S)
    with pytest.raises(LLMError):
        llm.generate("p")
    assert len(call.prompts) == 2
    assert clock.sleeps[0] == 10.0
    assert sum(clock.sleeps) == pytest.approx(13.0)  # 10 s retry wait + 3 s more for the pacer


@pytest.mark.parametrize(
    "error",
    [
        FakeAPIError(504, "DEADLINE_EXCEEDED"),
        ConnectionError("reset"),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("dropped"),
        TimeoutError("timeout"),
    ],
)
def test_504_and_dropped_connections_are_retried(error):
    call = ScriptedCall(error, "ok")
    llm, _ = _llm(call)
    assert llm.generate("p") == "ok"
    assert len(call.prompts) == 2


@pytest.mark.parametrize("error", [FakeAPIError(400), FakeAPIError(403), ValueError("bad")])
def test_other_errors_are_not_retried(error):
    assert not is_retryable(error)
    call = ScriptedCall(error, "never reached")
    llm, _ = _llm(call)
    with pytest.raises(LLMError):
        llm.generate("p")
    assert len(call.prompts) == 1


def test_empty_reply_is_an_error():
    llm, _ = _llm(ScriptedCall("   "))
    with pytest.raises(LLMError, match="empty"):
        llm.generate("p")


# --- timeout ------------------------------------------------------------------------------


def test_call_with_no_reply_times_out_after_60_s():
    import threading

    never = threading.Event()
    clock = FakeClock()
    with pytest.raises(TimeoutError):
        call_with_timeout(lambda p: never.wait(5) or "late", "p", GEMINI_TIMEOUT_S, clock.clock, clock.wait)
    assert clock.waits == [60.0]
    never.set()


def test_timeout_counts_as_dropped_connection_and_is_retried():
    import threading

    release = threading.Event()
    attempts = []

    def call(prompt):
        attempts.append(prompt)
        if len(attempts) == 1:
            release.wait(5)  # first attempt hangs until the test ends
            return "late"
        return "ok"

    llm, clock = _llm(call)
    try:
        assert llm.generate("p") == "ok"
    finally:
        release.set()
    assert len(attempts) == 2
    assert clock.waits == [60.0]
    assert clock.sleeps == [30.0]


# --- pacer --------------------------------------------------------------------------------


def test_pacer_spaces_calls_13_s_apart():
    clock = FakeClock()
    pacer = Pacer(13.0, clock.clock, clock.sleep)
    pacer.wait_turn()
    clock.t += 5
    pacer.wait_turn()
    assert clock.sleeps == [8.0]


def test_pacer_is_shared_across_calls():
    clock = FakeClock()
    pacer = Pacer(13.0, clock.clock, clock.sleep)
    first = GeminiLLM("m", ScriptedCall("a"), pacer, sleep=clock.sleep, clock=clock.clock, wait=clock.wait)
    second = GeminiLLM("m", ScriptedCall("b"), pacer, sleep=clock.sleep, clock=clock.clock, wait=clock.wait)
    first.generate("p")
    second.generate("p")
    assert clock.sleeps == [13.0]
