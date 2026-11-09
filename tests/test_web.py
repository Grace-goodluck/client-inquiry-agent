"""Web demo with a fake Gemini client (SCOPE.md, Web). No real API calls."""

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from client_inquiry_agent.agent import BrandSource
from client_inquiry_agent.config import WEB_RETRY_WAITS_S
from client_inquiry_agent.gemini_client import GeminiLLM
from client_inquiry_agent.pacing import Pacer
from client_inquiry_agent.ports import LLMError
from client_inquiry_agent.web import app as web
from client_inquiry_agent.web.app import (
    BUSY_MESSAGE,
    FAILED_MESSAGE,
    FORMAT_MESSAGE,
    NOT_CONFIGURED_MESSAGE,
    QUOTA_MESSAGE,
    create_app,
    port_from_env,
)
from client_inquiry_agent.web.limits import Limiter, Limits
from fakes import (
    GOOD_REPLY,
    WEEKENDS_ID,
    FakeAPIError,
    FakeClock,
    FakeLLM,
    FakeNow,
    FakeSource,
    gemini_json,
)

MESSAGE = "Hi Grace, can we have calls on weekends?"


def _client(*replies, llm=None, limits=Limits(), now=None, source=None):
    llm = llm if llm is not None else FakeLLM(list(replies))
    limiter = Limiter(limits, now or FakeNow())
    app = create_app(llm=llm, limiter=limiter, source=source or FakeSource())
    return TestClient(app), llm


def _post(client, message=MESSAGE, ip="203.0.113.7"):
    return client.post("/draft", data={"message": message}, headers={"X-Forwarded-For": ip})


# --- page ---------------------------------------------------------------------------------


def test_page_has_title_note_textarea_counter_and_examples():
    client, _ = _client()
    html = client.get("/").text
    assert "<title>Client Inquiry Agent by TechByGrace</title>" in html
    assert "<h1>Client Inquiry Agent</h1>" in html
    assert '<p class="byline">by TechByGrace</p>' in html
    assert ".byline { color: var(--gold);" in html
    assert "Client Inquiry Agent —" not in html
    assert "Use made-up messages only. Don't enter personal details." in html
    assert 'maxlength="2000"' in html
    assert 'id="counter"' in html and "0 / 2,000" in html
    assert html.count('class="example"') == 3
    for label in ("Weekend question", "Price question", "New project"):
        assert label in html
    assert 'name="viewport"' in html


def test_page_uses_emerald_and_gold_colours():
    client, _ = _client()
    html = client.get("/").text
    for colour in ("#06281F", "#0B3B2E", "#D4AF37", "#E9D8A6", "#F3EEE2", "#9FB3A8"):
        assert colour in html


def test_example_messages_are_made_up_and_money_free():
    from client_inquiry_agent.guards import find_money_amounts

    for _, text in web.EXAMPLES:
        assert find_money_amounts(text) == []
        assert len(text) <= 2000


def test_get_draft_redirects_home():
    client, _ = _client()
    response = client.get("/draft", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"


# --- drafting -----------------------------------------------------------------------------


def test_good_draft_shows_category_reply_sources_needs_grace_and_not_sent():
    client, llm = _client(gemini_json(needs_grace=["Evening calls?"]))
    response = _post(client)
    html = response.text
    assert response.status_code == 200
    assert '<span class="badge">Policy</span>' in html
    assert "I work Monday to Friday" in html
    assert WEEKENDS_ID.replace(">", "&gt;") in html
    assert "Evening calls?" in html
    assert 'id="copy"' in html
    assert "Draft only — nothing has been sent." in html
    assert len(llm.prompts) == 1


def test_message_is_kept_in_the_textarea():
    client, _ = _client(gemini_json())
    assert f">{MESSAGE}</textarea>" in _post(client).text.replace("&#39;", "'")


def test_reply_is_html_escaped():
    reply = "Hi <script>alert(document.cookie)</script>\n\nGrace, TechByGrace"  # no digits: passes the number check
    client, _ = _client(gemini_json(reply=reply))
    html = _post(client).text
    assert "<script>alert(document.cookie)</script>" not in html
    assert "&lt;script&gt;alert(document.cookie)&lt;/script&gt;" in html


def test_guard_notes_shown():
    reply = "It costs ₦250,000. Delivery in 3 weeks.\n\nGrace, TechByGrace"
    client, _ = _client(gemini_json(category="pricing", reply=reply))
    html = _post(client).text
    assert "price guard replaced a money amount" in html.lower()
    assert "safe reply was used instead" in html
    assert "₦" not in html.split('id="draft"')[1].split("</div>")[0]


def test_web_always_uses_public_brand_never_the_brain():
    app = create_app(llm=FakeLLM([]), limiter=Limiter())
    assert isinstance(app.state.source, BrandSource)


# --- input validation ---------------------------------------------------------------------


@pytest.mark.parametrize("message", ["", "   ", "\r\n\r\n"])
def test_empty_message_is_rejected_without_using_quota(message):
    client, llm = _client(gemini_json(), limits=Limits(drafts_per_visitor_per_hour=1))
    response = _post(client, message)
    assert response.status_code == 400
    assert "empty" in response.text
    assert llm.prompts == []
    assert _post(client).status_code == 200  # the quota of 1 is still available


def test_too_long_message_is_rejected():
    client, llm = _client()
    response = _post(client, "a" * 2001)
    assert response.status_code == 400
    assert "2,000" in response.text
    assert llm.prompts == []


def test_crlf_line_breaks_count_as_one_character():
    # 1,000 lines of "a" + line break = 2,000 characters in the browser, 3,000 bytes as CRLF.
    message = "a\r\n" * 999 + "a"
    client, llm = _client(gemini_json())
    assert _post(client, message).status_code == 200
    assert len(llm.prompts) == 1


def test_oversized_body_is_rejected_before_reading():
    client, llm = _client()
    response = _post(client, "é" * 9000)  # far over 32 KB once URL-encoded
    assert response.status_code == 413
    assert llm.prompts == []


def test_missing_message_field_is_empty():
    client, _ = _client()
    response = client.post("/draft", data={"other": "x"})
    assert response.status_code == 400


# --- rate limits --------------------------------------------------------------------------


def test_sixth_draft_in_an_hour_is_refused_with_friendly_message():
    client, llm = _client(*[gemini_json()] * 5)
    for _ in range(5):
        assert _post(client).status_code == 200
    response = _post(client)
    assert response.status_code == 429
    assert "You&#39;ve used your 5 drafts for this hour" in response.text
    assert int(response.headers["Retry-After"]) > 0
    assert len(llm.prompts) == 5


def test_visitor_limit_resets_after_an_hour():
    now = FakeNow()
    client, _ = _client(*[gemini_json()] * 6, now=now)
    for _ in range(5):
        _post(client)
    now.advance(hours=1, seconds=1)
    assert _post(client).status_code == 200


def test_visitors_are_counted_separately():
    client, _ = _client(*[gemini_json()] * 6)
    for _ in range(5):
        _post(client, ip="198.51.100.1")
    assert _post(client, ip="198.51.100.2").status_code == 200


def test_twenty_first_draft_of_the_day_is_refused():
    client, llm = _client(*[gemini_json()] * 20)
    for i in range(20):
        assert _post(client, ip=f"198.51.100.{i}").status_code == 200
    response = _post(client, ip="198.51.100.99")
    assert response.status_code == 429
    assert "limit of 20 drafts for today" in response.text
    assert len(llm.prompts) == 20


# --- Gemini failures ----------------------------------------------------------------------


def _llm_error(cause):
    try:
        raise cause
    except Exception as exc:
        try:
            raise LLMError(f"Gemini call failed: {exc}") from exc
        except LLMError as wrapped:
            return wrapped


@pytest.mark.parametrize(
    ("cause", "status", "message"),
    [
        (FakeAPIError(429, "RESOURCE_EXHAUSTED"), 429, QUOTA_MESSAGE),
        (TimeoutError("no reply within 60 s"), 503, BUSY_MESSAGE),
        (FakeAPIError(503, "UNAVAILABLE"), 503, BUSY_MESSAGE),
        (FakeAPIError(504, "DEADLINE_EXCEEDED"), 503, BUSY_MESSAGE),
        (httpx.ConnectError("dropped"), 503, BUSY_MESSAGE),
        (FakeAPIError(400, "INVALID_ARGUMENT"), 502, FAILED_MESSAGE),
    ],
)
def test_gemini_failures_have_friendly_messages(cause, status, message):
    client, _ = _client(_llm_error(cause))
    response = _post(client)
    assert response.status_code == status
    assert message.replace("'", "&#39;") in response.text
    assert "RESOURCE_EXHAUSTED" not in response.text and "Gemini call failed" not in response.text


def test_broken_json_has_friendly_message():
    client, _ = _client("not json")
    response = _post(client)
    assert response.status_code == 502
    assert FORMAT_MESSAGE.replace("'", "&#39;") in response.text


def test_not_configured_without_key():
    client = TestClient(create_app(llm=None, limiter=Limiter(), source=FakeSource()))
    response = _post(client)
    assert response.status_code == 503
    assert NOT_CONFIGURED_MESSAGE in response.text


def _real_llm(call):
    clock = FakeClock()
    pacer = Pacer(13.0, clock.clock, clock.sleep)
    llm = GeminiLLM("gemini-test", call, pacer, WEB_RETRY_WAITS_S, sleep=clock.sleep, clock=clock.clock, wait=clock.wait)
    return llm, clock


def test_web_retries_503_once_after_10_s():
    replies = [FakeAPIError(503, "UNAVAILABLE"), gemini_json()]
    calls = []

    def call(prompt):
        calls.append(prompt)
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    llm, clock = _real_llm(call)
    client, _ = _client(llm=llm)
    assert _post(client).status_code == 200
    assert len(calls) == 2
    assert clock.sleeps[0] == 10.0


def test_web_never_retries_429():
    calls = []

    def call(prompt):
        calls.append(prompt)
        raise FakeAPIError(429, "RESOURCE_EXHAUSTED")

    llm, clock = _real_llm(call)
    client, _ = _client(llm=llm)
    response = _post(client)
    assert response.status_code == 429
    assert QUOTA_MESSAGE.replace("'", "&#39;") in response.text
    assert len(calls) == 1
    assert clock.sleeps == []


def test_web_gives_up_after_one_retry():
    calls = []

    def call(prompt):
        calls.append(prompt)
        raise FakeAPIError(503, "UNAVAILABLE")

    llm, _ = _real_llm(call)
    client, _ = _client(llm=llm)
    assert _post(client).status_code == 503
    assert len(calls) == 2


# --- privacy: never log the message -------------------------------------------------------


def test_message_text_is_never_logged(caplog):
    secret = "Zebra-Tangerine-4417 wants weekend calls"
    caplog.set_level(logging.DEBUG)
    client, _ = _client(gemini_json(), _llm_error(FakeAPIError(429)), "not json")
    _post(client, secret)
    _post(client, secret)
    _post(client, secret)
    _post(client, "")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Zebra-Tangerine" not in logged
    assert "category=policy" in logged
    assert "seconds=" in logged
    assert "error=quota" in logged
    assert "error=format" in logged


# --- health and port ----------------------------------------------------------------------


def test_health_get_and_head():
    client, llm = _client()
    assert client.get("/health").text == "ok"
    head = client.head("/health")
    assert head.status_code == 200
    assert llm.prompts == []


def test_port_from_env():
    assert port_from_env({}) == 8000
    assert port_from_env({"PORT": "10000"}) == 10000
    with pytest.raises(ValueError):
        port_from_env({"PORT": "abc"})


def test_build_default_app_reads_environment_only(monkeypatch):
    seen = {}

    def fake_make(model, api_key, timeout_s=60.0):
        seen.update(model=model, key=api_key)
        return lambda prompt: gemini_json()

    monkeypatch.setattr(web, "make_gemini_call", fake_make)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("CLIENT_AGENT_MODEL", "gemini-test-model")
    app = web.build_default_app()
    assert seen == {"model": "gemini-test-model", "key": "test-key"}
    assert isinstance(app.state.source, BrandSource)


def test_build_default_app_without_key_still_serves_the_page(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = TestClient(web.build_default_app())
    assert client.get("/").status_code == 200
    assert client.post("/draft", data={"message": MESSAGE}).status_code == 503
