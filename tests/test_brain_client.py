"""Second Brain /ask parsing (same contract as job_application_agent). No network."""

import io
import json
import urllib.error

from client_inquiry_agent.brain_client import HttpBrain, parse_ask_response


def test_answered():
    reply = parse_ask_response(200, json.dumps({"status": "answered", "answer": " Yes. ", "sources": ["a.md", " "]}))
    assert (reply.status, reply.answer, reply.sources) == ("answered", "Yes.", ("a.md",))


def test_not_found():
    reply = parse_ask_response(200, json.dumps({"status": "not_found", "note": "no match"}))
    assert (reply.status, reply.note) == ("not_found", "no match")


def test_errors_never_raise():
    assert parse_ask_response(200, "not json").status == "error"
    assert parse_ask_response(200, json.dumps({"status": "answered", "answer": ""})).status == "error"
    assert parse_ask_response(502, json.dumps({"status": "invalid", "reason": "bad"})).status == "error"
    assert parse_ask_response(503, json.dumps({"status": "error", "detail": "down"})).status == "error"
    assert parse_ask_response(422, json.dumps({"detail": []})).status == "error"


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_brain_posts_question_to_ask():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return _Response(json.dumps({"status": "answered", "answer": "Hi", "sources": []}).encode())

    reply = HttpBrain("http://127.0.0.1:8000/", opener=opener).ask("Weekends?")
    assert seen == {"url": "http://127.0.0.1:8000/ask", "body": {"question": "Weekends?"}}
    assert reply.answer == "Hi"


def test_http_brain_unreachable_is_an_error_reply():
    def opener(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    reply = HttpBrain("http://127.0.0.1:8000", opener=opener).ask("q")
    assert reply.status == "error"
    assert "could not reach the brain" in reply.error
