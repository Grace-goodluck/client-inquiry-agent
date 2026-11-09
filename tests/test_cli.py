"""Command line with a fake Gemini client and a fake knowledge source."""

import json

import pytest

from client_inquiry_agent import cli
from client_inquiry_agent.ports import LLMError
from fakes import GOOD_REPLY, WEEKENDS_ID, FakeLLM, FakeSource, gemini_json


def _main(argv, *replies, source=None):
    return cli.main(argv, llm=FakeLLM(list(replies)), source=source or FakeSource())


def test_prints_category_draft_sources_and_needs_grace(capsys):
    code = _main(["Do you work weekends?"], gemini_json(needs_grace=["Evening calls?"]))
    out = capsys.readouterr().out
    assert code == 0
    assert "Category: policy" in out
    assert GOOD_REPLY in out
    assert f"  - {WEEKENDS_ID}" in out
    assert "Needs Grace:\n  - Evening calls?" in out
    assert "nothing has been sent" in out


def test_json_output(capsys):
    assert _main(["Do you work weekends?", "--json"], gemini_json()) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["category"] == "policy"
    assert data["reply"] == GOOD_REPLY
    assert data["sources"] == [WEEKENDS_ID]
    assert data["needs_grace"] == []
    assert data["price_guard_applied"] is False
    assert data["number_check_passed"] is True


def test_message_from_file(tmp_path, capsys):
    path = tmp_path / "message.txt"
    path.write_text("Do you work weekends?\n", encoding="utf-8")
    source = FakeSource()
    assert _main(["--file", str(path)], gemini_json(), source=source) == 0
    assert source.queries[0][0] == "Do you work weekends?"


def test_notes_shown_when_guards_fire(capsys):
    reply = "It costs ₦250,000. Delivery in 3 weeks.\n\nGrace, TechByGrace"
    assert _main(["How much?"], gemini_json(category="pricing", reply=reply)) == 0
    out = capsys.readouterr().out
    assert "Notes:" in out
    assert "number check failed (3), safe fallback used" in out


def test_empty_message_exit_2(capsys):
    assert _main(["   "]) == 2
    assert "empty" in capsys.readouterr().err


def test_too_long_message_exit_2(capsys):
    assert _main(["a" * 2001]) == 2
    assert "2,000" in capsys.readouterr().err


def test_message_and_file_together_is_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _main(["hi", "--file", str(tmp_path / "x.txt")])
    assert exc.value.code == 2


def test_no_message_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        _main([])
    assert exc.value.code == 2


def test_missing_file_exit_2(tmp_path, capsys):
    assert _main(["--file", str(tmp_path / "missing.txt")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_gemini_failure_exit_1(capsys):
    assert _main(["Weekends?"], LLMError("Gemini call failed: 429")) == 1
    assert "429" in capsys.readouterr().err


def test_broken_json_exit_1(capsys):
    assert _main(["Weekends?"], "not json") == 1
    assert "valid JSON" in capsys.readouterr().err


def test_missing_api_key_exit_2(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cli.main(["Weekends?"], source=FakeSource()) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
