"""Evaluation file format, checks and report rendering. No Gemini calls."""

import pytest

from client_inquiry_agent.agent import InquiryResult
from client_inquiry_agent.evaluation import (
    CaseOutcome,
    EvaluationFormatError,
    check_result,
    load_inquiries,
    parse_inquiries,
    render_results,
)

INQUIRIES = load_inquiries()
BY_ID = {i.id: i for i in INQUIRIES}


def _result(reply, category="policy", needs_grace=(), **kw):
    return InquiryResult(category, reply, (), tuple(needs_grace), False, True, **kw)


def test_ten_inquiries_with_required_cases():
    assert len(INQUIRIES) == 10
    for required in ("E01-weekend", "E02-refund", "E03-code-ownership", "E04-price", "E08-unanswerable", "E10-vague"):
        assert required in BY_ID
    assert BY_ID["E10-vague"].message == "hi, interested"
    assert BY_ID["E08-unanswerable"].needs_grace == "yes"


def test_price_case_requires_quote_line_and_bans_currency():
    case = BY_ID["E04-price"]
    assert case.categories == ("pricing",)
    assert ("Pricing depends on scope",) in case.must_include
    assert "₦" in case.must_not_include and "$" in case.must_not_include


def test_alternatives_and_items_parse():
    case = BY_ID["E01-weekend"]
    assert case.must_include == (("Monday to Friday", "weekdays", "working days"),)
    assert BY_ID["E08-unanswerable"].must_include == ()
    assert BY_ID["E09-mixed-topics"].categories == ("policy", "timeline")


def test_good_weekend_reply_passes_all_checks():
    reply = "Thanks! I work Monday to Friday and don't work weekends.\n\nGrace, TechByGrace"
    checks = check_result(BY_ID["E01-weekend"], _result(reply))
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


def test_checks_are_case_insensitive():
    reply = "I work MONDAY TO FRIDAY only."
    assert all(c.passed for c in check_result(BY_ID["E01-weekend"], _result(reply)))


def test_wrong_category_missing_fact_banned_phrase_and_money_fail():
    reply = "Sure, Saturday works! It costs ₦50,000."
    checks = {c.name: c.passed for c in check_result(BY_ID["E01-weekend"], _result(reply, category="other"))}
    assert checks["category"] is False
    assert checks["includes: Monday to Friday | weekdays | working days"] is False
    assert checks["excludes: Saturday works"] is False
    assert checks["no money amounts"] is False


def test_needs_grace_required_for_unanswerable():
    case = BY_ID["E08-unanswerable"]
    empty = check_result(case, _result("Grace will follow up.", category="other"))
    filled = check_result(case, _result("Grace will follow up.", category="other", needs_grace=["Mobile games?"]))
    assert not all(c.passed for c in empty)
    assert all(c.passed for c in filled)


def test_bad_format_is_rejected():
    with pytest.raises(EvaluationFormatError, match="missing"):
        parse_inquiries("## X\n- message: hi\n- category: other\n")
    with pytest.raises(EvaluationFormatError, match="category"):
        parse_inquiries(
            "## X\n- message: hi\n- category: sales\n- must_include: -\n- must_not_include: -\n- needs_grace: any\n"
        )


def test_report_has_table_drafts_and_empty_verdict_lines():
    good = CaseOutcome(BY_ID["E01-weekend"], _result("I work Monday to Friday.\n\nGrace, TechByGrace"))
    good.checks = check_result(good.inquiry, good.result)
    failed = CaseOutcome(BY_ID["E02-refund"], error="Gemini call failed: 429")
    report = render_results([good, failed], "2026-11-09", "gemini-3.6-flash")
    assert "| id | category expected / got | checks | needs_grace |" in report
    assert "| E01-weekend | policy / policy |" in report
    assert "| E02-refund | policy / — | ERROR | — |" in report
    assert "I work Monday to Friday.\n\nGrace, TechByGrace" in report
    assert report.count("**Grace's verdict:** \n") == 2
    assert "All checks passed: 1/2" in report
