"""(D) Source ids: E03's sources came back empty. Gemini's ids are now matched leniently,
and anything that still matches nothing is reported in dropped_sources instead of vanishing."""

import pytest

from client_inquiry_agent.agent import BrandSource, answer_inquiry, chunk_id, match_sources
from client_inquiry_agent.config import TOP_K
from client_inquiry_agent.evaluation import load_inquiries
from fakes import FakeLLM, FakeSource, WEEKENDS_ID, gemini_json

E03 = next(i.message for i in load_inquiries() if i.id == "E03-code-ownership")
FAQ_OWN = "2026-11-03_faq.md > Will I own my website and its code?"
HOW_OWN = "2026-11-03_how-i-work.md > Who owns the work"
OWN_REPLY = (
    "Hi there,\n\nAfter final payment, you own the app, its code and its accounts.\n\nGrace, TechByGrace"
)


def test_e03_was_given_both_ownership_chunks():
    ids = [chunk_id(c) for c in BrandSource().retrieve(E03, TOP_K)]
    assert FAQ_OWN in ids and HOW_OWN in ids


@pytest.mark.parametrize(
    "returned",
    [
        FAQ_OWN,
        f"[id: {FAQ_OWN}]",
        f"id: {FAQ_OWN}",
        f"[{FAQ_OWN}]",
        FAQ_OWN.rstrip("?"),
        FAQ_OWN.upper(),
        f"  {FAQ_OWN.replace(' > ', '  >  ')} ",
        f'"{FAQ_OWN}"',
        f"`{FAQ_OWN}`",
    ],
)
def test_e03_id_variants_are_matched(returned):
    llm = FakeLLM([gemini_json(reply=OWN_REPLY, sources=[returned, HOW_OWN])])
    result = answer_inquiry(E03, BrandSource(), llm)
    assert result.sources == (FAQ_OWN, HOW_OWN)
    assert result.dropped_sources == ()


def test_ids_that_match_nothing_are_reported_not_silently_lost():
    returned = ["2026-11-03_faq.md", "Who owns the work?", HOW_OWN]
    llm = FakeLLM([gemini_json(reply=OWN_REPLY, sources=returned)])
    result = answer_inquiry(E03, BrandSource(), llm)
    assert result.sources == (HOW_OWN,)
    assert result.dropped_sources == ("2026-11-03_faq.md", "Who owns the work?")
    assert result.to_dict()["dropped_sources"] == ["2026-11-03_faq.md", "Who owns the work?"]


def test_empty_sources_from_gemini_show_as_empty_with_nothing_dropped():
    # Distinguishes "Gemini listed none" from "Gemini listed ids that matched nothing".
    llm = FakeLLM([gemini_json(reply=OWN_REPLY, sources=[])])
    result = answer_inquiry(E03, BrandSource(), llm)
    assert result.sources == () and result.dropped_sources == ()


def test_match_sources_keeps_given_order_of_return_and_drops_repeats():
    given = ["a.md > One", "b.md > Two?"]
    assert match_sources(["B.MD > two", "a.md > One", "[id: a.md > one]", "zzz", "zzz"], given) == (
        ["b.md > Two?", "a.md > One"],
        ["zzz"],
    )


def test_numbered_duplicate_ids_stay_distinct():
    given = ["a.md > H", "a.md > H (2)"]
    assert match_sources(["a.md > H (2)"], given) == (["a.md > H (2)"], [])


def test_dropped_sources_kept_when_fallback_reply_used():
    reply = "I can deliver in 3 weeks.\n\nGrace, TechByGrace"
    llm = FakeLLM([gemini_json(reply=reply, sources=[WEEKENDS_ID, "made-up.md"])])
    result = answer_inquiry("When can you deliver?", FakeSource(), llm)
    assert result.sources == ()
    assert result.dropped_sources == ("made-up.md",)


def test_prompt_asks_for_ids_without_brackets_and_non_empty_sources():
    llm = FakeLLM([gemini_json()])
    answer_inquiry("Weekends?", FakeSource(), llm)
    assert 'copied exactly as written after "id: " (without the brackets)' in llm.prompts[0]
    assert "this list must not be empty" in llm.prompts[0]
