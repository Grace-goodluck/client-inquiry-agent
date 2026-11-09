"""answer_inquiry with a fake Gemini client (SCOPE.md, Purpose and Safety)."""

import json

import pytest

from client_inquiry_agent.agent import (
    BRAIN_SOURCE,
    CATEGORIES,
    FALLBACK_REPLY,
    BrainSource,
    BrandSource,
    InquiryError,
    ReplyFormatError,
    SourceError,
    answer_inquiry,
    build_prompt,
    ensure_sign_off,
    label_chunks,
)
from client_inquiry_agent.brain_client import BrainReply
from client_inquiry_agent.config import MAX_MESSAGE_CHARS, SIGN_OFF, TOP_K
from client_inquiry_agent.guards import QUOTE_LINE
from client_inquiry_agent.ports import LLMError
from client_inquiry_agent.retrieval import Chunk
from fakes import (
    CHUNKS,
    GOOD_REPLY,
    PAYMENT_ID,
    REFUNDS_ID,
    WEEKENDS_ID,
    FakeBrain,
    FakeLLM,
    FakeSource,
    gemini_json,
)

MESSAGE = "Hi Grace, can we have calls on weekends?"


def _run(*replies, source=None, message=MESSAGE):
    llm = FakeLLM(list(replies))
    result = answer_inquiry(message, source or FakeSource(), llm)
    return result, llm


# --- normal reply -------------------------------------------------------------------------


def test_good_reply_passes_through():
    result, llm = _run(gemini_json())
    assert result.category == "policy"
    assert result.reply == GOOD_REPLY
    assert result.sources == (WEEKENDS_ID,)
    assert result.needs_grace == ()
    assert result.price_guard_applied is False
    assert result.number_check_passed is True
    assert len(llm.prompts) == 1  # one Gemini call per message


def test_needs_grace_is_kept():
    result, _ = _run(gemini_json(needs_grace=["Can she do a call at short notice?"]))
    assert result.needs_grace == ("Can she do a call at short notice?",)


def test_retrieves_top_5_using_the_message():
    source = FakeSource()
    _run(gemini_json(), source=source)
    assert TOP_K == 5
    assert source.queries == [(MESSAGE, 5)]


def test_message_is_stripped_before_use():
    source = FakeSource()
    _run(gemini_json(), source=source, message=f"  {MESSAGE}\n")
    assert source.queries[0][0] == MESSAGE


def test_json_in_code_fence_is_accepted():
    result, _ = _run(f"```json\n{gemini_json()}\n```")
    assert result.reply == GOOD_REPLY


def test_result_to_dict_is_json_ready():
    result, _ = _run(gemini_json())
    data = json.loads(json.dumps(result.to_dict()))
    assert data["category"] == "policy"
    assert data["sources"] == [WEEKENDS_ID]
    assert data["number_check_passed"] is True


# --- prompt -------------------------------------------------------------------------------


def test_prompt_labels_chunks_and_states_the_rules():
    _, llm = _run(gemini_json())
    prompt = llm.prompts[0]
    for cid, chunk in zip((WEEKENDS_ID, REFUNDS_ID, PAYMENT_ID), CHUNKS):
        assert f"[id: {cid}]\n{chunk.text}" in prompt
    assert MESSAGE in prompt
    assert "Use only the facts above" in prompt
    assert "needs_grace" in prompt
    assert "Never state prices" in prompt
    assert "under 180 words" in prompt
    assert SIGN_OFF in prompt
    for category in CATEGORIES:
        assert category in prompt


def test_prompt_asks_vague_messages_what_they_need():
    _, llm = _run(gemini_json())
    assert (
        'If the message is vague and does not say what the client needs (for example "hi, interested"), '
        "the reply should warmly ask what they need." in llm.prompts[0]
    )


def test_prompt_with_no_chunks_says_so():
    assert "(no facts were found for this message)" in build_prompt("Hello", [])


def test_duplicate_ids_are_made_unique():
    chunks = [Chunk("a.md", "H", "one"), Chunk("a.md", "H", "two"), Chunk("b.md", "", "three")]
    assert [cid for cid, _ in label_chunks(chunks)] == ["a.md > H", "a.md > H (2)", "b.md"]


# --- price guard --------------------------------------------------------------------------


def test_invented_price_is_replaced_with_quote_line():
    reply = (
        "Hi Ada,\n\nThanks for asking! A business website starts at ₦250,000. "
        "I take a 50% deposit before work starts.\n\nGrace, TechByGrace"
    )
    result, _ = _run(gemini_json(category="pricing", reply=reply, sources=[PAYMENT_ID]))
    assert result.price_guard_applied is True
    assert "₦" not in result.reply and "250,000" not in result.reply
    assert QUOTE_LINE in result.reply
    assert "50% deposit" in result.reply  # percentages are payment terms, kept
    assert result.number_check_passed is True
    assert result.sources == (PAYMENT_ID,)


# --- number check -------------------------------------------------------------------------


def test_invented_number_uses_fallback():
    reply = "Hi Ada,\n\nI can deliver your website in 3 weeks.\n\nGrace, TechByGrace"
    result, _ = _run(gemini_json(category="timeline", reply=reply, needs_grace=["Rush jobs?"]))
    assert result.reply == FALLBACK_REPLY
    assert result.number_check_passed is False
    assert result.unsupported_numbers == ("3",)
    assert result.needs_grace == ("Rush jobs?", MESSAGE)
    assert result.sources == ()
    assert result.category == "timeline"


def test_numbers_from_the_chunks_pass():
    reply = "I reply within 24 hours and take a 50% deposit.\n\nGrace, TechByGrace"
    result, _ = _run(gemini_json(reply=reply))
    assert result.number_check_passed is True


def test_price_guard_runs_before_number_check():
    # "300" is not in the chunks, but the guard removes the price sentence first.
    reply = "A website is $300. I reply within 24 hours.\n\nGrace, TechByGrace"
    result, _ = _run(gemini_json(category="pricing", reply=reply))
    assert result.price_guard_applied is True
    assert result.number_check_passed is True


# --- sources ------------------------------------------------------------------------------


def test_source_not_given_is_dropped():
    sources = [WEEKENDS_ID, "2026-11-03_portfolio-projects.md > Eden & Sunset", "made-up.md", WEEKENDS_ID]
    result, _ = _run(gemini_json(sources=sources))
    assert result.sources == (WEEKENDS_ID,)


# --- broken replies -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Sure! Here is my reply: Hi Ada...",
        '{"category": "policy", "reply": "Hi"',  # cut off
        "[]",
        json.dumps({"category": "policy", "reply": "Hi", "sources": []}),  # needs_grace missing
        json.dumps({"category": "policy", "reply": "", "sources": [], "needs_grace": []}),
        json.dumps({"category": "policy", "reply": "Hi", "sources": "faq.md", "needs_grace": []}),
        json.dumps({"category": "policy", "reply": "Hi", "sources": [], "needs_grace": [1, 2]}),
    ],
)
def test_broken_json_is_a_clear_error(text):
    with pytest.raises(ReplyFormatError):
        _run(text)


@pytest.mark.parametrize("category", ["refunds", "Policy", "", None, "pricing_question"])
def test_bad_category_is_an_error(category):
    with pytest.raises(ReplyFormatError, match="category"):
        _run(gemini_json(category=category))


def test_gemini_failure_propagates():
    with pytest.raises(LLMError):
        _run(LLMError("Gemini call failed: 429"))


# --- sign-off -----------------------------------------------------------------------------


def test_missing_sign_off_is_added():
    reply = "Hi Ada,\n\nI don't work on weekends.\n\nWarm regards,"
    result, _ = _run(gemini_json(reply=reply))
    assert result.reply == f"{reply}\n\n{SIGN_OFF}"


def test_sign_off_not_doubled():
    assert ensure_sign_off(f"Hi.\n\n{SIGN_OFF}.") == f"Hi.\n\n{SIGN_OFF}."
    assert ensure_sign_off(FALLBACK_REPLY) == FALLBACK_REPLY


def test_fallback_ends_with_sign_off():
    assert FALLBACK_REPLY.endswith(SIGN_OFF)


# --- message validation -------------------------------------------------------------------


@pytest.mark.parametrize("message", ["", "   ", "\n\t", None])
def test_empty_message_rejected(message):
    with pytest.raises(InquiryError, match="empty"):
        answer_inquiry(message, FakeSource(), FakeLLM([]))


def test_too_long_message_rejected():
    with pytest.raises(InquiryError, match="2,000"):
        answer_inquiry("a" * (MAX_MESSAGE_CHARS + 1), FakeSource(), FakeLLM([]))


def test_message_at_the_limit_is_accepted():
    result = answer_inquiry("a" * MAX_MESSAGE_CHARS, FakeSource(), FakeLLM([gemini_json()]))
    assert result.category == "policy"


# --- knowledge sources --------------------------------------------------------------------


def test_brand_source_uses_public_brand():
    chunks = BrandSource().retrieve("Do you offer refunds?", 4)
    assert 0 < len(chunks) <= 4
    assert any("non-refundable" in c.text for c in chunks)


def test_brain_answer_becomes_one_chunk():
    brain = FakeBrain(BrainReply(status="answered", answer="Grace does not work weekends.", sources=("x.md",)))
    source = BrainSource(brain)
    result, llm = _run(gemini_json(sources=[BRAIN_SOURCE], reply=f"No weekends, sorry!\n\n{SIGN_OFF}"), source=source)
    assert brain.questions == [MESSAGE]
    assert "[id: second-brain]\nGrace does not work weekends." in llm.prompts[0]
    assert result.sources == (BRAIN_SOURCE,)


def test_brain_not_found_gives_no_chunks():
    source = BrainSource(FakeBrain(BrainReply(status="not_found", note="nothing")))
    assert source.retrieve("anything", 4) == []


def test_brain_error_is_a_source_error():
    source = BrainSource(FakeBrain(BrainReply(status="error", error="could not reach the brain")))
    with pytest.raises(SourceError, match="could not reach the brain"):
        source.retrieve("anything", 4)
