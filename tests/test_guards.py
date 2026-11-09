"""Unit tests for the price guard and the number check (SCOPE.md, Safety)."""

import pytest

from client_inquiry_agent.guards import (
    QUOTE_LINE,
    apply_price_guard,
    check_numbers,
    find_money_amounts,
)
from client_inquiry_agent.retrieval import Chunk, SearchResult

# --- find_money_amounts -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It costs ₦500.", "₦500"),
        ("About $1,200 in total.", "$1,200"),
        ("Budget: NGN 50,000", "NGN 50,000"),
        ("Budget: NGN50,000", "NGN50,000"),
        ("Roughly USD 300.", "USD 300"),
        ("Roughly usd 300.", "usd 300"),
        ("It is 500 naira.", "500 naira"),
        ("It is 200 dollars.", "200 dollars"),
        ("Just 1 dollar.", "1 dollar"),
        ("Around 50k for this.", "50k"),
        ("Around 50K for this.", "50K"),
        ("Maybe 1.5m overall.", "1.5m"),
        ("Maybe 2bn overall.", "2bn"),
        ("Between ₦100k–₦200k.", "₦100k–₦200k"),
        ("Between ₦100k-200k.", "₦100k-200k"),
        ("From $300 to $500.", "$300 to $500"),
        ("From 100k to 200k.", "100k to 200k"),
        ("From 100–200 dollars.", "100–200 dollars"),
        ("A total of 150,000.", "150,000"),
        ("A total of 1,250,000.50 today", "1,250,000.50"),
        ("Pay N250,000 now.", "N250,000"),
        ("About $1.5 million.", "$1.5 million"),
        ("About ₦ 75,000 monthly.", "₦ 75,000"),
        ("£400 or so", "£400"),
        ("€400 or so", "€400"),
        ("150,000, as agreed", "150,000"),
    ],
)
def test_finds_money_amounts(text, expected):
    assert find_money_amounts(text) == [expected]


def test_finds_several_amounts():
    assert find_money_amounts("₦250,000 / $300 per site") == ["₦250,000", "$300"]


@pytest.mark.parametrize(
    "text",
    [
        "50% upfront and 50% on delivery.",
        "50% to start, 25% at a midpoint review and 25% on delivery.",
        "Typical timeline: 2 weeks.",
        "Support lasts 1 to 2 months.",
        "It takes 4 to 8 weeks.",
        "Prices are in Nigerian naira for clients in Nigeria and in US dollars abroad.",
        "naira",
        "dollars",
        "$",
        "I reply within 24 hours.",
        "West Africa Time, UTC+1.",
        "Built test-first with 269 automated tests.",
        "Improved from 10/12 to 11/12.",
        "BSc Economics (2016).",
        "Across Nigeria's 36 states.",
        "A 5km walk at 9am.",
        "Up to 5 pages and 2 rounds of revisions.",
        "In 2 minutes.",
        "Call me on Monday.",
        "Increase of 1,000% overall.",
    ],
)
def test_does_not_match_non_money(text):
    assert find_money_amounts(text) == []


# --- apply_price_guard --------------------------------------------------------------------


def test_replaces_sentence_with_amount():
    text = "Thanks for reaching out. A website costs ₦250,000. It takes 1 to 2 weeks."
    assert apply_price_guard(text) == (
        f"Thanks for reaching out. {QUOTE_LINE} It takes 1 to 2 weeks.",
        True,
    )


def test_quote_line_appears_only_once():
    text = "A site is ₦250,000. An app is $1,500. Thanks!"
    guarded, replaced = apply_price_guard(text)
    assert replaced
    assert guarded == f"{QUOTE_LINE} Thanks!"
    assert guarded.count(QUOTE_LINE) == 1


def test_existing_quote_line_is_not_repeated():
    text = f"{QUOTE_LINE} Most sites are around 300k."
    guarded, replaced = apply_price_guard(text)
    assert replaced
    assert guarded == QUOTE_LINE


def test_no_amount_leaves_text_unchanged():
    text = "Hi Ada,\n\nWe take 50% upfront.\n\nGrace, TechByGrace\n"
    assert apply_price_guard(text) == (text, False)


def test_keeps_line_structure():
    text = "Hi Ada,\n\nThe price is ₦500k.\n\nGrace, TechByGrace"
    assert apply_price_guard(text) == (
        f"Hi Ada,\n\n{QUOTE_LINE}\n\nGrace, TechByGrace",
        True,
    )


def test_dropped_sentence_keeps_paragraph_break():
    text = "Hi. It is ₦5,000. Also $60.\n\nGrace, TechByGrace"
    assert apply_price_guard(text) == (f"Hi. {QUOTE_LINE}\n\nGrace, TechByGrace", True)


def test_amount_on_line_without_full_stop():
    text = "Price: 50k\nThanks"
    assert apply_price_guard(text) == (f"{QUOTE_LINE}\nThanks", True)


def test_guarded_text_has_no_amounts_left():
    text = "Websites start at ₦250,000 / $300. Apps from 1.2m. Data work from 200 dollars."
    guarded, _ = apply_price_guard(text)
    assert find_money_amounts(guarded) == []


# --- check_numbers ------------------------------------------------------------------------

CHUNKS = [
    Chunk("how-i-work.md", "Communication", "I reply within 24 hours on working days."),
    Chunk("faq.md", "How do I start?", "Within 3 working days I send a proposal. 50% deposit."),
    "Typical timeline: 1 to 2 weeks. Version 1.5 of 1200 items. Phone: +2348073478437",
]


def test_supported_numbers_pass():
    draft = "I reply within 24 hours, send a proposal within 3 working days, and take a 50% deposit."
    assert check_numbers(draft, CHUNKS) == []


def test_unsupported_numbers_are_returned_as_written_once():
    draft = "It takes 3 weeks, maybe 6 weeks, or 6 at most. Delivery in 2 weeks."
    assert check_numbers(draft, CHUNKS) == ["6"]


def test_unsupported_number_found():
    assert check_numbers("Support lasts 5 months.", CHUNKS) == ["5"]


def test_list_markers_at_line_start_are_ignored():
    draft = "Next steps:\n7. Book a call\n  8) Get a proposal"
    assert check_numbers(draft, CHUNKS) == []


def test_numbers_mid_line_after_marker_still_checked():
    assert check_numbers("7. Expect 9 updates", CHUNKS) == ["9"]


def test_comma_and_plain_forms_match():
    assert check_numbers("We handle 1,200 items.", CHUNKS) == []


def test_decimals_must_match_value():
    assert check_numbers("Version 1.50 is out.", CHUNKS) == []
    assert check_numbers("Version 1.6 is out.", CHUNKS) == ["1.6"]


def test_digits_attached_to_letters_are_checked():
    assert check_numbers("Call at 4pm.", CHUNKS) == ["4"]


def test_long_numbers_compare_exactly():
    assert check_numbers("Call +2348073478437.", CHUNKS) == []
    assert check_numbers("Call +2348073478438.", CHUNKS) == ["2348073478438"]


def test_numbers_in_chunk_headings_count():
    chunks = [Chunk("how-i-work.md", "7. Handover and support", "Free support for bugs.")]
    assert check_numbers("Step 7 is handover.", chunks) == []


def test_accepts_search_results():
    results = [SearchResult(Chunk("faq.md", "Weekends", "Monday to Friday, 24 hours."), 1.0)]
    assert check_numbers("Replies within 24 hours.", results) == []


def test_numbers_written_as_words_are_not_checked_in_v1():
    assert check_numbers("It takes three weeks.", CHUNKS) == []


def test_no_chunks_rejects_every_number():
    assert check_numbers("2 weeks and 50% upfront.", []) == ["2", "50"]
