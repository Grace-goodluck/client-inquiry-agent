"""Tests against the real public_brand/ corpus."""

import pytest

from client_inquiry_agent.evaluation import load_inquiries
from client_inquiry_agent.guards import find_money_amounts
from client_inquiry_agent.retrieval import BRAND_DIR, load_chunks, search

BRAND_FILES = sorted(BRAND_DIR.glob("*.md"))


def test_corpus_has_the_five_brand_files_and_no_lessons_note():
    names = [p.name for p in BRAND_FILES]
    assert len(names) == 5
    assert not any("lessons" in name for name in names)


def test_corpus_loads_into_chunks():
    chunks = load_chunks()
    assert len(chunks) > 20
    assert all(c.file and c.text for c in chunks)


@pytest.mark.parametrize("path", BRAND_FILES, ids=lambda p: p.name)
def test_no_money_amounts_anywhere_in_public_brand(path):
    """Regression: a price must never sneak back into the public files."""
    assert find_money_amounts(path.read_text(encoding="utf-8")) == []


def _hits(query):
    return [(r.chunk.file, r.chunk.heading, r.chunk.text) for r in search(query)]


def test_weekend_question_finds_how_i_work_chunk():
    hits = _hits("Do you work on weekends?")
    assert any(
        file == "2026-11-03_how-i-work.md" and heading == "Communication"
        for file, heading, _ in hits
    )


def test_refund_question_finds_no_refund_text():
    hits = _hits("Do you offer refunds?")
    assert any("non-refundable" in text for _, _, text in hits)


def test_singular_refund_question_finds_no_refund_text():
    hits = _hits("Can I get a refund?")
    assert any("non-refundable" in text for _, _, text in hits)


def test_singular_weekend_question_finds_weekends_text():
    hits = _hits("Do you work on the weekend?")
    assert any("weekends" in text for _, _, text in hits)


def test_code_ownership_question_finds_ownership_text():
    hits = _hits("Who owns the code?")
    assert any("After final payment, the client owns" in text for _, _, text in hits)


def test_e05_get_started_message_finds_the_proposal_chunk():
    message = next(i.message for i in load_inquiries() if i.id == "E05-new-project")
    ids = [f"{r.chunk.file} > {r.chunk.heading}" for r in search(message)]
    assert "2026-11-03_how-i-work.md > 2. Written proposal and quote" in ids


def test_search_returns_at_most_five_by_default_best_first():
    results = search("website timeline payment")
    assert 0 < len(results) <= 5
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
