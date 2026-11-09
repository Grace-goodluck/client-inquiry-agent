"""(A) query-only synonyms and (B) fallback chunks when the best BM25 score is low."""

import pytest

from client_inquiry_agent.agent import BrandSource, chunk_id
from client_inquiry_agent.config import TOP_K
from client_inquiry_agent.evaluation import load_inquiries
from client_inquiry_agent.retrieval import (
    LOW_SCORE,
    QUERY_SYNONYMS,
    BM25Index,
    Chunk,
    expand_query,
    fallback_chunks,
    load_chunks,
    retrieve_chunks,
    search,
    tokenize,
)

MESSAGES = {i.id: i.message for i in load_inquiries()}

# --- (A) synonyms: unit -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "added"),
    [
        ("Saturdays", "weekend"),
        ("Sunday", "weekend"),
        ("get started", "start"),
        ("starting soon", "start"),
        ("What does it cost?", "pricing"),
        ("your prices", "pricing"),
        ("How much do you charge?", "pricing"),
        ("Can I get a quote?", "pricing"),
        ("after delivery", "support"),
        ("after launch", "support"),
        ("once it launched", "support"),
    ],
)
def test_query_words_add_their_synonym(query, added):
    terms = expand_query(tokenize(query))
    assert added in terms


def test_original_words_are_kept_and_not_repeated():
    assert expand_query(["saturday", "sunday", "call"]) == ["saturday", "sunday", "call", "weekend"]


def test_synonym_values_are_normalised_corpus_words():
    corpus_words = {w for c in load_chunks() for w in tokenize(f"{c.heading}\n{c.text}")}
    for values in QUERY_SYNONYMS.values():
        for word in values:
            assert word in corpus_words, word


def test_synonyms_apply_to_query_only_not_chunks():
    # A chunk that says "Saturday" is NOT found by a query for "weekend":
    # chunks are indexed as written, only the query is expanded.
    index = BM25Index([Chunk("a.md", "", "Open on Saturday."), Chunk("b.md", "", "Other text.")])
    assert index.search("weekend") == []
    assert index.search("Saturday")[0].chunk.file == "a.md"


def test_search_uses_synonyms():
    index = BM25Index([Chunk("a.md", "Hours", "I do not work on weekends."), Chunk("b.md", "", "Payments.")])
    assert index.search("Can we meet on Saturday?")[0].chunk.file == "a.md"


# --- (A) synonyms: real corpus ------------------------------------------------------------


def test_e01_saturday_sunday_message_finds_weekend_chunks():
    ids = [chunk_id(r.chunk) for r in search(MESSAGES["E01-weekend"])]
    assert "2026-11-03_how-i-work.md > Communication" in ids
    assert "2026-11-03_faq.md > Do you work weekends?" in ids


def test_cost_question_finds_pricing_text():
    texts = [r.chunk.text for r in search("What would a business website cost?")]
    assert any("Pricing depends on scope" in t for t in texts)


def test_launch_question_finds_support_text():
    ids = [chunk_id(r.chunk) for r in search("Will you help after launch?")]
    assert any("support" in cid.lower() or "maintenance" in cid.lower() for cid in ids)


# --- (B) fallback: unit -------------------------------------------------------------------

CORPUS = [
    Chunk("x_how-i-work.md", "1. Free scoping call", "We start with a short call."),
    Chunk("x_how-i-work.md", "2. Written proposal and quote", "I send a written proposal."),
    Chunk("x_how-i-work.md", "Communication", "I work Monday to Friday."),
    Chunk("x_bio.md", "What I do", "I build websites, web apps and AI agents."),
    Chunk("x_bio.md", "Headline", "Full stack developer."),
    Chunk("x_faq.md", "Refunds", "Payments are non-refundable. Refunds are not offered."),
]
INDEX = BM25Index(CORPUS)
FALLBACK = [CORPUS[0], CORPUS[1], CORPUS[3]]


def test_fallback_is_process_steps_then_services_overview():
    assert fallback_chunks(CORPUS) == FALLBACK


def test_strong_match_returns_hits_only():
    results = INDEX.search("refunds", 4)
    assert results[0].score >= LOW_SCORE
    assert retrieve_chunks("refunds", 4, INDEX) == [r.chunk for r in results]


def test_no_match_returns_fallback_only():
    assert retrieve_chunks("hi, interested", 4, INDEX) == FALLBACK


def test_weak_match_keeps_hits_then_adds_fallback_without_repeats():
    weak = BM25Index(CORPUS + [Chunk("y.md", "", " ".join(["filler"] * 200) + " zebra")])
    results = weak.search("zebra", 4)
    assert 0 < results[0].score < LOW_SCORE
    chunks = retrieve_chunks("zebra", 4, weak)
    assert chunks[0].file == "y.md"
    assert chunks[1:] == FALLBACK


def test_hit_that_is_also_a_fallback_chunk_is_not_repeated():
    weak = BM25Index(CORPUS + [Chunk("y.md", "", " ".join(["filler"] * 300))])
    chunks = retrieve_chunks("proposal", 4, weak)
    assert len(chunks) == len(set(chunks))


# --- (B) fallback: real corpus ------------------------------------------------------------


def test_real_fallback_has_seven_process_steps_and_what_i_do():
    ids = [chunk_id(c) for c in fallback_chunks(load_chunks())]
    assert len(ids) == 8
    assert ids[0] == "2026-11-03_how-i-work.md > 1. Free scoping call"
    assert ids[1] == "2026-11-03_how-i-work.md > 2. Written proposal and quote"
    assert ids[6] == "2026-11-03_how-i-work.md > 7. Handover and support"
    assert ids[7] == "2026-11-02_bio.md > What I do"


def test_e10_vague_message_gets_the_fallback_chunks():
    chunks = BrandSource().retrieve(MESSAGES["E10-vague"], TOP_K)
    assert chunks == fallback_chunks(load_chunks())


@pytest.mark.parametrize("eid", [k for k in MESSAGES if k != "E10-vague"])
def test_real_questions_score_above_the_threshold(eid):
    # Keeps LOW_SCORE honest: no evaluation question except the vague one triggers the fallback.
    assert search(MESSAGES[eid])[0].score >= LOW_SCORE
