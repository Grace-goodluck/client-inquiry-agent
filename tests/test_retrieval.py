"""Unit tests for chunking, tokenizing and BM25 search (SCOPE.md, Knowledge: Retrieval)."""

import math

import pytest

from client_inquiry_agent.retrieval import (
    B,
    K1,
    BM25Index,
    Chunk,
    load_chunks,
    normalise,
    split_markdown,
    tokenize,
)

DOC = """---
title: Example
---

# Example title

## First section
Line one.
Line two.

## Empty heading
### Sub section
Sub text.
"""


def test_parameters_match_scope():
    assert (K1, B) == (1.5, 0.75)


def test_split_skips_front_matter_and_keeps_file_heading_text():
    chunks = split_markdown("example.md", DOC)
    assert chunks == [
        Chunk("example.md", "First section", "Line one.\nLine two."),
        Chunk("example.md", "Sub section", "Sub text."),
    ]


def test_headings_without_body_produce_no_chunk():
    headings = [c.heading for c in split_markdown("example.md", DOC)]
    assert "Example title" not in headings
    assert "Empty heading" not in headings


def test_text_before_first_heading_is_kept():
    assert split_markdown("x.md", "Intro text.\n## H\nBody") == [
        Chunk("x.md", "", "Intro text."),
        Chunk("x.md", "H", "Body"),
    ]


def test_tokenize_lowercases_splits_and_drops_stopwords():
    assert tokenize("Do YOU work on the Weekends?") == ["work", "weekend"]
    assert tokenize("Client's code, v2!") == ["client", "code", "v2"]


def test_negations_are_not_stopwords():
    assert tokenize("no refunds, not ever") == ["no", "refund", "not", "ever"]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("refunds", "refund"),
        ("weekends", "weekend"),
        ("services", "service"),
        ("payments", "payment"),
        ("classes", "class"),
        ("boxes", "box"),
        ("matches", "match"),
        ("dishes", "dish"),
        ("uses", "use"),
        ("refund", "refund"),
        ("business", "business"),
        ("process", "process"),
        ("bus", "bus"),
        ("yes", "yes"),
        ("refundable", "refundable"),
    ],
)
def test_normalise_drops_trailing_plural(word, expected):
    assert normalise(word) == expected


def test_singular_and_plural_match_each_other():
    assert tokenize("refund weekend service") == tokenize("refunds weekends services")


def test_bm25_score_matches_formula():
    chunks = [Chunk("a.md", "", "apple banana"), Chunk("b.md", "", "banana banana cherry")]
    index = BM25Index(chunks)
    n, df, tf, length, avg = 2, 1, 1, 2, 2.5
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    expected = idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg))
    [result] = index.search("apple")
    assert result.chunk.file == "a.md"
    assert result.score == pytest.approx(expected)


def test_search_ranks_best_first_and_respects_top_k():
    chunks = [
        Chunk("a.md", "Weekends", "I do not work on weekends."),
        Chunk("b.md", "Payments", "Payment is by bank transfer."),
        Chunk("c.md", "Hours", "I work Monday to Friday."),
    ]
    results = BM25Index(chunks).search("work weekends", top_k=2)
    assert [r.chunk.file for r in results] == ["a.md", "c.md"]
    assert results[0].score > results[1].score


def test_heading_is_searchable():
    chunks = [Chunk("a.md", "Refunds", "Payments are final."), Chunk("b.md", "Other", "Text.")]
    assert BM25Index(chunks).search("refunds")[0].chunk.file == "a.md"


def test_no_match_and_stopword_only_queries_return_nothing():
    index = BM25Index([Chunk("a.md", "", "apple banana")])
    assert index.search("zebra") == []
    assert index.search("the and of") == []


def test_empty_index_returns_nothing():
    assert BM25Index([]).search("anything") == []


def test_load_chunks_reads_md_files_only(tmp_path):
    (tmp_path / "one.md").write_text("## A\nalpha", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("## B\nbeta", encoding="utf-8")
    assert load_chunks(tmp_path) == [Chunk("one.md", "A", "alpha")]


def test_load_chunks_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "missing")
