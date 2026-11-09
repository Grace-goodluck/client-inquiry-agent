"""Retrieval (SCOPE.md, Knowledge): BM25 keyword search over public_brand/ chunks.

- Every .md file in public_brand/ is split at markdown headings; YAML front matter is skipped.
  Each chunk keeps its file name, heading and body text. Headings with no body of their own
  (such as a document title directly followed by a sub-heading) produce no chunk.
- Words: lowercase, split on anything that is not a letter or digit, common stopwords dropped,
  then a trailing plural is removed (see normalise()), so "refund" matches "refunds" and
  "weekend" matches "weekends". The same rule runs on queries and chunks.
- Scoring: Okapi BM25 with k1=1.5, b=0.75. The heading is indexed together with the body.
- Plain Python, no embedding calls and no extra libraries.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BRAND_DIR = Path(__file__).resolve().parents[2] / "public_brand"

K1 = 1.5
B = 0.75

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its me my of on or our so than that the their them then there these they this
    to us was we were what when where which who whom will with would you your yours s t
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


@dataclass(frozen=True)
class Chunk:
    file: str
    heading: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


_ES_ENDINGS = ("ss", "x", "z", "ch", "sh")


def normalise(word: str) -> str:
    """Drop a trailing plural: refunds -> refund, classes -> class, matches -> match.

    Words of 3 letters or fewer and words ending in "ss" (business, process) are left alone.
    "es" is dropped only after ss/x/z/ch/sh, so services -> service, not servic.
    """
    if len(word) <= 3 or word.endswith("ss"):
        return word
    if word.endswith("es") and word[:-2].endswith(_ES_ENDINGS):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [normalise(word) for word in _WORD.findall(text.lower()) if word not in STOPWORDS]


def _strip_front_matter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return lines[i + 1 :]
    return lines


def split_markdown(file_name: str, markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            chunks.append(Chunk(file=file_name, heading=heading, text=text))

    for line in _strip_front_matter(markdown.splitlines()):
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group(2)
            body = []
        else:
            body.append(line)
    flush()
    return chunks


def load_chunks(folder: Path | str = BRAND_DIR) -> list[Chunk]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Brand folder not found: {folder}")
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(split_markdown(path.name, path.read_text(encoding="utf-8")))
    return chunks


class BM25Index:
    def __init__(self, chunks: Iterable[Chunk], k1: float = K1, b: float = B) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._term_freqs = [Counter(tokenize(f"{c.heading}\n{c.text}")) for c in self.chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        n = len(self.chunks)
        self._avg_length = sum(self._lengths) / n if n else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for term, df in doc_freq.items()
        }

    def score(self, query_terms: Iterable[str], index: int) -> float:
        tf = self._term_freqs[index]
        norm = self.k1 * (1 - self.b + self.b * self._lengths[index] / self._avg_length)
        total = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq:
                total += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
        return total

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks or self._avg_length == 0 or top_k <= 0:
            return []
        terms = list(dict.fromkeys(tokenize(query)))
        scored = [(self.score(terms, i), i) for i in range(len(self.chunks))]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [SearchResult(self.chunks[i], s) for s, i in scored[:top_k]]


@lru_cache(maxsize=1)
def _default_index() -> BM25Index:
    return BM25Index(load_chunks(BRAND_DIR))


def search(query: str, top_k: int = 4) -> list[SearchResult]:
    """Search public_brand/ and return up to top_k chunks with a positive score, best first."""
    return _default_index().search(query, top_k)
