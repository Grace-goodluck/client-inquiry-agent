"""Safety guards (SCOPE.md, Safety): the price guard and the number check.

Price guard
- find_money_amounts() finds: currency-prefixed amounts (₦500, $1,200, NGN 50,000, USD 300,
  N250,000, $1.5 million), amounts followed by a currency word (500 naira, 200 dollars),
  shorthand amounts (50k, 1.5m, 2bn), ranges of any of these (₦100k–₦200k, 100k to 200k),
  and bare thousands-comma numbers (150,000).
- It does not match percentages (50% upfront), durations (2 weeks, 1 to 2 months), or the
  words "naira"/"dollars" with no number.
- apply_price_guard() replaces the first sentence containing an amount with QUOTE_LINE and
  removes any later ones, so the quote line never appears twice.

Number check
- check_numbers() returns every number written in digits in the draft that does not appear
  in the retrieved chunks. List markers at the start of a line ("1.", "2)") are ignored.
  "1,200" and "1200" are the same number; so are "1.5" and "1.50".
- Known limitation (v1): numbers written as words ("three weeks") are not checked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import Decimal

QUOTE_LINE = "Pricing depends on scope. I'll send you a quote after a short call."

# --- price guard --------------------------------------------------------------------------

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_SCALE = r"(?:\s?(?:k|m|bn|thousand|million|billion)\b)"
_SHORT_SCALE = r"(?:k|m|bn)\b"
_PREFIX = r"(?:₦|\$|£|€|(?:NGN|USD)\s?|(?-i:\bN)(?=\d))"
_CURRENCY_WORD = r"(?:naira|dollars?|ngn|usd)\b"
_SEP = r"\s?(?:-|–|—|to)\s?"
_START = r"(?<![\w.,])"

_PREFIXED = rf"{_PREFIX}\s?{_NUM}{_SCALE}?(?:{_SEP}{_PREFIX}?\s?{_NUM}{_SCALE}?)?"
_WORDED = rf"{_START}{_NUM}{_SCALE}?(?:{_SEP}{_NUM}{_SCALE}?)?\s?{_CURRENCY_WORD}"
_SHORTHAND = rf"{_START}\d+(?:\.\d+)?{_SHORT_SCALE}(?:{_SEP}\d+(?:\.\d+)?{_SHORT_SCALE})?"
_THOUSANDS = rf"{_START}\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?(?!\d)(?!\s?%)"

_MONEY = re.compile(f"{_PREFIXED}|{_WORDED}|{_SHORTHAND}|{_THOUSANDS}", re.IGNORECASE)

# A sentence ends at . ! or ? followed by spaces, or at a line break.
_SENTENCE_BREAK = re.compile(r"((?<=[.!?])[ \t]+|[ \t]*\n\s*)")


def find_money_amounts(text: str) -> list[str]:
    return [match.group(0) for match in _MONEY.finditer(text)]


def apply_price_guard(text: str) -> tuple[str, bool]:
    """Return (guarded text, whether any sentence was replaced)."""
    parts = _SENTENCE_BREAK.split(text)
    sentences = parts[0::2]
    separators = parts[1::2] + [""]

    quote_present = QUOTE_LINE in text
    replaced = False
    out: list[list[str]] = []  # [sentence, separator after it]
    for sentence, separator in zip(sentences, separators):
        if not find_money_amounts(sentence):
            out.append([sentence, separator])
            continue
        replaced = True
        if not quote_present:
            out.append([QUOTE_LINE, separator])
            quote_present = True
        elif out and "\n" in separator and "\n" not in out[-1][1]:
            out[-1][1] = separator  # keep the paragraph break of the dropped sentence

    result = "".join(sentence + separator for sentence, separator in out)
    trailing = text[len(text.rstrip()) :]
    return result.rstrip() + trailing, replaced


# --- number check -------------------------------------------------------------------------

_DIGITS = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^[ \t]*\d+[.)](?=[ \t])", re.MULTILINE)


def _value(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def _chunk_text(chunk: object) -> str:
    if isinstance(chunk, str):
        return chunk
    inner = getattr(chunk, "chunk", chunk)  # SearchResult -> Chunk
    return f"{getattr(inner, 'heading', '')}\n{getattr(inner, 'text', '')}"


def check_numbers(draft: str, chunks: Iterable[object]) -> list[str]:
    """Return the numbers in the draft (as written, without repeats) not found in the chunks.

    chunks may be Chunk objects, SearchResult objects or plain strings.
    """
    allowed = {_value(raw) for c in chunks for raw in _DIGITS.findall(_chunk_text(c))}
    unsupported: list[str] = []
    for raw in _DIGITS.findall(_LIST_MARKER.sub("", draft)):
        if _value(raw) not in allowed and raw not in unsupported:
            unsupported.append(raw)
    return unsupported
