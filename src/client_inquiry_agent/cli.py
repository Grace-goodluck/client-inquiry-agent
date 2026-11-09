"""Command line: client-inquiry "message" [--file path.txt] [--brain] [--json].

Prints the category, the draft reply, the sources and the needs_grace list. Drafts only:
nothing is ever sent.

Exit codes: 0 draft produced; 1 Gemini, the Second Brain or Gemini's reply format failed;
2 bad arguments, bad message or missing configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from client_inquiry_agent.agent import (
    BrainSource,
    BrandSource,
    InquiryError,
    InquiryResult,
    KnowledgeSource,
    ReplyFormatError,
    SourceError,
    answer_inquiry,
    validate_message,
)
from client_inquiry_agent.brain_client import HttpBrain
from client_inquiry_agent.config import BRAIN_URL, ConfigError, load_config
from client_inquiry_agent.gemini_client import GeminiLLM, make_gemini_call
from client_inquiry_agent.pacing import Pacer
from client_inquiry_agent.ports import LLM, LLMError


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="client-inquiry",
        description="Draft a reply to one client message from Grace's brand files. Never sends anything.",
    )
    parser.add_argument("message", nargs="?", help="the client's message (or use --file)")
    parser.add_argument("--file", type=Path, help="read the client's message from a text file")
    parser.add_argument("--brain", action="store_true", help=f"use the Second Brain API at {BRAIN_URL}")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    if (args.message is None) == (args.file is None):
        parser.error("give either a message or --file, not both")
    return args


def _use_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def format_result(result: InquiryResult) -> str:
    lines = [f"Category: {result.category}", "", "Draft reply:", "-" * 60, result.reply, "-" * 60, "", "Sources:"]
    lines += [f"  - {s}" for s in result.sources] or ["  (none)"]
    lines += ["", "Needs Grace:"]
    lines += [f"  - {q}" for q in result.needs_grace] or ["  (nothing)"]
    notes = []
    if result.price_guard_applied:
        notes.append("price guard replaced a money amount with the quote line")
    if not result.number_check_passed:
        notes.append(
            "number check failed (" + ", ".join(result.unsupported_numbers) + "), safe fallback used"
        )
    if result.dropped_sources:
        notes.append("source ids from Gemini that matched nothing given: " + "; ".join(result.dropped_sources))
    if notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in notes]
    lines += ["", "Draft only: nothing has been sent."]
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, llm: LLM | None = None, source: KnowledgeSource | None = None) -> int:
    _use_utf8_output()
    args = _parse_args(argv)

    if args.file is not None:
        try:
            message = args.file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {args.file}: {exc}", file=sys.stderr)
            return 2
    else:
        message = args.message

    try:
        message = validate_message(message)
    except InquiryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pacer = Pacer()  # one instance: brain and Gemini calls share the spacing
    if llm is None:
        try:
            config = load_config()
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        llm = GeminiLLM(config.model, make_gemini_call(config.model, config.gemini_api_key), pacer)
    if source is None:
        source = BrainSource(HttpBrain(BRAIN_URL, pacer=pacer)) if args.brain else BrandSource()

    try:
        result = answer_inquiry(message, source, llm)
    except (LLMError, ReplyFormatError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
