"""Evaluation (SCOPE.md, Evaluation): parse evaluation/inquiries.md, check results, render the report.

The runner is scripts/run_evaluation.py; this module holds the logic so it can be tested
without calling Gemini.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from client_inquiry_agent.agent import CATEGORIES, InquiryResult
from client_inquiry_agent.guards import find_money_amounts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INQUIRIES_FILE = PROJECT_ROOT / "evaluation" / "inquiries.md"

_FIELD = re.compile(r"^-\s+(message|category|must_include|must_not_include|needs_grace):\s*(.*)$")


class EvaluationFormatError(ValueError):
    """inquiries.md is not in the expected format."""


@dataclass(frozen=True)
class Inquiry:
    id: str
    message: str
    categories: tuple[str, ...]  # any of these is accepted
    must_include: tuple[tuple[str, ...], ...]  # each item: alternatives, any one is enough
    must_not_include: tuple[str, ...]
    needs_grace: str  # "yes", "no" or "any"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseOutcome:
    inquiry: Inquiry
    result: InquiryResult | None = None
    error: str | None = None
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)


def _alternatives(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _items(text: str) -> list[str]:
    text = text.strip()
    if text in ("", "-"):
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def parse_inquiries(markdown: str) -> list[Inquiry]:
    sections: list[tuple[str, dict[str, str]]] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), {}))
            continue
        match = _FIELD.match(line.strip())
        if match and sections:
            sections[-1][1][match.group(1)] = match.group(2).strip()

    inquiries: list[Inquiry] = []
    for case_id, fields in sections:
        missing = [k for k in ("message", "category", "must_include", "must_not_include", "needs_grace") if k not in fields]
        if missing:
            raise EvaluationFormatError(f"{case_id}: missing {', '.join(missing)}")
        categories = _alternatives(fields["category"])
        unknown = [c for c in categories if c not in CATEGORIES]
        if not categories or unknown:
            raise EvaluationFormatError(f"{case_id}: unknown category {unknown or fields['category']!r}")
        needs_grace = fields["needs_grace"].lower()
        if needs_grace not in ("yes", "no", "any"):
            raise EvaluationFormatError(f"{case_id}: needs_grace must be yes, no or any")
        if not fields["message"]:
            raise EvaluationFormatError(f"{case_id}: empty message")
        inquiries.append(
            Inquiry(
                id=case_id,
                message=fields["message"],
                categories=categories,
                must_include=tuple(_alternatives(item) for item in _items(fields["must_include"])),
                must_not_include=tuple(_items(fields["must_not_include"])),
                needs_grace=needs_grace,
            )
        )
    ids = [i.id for i in inquiries]
    if len(set(ids)) != len(ids):
        raise EvaluationFormatError("duplicate inquiry ids")
    return inquiries


def load_inquiries(path: Path = INQUIRIES_FILE) -> list[Inquiry]:
    return parse_inquiries(Path(path).read_text(encoding="utf-8"))


def results_path(run_date: str, only: str | None = None) -> Path:
    """evaluation/results-<date>.md, or results-<date>-only-<ids>.md for a partial run."""
    name = f"results-{run_date}"
    if only:
        ids = "-".join(item.strip() for item in only.split(",") if item.strip())
        name += "-only-" + re.sub(r"[^A-Za-z0-9-]+", "_", ids)
    return PROJECT_ROOT / "evaluation" / f"{name}.md"


def select_inquiries(inquiries: list[Inquiry], only: str | None) -> list[Inquiry]:
    """Filter by a comma-separated list such as "E01,E05" or "E01-weekend".

    Each item matches an id exactly or as its prefix before "-" ("E01" matches "E01-weekend").
    Unknown items raise EvaluationFormatError. Order follows inquiries.md.
    """
    if not only:
        return list(inquiries)
    wanted = [item.strip() for item in only.split(",") if item.strip()]
    if not wanted:
        raise EvaluationFormatError("--only is empty")

    def matches(item: str, inquiry: Inquiry) -> bool:
        return inquiry.id == item or inquiry.id.startswith(item + "-")

    unknown = [item for item in wanted if not any(matches(item, i) for i in inquiries)]
    if unknown:
        raise EvaluationFormatError(f"unknown inquiry id(s): {', '.join(unknown)}")
    return [i for i in inquiries if any(matches(item, i) for item in wanted)]


def check_result(inquiry: Inquiry, result: InquiryResult) -> list[Check]:
    reply = result.reply.lower()
    checks = [
        Check(
            "category",
            result.category in inquiry.categories,
            f"expected {' or '.join(inquiry.categories)}, got {result.category}",
        )
    ]
    for alternatives in inquiry.must_include:
        found = any(alt.lower() in reply for alt in alternatives)
        checks.append(Check(f"includes: {' | '.join(alternatives)}", found))
    for phrase in inquiry.must_not_include:
        checks.append(Check(f"excludes: {phrase}", phrase.lower() not in reply))
    amounts = find_money_amounts(result.reply)
    checks.append(Check("no money amounts", not amounts, ", ".join(amounts)))
    if inquiry.needs_grace == "yes":
        checks.append(Check("needs_grace not empty", bool(result.needs_grace)))
    elif inquiry.needs_grace == "no":
        checks.append(Check("needs_grace empty", not result.needs_grace))
    return checks


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_results(outcomes: list[CaseOutcome], run_date: str, model: str) -> str:
    passed = sum(o.passed for o in outcomes)
    lines = [
        f"# Evaluation results — {run_date}",
        "",
        f"Model: `{model}` · Inquiries: {len(outcomes)} · All checks passed: {passed}/{len(outcomes)}",
        "",
        "Automatic checks: category, must_include, must_not_include, no money amounts, and needs_grace where required.",
        "",
        "| id | category expected / got | checks | needs_grace |",
        "|---|---|---|---|",
    ]
    for o in outcomes:
        expected = " or ".join(o.inquiry.categories)
        if o.error is not None:
            lines.append(f"| {o.inquiry.id} | {expected} / — | ERROR | — |")
            continue
        assert o.result is not None
        ok = sum(c.passed for c in o.checks)
        failed = [c.name for c in o.checks if not c.passed]
        checks = f"{ok}/{len(o.checks)} passed" + (f"; failed: {'; '.join(failed)}" if failed else "")
        needs = "; ".join(o.result.needs_grace) or "—"
        lines.append(f"| {o.inquiry.id} | {expected} / {o.result.category} | {_cell(checks)} | {_cell(needs)} |")

    for o in outcomes:
        lines += ["", "---", "", f"## {o.inquiry.id}", "", f"**Message:** {o.inquiry.message}", ""]
        if o.error is not None:
            lines += [f"**Error:** {o.error}", ""]
        else:
            assert o.result is not None
            r = o.result
            lines.append(f"**Category:** expected {' or '.join(o.inquiry.categories)}, got {r.category}")
            lines.append(f"**Sources:** {', '.join(r.sources) or '—'}")
            if r.dropped_sources:
                lines.append(f"**Dropped source ids (matched nothing given):** {'; '.join(r.dropped_sources)}")
            lines.append(f"**Needs Grace:** {'; '.join(r.needs_grace) or '—'}")
            flags = [
                f"price guard applied: {'yes' if r.price_guard_applied else 'no'}",
                f"number check passed: {'yes' if r.number_check_passed else 'no'}",
            ]
            if r.unsupported_numbers:
                flags.append(f"unsupported numbers: {', '.join(r.unsupported_numbers)}")
            lines.append(f"**Guards:** {'; '.join(flags)}")
            failed = [c for c in o.checks if not c.passed]
            if failed:
                lines.append("**Failed checks:** " + "; ".join(f"{c.name}" + (f" ({c.detail})" if c.detail else "") for c in failed))
            lines += ["", "**Draft:**", "", "```text", r.reply, "```", ""]
        lines += ["**Grace's verdict:** ", ""]
    return "\n".join(lines).rstrip("\n") + "\n"
