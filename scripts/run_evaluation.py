"""Run the 10 evaluation inquiries through the real agent and write evaluation/results-<date>.md.

Makes real Gemini calls (one per inquiry, at least 13 s apart through the shared pacer).
Reads GEMINI_API_KEY (and optionally CLIENT_AGENT_MODEL) from the environment; never loads .env.

Usage (from the project root):
    uv run python scripts/run_evaluation.py [--out PATH]

Exit codes: 0 every inquiry passed every check; 1 at least one failed or errored; 2 configuration error.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from client_inquiry_agent.agent import BrandSource, ReplyFormatError, SourceError, answer_inquiry
from client_inquiry_agent.config import ConfigError, load_config
from client_inquiry_agent.evaluation import (
    PROJECT_ROOT,
    CaseOutcome,
    check_result,
    load_inquiries,
    render_results,
)
from client_inquiry_agent.gemini_client import GeminiLLM, make_gemini_call
from client_inquiry_agent.pacing import Pacer
from client_inquiry_agent.ports import LLMError


def main(argv: list[str] | None = None) -> int:
    run_date = date.today().isoformat()
    parser = argparse.ArgumentParser(description="Run the evaluation inquiries through the real agent.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / f"results-{run_date}.md",
        help="where to write the results (default: evaluation/results-<today>.md)",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pacer = Pacer()  # shared: every Gemini call, retries included, starts at least 13 s after the last
    llm = GeminiLLM(config.model, make_gemini_call(config.model, config.gemini_api_key), pacer)
    source = BrandSource()

    inquiries = load_inquiries()
    outcomes: list[CaseOutcome] = []
    for n, inquiry in enumerate(inquiries, start=1):
        print(f"[{n}/{len(inquiries)}] {inquiry.id} ...", end=" ", flush=True)
        outcome = CaseOutcome(inquiry)
        try:
            outcome.result = answer_inquiry(inquiry.message, source, llm)
            outcome.checks = check_result(inquiry, outcome.result)
        except (LLMError, ReplyFormatError, SourceError) as exc:
            outcome.error = str(exc)
        outcomes.append(outcome)
        if outcome.error:
            print(f"ERROR: {outcome.error}")
        else:
            ok = sum(c.passed for c in outcome.checks)
            print(f"{outcome.result.category}, {ok}/{len(outcome.checks)} checks passed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_results(outcomes, run_date, config.model), encoding="utf-8")
    passed = sum(o.passed for o in outcomes)
    print(f"\n{passed}/{len(outcomes)} inquiries passed every check. Results written to {args.out}")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
