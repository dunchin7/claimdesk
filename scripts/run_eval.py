"""Week-4 eval runner CLI.

Loads the golden set, runs the full pipeline, writes a Markdown + JSON report,
and optionally enforces regression thresholds against a baseline.

Usage:
    # Run eval and write a fresh report card
    uv run python scripts/run_eval.py

    # Run eval and check against a baseline; non-zero exit if any metric
    # drops by more than 2pp without --override
    uv run python scripts/run_eval.py --baseline backend/app/evals/reports/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.evals.reports import write_report  # noqa: E402
from app.evals.runners import load_claims_jsonl, run_eval  # noqa: E402

DEFAULT_REPORTS_DIR = ROOT / "backend/app/evals/reports"
DEFAULT_CLAIMS = ROOT / "data/synthetic/claims.jsonl"

# Metrics whose regression we treat as a build-failure if they drop by
# more than `threshold_pp` without explicit override.
REGRESSION_GUARDED = (
    "decision_accuracy",
    "citation_verbatim_rate",
    "citation_precision_on_correct",
    "faithfulness_rate",
)


def detect_regressions(
    current: dict, baseline: dict, threshold_pp: float
) -> list[str]:
    """Return human-readable regression messages. Empty = no regressions."""
    msgs: list[str] = []
    for metric in REGRESSION_GUARDED:
        cur_v = current.get(metric)
        base_v = baseline.get(metric)
        if cur_v is None or base_v is None:
            continue
        delta_pp = (cur_v - base_v) * 100
        if delta_pp < -threshold_pp:
            msgs.append(
                f"{metric}: {base_v:.1%} → {cur_v:.1%} "
                f"({delta_pp:+.1f}pp, threshold -{threshold_pp:.1f}pp)"
            )
    return msgs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--threshold-pp",
        type=float,
        default=2.0,
        help="Allowed drop (in percentage points) before a regression is flagged.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Run the eval but exit 0 even if regressions are found.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Async concurrency for the eval. Default lowered from 8 to 4 "
        "post-Phase-3 — concurrency=8 burst-tripped API rate limits and "
        "produced 10/200 errors despite the retry wrapper. 4 leaves enough "
        "headroom for the wrapper to recover transient hiccups.",
    )
    parser.add_argument(
        "--judge-email-rate",
        type=float,
        default=0.1,
        help="Fraction of emails to LLM-judge. Default 0.1 = 10% sample.",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Skip the faithfulness judge (faster, cheaper).",
    )
    parser.add_argument(
        "--save-as-baseline",
        type=Path,
        default=None,
        help="If set, copy this run's JSON to the given path as a new baseline.",
    )
    args = parser.parse_args()

    claims = load_claims_jsonl(args.claims)
    if args.limit is not None:
        claims = claims[: args.limit]

    print(f"[eval] running on {len(claims)} claims (concurrency={args.concurrency})")
    report = asyncio.run(
        run_eval(
            claims=claims,
            concurrency=args.concurrency,
            judge_email_sample_rate=args.judge_email_rate,
            judge_faithfulness=not args.no_faithfulness,
        )
    )

    md_path, json_path = write_report(report, args.reports_dir)
    print(f"[eval] wrote {md_path}")
    print(f"[eval] wrote {json_path}")

    # Print headline
    print()
    print(f"  Decision accuracy:        {report.decision_accuracy:.1%}")
    print(f"  Citation verbatim:        {report.citation_verbatim_rate:.1%}")
    print(f"  Citation precision:       {report.citation_precision_on_correct:.1%}")
    print(f"  Faithfulness rate:        {report.faithfulness_rate:.1%}")
    if report.email_avg_score:
        print(f"  Email avg score:          {report.email_avg_score}/5")
    print(f"  p50/p95 latency:          {report.p50_latency_ms:.0f} / "
          f"{report.p95_latency_ms:.0f} ms")
    print(f"  Cost/accurate decision:   ${report.cost_per_accurate_decision_usd:.6f}")
    print(f"  Avg cost / claim:         ${report.avg_cost_usd:.6f}")
    # Week 17 routing metrics
    print(f"  Auto-resolve rate:        {report.auto_resolve_rate:.1%}")
    print(f"  Accuracy on auto-resolved:{report.accuracy_on_auto_resolved:.1%}")
    if report.by_route:
        print("  By route:")
        for route, stats in report.by_route.items():
            print(
                f"    {route:<14} n={stats['n']:>4} share={stats['share']:.1%} "
                f"accuracy={stats['accuracy']:.1%}"
            )
    print()

    # Regression check
    exit_code = 0
    if args.baseline is not None:
        if not args.baseline.is_file():
            print(f"[eval] baseline not found at {args.baseline} — skipping check")
        else:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            regressions = detect_regressions(
                asdict(report), baseline, args.threshold_pp
            )
            if regressions:
                print("[eval] REGRESSIONS DETECTED:")
                for r in regressions:
                    print(f"  - {r}")
                if not args.override:
                    print("[eval] failing build (use --override to ignore)")
                    exit_code = 1
                else:
                    print("[eval] --override set; passing despite regressions")
            else:
                print("[eval] no regressions vs baseline")

    if args.save_as_baseline is not None:
        args.save_as_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.save_as_baseline.write_text(
            json_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"[eval] saved as baseline → {args.save_as_baseline}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
