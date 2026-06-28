"""Week-2 extraction eval.

Reads `data/synthetic/claims.jsonl`, runs the extractor on each claim, and
compares to per-claim ground-truth labels (`expected_extraction`).

Reports:
- Per-field accuracy (fraction of claims where the field matched the label)
- Overall accuracy (weighted by scored-field count)
- Cost per call (USD), latency (ms)
- Output CSV at the given --report-csv path

Usage:
    uv run python scripts/eval_extraction.py \\
        --claims data/synthetic/claims.jsonl \\
        --report-csv data/synthetic/eval_extraction_report.csv \\
        --limit 100 [--temperature 0.0]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.llm import chat, cost_of  # noqa: E402
from app.ai.prompt_loader import render_prompt  # noqa: E402
from app.ai.schemas import ClaimExtraction  # noqa: E402

# Fields we score against ground truth. Free-text fields (customer_summary)
# are excluded — they need an LLM judge, which is Week 3+.
SCORED_FIELDS = (
    "sku",
    "failure_mode",
    "claim_type",
    "severity",
    "evidence_strength",
    "time_since_purchase_days",
    "mentioned_serial",
    "customer_emotion",
    "prior_contact_attempts",
)


def _normalize(v: Any) -> Any:
    """Compare values structurally; strings are case-folded and stripped."""
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _field_match(predicted: Any, expected: Any) -> bool:
    return _normalize(predicted) == _normalize(expected)


async def _extract_one(
    claim: dict[str, Any], prompt_name: str, model_alias: str, temperature: float
) -> tuple[ClaimExtraction | None, float, float, str | None]:
    """Returns (extraction | None, latency_ms, cost_usd, error)."""
    rendered = render_prompt(
        prompt_name,
        customer_text=claim["raw_input"],
        photo_descriptions=claim.get("photo_descriptions") or [],
    )
    t0 = time.perf_counter()
    try:
        # Use Instructor's structured output via the chat() abstraction.
        result = await chat(
            messages=[{"role": "user", "content": rendered}],
            model_alias=model_alias,
            response_model=ClaimExtraction,
            temperature=temperature,
        )
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        return None, latency_ms, 0.0, f"{type(e).__name__}: {e}"

    latency_ms = (time.perf_counter() - t0) * 1000
    # Instructor's `from_litellm` doesn't surface the raw response for
    # cost accounting. Best-effort: attempt to read attached usage if any.
    usage = getattr(result, "_raw_response", None)
    cost = cost_of(usage) if usage is not None else 0.0
    return result, latency_ms, cost, None


async def run(
    claims_path: Path,
    report_csv: Path,
    prompt_name: str,
    model_alias: str,
    temperature: float,
    limit: int | None,
    concurrency: int,
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    with claims_path.open("r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))
    if limit is not None:
        claims = claims[:limit]

    # Bound concurrency so we don't trip API rate limits.
    sem = asyncio.Semaphore(concurrency)

    async def gated(claim: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        async with sem:
            extraction, latency_ms, cost, err = await _extract_one(
                claim, prompt_name, model_alias, temperature
            )
        return claim, (extraction, latency_ms, cost, err)

    print(
        f"[eval] running {len(claims)} claims with prompt={prompt_name} "
        f"alias={model_alias} T={temperature} concurrency={concurrency}"
    )
    results = await asyncio.gather(*(gated(c) for c in claims))

    # Aggregate
    per_field_correct: dict[str, int] = defaultdict(int)
    per_field_scored: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_cost = 0.0
    n_errors = 0

    for claim, (extraction, latency_ms, cost, err) in results:
        latencies.append(latency_ms)
        total_cost += cost
        row: dict[str, Any] = {
            "claim_id": claim["claim_id"],
            "decision_kind": claim["decision_kind"],
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost, 6),
            "error": err or "",
        }

        if err is not None or extraction is None:
            n_errors += 1
            for fld in SCORED_FIELDS:
                row[f"pred_{fld}"] = ""
                row[f"exp_{fld}"] = ""
                row[f"match_{fld}"] = ""
            rows.append(row)
            continue

        expected = claim.get("expected_extraction") or {}
        pred_dump = extraction.model_dump(mode="json")

        for fld in SCORED_FIELDS:
            if fld not in expected:
                row[f"pred_{fld}"] = pred_dump.get(fld, "")
                row[f"exp_{fld}"] = ""
                row[f"match_{fld}"] = ""
                continue
            pred = pred_dump.get(fld)
            exp = expected[fld]
            ok = _field_match(pred, exp)
            per_field_scored[fld] += 1
            if ok:
                per_field_correct[fld] += 1
            row[f"pred_{fld}"] = pred
            row[f"exp_{fld}"] = exp
            row[f"match_{fld}"] = int(ok)
        rows.append(row)

    # Summary metrics
    summary: dict[str, Any] = {
        "n_claims": len(claims),
        "n_errors": n_errors,
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_usd": round(total_cost / max(len(claims), 1), 6),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(
            statistics.quantiles(latencies, n=20)[18], 1
        ) if len(latencies) >= 20 else round(max(latencies) if latencies else 0, 1),
        "per_field_accuracy": {
            fld: round(per_field_correct[fld] / max(per_field_scored[fld], 1), 3)
            for fld in SCORED_FIELDS
        },
        "per_field_scored": dict(per_field_scored),
    }
    total_correct = sum(per_field_correct.values())
    total_scored = sum(per_field_scored.values())
    summary["overall_accuracy"] = (
        round(total_correct / max(total_scored, 1), 3) if total_scored else 0.0
    )

    # Write CSV
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with report_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print(f"  Extraction eval — {summary['n_claims']} claims")
    print("=" * 60)
    print(f"Errors:           {summary['n_errors']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%}")
    print(f"Total cost:       ${summary['total_cost_usd']:.4f}")
    print(f"Avg cost/claim:   ${summary['avg_cost_usd']:.6f}")
    print(f"Latency p50/p95:  {summary['p50_latency_ms']:.0f} / "
          f"{summary['p95_latency_ms']:.0f} ms")
    print()
    print(f"{'Field':<28} {'Acc':>6} {'N':>5}")
    print("-" * 42)
    for fld, acc in summary["per_field_accuracy"].items():
        n = summary["per_field_scored"].get(fld, 0)
        print(f"{fld:<28} {acc:>6.1%} {n:>5}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims", type=Path, default=ROOT / "data/synthetic/claims.jsonl"
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=ROOT / "data/synthetic/eval_extraction_report.csv",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--prompt", default="extract_v2")
    parser.add_argument("--model", default="extractor")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    summary = asyncio.run(
        run(
            claims_path=args.claims,
            report_csv=args.report_csv,
            prompt_name=args.prompt,
            model_alias=args.model,
            temperature=args.temperature,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    )
    print_summary(summary)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[eval] wrote {args.report_json}")
    print(f"[eval] wrote {args.report_csv}")
    return 0 if summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
