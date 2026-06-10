"""Week-3 decision eval.

Runs the full extract→adjudicate→citation-check→email pipeline against the
golden set and reports decision accuracy, citation precision, and per-stratum
breakdowns.

Usage:
    uv run python scripts/eval_decision.py \\
        --claims data/synthetic/claims.jsonl \\
        --report-csv data/synthetic/eval_decision_report.csv \\
        --report-json data/synthetic/eval_decision_report.json
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

from app.adjudication.pipeline import process_claim  # noqa: E402


async def _process_one(claim: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = await process_claim(
            raw_input=claim["raw_input"],
            photo_descriptions=claim.get("photo_descriptions") or [],
        )
    except Exception as e:  # noqa: BLE001
        return {
            "claim_id": claim["claim_id"],
            "decision_kind": claim["decision_kind"],
            "expected_decision": claim["expected_decision"],
            "predicted_decision": "",
            "expected_resolution": claim["expected_resolution"],
            "predicted_resolution": "",
            "citation_verbatim": False,
            "citation_fuzzy": 0.0,
            "confidence": "",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "decision_match": False,
            "error": f"{type(e).__name__}: {e}",
        }

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "claim_id": claim["claim_id"],
        "decision_kind": claim["decision_kind"],
        "expected_decision": claim["expected_decision"],
        "predicted_decision": result.decision.outcome,
        "expected_resolution": claim["expected_resolution"],
        "predicted_resolution": result.decision.resolution,
        "citation_verbatim": result.citation_result.verbatim,
        "citation_fuzzy": result.citation_result.fuzzy_ratio,
        "confidence": result.decision.confidence,
        "latency_ms": round(latency_ms, 1),
        "decision_match": result.decision.outcome == claim["expected_decision"],
        "predicted_citation": result.decision.policy_citation,
        "rationale": result.decision.rationale,
        "error": "",
    }


async def run(
    claims_path: Path,
    report_csv: Path,
    limit: int | None,
    concurrency: int,
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    with claims_path.open("r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))
    if limit is not None:
        claims = claims[:limit]

    sem = asyncio.Semaphore(concurrency)

    async def gated(claim: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _process_one(claim)

    print(f"[eval] running pipeline on {len(claims)} claims, concurrency={concurrency}")
    rows = await asyncio.gather(*(gated(c) for c in claims))

    # Aggregate
    n = len(rows)
    n_errors = sum(1 for r in rows if r["error"])
    n_decision_correct = sum(1 for r in rows if r["decision_match"])
    n_citation_verbatim = sum(1 for r in rows if r["citation_verbatim"])
    n_citation_on_correct = sum(
        1 for r in rows if r["decision_match"] and r["citation_verbatim"]
    )
    latencies = [r["latency_ms"] for r in rows if not r["error"]]

    by_stratum_total: dict[str, int] = defaultdict(int)
    by_stratum_correct: dict[str, int] = defaultdict(int)
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    confidence_correct: dict[str, list[bool]] = defaultdict(list)

    for r in rows:
        by_stratum_total[r["decision_kind"]] += 1
        if r["decision_match"]:
            by_stratum_correct[r["decision_kind"]] += 1
        if not r["error"]:
            confusion[(r["expected_decision"], r["predicted_decision"])] += 1
            confidence_correct[r["confidence"]].append(bool(r["decision_match"]))

    summary: dict[str, Any] = {
        "n_claims": n,
        "n_errors": n_errors,
        "decision_accuracy": round(n_decision_correct / max(n, 1), 3),
        "citation_verbatim_rate": round(n_citation_verbatim / max(n, 1), 3),
        # The Week-3 gate: of decisions we got right, how often was the
        # citation also verbatim? 90%+ is the target.
        "citation_precision_on_correct": round(
            n_citation_on_correct / max(n_decision_correct, 1), 3
        ),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(
            statistics.quantiles(latencies, n=20)[18], 1
        ) if len(latencies) >= 20 else round(max(latencies) if latencies else 0, 1),
        "by_stratum": {
            k: {
                "n": by_stratum_total[k],
                "accuracy": round(by_stratum_correct[k] / max(by_stratum_total[k], 1), 3),
            }
            for k in sorted(by_stratum_total)
        },
        "confusion_matrix": {f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())},
        "calibration": {
            conf: {
                "n": len(matches),
                "accuracy": round(sum(matches) / max(len(matches), 1), 3),
            }
            for conf, matches in confidence_correct.items()
        },
    }

    # Write CSV
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        # Drop multiline rationale/citation from CSV to keep it clean;
        # they're available in the JSON.
        csv_fields = [
            "claim_id", "decision_kind", "expected_decision", "predicted_decision",
            "expected_resolution", "predicted_resolution", "citation_verbatim",
            "citation_fuzzy", "confidence", "latency_ms", "decision_match", "error",
        ]
        with report_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return summary


def print_summary(s: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print(f"  Decision eval — {s['n_claims']} claims")
    print("=" * 60)
    print(f"Errors:                    {s['n_errors']}")
    print(f"Decision accuracy:         {s['decision_accuracy']:.1%}")
    print(f"Citation verbatim rate:    {s['citation_verbatim_rate']:.1%}")
    print(f"Citation precision (correct decisions): "
          f"{s['citation_precision_on_correct']:.1%}")
    print(f"Latency p50/p95:           {s['p50_latency_ms']:.0f} / "
          f"{s['p95_latency_ms']:.0f} ms")
    print()
    print("By stratum:")
    for k, v in s["by_stratum"].items():
        print(f"  {k:<18} n={v['n']:>3}  acc={v['accuracy']:.1%}")
    print()
    print("Confusion (expected -> predicted):")
    for k, v in s["confusion_matrix"].items():
        print(f"  {k:<28} {v}")
    print()
    print("Calibration (confidence -> accuracy):")
    for conf, info in s["calibration"].items():
        print(f"  {conf:<10} n={info['n']:>3}  acc={info['accuracy']:.1%}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=ROOT / "data/synthetic/claims.jsonl")
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=ROOT / "data/synthetic/eval_decision_report.csv",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    summary = asyncio.run(
        run(
            claims_path=args.claims,
            report_csv=args.report_csv,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
