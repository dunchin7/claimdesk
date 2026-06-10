"""Week-11 agent eval.

Runs the ReAct agent against a stratified sample of the golden set. Reports:
- decision accuracy vs expected_decision (target ≥60% per design doc)
- avg / max cost per run (target avg ≤$0.30, max ≤$0.50)
- p50 / p95 latency
- n_tool_calls distribution
- status breakdown (completed / cost_capped / iter_capped / failed)
- per-stratum accuracy

Usage:
    uv run python scripts/eval_agent.py
    uv run python scripts/eval_agent.py --limit 30 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.agents.multi import run_multi_agent  # noqa: E402
from app.ai.agents.react import run_agent as run_react_agent  # noqa: E402

CLAIMS_PATH = ROOT / "data/synthetic/claims.jsonl"


def stratified_sample(claims: list[dict], n: int, seed: int) -> list[dict]:
    """Take a stratified sample preserving the proportion of decision_kinds."""
    rng = random.Random(seed)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for c in claims:
        by_kind[c["decision_kind"]].append(c)

    total = len(claims)
    out: list[dict] = []
    for kind, items in by_kind.items():
        target = max(1, round(len(items) / total * n))
        rng.shuffle(items)
        out.extend(items[:target])
    rng.shuffle(out)
    return out[:n]


async def _eval_one(
    claim: dict, cost_cap: float, max_iter: int, agent: str
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        if agent == "multi":
            result = await run_multi_agent(
                claim_id=claim["claim_id"],
                raw_input=claim["raw_input"],
                photo_urls=[],
                customer_id=claim.get("customer_id"),
            )
            predicted = (result.get("final_decision") or {}).get("outcome")
            return {
                "claim_id": claim["claim_id"],
                "decision_kind": claim["decision_kind"],
                "expected": claim["expected_decision"],
                "predicted": predicted,
                "match": predicted == claim["expected_decision"],
                "status": "completed" if predicted else "incomplete",
                "cost_usd": result["cost_usd"],
                "n_iterations": result.get("critic_iterations", 0),
                "n_tool_calls": len(result.get("step_log") or []),
                "latency_ms": result["latency_ms"],
                "trace_id": result.get("trace_id", ""),
                "error": None,
                # Week-14 extras
                "specialists_run": result.get("specialists_run") or [],
                "n_specialists": len(result.get("specialists_run") or []),
                "critic_disagreed": bool(result.get("critic_disagreed")),
                "replan_iterations": result.get("replan_iterations", 0),
                "plan_complexity": (result.get("plan") or {}).get("complexity"),
                "plan_fraud_concern": (result.get("plan") or {}).get("fraud_concern"),
            }

        result = await run_react_agent(
            claim_id=claim["claim_id"],
            raw_input=claim["raw_input"],
            photo_urls=[],
            cost_cap_usd=cost_cap,
            max_iter=max_iter,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "claim_id": claim["claim_id"],
            "decision_kind": claim["decision_kind"],
            "expected": claim["expected_decision"],
            "predicted": None,
            "status": "exception",
            "cost_usd": 0.0,
            "n_iterations": 0,
            "n_tool_calls": 0,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "error": f"{type(e).__name__}: {e}",
        }
    predicted = (result.final_decision or {}).get("outcome")
    return {
        "claim_id": claim["claim_id"],
        "decision_kind": claim["decision_kind"],
        "expected": claim["expected_decision"],
        "predicted": predicted,
        "match": predicted == claim["expected_decision"],
        "status": result.status,
        "cost_usd": result.cost_usd,
        "n_iterations": result.n_iterations,
        "n_tool_calls": result.n_tool_calls,
        "latency_ms": result.latency_ms,
        "trace_id": result.trace_id,
        "error": result.error,
    }


async def run(
    claims: list[dict], cost_cap: float, max_iter: int, concurrency: int, agent: str
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)

    async def gated(c: dict) -> dict[str, Any]:
        async with sem:
            return await _eval_one(c, cost_cap, max_iter, agent)

    print(f"[eval] running agent={agent} on {len(claims)} claims (concurrency={concurrency})")
    rows = await asyncio.gather(*(gated(c) for c in claims))

    n = len(rows)
    n_match = sum(1 for r in rows if r.get("match"))
    n_completed = sum(1 for r in rows if r.get("status") == "completed")
    by_status: dict[str, int] = defaultdict(int)
    by_stratum_total: dict[str, int] = defaultdict(int)
    by_stratum_match: dict[str, int] = defaultdict(int)
    confusion: dict[tuple[str | None, str | None], int] = defaultdict(int)
    costs = [r["cost_usd"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]
    tool_calls = [r["n_tool_calls"] for r in rows]
    iters = [r["n_iterations"] for r in rows]

    for r in rows:
        by_status[r["status"]] += 1
        by_stratum_total[r["decision_kind"]] += 1
        if r.get("match"):
            by_stratum_match[r["decision_kind"]] += 1
        confusion[(r["expected"], r["predicted"])] += 1

    summary: dict[str, Any] = {
        "n_claims": n,
        "accuracy": round(n_match / max(n, 1), 3),
        "completion_rate": round(n_completed / max(n, 1), 3),
        "avg_cost_usd": round(sum(costs) / max(n, 1), 6),
        "max_cost_usd": round(max(costs) if costs else 0.0, 6),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": (
            round(statistics.quantiles(latencies, n=20)[18], 1)
            if len(latencies) >= 20 else round(max(latencies) if latencies else 0, 1)
        ),
        "avg_tool_calls": round(sum(tool_calls) / max(n, 1), 1),
        "avg_iterations": round(sum(iters) / max(n, 1), 1),
        "by_status": dict(by_status),
        "by_stratum": {
            k: {"n": by_stratum_total[k],
                "accuracy": round(by_stratum_match[k] / max(by_stratum_total[k], 1), 3)}
            for k in sorted(by_stratum_total)
        },
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items(), key=lambda x: str(x[0]))},
        "per_claim": rows,
    }

    # Week-14 multi-agent metrics (only meaningful when run with --agent multi)
    if any("specialists_run" in r for r in rows):
        n_specialists_per_claim = [r.get("n_specialists", 0) for r in rows if r.get("specialists_run") is not None]
        n_critic_disagreed = sum(1 for r in rows if r.get("critic_disagreed"))
        n_replans = sum(1 for r in rows if (r.get("replan_iterations") or 0) > 0)
        complexity_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            c = r.get("plan_complexity")
            if c:
                complexity_counts[c] += 1
        summary["multi_agent"] = {
            "avg_specialists_run": round(sum(n_specialists_per_claim) / max(len(n_specialists_per_claim), 1), 2),
            "avg_specialists_skipped": round(
                5 - sum(n_specialists_per_claim) / max(len(n_specialists_per_claim), 1), 2
            ),
            "critic_correction_rate": round(n_critic_disagreed / max(n, 1), 3),
            "replan_rate": round(n_replans / max(n, 1), 3),
            "plan_complexity_distribution": dict(complexity_counts),
        }
    return summary


def print_summary(s: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print(f"  Agent eval — n={s['n_claims']}")
    print("=" * 60)
    print(f"Accuracy:           {s['accuracy']:.1%}")
    print(f"Completion rate:    {s['completion_rate']:.1%}")
    print(f"Avg cost / run:     ${s['avg_cost_usd']:.4f}")
    print(f"Max cost / run:     ${s['max_cost_usd']:.4f}")
    print(f"Latency p50/p95:    {s['p50_latency_ms']:.0f} / {s['p95_latency_ms']:.0f} ms")
    print(f"Avg tool calls:     {s['avg_tool_calls']}")
    print(f"Avg iterations:     {s['avg_iterations']}")
    print()
    print("By status:")
    for k, v in s["by_status"].items():
        print(f"  {k:<14} {v}")
    print()
    print("By stratum:")
    for k, v in s["by_stratum"].items():
        print(f"  {k:<18} n={v['n']:>3}  acc={v['accuracy']:.1%}")
    print()
    print("Confusion (expected → predicted):")
    for k, v in s["confusion"].items():
        print(f"  {k:<28} {v}")

    ma = s.get("multi_agent")
    if ma:
        print()
        print("Multi-agent metrics:")
        print(f"  avg specialists run:        {ma['avg_specialists_run']} / 5")
        print(f"  critic_correction_rate:     {ma['critic_correction_rate']:.1%} (target 5-15%)")
        print(f"  replan_rate:                {ma['replan_rate']:.1%}")
        print(f"  plan complexity dist:       {ma['plan_complexity_distribution']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--cost-cap", type=float, default=0.50)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument(
        "--agent", default="react", choices=["react", "multi"],
        help="Which agent to bench: react (Week 11 single ReAct) or multi (Week 12 LangGraph multi-agent)",
    )
    parser.add_argument(
        "--report-json", type=Path,
        default=None,
        help="Defaults to data/synthetic/eval_agent_<agent>_report.json",
    )
    args = parser.parse_args()

    if args.report_json is None:
        args.report_json = ROOT / f"data/synthetic/eval_agent_{args.agent}_report.json"

    all_claims = [json.loads(l) for l in CLAIMS_PATH.read_text().splitlines() if l.strip()]
    sample = stratified_sample(all_claims, args.limit, args.seed)

    summary = asyncio.run(run(sample, args.cost_cap, args.max_iter, args.concurrency, args.agent))
    print_summary(summary)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    # Strip per-claim to keep JSON readable on stdout summaries; keep it in file
    args.report_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[eval] wrote {args.report_json}")

    # Phase-3 gate per design doc: ≥60% accuracy, avg cost ≤$0.30, max ≤$0.50
    if summary["accuracy"] >= 0.60 and summary["avg_cost_usd"] <= 0.30 and summary["max_cost_usd"] <= 0.50:
        print("[gate] Week-11 acceptance criteria met.")
        return 0
    print("[gate] Week-11 acceptance criteria NOT fully met (see metrics above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
