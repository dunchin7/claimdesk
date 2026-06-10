"""Production-grade eval runner (Week 4).

Wraps the per-claim pipeline with full instrumentation:
- Decision accuracy + per-stratum breakdown
- Citation precision (verbatim + on-topic)
- Faithfulness (LLM-judge)
- Email quality (LLM-judge, sampled)
- Cost-per-claim and cost-per-accurate-decision
- Latency p50/p95
- Calibration table

The runner emits a Markdown report card and a JSON blob suitable for
regression comparison.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.adjudication.pipeline import process_claim
from app.ai.llm import cost_of
from app.ai.registry import load_registry
from app.core.logging import get_logger
from app.evals.judges import (
    EmailQualityScore,
    FaithfulnessScore,
    judge_email,
    judge_faithfulness,
)

log = get_logger(__name__)


@dataclass
class PerClaimResult:
    claim_id: str
    decision_kind: str
    expected_decision: str
    predicted_decision: str
    decision_match: bool
    expected_resolution: str
    predicted_resolution: str
    citation_verbatim: bool
    citation_fuzzy: float
    confidence: str
    cost_usd: float
    latency_ms: float
    faithfulness: str | None = None
    email_avg_score: float | None = None
    error: str = ""
    # Extraction fields captured for calibrator training (Week 14.5)
    extracted_evidence_strength: str | None = None
    extracted_severity: str | None = None
    extracted_failure_mode: str | None = None
    extracted_claim_type: str | None = None
    extracted_customer_emotion: str | None = None
    extracted_time_since_purchase_days: int | None = None
    extracted_mentioned_serial: str | None = None
    extracted_prior_contact_attempts: bool | None = None
    # Week 17: routing decision captured per claim
    calibrated_prob: float = 0.0
    route: str = ""
    route_reasons: list[str] = field(default_factory=list)
    # Phase-4 P1.5: XGBoost calibrated fraud probability when available
    fraud_score: float | None = None


@dataclass
class EvalRunReport:
    run_id: str
    timestamp: str
    n_claims: int
    n_errors: int
    decision_accuracy: float
    citation_verbatim_rate: float
    citation_precision_on_correct: float
    faithfulness_rate: float
    email_avg_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_cost_usd: float
    cost_per_accurate_decision_usd: float
    by_stratum: dict[str, dict[str, Any]] = field(default_factory=dict)
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    calibration: dict[str, dict[str, Any]] = field(default_factory=dict)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    per_claim: list[dict[str, Any]] = field(default_factory=list)
    # Week 17: routing metrics — the Phase-3 acceptance criteria
    auto_resolve_rate: float = 0.0
    accuracy_on_auto_resolved: float = 0.0
    by_route: dict[str, dict[str, Any]] = field(default_factory=dict)


async def _judge_email_safe(email: str, outcome: str, resolution: str) -> EmailQualityScore | None:
    try:
        return await judge_email(email, outcome, resolution)
    except Exception as e:  # noqa: BLE001
        log.warning("eval.judge_email_failed", error=str(e))
        return None


async def _judge_faithfulness_safe(
    claim_text: str, citation: str, rationale: str
) -> FaithfulnessScore | None:
    try:
        return await judge_faithfulness(claim_text, citation, rationale)
    except Exception as e:  # noqa: BLE001
        log.warning("eval.judge_faithfulness_failed", error=str(e))
        return None


async def _process_one(
    claim: dict[str, Any],
    judge_email_sample_rate: float,
    judge_faithfulness: bool,
) -> PerClaimResult:
    t0 = time.perf_counter()
    try:
        result = await process_claim(
            raw_input=claim["raw_input"],
            photo_descriptions=claim.get("photo_descriptions") or [],
            # Phase-4 P1.5: pass identifiers so the pipeline can look up
            # CustomerContext from the DB and score fraud. Loaded via
            # `scripts/load_eval_to_db.py` before the eval run.
            claim_id=claim.get("claim_id"),
            customer_email=claim.get("customer_email"),
        )
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        return PerClaimResult(
            claim_id=claim["claim_id"],
            decision_kind=claim["decision_kind"],
            expected_decision=claim["expected_decision"],
            predicted_decision="",
            decision_match=False,
            expected_resolution=claim["expected_resolution"],
            predicted_resolution="",
            citation_verbatim=False,
            citation_fuzzy=0.0,
            confidence="",
            cost_usd=0.0,
            latency_ms=latency_ms,
            error=f"{type(e).__name__}: {e}",
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    decision_match = result.decision.outcome == claim["expected_decision"]

    # Real cost from PipelineResult — extract + adjudicate + email summed via
    # `chat(return_cost=True)` + `litellm.completion_cost()` on the raw
    # Instructor completion (fixed in Week 10).
    claim_cost = result.cost_usd

    # Faithfulness judge (slow — only run if asked)
    faithfulness_verdict: str | None = None
    if judge_faithfulness and not decision_match:
        # Only judge mismatched cases; faithful-but-wrong is the interesting bucket
        score = await _judge_faithfulness_safe(
            claim["raw_input"], result.decision.policy_citation, result.decision.rationale
        )
        if score is not None:
            faithfulness_verdict = score.verdict

    # Email judge (sampled to keep cost down)
    email_avg: float | None = None
    if judge_email_sample_rate > 0:
        # Deterministic sampling by hash of claim_id
        h = abs(hash(claim["claim_id"])) % 1000
        if h / 1000.0 < judge_email_sample_rate:
            score = await _judge_email_safe(
                result.email, result.decision.outcome, result.decision.resolution
            )
            if score is not None:
                email_avg = score.average

    extraction = result.extraction
    return PerClaimResult(
        claim_id=claim["claim_id"],
        decision_kind=claim["decision_kind"],
        expected_decision=claim["expected_decision"],
        predicted_decision=result.decision.outcome,
        decision_match=decision_match,
        expected_resolution=claim["expected_resolution"],
        predicted_resolution=result.decision.resolution,
        citation_verbatim=result.citation_result.verbatim,
        citation_fuzzy=result.citation_result.fuzzy_ratio,
        confidence=result.decision.confidence,
        cost_usd=claim_cost,
        latency_ms=latency_ms,
        faithfulness=faithfulness_verdict,
        email_avg_score=email_avg,
        extracted_evidence_strength=extraction.evidence_strength,
        extracted_severity=extraction.severity,
        extracted_failure_mode=extraction.failure_mode,
        extracted_claim_type=extraction.claim_type,
        extracted_customer_emotion=extraction.customer_emotion,
        extracted_time_since_purchase_days=extraction.time_since_purchase_days,
        extracted_mentioned_serial=extraction.mentioned_serial,
        extracted_prior_contact_attempts=extraction.prior_contact_attempts,
        calibrated_prob=result.calibrated_prob,
        route=result.route,
        route_reasons=list(result.route_reasons),
        fraud_score=result.fraud_score,
    )


async def run_eval(
    claims: list[dict[str, Any]],
    *,
    concurrency: int = 8,
    judge_email_sample_rate: float = 0.1,
    judge_faithfulness: bool = True,
) -> EvalRunReport:
    """Run the full eval. Returns a populated `EvalRunReport`."""
    sem = asyncio.Semaphore(concurrency)

    async def gated(claim: dict[str, Any]) -> PerClaimResult:
        async with sem:
            return await _process_one(claim, judge_email_sample_rate, judge_faithfulness)

    log.info(
        "eval.run.start",
        n=len(claims),
        concurrency=concurrency,
        judge_email_sample_rate=judge_email_sample_rate,
        judge_faithfulness=judge_faithfulness,
    )
    results = await asyncio.gather(*(gated(c) for c in claims))

    n = len(results)
    n_errors = sum(1 for r in results if r.error)
    n_correct = sum(1 for r in results if r.decision_match)
    n_verbatim = sum(1 for r in results if r.citation_verbatim)
    n_verbatim_on_correct = sum(
        1 for r in results if r.decision_match and r.citation_verbatim
    )

    latencies = [r.latency_ms for r in results if not r.error]
    costs = [r.cost_usd for r in results]

    by_stratum_total: dict[str, int] = defaultdict(int)
    by_stratum_correct: dict[str, int] = defaultdict(int)
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    confidence_correct: dict[str, list[bool]] = defaultdict(list)

    faithful_n = 0
    faithful_total = 0
    email_scores: list[float] = []

    for r in results:
        by_stratum_total[r.decision_kind] += 1
        if r.decision_match:
            by_stratum_correct[r.decision_kind] += 1
        if not r.error:
            confusion[(r.expected_decision, r.predicted_decision)] += 1
            confidence_correct[r.confidence].append(r.decision_match)
        if r.faithfulness is not None:
            faithful_total += 1
            if r.faithfulness == "faithful":
                faithful_n += 1
        if r.email_avg_score is not None:
            email_scores.append(r.email_avg_score)

    decision_accuracy = round(n_correct / max(n, 1), 3)
    avg_cost = round(sum(costs) / max(n, 1), 6)
    cost_per_correct = (
        round(sum(costs) / n_correct, 6) if n_correct else 0.0
    )

    p50 = round(statistics.median(latencies), 1) if latencies else 0
    p95 = (
        round(statistics.quantiles(latencies, n=20)[18], 1)
        if len(latencies) >= 20
        else round(max(latencies) if latencies else 0, 1)
    )

    registry = load_registry()
    prompt_versions = {
        role: entry["prompt"] for role, entry in registry["active"].items()
    }

    # Week 17 routing metrics
    by_route_n: dict[str, int] = defaultdict(int)
    by_route_correct: dict[str, int] = defaultdict(int)
    for r in results:
        if not r.route:
            continue
        by_route_n[r.route] += 1
        if r.decision_match:
            by_route_correct[r.route] += 1

    n_auto = by_route_n.get("auto_resolve", 0)
    n_auto_correct = by_route_correct.get("auto_resolve", 0)
    auto_resolve_rate = round(n_auto / max(n, 1), 3)
    accuracy_on_auto_resolved = round(n_auto_correct / max(n_auto, 1), 3)
    by_route_summary = {
        route: {
            "n": by_route_n[route],
            "share": round(by_route_n[route] / max(n, 1), 3),
            "accuracy": round(by_route_correct[route] / max(by_route_n[route], 1), 3),
        }
        for route in sorted(by_route_n)
    }

    report = EvalRunReport(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        n_claims=n,
        n_errors=n_errors,
        decision_accuracy=decision_accuracy,
        citation_verbatim_rate=round(n_verbatim / max(n, 1), 3),
        citation_precision_on_correct=round(
            n_verbatim_on_correct / max(n_correct, 1), 3
        ),
        faithfulness_rate=(
            round(faithful_n / max(faithful_total, 1), 3) if faithful_total else 1.0
        ),
        email_avg_score=(
            round(sum(email_scores) / len(email_scores), 2) if email_scores else 0.0
        ),
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        avg_cost_usd=avg_cost,
        cost_per_accurate_decision_usd=cost_per_correct,
        by_stratum={
            k: {
                "n": by_stratum_total[k],
                "accuracy": round(
                    by_stratum_correct[k] / max(by_stratum_total[k], 1), 3
                ),
            }
            for k in sorted(by_stratum_total)
        },
        confusion_matrix={
            f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())
        },
        calibration={
            conf: {
                "n": len(matches),
                "accuracy": round(sum(matches) / max(len(matches), 1), 3),
            }
            for conf, matches in confidence_correct.items()
        },
        prompt_versions=prompt_versions,
        per_claim=[asdict(r) for r in results],
        auto_resolve_rate=auto_resolve_rate,
        accuracy_on_auto_resolved=accuracy_on_auto_resolved,
        by_route=by_route_summary,
    )
    log.info(
        "eval.run.done",
        run_id=report.run_id,
        accuracy=report.decision_accuracy,
        verbatim=report.citation_verbatim_rate,
    )
    return report


def load_claims_jsonl(path: Path) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))
    return claims
