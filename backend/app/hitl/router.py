"""HITL routing — enqueue claims that aren't safely auto-resolvable.

The pipeline returns `PipelineResult.route` ∈ {"auto_resolve", "assist", "review"}.
This module turns that into an `operator_queue` row when the route requires
human attention, and snapshots the agent's decision + signals so the operator
sees exactly what was decided at enqueue time.

The "auto-resolve" path bypasses this — `enqueue_if_needed()` returns None
and the caller proceeds to send the customer email + create the RMA.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.adjudication.pipeline import PipelineResult
from app.core.logging import get_logger
from app.db.models import OperatorQueueItem
from app.security.injection import detect_injection_signals

log = get_logger(__name__)


async def enqueue_if_needed(
    session: AsyncSession,
    *,
    claim_id: str | None,
    raw_input: str,
    result: PipelineResult,
    agent_run_id: str | None = None,
) -> OperatorQueueItem | None:
    """Insert an operator_queue row when `result.route != "auto_resolve"`.

    Returns the persisted row (caller commits the session) or None when
    the claim is safe to auto-resolve.
    """
    if result.route == "auto_resolve":
        log.info(
            "hitl.auto_resolve",
            claim_id=claim_id,
            calibrated_prob=result.calibrated_prob,
            outcome=result.decision.outcome,
        )
        return None

    # Capture all signals an operator might want to see — injection patterns,
    # fraud indicators, citation status. Aggregated into one JSON blob so
    # the operator UI doesn't have to join multiple tables.
    injection_signals = detect_injection_signals(raw_input)
    signals: dict[str, Any] = {
        "injection_signals": injection_signals,
        "citation_verbatim": result.citation_result.verbatim,
        "citation_fuzzy_ratio": result.citation_result.fuzzy_ratio,
        "calibrated_prob": result.calibrated_prob,
        "route": result.route,
        # Week 16: structured reasons (e.g. ["cost_cap_hit", "injection_signal"])
        # so the operator UI can sort/filter by what tripped the route demotion.
        "route_reasons": list(result.route_reasons),
        "cost_usd": result.cost_usd,
        "policy_version": result.policy_version,
        "prompt_versions": dict(result.prompt_versions),
    }

    item = OperatorQueueItem(
        claim_id=None,  # claim_id from the API is a string; not always a DB UUID
        agent_run_id=None,  # filled by multi-agent path; single-shot has no run row
        route=result.route,
        calibrated_prob=result.calibrated_prob,
        status="pending",
        agent_decision={
            "outcome": result.decision.outcome,
            "resolution": result.decision.resolution,
            "rationale": result.decision.rationale,
            "policy_citation": result.decision.policy_citation,
            "confidence": result.decision.confidence,
            "missing_info_questions": result.decision.missing_info_questions,
            "email_draft": result.email,
        },
        signals=signals,
        raw_input=raw_input[:10_000],
    )
    session.add(item)
    await session.flush()  # populate item.id
    log.info(
        "hitl.enqueued",
        queue_id=str(item.id),
        route=result.route,
        calibrated_prob=result.calibrated_prob,
        n_injection_signals=len(injection_signals),
    )
    return item


# Routing thresholds — duplicated from confidence.calibrator deliberately so
# that the router-level decision is documented at the routing call-site.
# A future change to calibrator thresholds should review this too.
ROUTE_AUTO_RESOLVE = "auto_resolve"
ROUTE_ASSIST = "assist"
ROUTE_REVIEW = "review"
