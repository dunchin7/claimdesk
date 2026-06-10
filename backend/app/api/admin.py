"""Operator / admin endpoints — Week-15 HITL routing.

The operator queue is the entry point. Items are listed via GET /queue,
claimed (assigned to an operator) via POST /queue/{id}/claim, then
resolved via approve / override / escalate. Every state transition
writes to AgentAction for audit + idempotency.

Frontend UI is deferred — these endpoints are designed for curl / a
thin admin page. A Next.js operator UI lands in Week 17 if scope allows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import AgentAction, OperatorQueueItem
from app.db.session import get_session
from app.notifications.email import send_email

router = APIRouter()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueueListItem(BaseModel):
    id: str
    route: str
    status: str
    calibrated_prob: float
    outcome: str
    confidence: str
    enqueued_at: datetime
    operator_id: str | None = None
    n_signals: int = 0


class QueueDetail(BaseModel):
    id: str
    claim_id: str | None
    agent_run_id: str | None
    route: str
    status: str
    calibrated_prob: float
    agent_decision: dict[str, Any]
    signals: dict[str, Any]
    raw_input: str | None
    enqueued_at: datetime
    claimed_at: datetime | None
    completed_at: datetime | None
    operator_id: str | None
    operator_decision: dict[str, Any] | None
    operator_notes: str | None


class ClaimBody(BaseModel):
    operator_id: str = Field(min_length=1, max_length=128)


class ApproveBody(BaseModel):
    operator_id: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=4000)
    # Week 16: hold-until-approval email send. If `customer_email` is
    # provided, the agent's drafted email is sent to that address as
    # part of the approve action; the (queue_id, action_type) idempotency
    # guarantees we never double-send. Omit when there's no customer
    # contact info yet — the email stays parked in agent_decision.
    customer_email: str | None = Field(default=None, max_length=320)
    email_subject: str | None = Field(default=None, max_length=320)


class OverrideBody(BaseModel):
    operator_id: str = Field(min_length=1, max_length=128)
    outcome: str = Field(description="approve / reject / needs_info")
    resolution: str = Field(
        default="none",
        description="refund / replacement / repair / store_credit / none",
    )
    rationale: str = Field(min_length=10, max_length=4000)
    notes: str | None = Field(default=None, max_length=4000)
    # Same as ApproveBody. If provided AND operator drafts a custom
    # email body, that's what gets sent. Otherwise we send the
    # agent's draft (which may not reflect the override — operator
    # discretion).
    customer_email: str | None = Field(default=None, max_length=320)
    email_subject: str | None = Field(default=None, max_length=320)
    email_body_override: str | None = Field(default=None, max_length=10_000)


class EscalateBody(BaseModel):
    operator_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=10, max_length=4000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_list_item(row: OperatorQueueItem) -> QueueListItem:
    decision = row.agent_decision or {}
    return QueueListItem(
        id=str(row.id),
        route=row.route,
        status=row.status,
        calibrated_prob=row.calibrated_prob,
        outcome=str(decision.get("outcome", "")),
        confidence=str(decision.get("confidence", "")),
        enqueued_at=row.enqueued_at,
        operator_id=row.operator_id,
        n_signals=len((row.signals or {}).get("injection_signals") or []),
    )


def _to_detail(row: OperatorQueueItem) -> QueueDetail:
    return QueueDetail(
        id=str(row.id),
        claim_id=str(row.claim_id) if row.claim_id else None,
        agent_run_id=str(row.agent_run_id) if row.agent_run_id else None,
        route=row.route,
        status=row.status,
        calibrated_prob=row.calibrated_prob,
        agent_decision=row.agent_decision or {},
        signals=row.signals or {},
        raw_input=row.raw_input,
        enqueued_at=row.enqueued_at,
        claimed_at=row.claimed_at,
        completed_at=row.completed_at,
        operator_id=row.operator_id,
        operator_decision=row.operator_decision,
        operator_notes=row.operator_notes,
    )


async def _audit(
    session: AsyncSession,
    *,
    action_type: str,
    queue_id: UUID,
    payload: dict[str, Any],
) -> None:
    """Write an idempotency-protected audit row. UNIQUE constraint on
    (action_type, idempotency_key) means re-submits are no-ops, not duplicates."""
    session.add(AgentAction(
        action_type=action_type,
        idempotency_key=f"queue:{queue_id}",
        payload=payload,
    ))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/queue", response_model=list[QueueListItem])
async def list_queue(
    status_filter: str | None = Query(default=None, alias="status"),
    route: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[QueueListItem]:
    """List queue items, newest first. Filter by status / route."""
    stmt = (
        select(OperatorQueueItem)
        .order_by(OperatorQueueItem.enqueued_at.desc())
        .limit(limit)
    )
    if status_filter:
        stmt = stmt.where(OperatorQueueItem.status == status_filter)
    if route:
        stmt = stmt.where(OperatorQueueItem.route == route)
    rows = (await session.execute(stmt)).scalars().all()
    return [_to_list_item(r) for r in rows]


@router.get("/queue/{queue_id}", response_model=QueueDetail)
async def get_queue_item(
    queue_id: UUID, session: AsyncSession = Depends(get_session)
) -> QueueDetail:
    row = await session.get(OperatorQueueItem, queue_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return _to_detail(row)


@router.post("/queue/{queue_id}/claim", response_model=QueueDetail)
async def claim_queue_item(
    queue_id: UUID,
    body: ClaimBody,
    session: AsyncSession = Depends(get_session),
) -> QueueDetail:
    """Assign this item to an operator."""
    row = await session.get(OperatorQueueItem, queue_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if row.status not in ("pending", "in_review"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"queue item is {row.status}; cannot claim",
        )
    if row.operator_id and row.operator_id != body.operator_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"already claimed by {row.operator_id}",
        )
    row.operator_id = body.operator_id
    row.status = "in_review"
    if row.claimed_at is None:
        row.claimed_at = datetime.now(tz=timezone.utc)
    await session.commit()
    log.info("hitl.claimed", queue_id=str(queue_id), operator_id=body.operator_id)
    return _to_detail(row)


@router.post("/queue/{queue_id}/approve", response_model=QueueDetail)
async def approve(
    queue_id: UUID,
    body: ApproveBody,
    session: AsyncSession = Depends(get_session),
) -> QueueDetail:
    """Operator confirms the agent's decision. Idempotent.

    If `customer_email` is provided, the agent's drafted email is sent
    via the configured transport (Resend in prod, log transport in dev).
    The send is itself idempotent — the (action_type='email_sent',
    'queue:{id}:email') row blocks double-sends if the API is replayed.
    """
    row = await session.get(OperatorQueueItem, queue_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if row.status in ("approved", "overridden", "escalated"):
        return _to_detail(row)

    row.status = "approved"
    row.operator_id = body.operator_id
    row.operator_decision = dict(row.agent_decision)
    row.operator_notes = body.notes
    row.completed_at = datetime.now(tz=timezone.utc)

    await _audit(session, action_type="queue_approve", queue_id=queue_id, payload={
        "operator_id": body.operator_id,
        "agent_outcome": (row.agent_decision or {}).get("outcome"),
        "notes": body.notes,
    })

    # Week 16: send the held email if the operator provided contact info.
    if body.customer_email:
        email_body = (row.agent_decision or {}).get("email_draft") or ""
        if email_body:
            await send_email(
                session,
                to=body.customer_email,
                subject=body.email_subject or "Update on your warranty claim",
                body_text=email_body,
                idempotency_key=f"queue:{queue_id}:email",
            )

    await session.commit()
    log.info(
        "hitl.approved",
        queue_id=str(queue_id),
        operator_id=body.operator_id,
        email_sent=bool(body.customer_email),
    )
    return _to_detail(row)


@router.post("/queue/{queue_id}/override", response_model=QueueDetail)
async def override(
    queue_id: UUID,
    body: OverrideBody,
    session: AsyncSession = Depends(get_session),
) -> QueueDetail:
    """Operator replaces the agent's decision. Idempotent.

    For overrides the operator typically writes their own email body
    (the agent's draft reflected the wrong decision). Pass that in
    `email_body_override`; if omitted but `customer_email` is set, the
    agent's draft is sent — usually a mistake, but we honor it.
    """
    row = await session.get(OperatorQueueItem, queue_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if row.status in ("approved", "overridden", "escalated"):
        return _to_detail(row)

    row.status = "overridden"
    row.operator_id = body.operator_id
    row.operator_decision = {
        "outcome": body.outcome,
        "resolution": body.resolution,
        "rationale": body.rationale,
        "source": "operator_override",
    }
    row.operator_notes = body.notes
    row.completed_at = datetime.now(tz=timezone.utc)

    await _audit(session, action_type="queue_override", queue_id=queue_id, payload={
        "operator_id": body.operator_id,
        "agent_outcome": (row.agent_decision or {}).get("outcome"),
        "operator_outcome": body.outcome,
        "operator_resolution": body.resolution,
        "notes": body.notes,
    })

    # Week 16: hold-until-approval send. Use the operator's override
    # body if they supplied one, else fall back to the agent's draft.
    if body.customer_email:
        email_body = body.email_body_override or (
            (row.agent_decision or {}).get("email_draft") or ""
        )
        if email_body:
            await send_email(
                session,
                to=body.customer_email,
                subject=body.email_subject or "Update on your warranty claim",
                body_text=email_body,
                idempotency_key=f"queue:{queue_id}:email",
            )

    await session.commit()
    log.info(
        "hitl.overridden",
        queue_id=str(queue_id),
        agent_outcome=(row.agent_decision or {}).get("outcome"),
        operator_outcome=body.outcome,
        email_sent=bool(body.customer_email),
    )
    return _to_detail(row)


@router.post("/queue/{queue_id}/escalate", response_model=QueueDetail)
async def escalate(
    queue_id: UUID,
    body: EscalateBody,
    session: AsyncSession = Depends(get_session),
) -> QueueDetail:
    """Escalate to higher review. Idempotent."""
    row = await session.get(OperatorQueueItem, queue_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if row.status in ("approved", "overridden", "escalated"):
        return _to_detail(row)

    row.status = "escalated"
    row.operator_id = body.operator_id
    row.operator_notes = body.reason
    row.completed_at = datetime.now(tz=timezone.utc)

    await _audit(session, action_type="queue_escalate", queue_id=queue_id, payload={
        "operator_id": body.operator_id,
        "reason": body.reason,
    })
    await session.commit()
    log.info("hitl.escalated", queue_id=str(queue_id), operator_id=body.operator_id)
    return _to_detail(row)
