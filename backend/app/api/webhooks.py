"""Inbound webhooks.

Two providers planned:
- Shopify: order / customer / refund events. HMAC-verified.
- Resend inbound email: customer replies. Will resume paused agent runs.

This module verifies signatures, dedupes by (provider, event_id), and
stores the raw payload in `webhook_events` for audit + replay. Heavy
downstream processing (mirroring orders, resuming agents) is dispatched
from here but not done inline — webhooks must ack quickly or Shopify
will retry.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.limits import LIMIT_WEBHOOK, limiter
from app.core.logging import get_logger
from app.db.models import OperatorQueueItem, WebhookEvent
from app.db.session import get_session
from app.webhooks.email_inbound import parse_inbound, verify_resend_signature
from app.webhooks.shopify import (
    KNOWN_TOPICS,
    extract_event_id,
    verify_shopify_hmac,
)

router = APIRouter()
log = get_logger(__name__)


@router.post("/shopify", status_code=status.HTTP_200_OK)
@limiter.limit(LIMIT_WEBHOOK)
async def shopify_webhook(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Receive a Shopify webhook.

    Returns:
        200 + `{"status": "accepted"}` on first delivery.
        200 + `{"status": "duplicate"}` on retry of an event we already saw
          (Shopify treats 200 as "stop retrying", which is what we want).
        401 on missing or invalid HMAC.
        503 if the server isn't configured with a webhook secret.
        422 if the body isn't JSON.

    We read the raw body BEFORE parsing JSON so the HMAC is verified
    against the bytes Shopify actually sent.
    """
    raw_body = await request.body()

    signature = request.headers.get("X-Shopify-Hmac-Sha256")
    topic = request.headers.get("X-Shopify-Topic", "").strip()
    webhook_id = request.headers.get("X-Shopify-Webhook-Id")

    # Verify HMAC first — never look at the body before we trust it.
    from app.core.config import get_settings
    if not get_settings().shopify_api_secret:
        log.error("shopify.webhook.unconfigured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook secret not configured",
        )
    if not verify_shopify_hmac(raw_body, signature):
        log.warning(
            "shopify.webhook.invalid_hmac",
            topic=topic,
            ip=request.client.host if request.client else None,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="HMAC verification failed",
        )

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Shopify-Topic header",
        )

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON body: {type(e).__name__}",
        ) from e
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Webhook body must be a JSON object",
        )

    event_id = extract_event_id(topic, payload, webhook_id)

    # Dedupe insert. We rely on the UNIQUE constraint, not a pre-SELECT,
    # because the SELECT-then-INSERT race is real even at low volumes.
    event = WebhookEvent(
        provider="shopify",
        topic=topic,
        event_id=event_id,
        payload=payload,
        processed=False,
    )
    session.add(event)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        log.info(
            "shopify.webhook.duplicate",
            topic=topic,
            event_id=event_id,
        )
        return {"status": "duplicate", "topic": topic, "event_id": event_id}

    if topic not in KNOWN_TOPICS:
        log.info(
            "shopify.webhook.unhandled_topic",
            topic=topic,
            event_id=event_id,
        )
        # Still 200 — we've persisted, and we don't want Shopify retrying.
        return {"status": "accepted", "topic": topic, "event_id": event_id, "handled": False}

    # Real downstream mirroring (writing to `customers` / `orders` tables)
    # is left as a stub. The verified event row is the durable handoff:
    # a background worker can replay `processed=False` events when the
    # mirror code lands.
    log.info(
        "shopify.webhook.accepted",
        topic=topic,
        event_id=event_id,
        payload_id=payload.get("id"),
    )
    return {"status": "accepted", "topic": topic, "event_id": event_id, "handled": True}


@router.post("/email-reply", status_code=status.HTTP_200_OK)
@limiter.limit(LIMIT_WEBHOOK)
async def email_reply_webhook(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Resend inbound — customer reply to a held email.

    HMAC-verified via `RESEND_WEBHOOK_SECRET`. The thread_id is parsed
    from the subject's `[Claim #...]` suffix or the reply+<id>@ subaddress.
    If we find a matching `operator_queue` row, we append the inbound text
    to `operator_notes` and reopen the item (status → in_review).

    The graph-resume hook (calling `Command(resume=...)`) lands when the
    multi-agent graph adds an `interrupt()` step — today this endpoint
    is the durable receiver but doesn't itself wake an agent.
    """
    raw_body = await request.body()
    signature = request.headers.get("svix-signature") or request.headers.get(
        "X-Resend-Signature"
    )

    from app.core.config import get_settings
    if not get_settings().resend_webhook_secret:
        log.error("resend.webhook.unconfigured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Resend webhook secret not configured",
        )
    if not verify_resend_signature(raw_body, signature):
        log.warning(
            "resend.webhook.invalid_signature",
            ip=request.client.host if request.client else None,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="HMAC verification failed",
        )

    import json as _json
    try:
        payload = _json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON body: {type(e).__name__}",
        ) from e
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Webhook body must be a JSON object",
        )

    # Dedupe via the Svix message id if present, else hash of headers
    msg_id = request.headers.get("svix-id") or request.headers.get(
        "X-Resend-Event-Id"
    )
    event = WebhookEvent(
        provider="resend",
        topic=str(payload.get("type") or "email.received"),
        event_id=msg_id or f"resend-anon:{hash(raw_body)}",
        payload=payload,
        processed=False,
    )
    session.add(event)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        log.info("resend.webhook.duplicate", event_id=msg_id)
        return {"status": "duplicate"}

    inbound = parse_inbound(payload)
    if inbound.thread_id is None:
        log.info(
            "resend.webhook.no_thread_id",
            from_=inbound.from_email,
            subject_preview=inbound.subject[:60],
        )
        return {"status": "accepted", "matched": False}

    # Look up the queue row by id (thread_id == queue_id for now).
    from uuid import UUID
    try:
        qid = UUID(inbound.thread_id)
    except ValueError:
        log.info("resend.webhook.thread_id_not_uuid", thread_id=inbound.thread_id)
        return {"status": "accepted", "matched": False}

    row = (
        await session.execute(
            select(OperatorQueueItem).where(OperatorQueueItem.id == qid)
        )
    ).scalar_one_or_none()
    if row is None:
        log.info("resend.webhook.no_matching_queue", thread_id=inbound.thread_id)
        return {"status": "accepted", "matched": False}

    # Append to operator_notes and reopen
    prior_notes = row.operator_notes or ""
    addendum = (
        f"\n\n--- customer reply ({inbound.from_email}) ---\n"
        f"Subject: {inbound.subject}\n\n{inbound.text}"
    )
    row.operator_notes = (prior_notes + addendum).strip()[:32_000]
    # Reopen if the operator had already completed this — they need to
    # see the new info before re-resolving.
    if row.status in ("approved", "overridden", "escalated", "in_review"):
        row.status = "in_review"
    event.processed = True
    await session.commit()

    log.info(
        "resend.webhook.matched",
        thread_id=inbound.thread_id,
        from_=inbound.from_email,
        prior_status=row.status,
    )
    return {
        "status": "accepted",
        "matched": True,
        "queue_id": inbound.thread_id,
    }


@router.get("/__placeholder")
async def _placeholder() -> dict[str, str]:
    """Legacy mounting probe."""
    return {"status": "ok"}
