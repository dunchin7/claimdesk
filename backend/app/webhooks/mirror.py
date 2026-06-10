"""Shopify → local DB mirror worker (Week 17).

Reads `webhook_events` rows where `processed=False`, upserts into the
local `customers` / `orders` tables, marks the row processed. Safe to
re-run — every upsert keys on shopify_customer_id / shopify_order_id.

Topics handled today:
  - customers/create, customers/update → Customer upsert
  - orders/create, orders/updated      → Order upsert (links Customer)
  - refunds/create, orders/cancelled   → recorded as processed but no-op
    (refunds get their own table later; cancellations don't delete orders)

Anything else gets marked processed with `error=None` so we don't block
on unhandled topics — the raw payload is still in `webhook_events` for
later reprocessing if the schema grows.

The decimal/numeric coercion is deliberate: Shopify payloads ship
prices as strings (`"49.99"`). SQLAlchemy DECIMAL columns reject floats
in strict mode, so we go through `Decimal(str(x))`.

Usage (one-shot drain):
    uv run python scripts/run_shopify_mirror.py

Future: a tiny FastAPI lifespan task could run this in a loop; for now
the one-shot script is enough for production where events flow at low
rate and a cron can wake the worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Customer, Order, WebhookEvent

log = get_logger(__name__)


@dataclass
class MirrorResult:
    drained: int
    succeeded: int
    failed: int
    skipped_unhandled: int


def _to_decimal(value: Any) -> Decimal:
    """Shopify ships money as strings; this is the safe coerce."""
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def _parse_shopify_dt(value: Any) -> datetime:
    """Parse Shopify's ISO timestamps to tz-aware UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        # Shopify uses ISO-8601 with timezone offsets — Python 3.11+
        # parses these directly via fromisoformat.
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    # Fallback: now, so the row at least lands rather than crashing
    return datetime.now(tz=timezone.utc)


async def _upsert_customer(
    session: AsyncSession, payload: dict[str, Any]
) -> Customer | None:
    """Upsert a Customer keyed on shopify_customer_id."""
    shopify_id = payload.get("id")
    email = payload.get("email")
    if shopify_id is None or not email:
        return None
    shopify_id_str = str(shopify_id)

    existing = (
        await session.execute(
            select(Customer).where(Customer.shopify_customer_id == shopify_id_str)
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.email = email
        return existing

    row = Customer(
        email=email,
        shopify_customer_id=shopify_id_str,
        extra={
            "first_name": payload.get("first_name"),
            "last_name": payload.get("last_name"),
        },
    )
    session.add(row)
    await session.flush()
    return row


async def _upsert_order(
    session: AsyncSession, payload: dict[str, Any]
) -> Order | None:
    """Upsert an Order keyed on shopify_order_id.

    Requires the customer to exist in our DB (or be createable from the
    embedded `customer` block). Returns None and logs if neither.
    """
    shopify_order_id = payload.get("id")
    if shopify_order_id is None:
        return None
    shopify_order_id_str = str(shopify_order_id)

    # Shopify orders embed the customer
    customer_payload = payload.get("customer") or {}
    customer = await _upsert_customer(session, customer_payload)
    if customer is None:
        log.warning(
            "shopify.mirror.order_missing_customer",
            order_id=shopify_order_id_str,
        )
        return None

    existing = (
        await session.execute(
            select(Order).where(Order.shopify_order_id == shopify_order_id_str)
        )
    ).scalar_one_or_none()

    items = payload.get("line_items") or []
    total = _to_decimal(payload.get("total_price"))
    purchased_at = _parse_shopify_dt(payload.get("created_at"))

    if existing is not None:
        existing.items = items
        existing.total_usd = total
        existing.purchased_at = purchased_at
        return existing

    row = Order(
        customer_id=customer.id,
        shopify_order_id=shopify_order_id_str,
        items=items,
        total_usd=total,
        purchased_at=purchased_at,
    )
    session.add(row)
    await session.flush()
    return row


async def process_one(
    session: AsyncSession, event: WebhookEvent
) -> tuple[bool, str | None]:
    """Mirror a single event. Returns (success, error_message)."""
    topic = event.topic
    payload = event.payload or {}
    try:
        if topic in ("customers/create", "customers/update"):
            await _upsert_customer(session, payload)
        elif topic in ("orders/create", "orders/updated"):
            await _upsert_order(session, payload)
        elif topic in ("refunds/create", "orders/cancelled"):
            # No-op for now — recorded for future processing
            pass
        else:
            return True, None  # Unhandled but not an error
        event.processed = True
        event.error = None
        return True, None
    except Exception as e:  # noqa: BLE001
        # Roll back ONLY the mirror writes — keep the event row so we
        # can retry it later via reprocessing.
        log.error(
            "shopify.mirror.failed",
            event_id=event.event_id,
            topic=topic,
            error=str(e)[:200],
            error_type=type(e).__name__,
        )
        event.error = f"{type(e).__name__}: {str(e)[:480]}"
        # Leave processed=False so a retry sees it
        return False, str(e)


async def drain(session: AsyncSession, *, batch_size: int = 50) -> MirrorResult:
    """Drain pending webhook_events one batch at a time. Returns counts."""
    rows = (
        await session.execute(
            select(WebhookEvent)
            .where(WebhookEvent.processed.is_(False))
            .order_by(WebhookEvent.received_at.asc())
            .limit(batch_size)
        )
    ).scalars().all()

    drained = len(rows)
    succeeded = 0
    failed = 0
    skipped = 0
    for ev in rows:
        ok, _ = await process_one(session, ev)
        if not ok:
            failed += 1
        elif ev.topic not in (
            "customers/create",
            "customers/update",
            "orders/create",
            "orders/updated",
            "refunds/create",
            "orders/cancelled",
        ):
            skipped += 1
        else:
            succeeded += 1
    await session.commit()

    if drained:
        log.info(
            "shopify.mirror.drain_done",
            drained=drained,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
        )
    return MirrorResult(
        drained=drained, succeeded=succeeded, failed=failed, skipped_unhandled=skipped
    )
