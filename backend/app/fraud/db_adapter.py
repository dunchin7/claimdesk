"""DB → CustomerContext adapter (Phase-4 P1.5).

The fraud scorer expects a `CustomerContext` built from a list of claim dicts.
In Week 9 those came straight from the training JSONL. In production they
come from the `customers` + `claims` + `orders` tables.

This adapter does the SQL look-up: given a customer email (or claim_id),
returns the `CustomerContext` the scorer needs. If the customer isn't in
the DB yet (a brand-new email — common in early production), returns None
and the pipeline skips fraud scoring honestly rather than hallucinating
features.

Features that need columns we don't have today (shipping_address per claim,
claim_value_usd) are returned as None / NaN — XGBoost handles them natively
as "missing".
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Claim, Customer, Order
from app.fraud.features import CustomerContext

log = get_logger(__name__)


async def build_context_by_email(
    session: AsyncSession, email: str
) -> CustomerContext | None:
    """Build a CustomerContext from `customers` + `claims` tables, keyed by email.

    Returns None if the customer doesn't exist in the DB.
    """
    customer = (
        await session.execute(select(Customer).where(Customer.email == email))
    ).scalar_one_or_none()
    if customer is None:
        return None

    # Pull every claim by this customer, joined with the originating order
    # for the primary address (we don't have shipping_address on Claim yet).
    rows = (
        await session.execute(
            select(Claim, Order)
            .outerjoin(Order, Claim.order_id == Order.id)
            .where(Claim.customer_id == customer.id)
            .order_by(Claim.created_at.asc())
        )
    ).all()

    claims_list: list[dict] = []
    for claim, order in rows:
        claims_list.append({
            "claim_id": str(claim.id),
            "claim_date": claim.created_at.date(),
            # We don't track per-claim shipping_address yet — feature returns
            # None / 0.0 for these on `same_email_diff_address_count` and
            # `address_mismatch_score`. Acceptable for P1.
            "shipping_address": None,
            "claim_value_usd": float(order.total_usd) if order else None,
            "expected_decision": claim.decision,
            "is_fraud": False,  # we don't know this at runtime; only the
            # ground-truth label was set at load time, not exposed to scorer
            "raw_input": claim.raw_input or "",
        })

    return CustomerContext(
        customer_id=str(customer.id),
        email=customer.email,
        first_seen_date=(
            customer.created_at.date() if customer.created_at else None
        ),
        primary_address=None,  # not tracked on Customer today
        claims=claims_list,
    )


async def build_context_by_claim_id(
    session: AsyncSession, claim_id: str | UUID
) -> CustomerContext | None:
    """Build context by looking up the claim, then its customer.

    Use this in the eval flow where we have the claim UUID and want the
    same context the customer themselves would have.
    """
    cid = UUID(str(claim_id))
    claim = await session.get(Claim, cid)
    if claim is None:
        return None
    customer = await session.get(Customer, claim.customer_id)
    if customer is None:
        return None
    return await build_context_by_email(session, customer.email)
