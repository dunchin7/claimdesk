"""Shopify-shaped tools (Week 11 stub, Week 16 real).

Today these return synthetic data from the training-set JSONL so the agent
has something realistic to reason over. Week 16 wires real Shopify API
calls behind the same interface — agents don't have to change.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.ai.tools.registry import ToolSpec, register_tool

_TRAINING_PATH = Path(__file__).resolve().parents[3].parent / "data/synthetic/claims_training.jsonl"


# ---------------------------------------------------------------------------
# query_shopify_orders
# ---------------------------------------------------------------------------


class QueryShopifyOrdersInput(BaseModel):
    customer_id: str = Field(description="The customer's UUID to look up orders for.")


class OrderRecord(BaseModel):
    order_id: str
    sku: str
    purchase_date: str
    value_usd: float
    shipping_address: str


class QueryShopifyOrdersOutput(BaseModel):
    status: str = "ok"  # "ok" or "error"
    message: str = ""
    customer_id: str = ""
    orders: list[OrderRecord] = []


async def _query_shopify_orders(inp: QueryShopifyOrdersInput) -> QueryShopifyOrdersOutput:
    """Stub backed by the synthetic training JSONL.

    Returns each claim row as a synthetic "order" — purchase + ship-to + value.
    Sorted oldest-first. Week 16 swaps this body for a real Shopify call.
    """
    if not _TRAINING_PATH.is_file():
        return QueryShopifyOrdersOutput(
            status="error",
            message=f"Customer DB not available (training data missing at {_TRAINING_PATH})",
            customer_id=inp.customer_id,
        )
    orders: list[OrderRecord] = []
    with _TRAINING_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("customer_id") != inp.customer_id:
                continue
            orders.append(OrderRecord(
                order_id=row["claim_id"],  # synthetic: one order per claim
                sku=row["sku"],
                purchase_date=row["purchase_date"],
                value_usd=float(row.get("claim_value_usd", 0)),
                shipping_address=row.get("shipping_address", ""),
            ))
    orders.sort(key=lambda o: o.purchase_date)
    return QueryShopifyOrdersOutput(
        status="ok",
        customer_id=inp.customer_id,
        orders=orders,
    )


register_tool(ToolSpec(
    name="query_shopify_orders",
    description=(
        "Retrieve the order history for a specific customer by their customer_id. "
        "Returns a list of orders with SKU, purchase date, value, and shipping "
        "address. Useful for understanding what the customer has bought before "
        "and whether the current claim's product is on their order history."
    ),
    input_model=QueryShopifyOrdersInput,
    output_model=QueryShopifyOrdersOutput,
    handler=_query_shopify_orders,
))


# ---------------------------------------------------------------------------
# lookup_customer_history
# ---------------------------------------------------------------------------


class LookupCustomerHistoryInput(BaseModel):
    customer_id: str = Field(description="The customer's UUID.")


class PriorClaim(BaseModel):
    claim_id: str
    claim_date: str
    sku: str
    expected_decision: str
    is_fraud: bool


class LookupCustomerHistoryOutput(BaseModel):
    status: str = "ok"
    message: str = ""
    customer_id: str = ""
    first_seen_date: str | None = None
    n_prior_claims: int = 0
    prior_claims: list[PriorClaim] = []
    distinct_shipping_addresses: int = 0
    approved_count: int = 0
    rejected_count: int = 0


async def _lookup_customer_history(
    inp: LookupCustomerHistoryInput,
) -> LookupCustomerHistoryOutput:
    """Aggregate the customer's prior claim record.

    This is the structured cousin of `query_shopify_orders` — instead of
    raw orders it returns a fraud-relevant summary (n claims, distinct
    addresses, won/lost ratio, first-seen date). The agent should call
    this BEFORE deciding on a high-value claim from an unfamiliar
    customer.
    """
    if not _TRAINING_PATH.is_file():
        return LookupCustomerHistoryOutput(
            status="error",
            message="Customer history not available",
            customer_id=inp.customer_id,
        )
    prior: list[PriorClaim] = []
    addresses: set[str] = set()
    approved = 0
    rejected = 0
    first_seen: str | None = None
    with _TRAINING_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("customer_id") != inp.customer_id:
                continue
            if first_seen is None:
                first_seen = row.get("customer_first_seen_date")
            addresses.add((row.get("shipping_address") or "").strip().lower())
            prior.append(PriorClaim(
                claim_id=row["claim_id"],
                claim_date=row["claim_date"],
                sku=row["sku"],
                expected_decision=row["expected_decision"],
                is_fraud=bool(row.get("is_fraud", False)),
            ))
            if row["expected_decision"] == "approve":
                approved += 1
            elif row["expected_decision"] == "reject":
                rejected += 1

    prior.sort(key=lambda p: p.claim_date)
    addresses.discard("")
    return LookupCustomerHistoryOutput(
        customer_id=inp.customer_id,
        first_seen_date=first_seen,
        n_prior_claims=len(prior),
        prior_claims=prior,
        distinct_shipping_addresses=len(addresses),
        approved_count=approved,
        rejected_count=rejected,
    )


register_tool(ToolSpec(
    name="lookup_customer_history",
    description=(
        "Summarize a customer's prior claim history: how many claims, "
        "distinct shipping addresses on file, approve/reject ratio, account "
        "tenure. Use this for any high-value claim or any claim where the "
        "customer's behavior pattern matters (fraud signal triage)."
    ),
    input_model=LookupCustomerHistoryInput,
    output_model=LookupCustomerHistoryOutput,
    handler=_lookup_customer_history,
))
