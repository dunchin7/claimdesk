"""Generate the Week-9 fraud training set.

Distinct from `generate_synthetic_claims.py` (the 100-claim eval set, which we
DO NOT touch — the Phase-1 baseline depends on it). This generator:

- Produces ~300 claims with 15% fraud (~45 fraud, ~255 legit)
- Spreads claim_date over a 365-day window (so claims_per_customer_30d is meaningful)
- Generates 180 customers with realistic claim density (most have 0-2 claims;
  fraud-prone customers have 3-6 over a short window)
- Each claim has a `shipping_address` that may vary across a customer's claims
  (fraud signal: same email, multiple shipping addresses)
- Each claim has a `claim_value_usd` derived from SKU + resolution
- Each customer has a `first_seen_date` (tenure)
- Customers can have a `prior_outcomes` distribution (won/lost ratio)

Output is denormalized — every claim row contains everything the feature
extractor needs without a join.

Usage:
    uv run python scripts/generate_fraud_training_set.py \\
        --out data/synthetic/claims_training.jsonl --seed 42 --n 300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))  # so we can import `scripts.*`

from faker import Faker  # noqa: E402

# Reuse the SKU catalog from the eval-set generator so the worlds line up.
from scripts.generate_synthetic_claims import SKUS, TEMPLATES  # noqa: E402


# SKU → approximate retail price (USD). Used to compute claim_value_usd.
SKU_PRICES: dict[str, float] = {
    "EB-PACE-500": 1499.0,
    "EB-PACE-350": 1199.0,
    "EB-LEVEL-2": 1899.0,
    "EB-LEVEL-3": 2299.0,
    "EB-AVENT-X": 2099.0,
    "EB-AVENT-Y": 1599.0,
    "EB-CITY-1": 999.0,
    "EB-CITY-2": 1199.0,
    "EB-CARGO-1": 2499.0,
    "EB-MTN-1": 2199.0,
    "EB-MTN-2": 1799.0,
    "EB-FOLD-1": 1099.0,
    "EB-FOLD-2": 899.0,
    "BAT-PACE": 499.0,
    "BAT-LEVEL": 599.0,
    "BAT-AVENT": 549.0,
    "CHG-STD": 89.0,
    "CHG-FAST": 149.0,
    "ACC-LIGHT": 49.0,
    "ACC-RACK": 39.0,
}


Decision = Literal["approve", "reject", "needs_info"]
DecisionKind = Literal[
    "clear_approve", "clear_reject", "gray", "needs_info", "fraud_suspect"
]

# Fraud patterns we want the classifier to learn:
FRAUD_PATTERNS = [
    "same_email_multi",       # one customer, 4+ claims in <30 days
    "address_mismatch",       # claim ship-to ≠ original order ship-to
    "address_hopper",         # customer ships to 3+ different addresses over time
    "value_spike",            # claim value >> customer's AOV
    "tenure_spike",           # brand-new customer, immediate high-value claim
    "near_window_pattern",    # claim filed ~28 days into 30-day return window, multiple times
    "exif_mismatch",          # photo-based (Week 8 territory)
]


@dataclass
class TrainingClaim:
    claim_id: str
    customer_id: str
    customer_email: str
    customer_first_seen_date: str         # NEW
    sku: str
    product_name: str
    purchase_date: str
    claim_date: str                        # NEW: now spread over a year
    days_since_purchase: int
    raw_input: str
    shipping_address: str                  # NEW
    claim_value_usd: float                 # NEW
    decision_kind: str
    expected_decision: str
    is_fraud: bool
    fraud_pattern: str | None
    tags: list[str] = field(default_factory=list)
    notes: str = ""


def _pick(rng: random.Random, options: list[Any]) -> Any:
    return options[rng.randrange(len(options))]


def _generate_customers(rng: random.Random, fake: Faker, n: int) -> list[dict[str, Any]]:
    today = date(2026, 5, 1)
    customers: list[dict[str, Any]] = []
    for _ in range(n):
        # Customer "first_seen" anywhere from 7 days to 1100 days ago. Skewed
        # toward older customers (real distributions have a long tail).
        tenure_days = int(rng.triangular(7, 1100, 300))
        first_seen = today - timedelta(days=tenure_days)
        # Primary address — most customers have just one
        primary_address = fake.address().replace("\n", ", ")
        customers.append({
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "email": fake.email(),
            "name": fake.name(),
            "primary_address": primary_address,
            "secondary_addresses": [],
            "first_seen_date": first_seen.isoformat(),
            "tenure_days": tenure_days,
        })
    return customers


def _claim_value_for(sku: str, decision_kind: DecisionKind, rng: random.Random) -> float:
    """Estimate USD value at stake on this claim.

    For approve/replacement → full retail. For repair → 30-40% of retail.
    For reject/needs_info → smaller, the customer's *claimed* value. The
    feature `claim_value_to_aov_ratio` keys on this number — fraud cases
    tend to claim more than their order history justifies.
    """
    base = SKU_PRICES.get(sku, 500.0)
    if decision_kind == "clear_approve":
        return round(base * rng.uniform(0.9, 1.05), 2)
    if decision_kind == "clear_reject":
        return round(base * rng.uniform(0.10, 0.40), 2)
    if decision_kind == "fraud_suspect":
        # Fraud claims inflate value
        return round(base * rng.uniform(0.95, 1.35), 2)
    return round(base * rng.uniform(0.20, 0.70), 2)


def _build_claim(
    rng: random.Random,
    fake: Faker,
    today: date,
    customer: dict[str, Any],
    decision_kind: DecisionKind,
    *,
    fraud_pattern: str | None = None,
    nth_in_run: int = 1,
    claim_date_offset_days: int | None = None,
    shipping_address: str | None = None,
) -> TrainingClaim:
    product = _pick(rng, SKUS)
    sku = product["sku"]

    if decision_kind == "clear_approve":
        purchase_days_ago = rng.randint(15, 320)
    elif decision_kind == "clear_reject":
        purchase_days_ago = rng.randint(180, 540)
    elif decision_kind == "gray":
        purchase_days_ago = rng.randint(330, 380)
    elif decision_kind == "needs_info":
        purchase_days_ago = rng.randint(20, 200)
    else:  # fraud_suspect
        if fraud_pattern == "tenure_spike":
            # Fresh customer, near-immediate claim
            purchase_days_ago = rng.randint(5, 20)
        elif fraud_pattern == "near_window_pattern":
            purchase_days_ago = rng.randint(355, 365)
        else:
            purchase_days_ago = rng.randint(20, 90)

    # Spread claim dates across the year. By default uniform; for fraud
    # patterns that require clustering we use a per-pattern offset.
    if claim_date_offset_days is None:
        claim_date_offset_days = rng.randint(0, 364)
    claim_dt = today - timedelta(days=claim_date_offset_days)
    purchase = claim_dt - timedelta(days=purchase_days_ago)

    # Bucket selection per decision kind (same logic as the eval generator)
    if decision_kind == "clear_approve":
        bucket = _pick(rng, ["battery", "motor", "shipping_damage"])
        expected_decision = "approve"
    elif decision_kind == "clear_reject":
        bucket = _pick(rng, ["wear_tear", "accidental_damage", "other"])
        expected_decision = "reject"
    elif decision_kind == "gray":
        bucket = _pick(rng, ["battery", "electrical", "frame"])
        expected_decision = rng.choices(["approve", "needs_info"], weights=[0.4, 0.6])[0]
    elif decision_kind == "needs_info":
        bucket = "any"
        expected_decision = "needs_info"
    else:  # fraud_suspect
        bucket_map = {
            "same_email_multi": "same_email_multi",
            "address_mismatch": "address_mismatch",
            "address_hopper": "same_email_multi",
            "value_spike": "same_email_multi",
            "tenure_spike": "address_mismatch",
            "near_window_pattern": "same_email_multi",
            "exif_mismatch": "exif_mismatch",
        }
        bucket = bucket_map.get(fraud_pattern or "same_email_multi", "same_email_multi")
        expected_decision = "reject"

    template_options = TEMPLATES.get((decision_kind, bucket))
    if not template_options:
        # Fallback for fraud patterns without explicit templates
        template_options = TEMPLATES[("clear_approve", "battery")]
    template = _pick(rng, template_options)
    raw_input = template.format(
        product=product["name"],
        sku=sku,
        purchase_date=purchase.isoformat(),
        days=purchase_days_ago,
        nth=nth_in_run,
    )

    return TrainingClaim(
        claim_id=str(uuid.UUID(int=rng.getrandbits(128))),
        customer_id=customer["id"],
        customer_email=customer["email"],
        customer_first_seen_date=customer["first_seen_date"],
        sku=sku,
        product_name=product["name"],
        purchase_date=purchase.isoformat(),
        claim_date=claim_dt.isoformat(),
        days_since_purchase=purchase_days_ago,
        raw_input=raw_input,
        shipping_address=shipping_address or customer["primary_address"],
        claim_value_usd=_claim_value_for(sku, decision_kind, rng),
        decision_kind=decision_kind,
        expected_decision=expected_decision,
        is_fraud=(decision_kind == "fraud_suspect"),
        fraud_pattern=fraud_pattern,
        tags=[decision_kind] + ([f"fraud_{fraud_pattern}"] if fraud_pattern else []),
        notes=f"auto-generated kind={decision_kind} bucket={bucket}",
    )


def generate(seed: int, n: int) -> list[TrainingClaim]:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)
    today = date(2026, 5, 1)

    # Customers: roughly n/1.5 so we get realistic 0-3 claims/customer for legit
    # users; fraud-prone customers get loaded explicitly below.
    customers = _generate_customers(rng, fake, n=max(60, n // 2))

    # 85% legit, 15% fraud. Split the 85% across the usual decision kinds.
    n_fraud = round(n * 0.15)
    n_legit = n - n_fraud
    legit_split = {
        "clear_approve": round(n_legit * 0.40),
        "clear_reject": round(n_legit * 0.30),
        "gray": round(n_legit * 0.15),
        "needs_info": round(n_legit * 0.15),
    }
    # absorb rounding drift
    drift = n_legit - sum(legit_split.values())
    legit_split["clear_approve"] += drift

    claims: list[TrainingClaim] = []

    # --- Legit claims ---
    for kind, count in legit_split.items():
        for _ in range(count):
            customer = _pick(rng, customers)
            claims.append(_build_claim(rng, fake, today, customer, kind))  # type: ignore[arg-type]

    # --- Fraud claims, distributed across patterns ---
    pattern_dist: list[str] = []
    for p in FRAUD_PATTERNS:
        # Roughly equal allocation
        pattern_dist.extend([p] * (n_fraud // len(FRAUD_PATTERNS)))
    while len(pattern_dist) < n_fraud:
        pattern_dist.append("same_email_multi")
    pattern_dist = pattern_dist[:n_fraud]
    rng.shuffle(pattern_dist)

    # For patterns that need *multiple claims per customer*, we anchor on a
    # dedicated fraud-customer per cluster.
    fraud_customers_pool = customers[: max(8, n_fraud // 3)]

    same_email_clusters: dict[str, dict[str, Any]] = {}

    for i, pattern in enumerate(pattern_dist):
        if pattern == "same_email_multi":
            cluster_id = f"cluster_{i // 4}"  # 4 claims per cluster
            if cluster_id not in same_email_clusters:
                anchor = _pick(rng, fraud_customers_pool)
                same_email_clusters[cluster_id] = {
                    "anchor": anchor,
                    "start_offset": rng.randint(30, 200),  # claim cluster window start
                    "nth": 0,
                }
            cluster = same_email_clusters[cluster_id]
            cluster["nth"] += 1
            # Claims cluster within 20 days
            offset = cluster["start_offset"] - rng.randint(0, 20)
            claims.append(
                _build_claim(
                    rng, fake, today, cluster["anchor"], "fraud_suspect",
                    fraud_pattern=pattern,
                    nth_in_run=cluster["nth"],
                    claim_date_offset_days=offset,
                )
            )
        elif pattern == "address_hopper":
            customer = _pick(rng, fraud_customers_pool)
            # Generate a different shipping address each time
            ship_addr = fake.address().replace("\n", ", ")
            claims.append(
                _build_claim(
                    rng, fake, today, customer, "fraud_suspect",
                    fraud_pattern=pattern,
                    shipping_address=ship_addr,
                )
            )
        elif pattern == "address_mismatch":
            customer = _pick(rng, fraud_customers_pool)
            ship_addr = fake.address().replace("\n", ", ")  # differs from primary
            claims.append(
                _build_claim(
                    rng, fake, today, customer, "fraud_suspect",
                    fraud_pattern=pattern,
                    shipping_address=ship_addr,
                )
            )
        elif pattern == "tenure_spike":
            # Inject a fresh customer just for this claim
            fresh = _generate_customers(rng, fake, 1)[0]
            fresh["first_seen_date"] = (today - timedelta(days=rng.randint(2, 10))).isoformat()
            customers.append(fresh)
            claims.append(
                _build_claim(
                    rng, fake, today, fresh, "fraud_suspect",
                    fraud_pattern=pattern,
                )
            )
        else:  # value_spike, near_window_pattern, exif_mismatch
            customer = _pick(rng, fraud_customers_pool)
            claims.append(
                _build_claim(
                    rng, fake, today, customer, "fraud_suspect",
                    fraud_pattern=pattern,
                )
            )

    rng.shuffle(claims)
    return claims


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/synthetic/claims_training.jsonl"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    claims = generate(args.seed, args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"[ok] wrote {args.out}  ({len(claims)} claims)")

    # Summary
    fraud_n = sum(1 for c in claims if c.is_fraud)
    by_pattern: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    customers_seen: set[str] = set()
    for c in claims:
        by_kind[c.decision_kind] = by_kind.get(c.decision_kind, 0) + 1
        if c.fraud_pattern:
            by_pattern[c.fraud_pattern] = by_pattern.get(c.fraud_pattern, 0) + 1
        customers_seen.add(c.customer_id)
    print(f"  fraud rate:           {fraud_n}/{len(claims)} = {fraud_n/len(claims):.1%}")
    print(f"  unique customers:     {len(customers_seen)}")
    print(f"  claims/customer avg:  {len(claims) / max(len(customers_seen), 1):.2f}")
    print(f"  by decision_kind:     {by_kind}")
    print(f"  by fraud_pattern:     {by_pattern}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
