"""Generate synthetic warranty claims for ClaimDesk.

Produces a stratified golden set:
    30 clear-approve
    25 clear-reject
    20 gray
    15 needs-info
    10 fraud-suspect (orthogonal: each has a real decision label too)

Two modes:
    --llm   Uses Azure GPT-4o-mini to write customer narratives. Realistic
            but costs ~$0.05 per full run.
    (default) Template mode. Free, deterministic, fast. Identical metadata
            to LLM mode for the same seed.

Usage:
    uv run python scripts/generate_synthetic_claims.py \\
        --out data/synthetic/claims.jsonl --seed 42 [--llm] [--n 100]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

# Make `app.*` importable when running from workspace root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from faker import Faker  # noqa: E402

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# 20 fictional e-bike SKUs. Names are deliberately generic to avoid trademarks.
SKUS: list[dict[str, Any]] = [
    {"sku": "EB-PACE-500", "name": "PaceLine 500", "category": "ebike", "motor_w": 500, "battery_wh": 614},
    {"sku": "EB-PACE-350", "name": "PaceLine 350", "category": "ebike", "motor_w": 350, "battery_wh": 460},
    {"sku": "EB-LEVEL-2",  "name": "LevelUp 2",    "category": "ebike", "motor_w": 500, "battery_wh": 672},
    {"sku": "EB-LEVEL-3",  "name": "LevelUp 3",    "category": "ebike", "motor_w": 750, "battery_wh": 720},
    {"sku": "EB-AVENT-X",  "name": "Aventyr X",    "category": "ebike", "motor_w": 750, "battery_wh": 700},
    {"sku": "EB-AVENT-Y",  "name": "Aventyr Y",    "category": "ebike", "motor_w": 500, "battery_wh": 540},
    {"sku": "EB-CITY-1",   "name": "CityHop 1",    "category": "ebike", "motor_w": 250, "battery_wh": 360},
    {"sku": "EB-CITY-2",   "name": "CityHop 2",    "category": "ebike", "motor_w": 350, "battery_wh": 420},
    {"sku": "EB-CARGO-1",  "name": "HaulPro Cargo","category": "ebike", "motor_w": 750, "battery_wh": 840},
    {"sku": "EB-MTN-1",    "name": "TrailBolt 1",  "category": "ebike", "motor_w": 750, "battery_wh": 720},
    {"sku": "EB-MTN-2",    "name": "TrailBolt 2",  "category": "ebike", "motor_w": 500, "battery_wh": 614},
    {"sku": "EB-FOLD-1",   "name": "FoldStep 1",   "category": "ebike", "motor_w": 350, "battery_wh": 374},
    {"sku": "EB-FOLD-2",   "name": "FoldStep 2",   "category": "ebike", "motor_w": 250, "battery_wh": 280},
    {"sku": "BAT-PACE",    "name": "PaceLine Battery", "category": "ebike_battery", "battery_wh": 614},
    {"sku": "BAT-LEVEL",   "name": "LevelUp Battery",  "category": "ebike_battery", "battery_wh": 672},
    {"sku": "BAT-AVENT",   "name": "Aventyr Battery",  "category": "ebike_battery", "battery_wh": 700},
    {"sku": "CHG-STD",     "name": "Standard Charger", "category": "charger", "watts": 100},
    {"sku": "CHG-FAST",    "name": "Fast Charger",     "category": "charger", "watts": 200},
    {"sku": "ACC-LIGHT",   "name": "Headlight Pro",    "category": "accessory"},
    {"sku": "ACC-RACK",    "name": "Rear Rack",        "category": "accessory"},
]

DECISION_KIND = Literal["clear_approve", "clear_reject", "gray", "needs_info", "fraud_suspect"]

# Stratified counts. Sum must equal `--n` (default 100).
STRATA: dict[DECISION_KIND, int] = {
    "clear_approve": 30,
    "clear_reject": 25,
    "gray": 20,
    "needs_info": 15,
    "fraud_suspect": 10,
}

# Templated claim narratives keyed by (decision_kind, failure_mode_or_pattern).
# These provide deterministic text without needing the LLM.
TEMPLATES: dict[tuple[DECISION_KIND, str], list[str]] = {
    ("clear_approve", "battery"): [
        "Hi, my {product} battery ({sku}) won't hold a charge anymore. I bought it on {purchase_date}, only {days} days ago, and I have maybe 30 charge cycles. The charger shows green but range dropped from 35 miles to about 4. I have receipts and the original packaging. Photos attached showing battery, charger LEDs, and serial sticker.",
        "My {sku} stopped powering on after {days} days of normal commuting use. Receipt from {purchase_date} attached. Battery was always charged indoors, never below freezing. Photos: battery terminals, dashboard error E-04.",
    ],
    ("clear_approve", "motor"): [
        "Hi support — my {product} ({sku}) motor started making a grinding noise after {days} days. It now cuts out intermittently above 8 mph. Bike has been stored indoors. Purchased {purchase_date}. Photos of the motor housing and a short video showing the cutout attached.",
    ],
    ("clear_approve", "shipping_damage"): [
        "Box arrived dented yesterday. Inside, the {product} ({sku}) frame has a clear dent on the down tube and the rear derailleur is bent. Order date {purchase_date}, delivered {days} days later. Photos show the box damage, frame dent, and derailleur. I have not assembled it.",
    ],
    ("clear_reject", "wear_tear"): [
        "Hi, my {product} ({sku}) tires are bald and the brake pads are worn out. I've put about 2,400 miles on it since {purchase_date} ({days} days). Can you replace them under warranty?",
        "The grips on my {product} are starting to peel and the chain is rusty. Bought {purchase_date}. I ride it daily in the rain. Asking for warranty replacement of grips and chain.",
    ],
    ("clear_reject", "accidental_damage"): [
        "I crashed my {sku} into a curb yesterday. The front wheel is bent and the handlebars are scraped up. Bought {purchase_date}. The bike still works. Can you cover this under warranty? I do not have the extended damage protection plan.",
        "Dropped my {product} off the back of my truck at the trailhead. Frame has a crack. Purchased {purchase_date}. Hoping warranty covers it — I have only had it {days} days.",
    ],
    ("clear_reject", "other"): [
        "I want to return my {product} ({sku}) because I changed my mind. I bought it {purchase_date}, {days} days ago. I have ridden it twice. Please process a full refund.",
    ],
    ("gray", "battery"): [
        "Hi, the battery on my {sku} is getting noticeably weaker. I bought it on {purchase_date} ({days} days ago) and I have done about 180 charge cycles. Range used to be 40 miles and now it is closer to 28. Is this covered? I do not have an exact capacity number.",
    ],
    ("gray", "electrical"): [
        "My {product} display flickers and sometimes the assist drops out for a second or two. Started about 3 weeks ago. Purchased {purchase_date}. I rode through some heavy rain a couple of times. No water damage I can see, but maybe related?",
    ],
    ("gray", "frame"): [
        "There is a small hairline crack near the {sku} bottom bracket. I noticed it during a clean. Bike is {days} days old (bought {purchase_date}). I ride mixed pavement and gravel. I have not crashed it. Photos attached but the crack is hard to see.",
    ],
    ("needs_info", "any"): [
        "the bike is broken can you fix it",
        "my new ebike doesnt work, please help. it was expensive",
        "Hi, I'd like to file a warranty claim. The thing isn't working right.",
        "battery problem on the ebike i bought from you. need replacement asap",
    ],
    ("fraud_suspect", "same_email_multi"): [
        "Hi support, third time this is happening. My new {product} battery died again. Need urgent replacement. This is the {nth} battery I've claimed in 6 weeks. Different addresses because I move around for work.",
    ],
    ("fraud_suspect", "address_mismatch"): [
        "My {sku} arrived damaged. Please ship the replacement to a different address than the original order — I'm staying with a friend. Frame is cracked, photos attached. Purchased {purchase_date}.",
    ],
    ("fraud_suspect", "exif_mismatch"): [
        "Battery on my {product} ({sku}) is not working. Photos attached. I bought it {purchase_date}. Need a refund or replacement.",
    ],
}


@dataclass
class GoldenClaim:
    claim_id: str
    customer_id: str
    customer_email: str
    sku: str
    product_name: str
    purchase_date: str        # ISO date
    claim_date: str           # ISO date
    days_since_purchase: int
    raw_input: str
    photo_descriptions: list[str]
    decision_kind: str
    expected_decision: Literal["approve", "reject", "needs_info"]
    expected_resolution: Literal["refund", "replacement", "repair", "store_credit", "none"]
    expected_citation: str
    is_fraud: bool
    fraud_pattern: str | None
    # Ground truth for the extraction eval (Week 2). Each entry is the value
    # the LLM should produce for that field given `raw_input`. `None` here
    # means "either null or any value is acceptable" — the eval treats those
    # fields as un-scorable rather than wrong.
    expected_extraction: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _pick(rng: random.Random, options: list[Any]) -> Any:
    return options[rng.randrange(len(options))]


# ---------------------------------------------------------------------------
# Per-template expected extraction labels.
#
# Maps (decision_kind, bucket) → partial ClaimExtraction expected by the
# auto-eval. Fields not listed are not scored. Fields explicitly set to None
# are scored ("null is the right answer").
# ---------------------------------------------------------------------------
_EXTRACTION_LABELS: dict[tuple[DECISION_KIND, str], dict[str, Any]] = {
    ("clear_approve", "battery"): {
        "failure_mode": "battery",
        "claim_type": "defect",
        "severity": "functional",
        "evidence_strength": "strong",
        "prior_contact_attempts": False,
    },
    ("clear_approve", "motor"): {
        "failure_mode": "motor",
        "claim_type": "defect",
        "severity": "functional",
        "evidence_strength": "strong",
        "prior_contact_attempts": False,
    },
    ("clear_approve", "shipping_damage"): {
        "failure_mode": "shipping_damage",
        "claim_type": "shipping",
        "evidence_strength": "strong",
        "prior_contact_attempts": False,
    },
    ("clear_reject", "wear_tear"): {
        "claim_type": "wear_tear",
        "prior_contact_attempts": False,
    },
    ("clear_reject", "accidental_damage"): {
        "claim_type": "accidental_damage",
        "prior_contact_attempts": False,
    },
    ("clear_reject", "other"): {
        "prior_contact_attempts": False,
    },
    ("gray", "battery"): {
        "failure_mode": "battery",
        "claim_type": "defect",
        "prior_contact_attempts": False,
    },
    ("gray", "electrical"): {
        "failure_mode": "electrical",
        "claim_type": "defect",
        "prior_contact_attempts": False,
    },
    ("gray", "frame"): {
        "failure_mode": "frame",
        "claim_type": "defect",
        "prior_contact_attempts": False,
    },
    ("needs_info", "any"): {
        "evidence_strength": "weak",
        "prior_contact_attempts": False,
    },
    ("fraud_suspect", "same_email_multi"): {
        "failure_mode": "battery",
        "prior_contact_attempts": True,
    },
    ("fraud_suspect", "address_mismatch"): {
        "claim_type": "shipping",
        "prior_contact_attempts": False,
    },
    ("fraud_suspect", "exif_mismatch"): {
        "failure_mode": "battery",
        "prior_contact_attempts": False,
    },
}


def _build_claim(
    rng: random.Random,
    fake: Faker,
    decision_kind: DECISION_KIND,
    customer: dict[str, Any],
    products: list[dict[str, Any]],
    fraud_pattern: str | None = None,
    nth: int = 1,
) -> GoldenClaim:
    today = date(2026, 5, 1)
    product = _pick(rng, products)
    sku = product["sku"]

    # Days since purchase varies by decision kind.
    if decision_kind == "clear_approve":
        days = rng.randint(15, 300)
    elif decision_kind == "clear_reject":
        days = rng.randint(180, 540)  # often outside window or high mileage
    elif decision_kind == "gray":
        days = rng.randint(330, 380)  # near boundary
    elif decision_kind == "needs_info":
        days = rng.randint(20, 200)
    else:  # fraud_suspect
        days = rng.randint(20, 28)  # late-window pattern

    purchase = today - timedelta(days=days)
    claim_dt = today - timedelta(days=rng.randint(0, 5))

    # Pick a template bucket
    if decision_kind == "clear_approve":
        bucket = _pick(rng, ["battery", "motor", "shipping_damage"])
        decision = "approve"
        resolution = (
            "replacement" if bucket in ("battery", "shipping_damage") else "repair"
        )
        citation = (
            "Manufacturer defects in materials or workmanship are covered for "
            "twelve (12) months from the date of purchase."
        )
    elif decision_kind == "clear_reject":
        bucket = _pick(rng, ["wear_tear", "accidental_damage", "other"])
        decision = "reject"
        resolution = "none"
        citation = (
            "Normal wear and tear, including tires, brake pads, grips, chains, "
            "and cables, is excluded from warranty coverage."
            if bucket == "wear_tear"
            else "Accidental damage is not covered under the standard warranty; "
            "extended damage protection (sold separately) is required."
            if bucket == "accidental_damage"
            else "Buyer's-remorse returns are limited to 14 days from delivery "
            "for unused product in original packaging."
        )
    elif decision_kind == "gray":
        bucket = _pick(rng, ["battery", "electrical", "frame"])
        decision = rng.choices(["approve", "needs_info"], weights=[0.4, 0.6])[0]
        resolution = "replacement" if decision == "approve" else "none"
        citation = (
            "Battery capacity is warranted at 70% of original after 24 months "
            "or 800 charge cycles, whichever comes first."
        )
    elif decision_kind == "needs_info":
        bucket = "any"
        decision = "needs_info"
        resolution = "none"
        citation = (
            "Claims must include the serial number, date of purchase, "
            "and photographs documenting the issue."
        )
    else:  # fraud_suspect
        bucket = fraud_pattern or "same_email_multi"
        # Fraud claims: 6 reject, 3 needs_info, 1 gray
        decision = "reject"
        resolution = "none"
        citation = (
            "Claims involving inconsistent shipping addresses, repeat patterns, "
            "or evidence of misrepresentation are subject to additional review."
        )

    template_options = TEMPLATES[(decision_kind, bucket)]
    template = _pick(rng, template_options)
    raw_input = template.format(
        product=product["name"],
        sku=sku,
        purchase_date=purchase.isoformat(),
        days=days,
        nth=nth,
    )

    photo_descriptions: list[str] = []
    if decision_kind in ("clear_approve", "clear_reject", "gray"):
        photo_descriptions = [
            f"close-up of {bucket} area showing the issue",
            f"{product['name']} serial number sticker",
        ]
    if decision_kind == "fraud_suspect" and bucket == "exif_mismatch":
        photo_descriptions = [
            "stock-photo-style image with no EXIF metadata",
            "photo timestamp predates the claim date by 8 months",
        ]

    tags = [decision_kind]
    if decision_kind == "fraud_suspect":
        tags.append(f"fraud_pattern_{bucket}")

    # Ground-truth extraction labels for the Week 2 auto-eval.
    expected = dict(_EXTRACTION_LABELS.get((decision_kind, bucket), {}))
    # Always assert the SKU when the customer text uses one (which our
    # templates do for every kind except "needs_info").
    if decision_kind != "needs_info" and "{sku}" in template:
        expected["sku"] = sku

    return GoldenClaim(
        claim_id=str(uuid.UUID(int=rng.getrandbits(128))),
        customer_id=customer["id"],
        customer_email=customer["email"],
        sku=sku,
        product_name=product["name"],
        purchase_date=purchase.isoformat(),
        claim_date=claim_dt.isoformat(),
        days_since_purchase=days,
        raw_input=raw_input,
        photo_descriptions=photo_descriptions,
        decision_kind=decision_kind,
        expected_decision=decision,  # type: ignore[arg-type]
        expected_resolution=resolution,  # type: ignore[arg-type]
        expected_citation=citation,
        is_fraud=(decision_kind == "fraud_suspect"),
        fraud_pattern=bucket if decision_kind == "fraud_suspect" else None,
        expected_extraction=expected,
        tags=tags,
        notes=f"auto-generated stratum={decision_kind} bucket={bucket}",
    )


def _generate_customers(rng: random.Random, fake: Faker, n: int = 60) -> list[dict[str, Any]]:
    customers = []
    for _ in range(n):
        customers.append(
            {
                "id": str(uuid.UUID(int=rng.getrandbits(128))),
                "email": fake.email(),
                "name": fake.name(),
                "address": fake.address().replace("\n", ", "),
            }
        )
    return customers


def generate_claims(seed: int, n: int) -> list[GoldenClaim]:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    customers = _generate_customers(rng, fake, n=60)

    # Honor the strata if n=100; otherwise scale proportionally.
    scale = n / 100
    counts: dict[DECISION_KIND, int] = {
        k: max(1, round(v * scale)) for k, v in STRATA.items()
    }
    # Adjust rounding drift
    drift = n - sum(counts.values())
    if drift != 0:
        counts["clear_approve"] += drift  # absorb in largest bucket

    claims: list[GoldenClaim] = []

    for kind, count in counts.items():
        if kind == "fraud_suspect":
            # Distribute across the 3 patterns: 4 same-email-multi, 3 address-mismatch, 3 exif-mismatch
            pattern_dist = (
                ["same_email_multi"] * round(count * 0.4)
                + ["address_mismatch"] * round(count * 0.3)
                + ["exif_mismatch"] * round(count * 0.3)
            )
            # Pad/truncate to count
            while len(pattern_dist) < count:
                pattern_dist.append("same_email_multi")
            pattern_dist = pattern_dist[:count]

            # For same-email-multi, reuse one customer across multiple claims
            shared_customer = customers[0]
            for i, pattern in enumerate(pattern_dist):
                customer = (
                    shared_customer if pattern == "same_email_multi" else _pick(rng, customers)
                )
                claims.append(
                    _build_claim(
                        rng, fake, kind, customer, SKUS, fraud_pattern=pattern, nth=i + 1
                    )
                )
        else:
            for _ in range(count):
                customer = _pick(rng, customers)
                claims.append(_build_claim(rng, fake, kind, customer, SKUS))

    rng.shuffle(claims)
    return claims


# ---------------------------------------------------------------------------
# Optional: LLM-rewritten narratives
# ---------------------------------------------------------------------------


async def llm_rewrite_narratives(claims: list[GoldenClaim]) -> list[GoldenClaim]:
    """Rewrite each claim's `raw_input` via Azure GPT-4o-mini to sound more
    natural. Metadata (decision, citation, etc.) is unchanged.
    """
    from app.ai.llm import chat  # noqa: PLC0415

    system = (
        "You rewrite warranty-claim messages from the customer's point of view. "
        "Keep ALL factual details (SKU, dates, days, mileage, symptoms) exactly as "
        "given. Vary tone, length, and word choice. Return only the rewritten "
        "message. No preamble."
    )

    async def rewrite_one(c: GoldenClaim) -> GoldenClaim:
        try:
            resp = await chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": c.raw_input},
                ],
                model_alias="extractor",
                temperature=0.7,
                max_tokens=400,
            )
            text = resp.choices[0].message.content.strip()  # type: ignore[union-attr]
            if text:
                c.raw_input = text
        except Exception as e:  # noqa: BLE001
            print(f"[warn] LLM rewrite failed for {c.claim_id}: {e}", file=sys.stderr)
        return c

    # Cap concurrency at 5 to be polite to Azure quotas.
    sem = asyncio.Semaphore(5)

    async def gated(c: GoldenClaim) -> GoldenClaim:
        async with sem:
            return await rewrite_one(c)

    return await asyncio.gather(*(gated(c) for c in claims))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, claims: list[GoldenClaim]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def print_distribution(claims: list[GoldenClaim]) -> None:
    print(f"\nGenerated {len(claims)} claims")
    print("-" * 50)
    by_kind: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    fraud = 0
    for c in claims:
        by_kind[c.decision_kind] = by_kind.get(c.decision_kind, 0) + 1
        by_decision[c.expected_decision] = by_decision.get(c.expected_decision, 0) + 1
        if c.is_fraud:
            fraud += 1
    print("By stratum:")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:<18} {v:>3}")
    print("By expected_decision:")
    for k, v in sorted(by_decision.items()):
        print(f"  {k:<18} {v:>3}")
    print(f"Fraud-flagged: {fraud}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/synthetic/claims.jsonl"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Rewrite narratives via Azure GPT-4o-mini for variety (~$0.05/run).",
    )
    args = parser.parse_args()

    claims = generate_claims(seed=args.seed, n=args.n)

    if args.llm:
        print(f"[info] Rewriting {len(claims)} narratives via Azure GPT-4o-mini...")
        claims = asyncio.run(llm_rewrite_narratives(claims))

    write_jsonl(args.out, claims)
    print(f"[ok] wrote {args.out}")
    print_distribution(claims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
