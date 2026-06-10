"""Fraud feature extraction (Week 9).

The 11 features the design doc calls out, computed from a claim record +
its customer's prior history. Each feature is a function on serializable
inputs (dicts / scalars) so we can:
- Run on the training jsonl without DB access
- Run on a live claim row via a thin adapter

Photo-derived features (`photo_ai_generated_likelihood`, `exif_consistency_score`)
return None when no photo is attached. XGBoost handles None natively as
"missing"; the LLM judge sees the absence as no-signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any


@dataclass
class CustomerContext:
    """All the historical context we need for a single claim's features.

    Pre-built from the training jsonl (or DB rows) once per customer, then
    sliced per-claim by `compute_features`.
    """

    customer_id: str
    email: str
    first_seen_date: date | None
    primary_address: str | None
    # All claims by this customer, in any order, including the one being scored.
    # Each entry: {claim_id, claim_date (date), shipping_address, claim_value_usd,
    #              expected_decision, is_fraud, raw_input}
    claims: list[dict[str, Any]]


def _parse_date(s: str | date | None) -> date | None:
    if s is None or isinstance(s, date):
        return s  # already a date or None
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            return None


def _normalize_address(s: str | None) -> str:
    if not s:
        return ""
    # Lowercase, collapse whitespace + commas. Fuzzy enough for our use.
    return " ".join(s.lower().replace(",", " ").split())


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def claims_per_customer_30d(
    target_claim_date: date, prior_claims: list[dict[str, Any]]
) -> int:
    """Count of OTHER claims by this customer in the 30 days BEFORE the target.

    Excludes the target claim itself. `prior_claims` should be the full
    customer history; we filter to claims whose date < target.
    """
    n = 0
    for c in prior_claims:
        cd = _parse_date(c.get("claim_date"))
        if cd is None or cd >= target_claim_date:
            continue
        if (target_claim_date - cd).days <= 30:
            n += 1
    return n


def same_email_diff_address_count(ctx: CustomerContext) -> int:
    """Distinct shipping addresses on file for this customer (normalized).

    Heavy fraud signal at 3+: address-hopper / shell-game pattern.
    """
    addrs = {_normalize_address(c.get("shipping_address")) for c in ctx.claims}
    addrs.discard("")  # ignore empties
    return len(addrs)


def address_mismatch_score(this_claim: dict[str, Any], ctx: CustomerContext) -> float:
    """0.0 if shipping address matches the customer's primary; 1.0 if completely different.

    Uses SequenceMatcher ratio on normalized strings. 1.0 - ratio gives us
    a "mismatch distance" in [0, 1].
    """
    primary = _normalize_address(ctx.primary_address)
    here = _normalize_address(this_claim.get("shipping_address"))
    if not primary or not here:
        return 0.0
    return 1.0 - SequenceMatcher(a=primary, b=here, autojunk=False).ratio()


def claim_value_to_aov_ratio(
    this_claim: dict[str, Any], prior_claims: list[dict[str, Any]]
) -> float | None:
    """This claim's value ÷ average of the customer's prior claim values.

    Returns None when the customer has no prior claims (we'd be dividing
    by zero and there's no signal to compute). XGBoost handles None.
    """
    target_value = float(this_claim.get("claim_value_usd") or 0.0)
    prior_values = [
        float(c.get("claim_value_usd") or 0.0)
        for c in prior_claims
        if (c.get("claim_id") != this_claim.get("claim_id")) and c.get("claim_value_usd")
    ]
    if not prior_values:
        return None
    avg = sum(prior_values) / len(prior_values)
    if avg <= 0:
        return None
    return target_value / avg


def customer_tenure_days(
    claim_date_val: date, first_seen_date: date | None
) -> int | None:
    if first_seen_date is None:
        return None
    return max((claim_date_val - first_seen_date).days, 0)


def prior_claim_outcomes_won_ratio(
    target_claim_id: str, prior_claims: list[dict[str, Any]]
) -> float | None:
    """approved / (approved + rejected) over the customer's prior claims.

    needs_info doesn't count toward either side. None if no prior outcomes.
    """
    approved = 0
    rejected = 0
    for c in prior_claims:
        if c.get("claim_id") == target_claim_id:
            continue
        outcome = c.get("expected_decision")
        if outcome == "approve":
            approved += 1
        elif outcome == "reject":
            rejected += 1
    total = approved + rejected
    if total == 0:
        return None
    return approved / total


def _tokenize(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def claim_text_similarity_max(
    target_text: str,
    target_claim_id: str,
    prior_claims: list[dict[str, Any]],
) -> float:
    """Max Jaccard similarity between target claim text and any prior claim text.

    We use Jaccard (set-based) rather than embedding cosine to keep the
    feature extractor synchronous and dependency-free. The Week-13 memory
    work can swap this for a real embedding similarity if it improves AUC.
    Returns 0.0 if no priors.
    """
    target_tokens = _tokenize(target_text)
    if not target_tokens:
        return 0.0
    best = 0.0
    for c in prior_claims:
        if c.get("claim_id") == target_claim_id:
            continue
        other_tokens = _tokenize(str(c.get("raw_input", "")))
        if not other_tokens:
            continue
        intersection = len(target_tokens & other_tokens)
        union = len(target_tokens | other_tokens)
        if union == 0:
            continue
        sim = intersection / union
        if sim > best:
            best = sim
    return round(best, 3)


def evidence_strength_score(extraction: dict[str, Any] | None) -> float | None:
    """Map evidence_strength categorical to a float."""
    if extraction is None:
        return None
    raw = extraction.get("evidence_strength")
    return {"strong": 1.0, "moderate": 0.5, "weak": 0.0}.get(raw)  # type: ignore[arg-type]


# Photo-derived features. These accept Week-8 outputs directly; for text-only
# claims, callers pass None.


def photo_ai_generated_likelihood(ai_score: float | None) -> float | None:
    """Pass-through. None when no photo was analyzed."""
    return ai_score


def exif_consistency_score(
    captured_at: date | datetime | None,
    claim_date_val: date,
) -> float | None:
    """1.0 = consistent (captured shortly before claim); 0.0 = impossible.

    Heuristic curve:
    - photo dated AFTER claim → 0.0 (physically impossible)
    - 0-30 days before claim → 1.0 (normal)
    - 30-180 days before → linear decay to 0.5
    - 180+ days before → 0.2 (very old photo, suspicious but legit "I forgot to file")
    - no exif date → None
    """
    if captured_at is None:
        return None
    captured_date = captured_at.date() if isinstance(captured_at, datetime) else captured_at
    days_before = (claim_date_val - captured_date).days
    if days_before < 0:
        return 0.0
    if days_before <= 30:
        return 1.0
    if days_before <= 180:
        return round(1.0 - 0.5 * ((days_before - 30) / 150), 3)
    return 0.2


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


# Ordered list of feature names. The trainer pins this order; the scorer
# rebuilds the same vector at inference time. DO NOT reorder without
# retraining — XGBoost indexes by position.
FEATURE_NAMES = [
    "claims_per_customer_30d",
    "same_email_diff_address_count",
    "address_mismatch_score",
    "claim_value_to_aov_ratio",
    "time_since_purchase_days",
    "customer_tenure_days",
    "prior_claim_outcomes_won_ratio",
    "claim_text_similarity_max",
    "photo_ai_generated_likelihood",
    "exif_consistency_score",
    "evidence_strength_from_extraction",
]


def compute_features(
    claim: dict[str, Any],
    ctx: CustomerContext,
    *,
    extraction: dict[str, Any] | None = None,
    ai_score: float | None = None,
    photo_captured_at: date | datetime | None = None,
) -> dict[str, float | None]:
    """Compute all 11 features for a single claim.

    Args:
        claim: the claim row (training jsonl entry or DB row dict).
        ctx: this customer's full context (other claims, primary address, tenure).
        extraction: optional ClaimExtraction.model_dump() output.
        ai_score: optional AI-image likelihood [0,1] from Week 8.
        photo_captured_at: optional EXIF capture timestamp from Week 8.
    """
    target_date = _parse_date(claim.get("claim_date"))
    target_id = claim.get("claim_id")
    if target_date is None:
        target_date = date(2026, 5, 1)  # fall back to "today" reference

    # All claims for this customer except the target one.
    prior_claims = [c for c in ctx.claims if c.get("claim_id") != target_id]

    return {
        "claims_per_customer_30d": float(claims_per_customer_30d(target_date, prior_claims)),
        "same_email_diff_address_count": float(same_email_diff_address_count(ctx)),
        "address_mismatch_score": address_mismatch_score(claim, ctx),
        "claim_value_to_aov_ratio": claim_value_to_aov_ratio(claim, prior_claims),
        "time_since_purchase_days": float(claim.get("days_since_purchase") or math.nan),
        "customer_tenure_days": (
            float(t) if (t := customer_tenure_days(target_date, ctx.first_seen_date)) is not None
            else None
        ),
        "prior_claim_outcomes_won_ratio": prior_claim_outcomes_won_ratio(
            str(target_id), prior_claims
        ),
        "claim_text_similarity_max": claim_text_similarity_max(
            str(claim.get("raw_input", "")), str(target_id), prior_claims
        ),
        "photo_ai_generated_likelihood": photo_ai_generated_likelihood(ai_score),
        "exif_consistency_score": exif_consistency_score(photo_captured_at, target_date),
        "evidence_strength_from_extraction": evidence_strength_score(extraction),
    }


def build_customer_contexts(
    claims_data: list[dict[str, Any]],
) -> dict[str, CustomerContext]:
    """Build per-customer contexts from the training jsonl."""
    by_customer: dict[str, CustomerContext] = {}
    for c in claims_data:
        cid = c["customer_id"]
        if cid not in by_customer:
            by_customer[cid] = CustomerContext(
                customer_id=cid,
                email=c["customer_email"],
                first_seen_date=_parse_date(c.get("customer_first_seen_date")),
                primary_address=c.get("shipping_address"),
                claims=[],
            )
        by_customer[cid].claims.append(c)

    # The "primary_address" we set first might not be the customer's primary.
    # Take the most common shipping_address per customer as their primary.
    from collections import Counter

    for ctx in by_customer.values():
        addresses = Counter(_normalize_address(c.get("shipping_address")) for c in ctx.claims)
        addresses.pop("", None)
        if addresses:
            most_common_norm = addresses.most_common(1)[0][0]
            # Find the original (non-normalized) string for that norm
            for c in ctx.claims:
                if _normalize_address(c.get("shipping_address")) == most_common_norm:
                    ctx.primary_address = c.get("shipping_address")
                    break

    return by_customer
