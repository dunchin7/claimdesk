"""Unit tests for app/fraud/features.py — pure-function tests, no DB."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from app.fraud.features import (
    CustomerContext,
    FEATURE_NAMES,
    address_mismatch_score,
    build_customer_contexts,
    claim_text_similarity_max,
    claim_value_to_aov_ratio,
    claims_per_customer_30d,
    compute_features,
    customer_tenure_days,
    evidence_strength_score,
    exif_consistency_score,
    prior_claim_outcomes_won_ratio,
    same_email_diff_address_count,
)


# --- helpers ----------------------------------------------------------------


def _claim(
    cid: str, date_str: str, addr: str = "1 Main St", value: float = 500.0,
    decision: str = "approve", raw: str = "test text", customer_id: str = "cust1",
) -> dict:
    return {
        "claim_id": cid,
        "customer_id": customer_id,
        "claim_date": date_str,
        "shipping_address": addr,
        "claim_value_usd": value,
        "expected_decision": decision,
        "raw_input": raw,
    }


def _ctx(claims: list[dict], first_seen: str | None = "2025-01-01",
         primary_addr: str | None = "1 Main St") -> CustomerContext:
    return CustomerContext(
        customer_id="cust1",
        email="x@y.com",
        first_seen_date=date.fromisoformat(first_seen) if first_seen else None,
        primary_address=primary_addr,
        claims=claims,
    )


# --- tests ------------------------------------------------------------------


def test_claims_per_customer_30d_counts_only_priors_in_window() -> None:
    target = date(2026, 5, 1)
    prior = [
        _claim("a", "2026-04-15"),  # 16 days before — in window
        _claim("b", "2026-04-05"),  # 26 days before — in window
        _claim("c", "2026-03-01"),  # 61 days before — out of window
        _claim("d", "2026-05-02"),  # AFTER target — excluded
    ]
    assert claims_per_customer_30d(target, prior) == 2


def test_same_email_diff_address_count() -> None:
    ctx = _ctx([
        _claim("a", "2026-04-15", addr="1 Main St"),
        _claim("b", "2026-04-20", addr="1 Main St"),  # exact duplicate
        _claim("c", "2026-04-25", addr="1 Main St,"),  # only trailing comma differs
        _claim("d", "2026-04-26", addr="99 Beach Rd"),
    ])
    # "1 Main St" ≡ "1 Main St," after comma+whitespace normalization → 1 addr
    # "99 Beach Rd" → 2nd addr
    n = same_email_diff_address_count(ctx)
    assert n == 2


def test_address_mismatch_score_identical() -> None:
    ctx = _ctx([_claim("a", "2026-04-15")], primary_addr="1 Main St")
    score = address_mismatch_score({"shipping_address": "1 Main St"}, ctx)
    assert score < 0.05


def test_address_mismatch_score_different() -> None:
    ctx = _ctx([_claim("a", "2026-04-15")], primary_addr="1 Main St, Boston, MA")
    score = address_mismatch_score(
        {"shipping_address": "999 Different Way, Los Angeles, CA"}, ctx
    )
    assert score > 0.5


def test_claim_value_to_aov_ratio() -> None:
    target = _claim("target", "2026-05-01", value=2000.0)
    prior = [
        _claim("a", "2026-03-01", value=500.0),
        _claim("b", "2026-04-01", value=500.0),
    ]
    ratio = claim_value_to_aov_ratio(target, prior + [target])
    assert ratio == 4.0  # 2000 / 500


def test_claim_value_to_aov_ratio_no_priors() -> None:
    target = _claim("target", "2026-05-01", value=2000.0)
    assert claim_value_to_aov_ratio(target, [target]) is None


def test_customer_tenure_days() -> None:
    days = customer_tenure_days(date(2026, 5, 1), date(2026, 1, 1))
    assert days == 120


def test_customer_tenure_days_unknown() -> None:
    assert customer_tenure_days(date(2026, 5, 1), None) is None


def test_prior_claim_outcomes_won_ratio() -> None:
    prior = [
        _claim("a", "2026-01-01", decision="approve"),
        _claim("b", "2026-02-01", decision="approve"),
        _claim("c", "2026-03-01", decision="reject"),
        _claim("d", "2026-03-15", decision="needs_info"),  # excluded
    ]
    ratio = prior_claim_outcomes_won_ratio("target", prior)
    assert ratio == 2 / 3


def test_claim_text_similarity_cookie_cutter() -> None:
    target = "My battery died after only 30 charge cycles"
    prior = [
        _claim("a", "2026-04-01", raw="My battery died after only 30 charge cycles"),
        _claim("b", "2026-04-15", raw="The wheel is bent on my bike"),
    ]
    sim = claim_text_similarity_max(target, "target", prior)
    assert sim > 0.9  # near-identical to claim 'a'


def test_claim_text_similarity_no_priors() -> None:
    sim = claim_text_similarity_max("text", "target", [])
    assert sim == 0.0


def test_evidence_strength_score_categorical() -> None:
    assert evidence_strength_score({"evidence_strength": "strong"}) == 1.0
    assert evidence_strength_score({"evidence_strength": "moderate"}) == 0.5
    assert evidence_strength_score({"evidence_strength": "weak"}) == 0.0
    assert evidence_strength_score(None) is None
    assert evidence_strength_score({}) is None


def test_exif_consistency_score_buckets() -> None:
    claim_dt = date(2026, 5, 1)
    # Photo taken AFTER claim → impossible
    assert exif_consistency_score(date(2026, 6, 1), claim_dt) == 0.0
    # 5 days before → normal
    assert exif_consistency_score(date(2026, 4, 25), claim_dt) == 1.0
    # 90 days before → middle band
    score = exif_consistency_score(date(2026, 2, 1), claim_dt)
    assert score is not None
    assert 0.4 < score < 1.0
    # 365 days before → very old
    assert exif_consistency_score(date(2025, 5, 1), claim_dt) == 0.2
    # No date → None
    assert exif_consistency_score(None, claim_dt) is None


def test_compute_features_returns_all_named() -> None:
    target = _claim("target", "2026-05-01", value=1000.0, raw="My battery died")
    ctx = _ctx([target, _claim("p1", "2026-04-15", value=500.0, decision="approve")])
    feats = compute_features(
        target, ctx,
        extraction={"evidence_strength": "strong"},
        ai_score=0.3,
        photo_captured_at=date(2026, 4, 28),
    )
    for name in FEATURE_NAMES:
        assert name in feats
    assert feats["evidence_strength_from_extraction"] == 1.0
    assert feats["photo_ai_generated_likelihood"] == 0.3
    assert feats["claims_per_customer_30d"] == 1.0


def test_compute_features_handles_missing_photo() -> None:
    target = _claim("target", "2026-05-01")
    ctx = _ctx([target])
    feats = compute_features(target, ctx)
    assert feats["photo_ai_generated_likelihood"] is None
    assert feats["exif_consistency_score"] is None


def test_build_customer_contexts_groups_correctly() -> None:
    rows = [
        {"claim_id": "a", "customer_id": "c1", "customer_email": "c1@x",
         "customer_first_seen_date": "2025-01-01", "shipping_address": "1 A St",
         "claim_date": "2026-04-01", "claim_value_usd": 100.0,
         "expected_decision": "approve", "raw_input": "x"},
        {"claim_id": "b", "customer_id": "c1", "customer_email": "c1@x",
         "customer_first_seen_date": "2025-01-01", "shipping_address": "1 A St",
         "claim_date": "2026-04-15", "claim_value_usd": 200.0,
         "expected_decision": "reject", "raw_input": "y"},
        {"claim_id": "c", "customer_id": "c2", "customer_email": "c2@x",
         "customer_first_seen_date": "2025-06-01", "shipping_address": "2 B St",
         "claim_date": "2026-05-01", "claim_value_usd": 300.0,
         "expected_decision": "approve", "raw_input": "z"},
    ]
    ctxs = build_customer_contexts(rows)
    assert set(ctxs) == {"c1", "c2"}
    assert len(ctxs["c1"].claims) == 2
    assert ctxs["c1"].primary_address == "1 A St"
    assert ctxs["c2"].first_seen_date == date(2025, 6, 1)
