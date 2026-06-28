"""Tests for the Reserve & Leakage Ledger cost model and book pricing."""

from __future__ import annotations

from app.ledger import (
    CostModel,
    ScoredClaim,
    estimate_claim_value,
    price_book,
    sweep_threshold,
)


def _cm() -> CostModel:
    return CostModel(review_labor=12.0, leakage_multiplier=1.0, dispute_ev=120.0, churn_cost=80.0)


def test_correct_auto_resolution_saves_labor_no_error() -> None:
    cm = _cm()
    c = cm.price_claim("approve", "approve", 800.0)
    assert c.auto_resolved
    assert c.lae_saved == 12.0
    assert c.leakage == 0.0 and c.false_denial == 0.0
    assert c.error_type is None
    assert c.net_benefit == 12.0


def test_leakage_wrong_approval() -> None:
    cm = _cm()
    c = cm.price_claim("approve", "reject", 800.0)
    assert c.error_type == "leakage"
    assert c.leakage == 800.0  # claim_value * 1.0
    assert c.false_denial == 0.0
    # net = labor saved - leakage
    assert c.net_benefit == 12.0 - 800.0


def test_false_denial_costs_more_than_leakage_for_same_claim() -> None:
    cm = _cm()
    leak = cm.price_claim("approve", "reject", 800.0)
    deny = cm.price_claim("reject", "approve", 800.0)
    # false denial = owed payout + dispute_ev + churn = 800 + 120 + 80
    assert deny.error_type == "false_denial"
    assert deny.false_denial == 1000.0
    # the asymmetry the instrument exists to surface
    assert deny.false_denial > leak.leakage


def test_routed_claim_is_net_zero() -> None:
    cm = _cm()
    c = cm.price_claim("needs_info", "approve", 800.0)
    assert not c.auto_resolved
    assert c.lae_saved == 0.0
    assert c.leakage == 0.0 and c.false_denial == 0.0
    assert c.net_benefit == 0.0


def test_ambiguous_gold_weights_error_down() -> None:
    cm = _cm()
    # auto-rejected something whose gold was needs_info: no full payout owed,
    # weighted by ambiguous_error_weight (0.5 default)
    c = cm.price_claim("reject", "needs_info", 800.0)
    assert c.error_type == "false_denial"
    assert c.false_denial == (0.0 + 120.0 + 80.0) * 0.5


def test_coefficients_are_editable() -> None:
    cm = CostModel(dispute_ev=500.0, churn_cost=300.0)
    c = cm.price_claim("reject", "approve", 1000.0)
    assert c.false_denial == 1000.0 + 500.0 + 300.0


def test_price_book_aggregates_two_sides() -> None:
    cm = _cm()
    claims = [
        ScoredClaim("ok", "approve", "approve", 800.0),       # correct -> saves labor
        ScoredClaim("leak", "approve", "reject", 800.0),      # leakage
        ScoredClaim("deny", "reject", "approve", 800.0),      # false denial
        ScoredClaim("route", "needs_info", "approve", 800.0), # routed
    ]
    book = price_book(claims, cm, label="t")
    assert book.n == 4
    assert book.n_auto_resolved == 3
    assert book.n_leakage_events == 1
    assert book.n_false_denial_events == 1
    assert book.leakage == 800.0
    assert book.false_denial == 1000.0
    assert book.lae_saved == 3 * 12.0
    assert book.net == book.lae_saved - 800.0 - 1000.0
    # per-1000 scaling
    assert round(book.per_1000(book.net), 2) == round(book.net / 4 * 1000, 2)


def test_threshold_sweep_routes_low_confidence_and_finds_crossover() -> None:
    cm = _cm()
    # one confident-correct claim (saves labor) and one confident-but-WRONG
    # denial (huge liability). At a high threshold both route to humans -> no
    # liability but no saving.
    claims = [
        ScoredClaim("good", "approve", "approve", 800.0, confidence=0.95),
        ScoredClaim("bad_deny", "reject", "approve", 800.0, confidence=0.90),
    ]
    curve = sweep_threshold(claims, cm, thresholds=[0.5, 0.92, 0.99])
    by_t = {p.threshold: p for p in curve}
    # t=0.5: both auto-resolved -> one false denial dominates -> net negative
    assert by_t[0.5].false_denial == 1000.0
    assert by_t[0.5].net < 0
    # t=0.92: the wrong denial (conf .90) routes out; the good one stays
    assert by_t[0.92].false_denial == 0.0
    assert by_t[0.92].auto_resolve_rate == 0.5
    # t=0.99: everything routes -> zero saving, zero liability
    assert by_t[0.99].auto_resolve_rate == 0.0
    assert by_t[0.99].net == 0.0


def test_estimate_claim_value_heuristics() -> None:
    assert estimate_claim_value("my laptop won't boot") == 1200.0
    assert estimate_claim_value("cracked my iPhone screen") == 800.0
    assert estimate_claim_value("my AirPods died") == 150.0
    assert estimate_claim_value("something unspecified broke") == 600.0  # default
