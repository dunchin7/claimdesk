"""Confidence-calibrator features (post-Week-14 hardening).

Maps decision-time signals to a feature vector for the logistic-regression
meta-classifier that produces calibrated_confidence ∈ [0, 1].

Features (all available BEFORE knowing the answer):
- confidence_is_high / confidence_is_medium       (one-hot of the LLM's self-report)
- citation_verbatim                                (post-validator output)
- citation_fuzzy                                   (continuous fallback score)
- evidence_strength_score                          (strong=1.0 / moderate=0.5 / weak=0.0)
- severity_total_loss / severity_structural        (high-severity one-hots)
- has_serial / has_time / has_dates                (evidence completeness)
- customer_emotion_calm                            (calmer claims tend to be cleaner)
- failure_mode_is_battery                          (battery is our hardest stratum)
- days_since_purchase_norm                         (normalized to [0, 1] over 0-540 days)
- predicted_decision_approve / predicted_needs_info (the model's chosen path —
  approves and needs_info have different calibration profiles in practice)

The mapping is pure: a dict in, an ordered list of floats out. Same keys as
the trainer expects.
"""

from __future__ import annotations

from typing import Any

# Fixed order. Trainer and scorer must agree. Adding a feature = retrain.
# Phase-4 P1.6: appended `fraud_score` as the 16th feature. The model loader
# refuses to load a pickle whose feature_names list differs from this one,
# so the calibrator must be re-pickled (calibrator_v2.pkl) after this change.
FEATURE_NAMES: list[str] = [
    "confidence_is_high",
    "confidence_is_medium",
    "citation_verbatim",
    "citation_fuzzy",
    "evidence_strength_score",
    "severity_total_loss",
    "severity_structural",
    "has_serial",
    "has_time",
    "customer_emotion_calm",
    "failure_mode_is_battery",
    "failure_mode_is_motor",
    "days_since_purchase_norm",
    "predicted_decision_approve",
    "predicted_decision_needs_info",
    "fraud_score",
]


def _bool01(v: Any) -> float:
    return 1.0 if bool(v) else 0.0


def _evidence_score(s: Any) -> float:
    return {"strong": 1.0, "moderate": 0.5, "weak": 0.0}.get(str(s), 0.5)


def _days_norm(d: Any) -> float:
    if d is None:
        return 0.5  # unknown → middle
    try:
        days = float(d)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, days / 540.0))


def build_features(
    *,
    confidence: str,
    citation_verbatim: bool,
    citation_fuzzy: float,
    evidence_strength: str | None,
    severity: str | None,
    failure_mode: str | None,
    customer_emotion: str | None,
    mentioned_serial: str | None,
    time_since_purchase_days: int | None,
    predicted_decision: str,
    fraud_score: float | None = None,
) -> dict[str, float]:
    """Build a feature dict from one decision's signals.

    Phase-4 P1.6: `fraud_score` is the XGBoost P(fraud). When None (no
    customer context), we default to 0.0 — same as "we saw a customer with
    no priors and assumed no fraud signal." The logistic regression learns
    whether that absent-signal default carries weight in either direction.
    """
    return {
        "confidence_is_high":           _bool01(confidence == "high"),
        "confidence_is_medium":         _bool01(confidence == "medium"),
        "citation_verbatim":            _bool01(citation_verbatim),
        "citation_fuzzy":               float(citation_fuzzy or 0.0),
        "evidence_strength_score":      _evidence_score(evidence_strength),
        "severity_total_loss":          _bool01(severity == "total_loss"),
        "severity_structural":          _bool01(severity == "structural"),
        "has_serial":                   _bool01(mentioned_serial),
        "has_time":                     _bool01(time_since_purchase_days is not None),
        "customer_emotion_calm":        _bool01(customer_emotion == "calm"),
        "failure_mode_is_battery":      _bool01(failure_mode == "battery"),
        "failure_mode_is_motor":        _bool01(failure_mode == "motor"),
        "days_since_purchase_norm":     _days_norm(time_since_purchase_days),
        "predicted_decision_approve":   _bool01(predicted_decision == "approve"),
        "predicted_decision_needs_info": _bool01(predicted_decision == "needs_info"),
        "fraud_score":                  float(fraud_score) if fraud_score is not None else 0.0,
    }


def features_to_row(feats: dict[str, float]) -> list[float]:
    """Convert a feature dict to an ordered row matching FEATURE_NAMES."""
    return [float(feats[name]) for name in FEATURE_NAMES]


def features_from_eval_row(row: dict[str, Any]) -> dict[str, float]:
    """Build features from a per_claim row in an EvalRunReport JSON.

    Used by `scripts/train_calibrator.py`. Eval JSONs from before P1.5
    won't have `fraud_score` — pass None and the default (0.0) lands.
    """
    return build_features(
        confidence=row.get("confidence", ""),
        citation_verbatim=bool(row.get("citation_verbatim", False)),
        citation_fuzzy=float(row.get("citation_fuzzy", 0.0)),
        evidence_strength=row.get("extracted_evidence_strength"),
        severity=row.get("extracted_severity"),
        failure_mode=row.get("extracted_failure_mode"),
        customer_emotion=row.get("extracted_customer_emotion"),
        mentioned_serial=row.get("extracted_mentioned_serial"),
        time_since_purchase_days=row.get("extracted_time_since_purchase_days"),
        predicted_decision=row.get("predicted_decision", ""),
        fraud_score=row.get("fraud_score"),
    )
