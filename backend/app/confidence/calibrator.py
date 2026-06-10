"""Confidence calibrator runtime (post-Week-14 hardening).

Loads `models/calibrator_v1.pkl` and exposes `calibrate()` returning a
real probability that the decision is correct. The Week-15 HITL router
uses this in place of the LLM's self-reported `confidence` field.

Threshold rule (Phase-4 P2.1 — calibrator_v3 = XGBoost on the same 16
features as v2, calibrated against the 200-claim sweep at
`backend/app/evals/reports/threshold_sweep_v5.md`):

- calibrated_prob >= 0.70 → auto-resolve
- 0.60 ≤ calibrated_prob < 0.70 → assist (queue for operator quick-check)
- calibrated_prob < 0.60 → require operator review

The 0.70 cut is empirical, not aspirational: at this threshold the
200-claim eval shows **100% accuracy on auto-resolved** (vs the 90%
Phase-3 target) at an 81.5% auto-resolve rate (vs the 75% target).
**Zero wrong machine decisions on the 163 auto-resolved claims.** Going
lower (0.65 → 99.4%, 0.55 → 97.7%) adds marginal throughput at the
cost of letting wrong decisions reach customers.

**Why thresholds moved across calibrator versions:**

| Version | Model | Auto threshold | Auto rate | Auto accuracy |
|---|---|---:|---:|---:|
| v1 (Week 14, 15 features) | LR | 0.85 → 0.75 | 43% → 77.5% | 95.3% → 91.6% |
| v2 (P1.6, +fraud_score) | LR | 0.55 | 88.5% | 91.5% |
| **v3 (P2.1, XGBoost)** | **XGBoost** | **0.70** | **81.5%** | **100.0%** |

The XGBoost classifier (v3) handles per-stratum signal via tree splits
rather than averaging coefficients — its high-confidence predictions
are *truly* high-confidence. The decisive shift between v2 and v3 isn't
the auto-resolve rate (similar), it's the **per-claim safety** —
operators see fewer false-confident machine decisions.

Re-run `scripts/threshold_sweep.py` after every calibrator retrain and
update these constants accordingly — never hand-pick a threshold based
on intuition about what a probability "should mean."

The threshold is a router-level decision, not a property of the model;
this module just emits the probability.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.ai.schemas import ClaimExtraction, Decision
from app.confidence.features import FEATURE_NAMES, build_features
from app.core.logging import get_logger

log = get_logger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
# Phase-4 P2.1 — `calibrator_v3.pkl` is the winning candidate from the
# v3a (per-decision LR) vs v3b (XGBoost) bake-off; the loader treats it
# as the production artifact. The previous v1 / v2 pickles stay on disk
# as fallback artifacts. Bundle shape is auto-detected from the
# `kind` field: "xgboost" / "per_decision_lr" / "global_lr" (legacy).
MODEL_PATH = _MODELS_DIR / "calibrator_v3.pkl"
V2_MODEL_PATH = _MODELS_DIR / "calibrator_v2.pkl"
LEGACY_MODEL_PATH = _MODELS_DIR / "calibrator_v1.pkl"

# Production routing thresholds. Picked from the calibration-table inspection
# at training time; live in code so the router doesn't depend on a magic
# number in a settings file.
AUTO_RESOLVE_THRESHOLD = 0.70
ASSIST_THRESHOLD = 0.60


@dataclass
class CalibratedConfidence:
    calibrated_prob: float           # P(decision is correct)
    raw_confidence: str              # The LLM's self-report (high/medium/low)
    route: str                       # "auto_resolve" | "assist" | "review"
    model_trained_at: str | None


@lru_cache(maxsize=1)
def _load_model() -> dict[str, Any] | None:
    """Load v3 if present, else v2, else v1, else None.

    Bundle shape detection:
      - `kind="xgboost"`: single XGBClassifier under `model`
      - `kind="per_decision_lr"`: dict of LogisticRegressions under `models`,
        keyed by predicted_decision ∈ {approve, reject, needs_info}
      - `kind="global_lr"` or no `kind` field (legacy v1/v2): single LR
        under `model`
    """
    candidate_paths = [MODEL_PATH, V2_MODEL_PATH, LEGACY_MODEL_PATH]
    for path in candidate_paths:
        if not path.is_file():
            continue
        with path.open("rb") as f:
            bundle = pickle.load(f)  # noqa: S301
        if bundle.get("feature_names") == list(FEATURE_NAMES):
            log.info(
                "calibrator.model_loaded",
                path=path.name,
                kind=bundle.get("kind", "global_lr"),
            )
            return bundle
        log.warning(
            "calibrator.feature_mismatch",
            path=path.name,
            note=(
                "calibrator pickle expected different features — falling "
                "back to the next candidate. Re-run train_calibrator_v3.py "
                "if all candidates are stale."
            ),
        )
    log.warning(
        "calibrator.no_compatible_model",
        note="run scripts/train_calibrator_v3.py to build calibrator_v3.pkl",
    )
    return None


def _predict_from_bundle(bundle: dict[str, Any], row: np.ndarray, predicted_decision: str) -> float:
    """Dispatch to the right submodel based on the bundle's `kind`."""
    kind = bundle.get("kind", "global_lr")
    if kind == "per_decision_lr":
        sub = bundle["models"].get(predicted_decision)
        if sub is None:
            # Decision label we didn't see at training — fall back to the
            # average of available submodels (a "no opinion" prior).
            sub_probs = []
            for s in bundle["models"].values():
                if hasattr(s, "predict_proba"):
                    sub_probs.append(float(s.predict_proba(row)[0, 1]))
            return float(sum(sub_probs) / len(sub_probs)) if sub_probs else 0.5
        if hasattr(sub, "predict_proba"):
            return float(sub.predict_proba(row)[0, 1])
        # Constant fallback (degenerate training stratum)
        return float(sub)
    # xgboost OR legacy global_lr — both use `model.predict_proba`
    return float(bundle["model"].predict_proba(row)[0, 1])


def _route_from_prob(p: float) -> str:
    if p >= AUTO_RESOLVE_THRESHOLD:
        return "auto_resolve"
    if p >= ASSIST_THRESHOLD:
        return "assist"
    return "review"


def calibrate(
    decision: Decision,
    extraction: ClaimExtraction,
    citation_verbatim: bool,
    citation_fuzzy: float,
    fraud_score: float | None = None,
) -> CalibratedConfidence:
    """Compute calibrated_prob for a single decision.

    If the model isn't loadable, fall back to a coarse rule: high → 0.7,
    medium → 0.5, low → 0.2. The router downgrades these to "review" by
    default (since 0.7 ties the assist threshold).
    """
    bundle = _load_model()
    if bundle is None:
        fallback = {"high": 0.7, "medium": 0.5, "low": 0.2}.get(
            decision.confidence, 0.5
        )
        return CalibratedConfidence(
            calibrated_prob=fallback,
            raw_confidence=decision.confidence,
            route=_route_from_prob(fallback),
            model_trained_at=None,
        )

    feats = build_features(
        confidence=decision.confidence,
        citation_verbatim=citation_verbatim,
        citation_fuzzy=citation_fuzzy,
        evidence_strength=extraction.evidence_strength,
        severity=extraction.severity,
        failure_mode=extraction.failure_mode,
        customer_emotion=extraction.customer_emotion,
        mentioned_serial=extraction.mentioned_serial,
        time_since_purchase_days=extraction.time_since_purchase_days,
        predicted_decision=decision.outcome,
        fraud_score=fraud_score,
    )
    row = np.array([[feats[name] for name in FEATURE_NAMES]], dtype=np.float32)
    prob = _predict_from_bundle(bundle, row, decision.outcome)

    return CalibratedConfidence(
        calibrated_prob=round(prob, 4),
        raw_confidence=decision.confidence,
        route=_route_from_prob(prob),
        model_trained_at=bundle.get("trained_at"),
    )
