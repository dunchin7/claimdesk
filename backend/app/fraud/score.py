"""Fraud scorer + LLM-judge narrative (Week 9).

`score_claim(claim, ctx, ...)` runs the trained XGBoost model and returns:
- xgboost_score:    the raw P(fraud) from the booster
- calibrated_prob:  the isotonic-calibrated probability
- top_3_features:   the highest-magnitude per-feature contributions
- llm_check:        a one-paragraph reasoning string from gpt-4o-mini that
                    looks at the top features and articulates whether the
                    pattern matches a known fraud template

The model is loaded lazily from `models/fraud_v1.pkl` and cached.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.ai.llm import chat
from app.core.logging import get_logger
from app.fraud.features import (
    FEATURE_NAMES,
    CustomerContext,
    compute_features,
)

log = get_logger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "fraud_v1.pkl"


@dataclass
class FraudScore:
    xgboost_score: float
    calibrated_prob: float
    top_3_features: list[dict[str, Any]] = field(default_factory=list)
    llm_check: str | None = None
    all_features: dict[str, float | None] = field(default_factory=dict)
    model_trained_at: str | None = None


@lru_cache(maxsize=1)
def _load_model() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            f"Run `uv run python scripts/train_fraud_model.py` first."
        )
    with MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)  # noqa: S301
    # Sanity: the feature ordering at training time must equal what we use now
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError(
            "Trained model's feature ordering differs from app.fraud.features.FEATURE_NAMES. "
            "Re-train after modifying the feature set."
        )
    return bundle


def _top_k_contributors(
    feats: dict[str, float | None], importances: dict[str, float], k: int = 3
) -> list[dict[str, Any]]:
    """Pick the top-k features by (value × importance), excluding NaN values.

    This is a proxy for SHAP. A real production scorer would compute SHAP
    values per-claim — meaningfully more accurate per-claim attribution. We
    skip that for now to keep deps tight; the LLM judge sees feature values
    + names + global importances and reasons from there.
    """
    ranked: list[tuple[float, str, float | None]] = []
    for name, raw_val in feats.items():
        if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
            continue
        # Normalize feature values into roughly comparable magnitudes.
        v = float(raw_val)
        if name == "claim_text_similarity_max":
            magnitude = v  # already 0..1
        elif name == "address_mismatch_score":
            magnitude = v  # already 0..1
        elif name == "claims_per_customer_30d":
            magnitude = min(v / 5.0, 1.0)  # 5+ claims → max
        elif name == "same_email_diff_address_count":
            magnitude = min((v - 1) / 4.0, 1.0)  # 1 addr is fine; 5+ is bad
        elif name == "claim_value_to_aov_ratio":
            magnitude = min(abs(v - 1.0), 2.0) / 2.0  # ~1.0 normal; far from 1 odd
        elif name == "customer_tenure_days":
            magnitude = max(0.0, 1.0 - v / 90.0)  # <30 days = high signal
        elif name == "time_since_purchase_days":
            # both extremes flag — bell-shape with peak at ~28-day window
            magnitude = abs(v - 90) / 360 if v > 28 else (28 - v) / 28
            magnitude = min(magnitude, 1.0)
        else:
            magnitude = v
        score = magnitude * importances.get(name, 0.0)
        ranked.append((score, name, raw_val))

    ranked.sort(key=lambda x: -x[0])
    return [
        {"name": n, "value": v, "contribution_score": round(s, 4)}
        for s, n, v in ranked[:k]
    ]


_JUDGE_PROMPT = """\
You are a fraud-pattern reviewer for an e-bike warranty claims system. A
gradient-boosted model just scored a claim. Based on the top contributing
features below, write **one short paragraph** (2-4 sentences) explaining:

1. Which fraud pattern (if any) this most resembles — same-email-multi,
   address-mismatch, address-hopper, value-spike, tenure-spike,
   near-window-pattern — and why.
2. Whether the feature values are *consistent* with that pattern or just
   *adjacent*. Be honest: a single feature firing weakly is not a clear
   pattern.
3. What additional check (if any) you'd recommend before action.

Do NOT speculate beyond the features. Don't repeat the feature values
verbatim — interpret them.

Model output:
- xgboost_score: {xgb_score:.3f}
- calibrated_prob: {calibrated:.3f}

Top contributing features (name, value):
{features_block}

Recent fraud-pattern catalog (use as reference, don't enumerate):
- same_email_multi: 4+ claims by one customer in <30 days
- address_mismatch: claim ships to a different address than the original order
- address_hopper: customer has 3+ distinct shipping addresses on file
- value_spike: claim value is much larger than customer's historical average
- tenure_spike: very new customer immediately files a high-value claim
- near_window_pattern: claim filed near end of return/coverage window, repeated
"""


async def _run_llm_judge(score: FraudScore) -> str:
    """One-paragraph natural-language analysis of the top features."""
    feature_lines = []
    for f in score.top_3_features:
        val_str = f"{f['value']:.3f}" if isinstance(f["value"], (int, float)) else str(f["value"])
        feature_lines.append(f"- {f['name']}: {val_str}")
    block = "\n".join(feature_lines) if feature_lines else "(none — no features fired)"

    prompt = _JUDGE_PROMPT.format(
        xgb_score=score.xgboost_score,
        calibrated=score.calibrated_prob,
        features_block=block,
    )
    try:
        resp = await chat(
            messages=[{"role": "user", "content": prompt}],
            model_alias="judge",  # gpt-4o-mini
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("fraud.judge_failed", error=str(e))
        return f"(LLM judge unavailable: {type(e).__name__})"
    text = (
        resp["choices"][0]["message"]["content"].strip()
        if isinstance(resp, dict)
        else resp.choices[0].message.content.strip()
    )
    return text


async def score_claim(
    claim: dict[str, Any],
    ctx: CustomerContext,
    *,
    extraction: dict[str, Any] | None = None,
    ai_score: float | None = None,
    photo_captured_at: date | datetime | None = None,
    include_llm_judge: bool = True,
) -> FraudScore:
    """Score a single claim. Returns the FraudScore bundle."""
    bundle = _load_model()
    calibrated_model = bundle["calibrated"]
    # The underlying booster is accessible for the raw (uncalibrated) score
    # and feature importances. For CalibratedClassifierCV the booster is
    # held in `calibrated_classifiers_[0].estimator` (FrozenEstimator wraps
    # it in newer sklearn; older versions expose `.base_estimator`).
    try:
        booster = calibrated_model.calibrated_classifiers_[0].estimator
        # FrozenEstimator wrapping?
        if hasattr(booster, "estimator"):
            booster = booster.estimator
    except (AttributeError, IndexError):
        booster = None

    feats = compute_features(
        claim, ctx,
        extraction=extraction,
        ai_score=ai_score,
        photo_captured_at=photo_captured_at,
    )
    row = np.array([
        [float(feats[name]) if feats[name] is not None else math.nan for name in FEATURE_NAMES]
    ], dtype=np.float32)

    calibrated_prob = float(calibrated_model.predict_proba(row)[0, 1])
    xgb_score = float(booster.predict_proba(row)[0, 1]) if booster is not None else calibrated_prob

    importances = {
        name: float(imp)
        for name, imp in zip(
            FEATURE_NAMES, booster.feature_importances_ if booster is not None else [0.0] * len(FEATURE_NAMES),
            strict=True,
        )
    }

    score = FraudScore(
        xgboost_score=round(xgb_score, 4),
        calibrated_prob=round(calibrated_prob, 4),
        top_3_features=_top_k_contributors(feats, importances, k=3),
        all_features={k: (None if v is None else round(float(v), 3)) for k, v in feats.items()},
        model_trained_at=bundle.get("trained_at"),
    )

    if include_llm_judge:
        score.llm_check = await _run_llm_judge(score)

    log.info(
        "fraud.score",
        claim_id=claim.get("claim_id"),
        calibrated_prob=score.calibrated_prob,
        top_feature=score.top_3_features[0]["name"] if score.top_3_features else None,
    )
    return score


def score_to_dict(score: FraudScore) -> dict[str, Any]:
    """JSON-safe serialization for API responses / DB writes."""
    return asdict(score)
