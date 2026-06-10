"""Train the confidence calibrator (post-Week-14 hardening).

Pools per_claim rows from recent eval JSONs that include extraction fields
(only post-2026-06-02 runs have these). Fits a logistic regression on the
15-feature row → P(decision_correct). Reports ROC-AUC + a calibration
table so we can verify "P=0.9 means 90% accurate" is approximately true.

Output:
- models/calibrator_v1.pkl
- models/calibrator_v1.json (metadata)

Usage:
    uv run python scripts/train_calibrator.py
    uv run python scripts/train_calibrator.py --reports-glob "backend/app/evals/reports/2026-06-*.json"
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.confidence.features import FEATURE_NAMES, features_from_eval_row  # noqa: E402

DEFAULT_REPORTS_GLOB = "backend/app/evals/reports/*.json"
MODEL_DIR = ROOT / "models"
# Phase-4 P1.6: bumped to v2 — fraud_score added as the 16th feature.
MODEL_PATH = MODEL_DIR / "calibrator_v2.pkl"
METADATA_PATH = MODEL_DIR / "calibrator_v2.json"


def _load_rows(reports_glob: str) -> tuple[list[dict], int]:
    """Pool per_claim rows from eval JSONs. Returns (rows, n_files_used)."""
    rows: list[dict] = []
    seen_claim_ids: set[tuple[str, str]] = set()  # (claim_id, predicted_decision) to allow same claim across runs
    n_files = 0
    for path in sorted(ROOT.glob(reports_glob)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        per_claim = data.get("per_claim", [])
        if not per_claim:
            continue
        # Only use rows that have the extraction fields (post-Week-14)
        has_extraction = any("extracted_evidence_strength" in r for r in per_claim)
        if not has_extraction:
            continue
        n_files += 1
        for r in per_claim:
            if r.get("error"):
                continue
            cid = r.get("claim_id", "")
            pred = r.get("predicted_decision", "")
            key = (cid, pred)
            if key in seen_claim_ids:
                continue
            seen_claim_ids.add(key)
            rows.append(r)
    return rows, n_files


def _calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    out: list[dict] = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        n = int(mask.sum())
        bucket = {
            "bucket_low": round(float(edges[i]), 3),
            "bucket_high": round(float(edges[i + 1]), 3),
            "n": n,
            "predicted_avg": round(float(y_prob[mask].mean()), 3) if n else 0.0,
            "observed_rate": round(float(y_true[mask].mean()), 3) if n else 0.0,
        }
        out.append(bucket)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-glob", default=DEFAULT_REPORTS_GLOB)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, n_files = _load_rows(args.reports_glob)
    print(f"[calib] loaded {len(rows)} rows from {n_files} eval files")
    if len(rows) < 30:
        print(f"[calib] too few rows ({len(rows)}) — re-run a full eval first")
        return 1

    X = np.array([
        [feat for feat in features_from_eval_row(r).values()]
        for r in rows
    ], dtype=np.float32)
    # features_from_eval_row returns dict; ordering must match FEATURE_NAMES.
    # Rebuild explicitly to be safe:
    X = np.array([
        [features_from_eval_row(r)[name] for name in FEATURE_NAMES]
        for r in rows
    ], dtype=np.float32)
    y = np.array([1 if r.get("decision_match") else 0 for r in rows], dtype=np.int32)
    print(f"[calib] feature matrix: {X.shape}, positive rate: {y.mean():.1%}")

    # k-fold cross-validated predictions for honest calibration metrics
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for fold, (tr_idx, te_idx) in enumerate(kfold.split(X, y)):
        clf = LogisticRegression(max_iter=1000, random_state=args.seed)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_pred[te_idx] = clf.predict_proba(X[te_idx])[:, 1]

    roc_auc = float(roc_auc_score(y, oof_pred))
    brier = float(brier_score_loss(y, oof_pred))
    cal_table = _calibration_table(y, oof_pred)

    # Final model fit on all data
    final = LogisticRegression(max_iter=1000, random_state=args.seed)
    final.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({
            "model": final,
            "feature_names": list(FEATURE_NAMES),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_train_rows": len(rows),
        }, f)

    coef = dict(zip(FEATURE_NAMES, final.coef_[0].tolist(), strict=True))
    intercept = float(final.intercept_[0])

    METADATA_PATH.write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": len(rows),
        "positive_rate": round(float(y.mean()), 3),
        "cv_roc_auc": round(roc_auc, 3),
        "cv_brier": round(brier, 3),
        "calibration_table": cal_table,
        "coefficients": {k: round(v, 4) for k, v in coef.items()},
        "intercept": round(intercept, 4),
        "seed": args.seed,
    }, indent=2))

    print()
    print("=" * 60)
    print(f"  Confidence calibrator — {len(rows)} rows")
    print("=" * 60)
    print(f"CV ROC-AUC:       {roc_auc:.3f}")
    print(f"CV Brier:         {brier:.3f}  (lower = better calibrated)")
    print()
    print(f"{'Bucket':<14} {'n':>4} {'pred avg':>10} {'observed':>10}")
    for row in cal_table:
        bucket = f"[{row['bucket_low']:.2f}–{row['bucket_high']:.2f}]"
        print(
            f"  {bucket:<12} {row['n']:>4} {row['predicted_avg']:>10.3f} "
            f"{row['observed_rate']:>10.3f}"
        )
    print()
    print("Top coefficients (positive → more confidence in decision-correctness):")
    for name, c in sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {name:<32} {c:+.3f}")
    print()
    print(f"[ok] saved {MODEL_PATH}")
    print(f"[ok] metadata {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
