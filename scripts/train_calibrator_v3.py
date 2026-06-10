"""Phase-4 P2.1 — train two candidate calibrators (v3a + v3b).

v3a: Per-predicted-decision logistic regression.
     Three models, one per `predicted_decision` ∈ {approve, reject, needs_info}.
     At inference, the pipeline routes to the matching submodel.

v3b: Single XGBoost classifier on the same 16 features.
     Tree splits implicitly partition the feature space, so we get
     per-stratum behavior without literally splitting the data.

Both train on the same pooled eval-row corpus that
`train_calibrator.py` reads (256 rows, 16 features). Cross-validation +
calibration tables are reported for each. The pipeline LOADER decides
which to use; this script saves both side-by-side so we can pick by
end-to-end eval, not by intermediate metric.

Saved artifacts:
- models/calibrator_v3a.pkl + .json (per-decision LR bundle)
- models/calibrator_v3b.pkl + .json (XGBoost bundle)
- models/calibrator_v3_compare.json (CV metrics side-by-side)

Usage:
    uv run python scripts/train_calibrator_v3.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.confidence.features import FEATURE_NAMES, features_from_eval_row  # noqa: E402

DEFAULT_REPORTS_GLOB = "backend/app/evals/reports/*.json"
MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Data loading — mirrors train_calibrator.py to keep the corpus identical
# ---------------------------------------------------------------------------


def _load_rows(reports_glob: str) -> tuple[list[dict], int]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    n_files = 0
    for path in sorted(ROOT.glob(reports_glob)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        per_claim = data.get("per_claim", [])
        if not per_claim:
            continue
        has_extraction = any("extracted_evidence_strength" in r for r in per_claim)
        if not has_extraction:
            continue
        n_files += 1
        for r in per_claim:
            if r.get("error"):
                continue
            key = (r.get("claim_id", ""), r.get("predicted_decision", ""))
            if key in seen:
                continue
            seen.add(key)
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
        out.append({
            "bucket_low": round(float(edges[i]), 3),
            "bucket_high": round(float(edges[i + 1]), 3),
            "n": n,
            "predicted_avg": round(float(y_prob[mask].mean()), 3) if n else 0.0,
            "observed_rate": round(float(y_true[mask].mean()), 3) if n else 0.0,
        })
    return out


def _build_X_y(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, predicted_decision_array) — predicted_decision used for
    v3a's per-submodel routing."""
    X = np.array(
        [[features_from_eval_row(r)[name] for name in FEATURE_NAMES] for r in rows],
        dtype=np.float32,
    )
    y = np.array([1 if r.get("decision_match") else 0 for r in rows], dtype=np.int32)
    preds = np.array([r.get("predicted_decision", "") for r in rows])
    return X, y, preds


# ---------------------------------------------------------------------------
# v3a: per-predicted-decision logistic regression
# ---------------------------------------------------------------------------


def train_v3a(
    X: np.ndarray, y: np.ndarray, preds: np.ndarray, seed: int
) -> tuple[dict, dict]:
    """Train one LR per predicted_decision. Returns (bundle, metrics)."""
    submodels: dict[str, LogisticRegression] = {}
    metrics: dict[str, dict] = {}

    # Cross-validated OOF predictions across the whole corpus, using the
    # submodel that matches each row's predicted_decision. This is the
    # honest evaluation — the routed-at-inference behavior.
    oof_pred = np.zeros(len(y), dtype=np.float32)
    counts = Counter(preds.tolist())

    for decision in sorted(counts):
        mask = preds == decision
        n = int(mask.sum())
        if n < 20:
            print(
                f"[v3a] skipping decision={decision!r}: only {n} rows (need >=20)"
            )
            continue

        X_sub, y_sub = X[mask], y[mask]
        if len(set(y_sub.tolist())) < 2:
            # All same label — LR can't fit. Fall back to constant.
            const = float(y_sub.mean())
            metrics[decision] = {
                "n": n,
                "cv_roc_auc": None,
                "cv_brier": None,
                "note": f"degenerate label distribution (all {int(y_sub.mean())})",
                "constant_prediction": const,
            }
            # No model — at inference we'll use constant. Encode that.
            submodels[decision] = const  # type: ignore[assignment]
            oof_pred[mask] = const
            continue

        # 5-fold CV OOF on this stratum
        # Use fewer splits if the minority class is too small for 5 folds.
        minority_count = min(int((y_sub == 0).sum()), int((y_sub == 1).sum()))
        n_splits = min(5, max(2, minority_count))
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        sub_oof = np.zeros(n, dtype=np.float32)
        for tr, te in kf.split(X_sub, y_sub):
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_sub[tr], y_sub[tr])
            sub_oof[te] = clf.predict_proba(X_sub[te])[:, 1]
        oof_pred[mask] = sub_oof

        roc = float(roc_auc_score(y_sub, sub_oof)) if len(set(y_sub.tolist())) == 2 else None
        brier = float(brier_score_loss(y_sub, sub_oof))
        metrics[decision] = {
            "n": n,
            "cv_roc_auc": round(roc, 3) if roc is not None else None,
            "cv_brier": round(brier, 3),
        }

        # Final fit on all rows in this stratum
        final = LogisticRegression(max_iter=1000, random_state=seed)
        final.fit(X_sub, y_sub)
        submodels[decision] = final

    overall_roc = float(roc_auc_score(y, oof_pred))
    overall_brier = float(brier_score_loss(y, oof_pred))
    cal_table = _calibration_table(y, oof_pred)

    bundle = {
        "kind": "per_decision_lr",
        "models": submodels,
        "feature_names": list(FEATURE_NAMES),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_rows": int(len(y)),
    }
    summary = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "per_decision_lr",
        "n_rows": int(len(y)),
        "positive_rate": round(float(y.mean()), 3),
        "overall_cv_roc_auc": round(overall_roc, 3),
        "overall_cv_brier": round(overall_brier, 3),
        "per_decision_metrics": metrics,
        "calibration_table": cal_table,
    }
    return bundle, summary


# ---------------------------------------------------------------------------
# v3b: single XGBoost classifier
# ---------------------------------------------------------------------------


def train_v3b(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict, dict]:
    """Train one XGBoost classifier. Returns (bundle, metrics)."""
    # Modest depth + early-stopping-ish learning rate; we have 256 rows so
    # easy to overfit. The numbers below mirror the Week-9 fraud model
    # which trained on a similar-scale corpus.
    common_params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.06,
        random_state=seed,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=2,
    )

    # 5-fold CV OOF
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for tr, te in kf.split(X, y):
        clf = xgb.XGBClassifier(**common_params)
        clf.fit(X[tr], y[tr])
        oof_pred[te] = clf.predict_proba(X[te])[:, 1]

    roc = float(roc_auc_score(y, oof_pred))
    brier = float(brier_score_loss(y, oof_pred))
    cal_table = _calibration_table(y, oof_pred)

    final = xgb.XGBClassifier(**common_params)
    final.fit(X, y)

    importances = dict(
        zip(FEATURE_NAMES, [float(v) for v in final.feature_importances_], strict=True)
    )

    bundle = {
        "kind": "xgboost",
        "model": final,
        "feature_names": list(FEATURE_NAMES),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_rows": int(len(y)),
    }
    summary = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "xgboost",
        "n_rows": int(len(y)),
        "positive_rate": round(float(y.mean()), 3),
        "cv_roc_auc": round(roc, 3),
        "cv_brier": round(brier, 3),
        "calibration_table": cal_table,
        "feature_importances": {
            k: round(v, 4) for k, v in sorted(importances.items(), key=lambda kv: -kv[1])
        },
    }
    return bundle, summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _print_table(title: str, table: list[dict]) -> None:
    print(f"\n[{title}] calibration table:")
    print(f"  {'bucket':<14} {'n':>4} {'pred avg':>10} {'observed':>10}")
    for row in table:
        bucket = f"[{row['bucket_low']:.2f}–{row['bucket_high']:.2f}]"
        print(
            f"  {bucket:<14} {row['n']:>4} {row['predicted_avg']:>10.3f} "
            f"{row['observed_rate']:>10.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-glob", default=DEFAULT_REPORTS_GLOB)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, n_files = _load_rows(args.reports_glob)
    print(f"[calib_v3] loaded {len(rows)} rows from {n_files} eval files")
    if len(rows) < 50:
        print("[calib_v3] too few rows — train on more eval data first")
        return 1

    X, y, preds = _build_X_y(rows)
    print(f"[calib_v3] feature matrix: {X.shape}")
    print(f"[calib_v3] predicted_decision distribution: {dict(Counter(preds.tolist()))}")

    # v3a — per-decision LR
    print("\n" + "=" * 60)
    print("v3a — per-predicted-decision logistic regression")
    print("=" * 60)
    v3a_bundle, v3a_summary = train_v3a(X, y, preds, seed=args.seed)
    print(f"Overall CV ROC-AUC: {v3a_summary['overall_cv_roc_auc']}")
    print(f"Overall CV Brier:   {v3a_summary['overall_cv_brier']}")
    print("Per-decision metrics:")
    for d, m in v3a_summary["per_decision_metrics"].items():
        print(f"  {d:<12} n={m['n']:>3} roc={m.get('cv_roc_auc')} brier={m.get('cv_brier')}")
    _print_table("v3a", v3a_summary["calibration_table"])

    # v3b — XGBoost
    print("\n" + "=" * 60)
    print("v3b — single XGBoost classifier")
    print("=" * 60)
    v3b_bundle, v3b_summary = train_v3b(X, y, seed=args.seed)
    print(f"CV ROC-AUC: {v3b_summary['cv_roc_auc']}")
    print(f"CV Brier:   {v3b_summary['cv_brier']}")
    _print_table("v3b", v3b_summary["calibration_table"])
    print("\nTop XGBoost feature importances:")
    for name, imp in list(v3b_summary["feature_importances"].items())[:8]:
        print(f"  {name:<32} {imp:.4f}")

    # Save both
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "calibrator_v3a.pkl").write_bytes(pickle.dumps(v3a_bundle))
    (MODEL_DIR / "calibrator_v3b.pkl").write_bytes(pickle.dumps(v3b_bundle))
    (MODEL_DIR / "calibrator_v3a.json").write_text(json.dumps(v3a_summary, indent=2))
    (MODEL_DIR / "calibrator_v3b.json").write_text(json.dumps(v3b_summary, indent=2))

    compare = {
        "v3a_per_decision_lr": {
            "roc_auc": v3a_summary["overall_cv_roc_auc"],
            "brier": v3a_summary["overall_cv_brier"],
        },
        "v3b_xgboost": {
            "roc_auc": v3b_summary["cv_roc_auc"],
            "brier": v3b_summary["cv_brier"],
        },
        "v2_logreg_baseline": {
            "roc_auc": 0.721,
            "brier": 0.174,
            "note": "from models/calibrator_v2.json — same corpus, single global LR",
        },
    }
    (MODEL_DIR / "calibrator_v3_compare.json").write_text(json.dumps(compare, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY (CV metrics — eval still decides the winner)")
    print("=" * 60)
    for k, v in compare.items():
        print(f"  {k:<28} roc_auc={v['roc_auc']}  brier={v['brier']}")
    print(
        "\n[ok] saved calibrator_v3a.pkl, calibrator_v3b.pkl, calibrator_v3_compare.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
