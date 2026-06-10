"""Unit tests for app/vision/ai_detection.py.

These are pure logic tests on `ExifSignals` inputs — no images, no model
loads. The fixture-library tests in test_exif.py validate the extractor
end-to-end on synthetic JPEGs.
"""

from __future__ import annotations

from datetime import datetime

from app.vision.ai_detection import ai_generated_likelihood
from app.vision.exif import ExifSignals


def _real_signals() -> ExifSignals:
    return ExifSignals(
        has_exif=True,
        has_gps=True,
        camera_make="Apple",
        camera_model="iPhone 14",
        software_field="iOS 17.4",
        captured_at=datetime(2026, 4, 20, 9, 15, 0),
        suspicious_software=False,
    )


def _sd_signals() -> ExifSignals:
    return ExifSignals(
        has_exif=True,
        has_gps=False,
        camera_make=None,
        camera_model=None,
        software_field="Stable Diffusion XL",
        captured_at=None,
        suspicious_software=True,
    )


def _no_exif_signals() -> ExifSignals:
    return ExifSignals(
        has_exif=False,
        has_gps=False,
        camera_make=None,
        camera_model=None,
        software_field=None,
        captured_at=None,
        suspicious_software=False,
    )


def test_real_phone_photo_scores_low() -> None:
    r = ai_generated_likelihood(_real_signals())
    assert r.score < 0.05
    assert any("consistent" in s.lower() for s in r.signals)


def test_stable_diffusion_scores_high() -> None:
    r = ai_generated_likelihood(_sd_signals())
    # Suspicious software (0.70) soft-OR'd with no_gps (0.05)
    assert r.score >= 0.70
    assert any("stable diffusion" in s.lower() for s in r.signals)


def test_no_exif_alone_is_only_mildly_suspicious() -> None:
    r = ai_generated_likelihood(_no_exif_signals())
    # Just no_exif (0.30); not conclusive on its own
    assert 0.25 < r.score < 0.35


def test_future_dated_photo_is_very_suspicious() -> None:
    sig = ExifSignals(
        has_exif=True, has_gps=True,
        camera_make="Apple", camera_model="iPhone 14",
        software_field="iOS 17.4",
        captured_at=datetime(2027, 5, 1),  # AFTER claim
        suspicious_software=False,
    )
    r = ai_generated_likelihood(sig, claim_date=datetime(2026, 5, 1))
    assert r.score >= 0.45
    assert any("after the claim" in s.lower() for s in r.signals)


def test_old_photo_predating_claim_by_more_than_180_days() -> None:
    sig = ExifSignals(
        has_exif=True, has_gps=True,
        camera_make="Apple", camera_model="iPhone 11",
        software_field=None,
        captured_at=datetime(2025, 1, 1),
        suspicious_software=False,
    )
    r = ai_generated_likelihood(sig, claim_date=datetime(2026, 5, 1))
    # timestamp_impossible weight (0.40) fires
    assert r.score >= 0.35
    assert any("predates" in s.lower() for s in r.signals)


def test_ml_score_soft_ors_with_heuristics() -> None:
    # Real photo + ML detector says 0.6 likelihood → final should be ~0.6
    r = ai_generated_likelihood(_real_signals(), ml_score=0.6)
    assert 0.55 < r.score < 0.65
    assert any("ml detector" in s.lower() for s in r.signals)


def test_multiple_signals_soft_or_to_higher_score() -> None:
    # SD software + no GPS + future date — all three fire
    sig = ExifSignals(
        has_exif=True, has_gps=False,
        camera_make=None, camera_model=None,
        software_field="Midjourney",
        captured_at=datetime(2027, 1, 1),
        suspicious_software=True,
    )
    r = ai_generated_likelihood(sig, claim_date=datetime(2026, 5, 1))
    # Soft-OR caps below 1.0 but should be well above any single weight
    assert r.score >= 0.85
    assert r.score <= 1.0
    assert len(r.signals) >= 2
