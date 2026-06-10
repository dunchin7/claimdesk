"""AI-generated-image likelihood (Week 8).

Combines EXIF signals into a 0–1 likelihood score. The score is *not*
calibrated — these are heuristic weights I picked, not learned from
labeled data. For production, the right next step is to label a few
hundred (real, AI-gen) pairs and fit a small classifier on these
features plus pixel-level features (JPEG quantization tables, ELA
residuals). That's a Week-9-onward task.

The output also includes a `signals` list — human-readable strings the
operator UI can show as "why this photo is flagged". Score without
explanation is worthless in claims review.

The ML detector (HuggingFace `umm-maybe/AI-image-detector` or similar)
is a documented seam in `_ml_detector_score` — wired but stubbed because
pulling 200MB+ of weights for every dev environment isn't justified
until we have a real fraud-photo corpus to evaluate against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging import get_logger
from app.vision.exif import ExifSignals

log = get_logger(__name__)


# Heuristic weights — picked by intuition, not learned. Each signal's
# contribution to the final score, capped at 1.0.
_WEIGHT_SUSPICIOUS_SOFTWARE = 0.70  # Software field names a known AI tool
_WEIGHT_NO_EXIF = 0.30              # No EXIF at all (Messenger / screenshot / AI)
_WEIGHT_NO_GPS = 0.05               # Mild — many real photos strip GPS
_WEIGHT_TIMESTAMP_IMPOSSIBLE = 0.40 # Photo predates claim by >180 days
_WEIGHT_TIMESTAMP_FUTURE = 0.50     # Photo dated AFTER claim filed


@dataclass
class AIDetectionResult:
    """Aggregated likelihood that the photo was AI-generated or staged."""

    score: float                   # 0.0 (almost certainly real) … 1.0 (almost certainly fake)
    signals: list[str]             # Human-readable reasons
    raw_signals: dict[str, float] = field(default_factory=dict)
    ml_detector_score: float | None = None  # Reserved for future HF model


def ai_generated_likelihood(
    exif: ExifSignals,
    *,
    claim_date: datetime | None = None,
    ml_score: float | None = None,
) -> AIDetectionResult:
    """Combine EXIF signals (and optional ML score) into a 0–1 likelihood.

    Args:
        exif: output of `extract_exif()`
        claim_date: date the claim was filed. Used to compute timestamp
            sanity (photo predating claim by >6mo = suspicious; photo
            dated after claim filed = very suspicious).
        ml_score: optional output of an ML AI-image detector in [0,1].
            If provided, it's combined with the heuristic score via
            soft-OR — neither alone is conclusive, but agreement is.

    Returns:
        AIDetectionResult with score, human-readable signals, and the
        raw per-signal contributions for auditing.
    """
    contributions: dict[str, float] = {}
    signals: list[str] = []

    # Strongest signal — software field names a known AI tool
    if exif.suspicious_software and exif.software_field:
        contributions["suspicious_software"] = _WEIGHT_SUSPICIOUS_SOFTWARE
        signals.append(
            f"EXIF Software field reads {exif.software_field!r} — "
            f"matches a known AI image tool."
        )

    # No EXIF at all
    if not exif.has_exif:
        contributions["no_exif"] = _WEIGHT_NO_EXIF
        signals.append(
            "Image has no EXIF metadata at all. Could be a screenshot, "
            "re-saved via chat app, or AI-generated; not conclusive."
        )

    # No GPS — mild signal; many legit photos strip GPS for privacy.
    if exif.has_exif and not exif.has_gps:
        contributions["no_gps"] = _WEIGHT_NO_GPS
        signals.append(
            "Image has EXIF but no GPS coordinates. Mild signal — many "
            "phones disable GPS by default."
        )

    # Timestamp checks vs the claim date
    if claim_date is not None and exif.captured_at is not None:
        days_before = (claim_date - exif.captured_at).days
        if days_before < 0:
            contributions["timestamp_future"] = _WEIGHT_TIMESTAMP_FUTURE
            signals.append(
                f"Photo's EXIF capture date ({exif.captured_at.date()}) is "
                f"AFTER the claim date ({claim_date.date()}) — "
                "physically impossible."
            )
        elif days_before > 180:
            contributions["timestamp_impossible"] = _WEIGHT_TIMESTAMP_IMPOSSIBLE
            signals.append(
                f"Photo predates claim by {days_before} days "
                "(>6 months). Verify with customer."
            )

    # Soft-OR combination: score = 1 - prod(1 - w_i) over all contributing
    # signals. Caps at 1.0 even if every weight fires.
    score = 0.0
    for w in contributions.values():
        score = score + w - score * w  # 1 - (1-score)*(1-w)

    # If an ML score was supplied, soft-OR it in as well.
    if ml_score is not None:
        score = score + ml_score - score * ml_score
        contributions["ml_detector"] = ml_score
        signals.append(
            f"ML detector reports {ml_score:.2f} likelihood of AI generation."
        )

    # Clamp and round for readability
    score = max(0.0, min(1.0, round(score, 3)))

    if not signals:
        signals = ["No suspicion signals — EXIF is consistent with a real photo."]

    log.info(
        "ai_detection.scored",
        score=score,
        n_signals=len(signals),
        contributions=contributions,
    )
    return AIDetectionResult(
        score=score,
        signals=signals,
        raw_signals=contributions,
        ml_detector_score=ml_score,
    )


def _ml_detector_score(image_bytes: bytes) -> float | None:  # noqa: ARG001
    """Seam for a future HuggingFace AI-image detector.

    Today: returns None (no model loaded). When we have a real fraud-photo
    corpus to evaluate against, wire `umm-maybe/AI-image-detector` (~200MB)
    or Hive AI (paid API) here. The return shape is a single [0,1] score
    that `ai_generated_likelihood()` will soft-OR with the heuristic.
    """
    return None
