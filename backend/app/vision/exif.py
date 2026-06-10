"""EXIF metadata extraction (Week 8).

We extract a small set of high-signal tags via PIL — capture timestamp,
camera make/model, software field, and GPS presence. These feed the
AI-image detector in `ai_detection.py` and the fraud features in the
Week-9 XGBoost classifier.

The function is pure: in = JPEG bytes, out = ExifSignals dataclass.
No I/O, no LLM. Cheap, deterministic, easy to test.

We don't parse every EXIF tag because most have no fraud-signal value;
sticking to a small set keeps the surface tight and the dataclass
trivially serializable for log / DB writes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime

from PIL import ExifTags, Image, UnidentifiedImageError

# Software-field substrings that strongly indicate AI generation. Case-
# insensitive substring match — we want to catch "Stable Diffusion XL",
# "AUTOMATIC1111 / Stable Diffusion", "Midjourney v6", "DALL-E 3", etc.
_AI_SOFTWARE_TOKENS = (
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "flux",
    "imagen",
    "firefly",
    "ideogram",
    "leonardo.ai",
    "ai-generated",
    "ai generated",
    "automatic1111",
    "comfyui",
)


@dataclass
class ExifSignals:
    """Structured read of the fraud-relevant EXIF tags."""

    has_exif: bool
    has_gps: bool
    camera_make: str | None
    camera_model: str | None
    software_field: str | None
    captured_at: datetime | None
    # Computed: substring of `software_field` matches a known AI tool tag.
    suspicious_software: bool
    # All raw tag names we found (for debugging / auditing). Not used in
    # scoring.
    raw_tag_names: list[str] = field(default_factory=list)

    def days_before(self, reference: datetime) -> int | None:
        """Days between `captured_at` and `reference`. None if unknown."""
        if self.captured_at is None:
            return None
        delta = reference - self.captured_at
        return delta.days


def _parse_exif_datetime(raw: str | None) -> datetime | None:
    """EXIF DateTime format is 'YYYY:MM:DD HH:MM:SS'. Some cameras vary."""
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def extract_exif(image_bytes: bytes) -> ExifSignals:
    """Extract a normalized EXIF signal bundle from JPEG bytes.

    On any decode error (corrupt file, non-image bytes), returns an
    "all-missing" signal bundle — caller can treat missing EXIF as its
    own (mild) suspicion signal.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError):
        return ExifSignals(
            has_exif=False, has_gps=False,
            camera_make=None, camera_model=None,
            software_field=None, captured_at=None,
            suspicious_software=False,
        )

    raw_exif = img.getexif()
    if raw_exif is None or len(raw_exif) == 0:
        return ExifSignals(
            has_exif=False, has_gps=False,
            camera_make=None, camera_model=None,
            software_field=None, captured_at=None,
            suspicious_software=False,
        )

    # `getexif()` only returns IFD0 entries; the meaty tags we care
    # about (DateTimeOriginal, ExposureTime, etc.) live in the Exif
    # sub-IFD at tag 0x8769. GPS lives in the GPS sub-IFD at 0x8825.
    # Pull both, then merge into a single name→value dict.
    tags: dict[str, object] = {}
    for tag_id, value in raw_exif.items():
        name = ExifTags.TAGS.get(tag_id, f"unknown_{tag_id}")
        tags[name] = value

    try:
        exif_ifd = raw_exif.get_ifd(ExifTags.IFD.Exif)
    except (AttributeError, KeyError):
        exif_ifd = {}
    for tag_id, value in (exif_ifd or {}).items():
        name = ExifTags.TAGS.get(tag_id, f"unknown_{tag_id}")
        # Don't clobber a top-level tag with the sub-IFD one if both exist.
        tags.setdefault(name, value)

    try:
        gps_ifd = raw_exif.get_ifd(ExifTags.IFD.GPSInfo)
    except (AttributeError, KeyError):
        gps_ifd = {}
    if gps_ifd:
        tags["GPSInfo"] = gps_ifd

    raw_tag_names = sorted(tags.keys())

    software = tags.get("Software")
    if isinstance(software, bytes):
        software = software.decode("utf-8", errors="replace")
    software_str = str(software).strip() if software is not None else None

    suspicious = False
    if software_str:
        lower = software_str.lower()
        suspicious = any(tok in lower for tok in _AI_SOFTWARE_TOKENS)

    make = tags.get("Make")
    model = tags.get("Model")
    if isinstance(make, bytes):
        make = make.decode("utf-8", errors="replace")
    if isinstance(model, bytes):
        model = model.decode("utf-8", errors="replace")

    captured_at = _parse_exif_datetime(
        tags.get("DateTimeOriginal") or tags.get("DateTime") or None  # type: ignore[arg-type]
    )

    # GPSInfo tag is itself a sub-IFD. Its presence alone is the signal.
    has_gps = "GPSInfo" in tags and bool(tags.get("GPSInfo"))

    return ExifSignals(
        has_exif=True,
        has_gps=has_gps,
        camera_make=str(make).strip() if make is not None else None,
        camera_model=str(model).strip() if model is not None else None,
        software_field=software_str,
        captured_at=captured_at,
        suspicious_software=suspicious,
        raw_tag_names=raw_tag_names,
    )
