"""Unit tests for app/vision/exif.py.

Each test generates a tiny in-memory JPEG with controlled EXIF and
verifies the extractor reads it back correctly. No network, no real
images committed to the repo.
"""

from __future__ import annotations

import io

import piexif
import pytest
from PIL import Image

from app.vision.exif import extract_exif


# --- helpers ----------------------------------------------------------------


def _make_jpeg(
    *,
    size: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] = (180, 180, 180),
    software: str | None = None,
    make: str | None = None,
    model: str | None = None,
    datetime_original: str | None = None,
    include_gps: bool = False,
) -> bytes:
    """Create a JPEG with exactly the EXIF tags we specify."""
    img = Image.new("RGB", size, color)

    zeroth: dict = {}
    exif: dict = {}
    gps: dict = {}

    if software is not None:
        zeroth[piexif.ImageIFD.Software] = software.encode("utf-8")
    if make is not None:
        zeroth[piexif.ImageIFD.Make] = make.encode("utf-8")
    if model is not None:
        zeroth[piexif.ImageIFD.Model] = model.encode("utf-8")
    if datetime_original is not None:
        exif[piexif.ExifIFD.DateTimeOriginal] = datetime_original.encode("utf-8")
    if include_gps:
        gps[piexif.GPSIFD.GPSLatitudeRef] = b"N"
        gps[piexif.GPSIFD.GPSLatitude] = ((37, 1), (46, 1), (29, 1))
        gps[piexif.GPSIFD.GPSLongitudeRef] = b"W"
        gps[piexif.GPSIFD.GPSLongitude] = ((122, 1), (25, 1), (10, 1))

    exif_bytes = piexif.dump({"0th": zeroth, "Exif": exif, "GPS": gps, "1st": {}, "thumbnail": None})

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_bytes)
    return buf.getvalue()


def _make_jpeg_no_exif() -> bytes:
    img = Image.new("RGB", (32, 32), (180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")  # no exif kwarg → no APP1 segment
    return buf.getvalue()


# --- tests ------------------------------------------------------------------


def test_no_exif_returns_empty_signals() -> None:
    sig = extract_exif(_make_jpeg_no_exif())
    assert sig.has_exif is False
    assert sig.has_gps is False
    assert sig.captured_at is None
    assert sig.software_field is None
    assert sig.suspicious_software is False


def test_normal_phone_photo() -> None:
    sig = extract_exif(_make_jpeg(
        make="Apple", model="iPhone 14",
        datetime_original="2026:05:01 14:30:22",
        include_gps=True,
    ))
    assert sig.has_exif is True
    assert sig.has_gps is True
    assert sig.camera_make == "Apple"
    assert sig.camera_model == "iPhone 14"
    assert sig.suspicious_software is False
    assert sig.captured_at is not None
    assert sig.captured_at.year == 2026
    assert sig.captured_at.month == 5


def test_stable_diffusion_software_field_flagged() -> None:
    sig = extract_exif(_make_jpeg(software="AUTOMATIC1111 / Stable Diffusion"))
    assert sig.has_exif is True
    assert sig.suspicious_software is True
    assert "stable diffusion" in (sig.software_field or "").lower()


def test_midjourney_software_field_flagged() -> None:
    sig = extract_exif(_make_jpeg(software="Midjourney v6"))
    assert sig.suspicious_software is True


def test_legitimate_software_not_flagged() -> None:
    # Photo editing apps are not AI generators
    sig = extract_exif(_make_jpeg(software="Adobe Lightroom 12.0"))
    assert sig.has_exif is True
    assert sig.suspicious_software is False


def test_corrupt_bytes_returns_empty_signals() -> None:
    sig = extract_exif(b"not a real image, just noise" * 10)
    assert sig.has_exif is False
    assert sig.suspicious_software is False


def test_days_before_with_known_capture() -> None:
    from datetime import datetime

    # 2026-01-01 12:00:00 → 2026-05-01 00:00:00 is 119 days + 12 hours;
    # timedelta.days truncates, so we expect 119 (not 120).
    sig = extract_exif(_make_jpeg(datetime_original="2026:01:01 12:00:00"))
    reference = datetime(2026, 5, 1)
    assert sig.days_before(reference) == 119


def test_days_before_without_capture_returns_none() -> None:
    from datetime import datetime

    sig = extract_exif(_make_jpeg_no_exif())
    assert sig.days_before(datetime(2026, 5, 1)) is None


@pytest.fixture
def fixture_data() -> dict[str, bytes]:
    """Build a small library of photo fixtures used across multiple tests."""
    return {
        "real_phone": _make_jpeg(
            make="Apple", model="iPhone 14",
            datetime_original="2026:04:20 09:15:00",
            include_gps=True,
        ),
        "ai_generated": _make_jpeg(
            software="Stable Diffusion XL",
            datetime_original="2026:05:01 10:00:00",
        ),
        "stripped": _make_jpeg_no_exif(),
        "future_dated": _make_jpeg(
            make="Apple", model="iPhone 14",
            datetime_original="2027:05:01 12:00:00",
        ),
        "old_photo": _make_jpeg(
            make="Apple", model="iPhone 11",
            datetime_original="2024:01:10 08:00:00",
            include_gps=True,
        ),
    }


def test_fixture_library_extracts_correctly(fixture_data: dict[str, bytes]) -> None:
    assert extract_exif(fixture_data["real_phone"]).suspicious_software is False
    assert extract_exif(fixture_data["ai_generated"]).suspicious_software is True
    assert extract_exif(fixture_data["stripped"]).has_exif is False
    assert extract_exif(fixture_data["future_dated"]).captured_at is not None
    assert extract_exif(fixture_data["old_photo"]).has_gps is True
