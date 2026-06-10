from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai.schemas import ClaimExtraction


def test_minimal_valid_extraction() -> None:
    e = ClaimExtraction(
        evidence_strength="moderate",
        customer_summary="Customer reports their battery stopped holding a charge.",
        customer_emotion="calm",
        prior_contact_attempts=False,
    )
    assert e.sku is None
    assert e.failure_mode is None
    assert e.evidence_strength == "moderate"
    assert e.mentioned_dates == []


def test_full_valid_extraction() -> None:
    from datetime import date

    e = ClaimExtraction(
        sku="EB-PACE-500",
        failure_mode="battery",
        claim_type="defect",
        severity="functional",
        evidence_strength="strong",
        customer_summary="Battery on PaceLine 500 fails to hold charge after 30 cycles.",
        time_since_purchase_days=120,
        mentioned_serial="PC500-887412-A",
        mentioned_dates=[date(2026, 1, 15)],
        customer_emotion="calm",
        prior_contact_attempts=False,
    )
    assert e.failure_mode == "battery"
    assert e.time_since_purchase_days == 120
    assert e.mentioned_serial == "PC500-887412-A"


def test_rejects_invalid_failure_mode() -> None:
    with pytest.raises(ValidationError):
        ClaimExtraction(
            failure_mode="exploded",  # type: ignore[arg-type]
            evidence_strength="weak",
            customer_summary="A reasonably long summary of the claim.",
            customer_emotion="calm",
            prior_contact_attempts=False,
        )


def test_rejects_too_short_summary() -> None:
    with pytest.raises(ValidationError):
        ClaimExtraction(
            evidence_strength="weak",
            customer_summary="too short",
            customer_emotion="calm",
            prior_contact_attempts=False,
        )


def test_rejects_negative_days() -> None:
    with pytest.raises(ValidationError):
        ClaimExtraction(
            evidence_strength="weak",
            customer_summary="Customer reports an issue with their bike.",
            customer_emotion="calm",
            prior_contact_attempts=False,
            time_since_purchase_days=-5,
        )
