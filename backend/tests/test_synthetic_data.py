from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_claims import STRATA, generate_claims  # noqa: E402


def test_generates_exactly_n() -> None:
    claims = generate_claims(seed=42, n=100)
    assert len(claims) == 100


def test_stratification_matches_spec() -> None:
    claims = generate_claims(seed=42, n=100)
    by_kind: dict[str, int] = {}
    for c in claims:
        by_kind[c.decision_kind] = by_kind.get(c.decision_kind, 0) + 1

    # n=100 case must hit exact strata counts.
    for kind, expected in STRATA.items():
        assert by_kind.get(kind, 0) == expected, (
            f"stratum {kind}: expected {expected}, got {by_kind.get(kind, 0)}"
        )


def test_seed_is_deterministic() -> None:
    a = generate_claims(seed=42, n=100)
    b = generate_claims(seed=42, n=100)
    assert [c.claim_id for c in a] == [c.claim_id for c in b]
    assert [c.raw_input for c in a] == [c.raw_input for c in b]


def test_fraud_patterns_distributed() -> None:
    claims = generate_claims(seed=42, n=100)
    fraud = [c for c in claims if c.is_fraud]
    assert len(fraud) == 10
    patterns = {c.fraud_pattern for c in fraud}
    assert patterns == {"same_email_multi", "address_mismatch", "exif_mismatch"}


def test_every_claim_has_required_fields() -> None:
    claims = generate_claims(seed=42, n=100)
    for c in claims:
        assert c.claim_id
        assert c.customer_id
        assert c.customer_email
        assert c.sku
        assert c.raw_input
        assert c.expected_decision in {"approve", "reject", "needs_info"}
        assert c.expected_resolution in {
            "refund",
            "replacement",
            "repair",
            "store_credit",
            "none",
        }
        assert c.expected_citation
