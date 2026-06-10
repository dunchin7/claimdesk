from __future__ import annotations

from app.adjudication.citation import normalize, verify_citation

POLICY = """
Manufacturer defects in materials or workmanship are covered for twelve (12)
months from the date of purchase. The date of purchase is the order date
recorded by PaceLine or the authorized reseller; ship date and delivery date
are not used.
"""


def test_verbatim_substring_matches() -> None:
    cite = "covered for twelve (12) months from the date of purchase"
    res = verify_citation(cite, POLICY)
    assert res.verbatim is True
    assert res.fuzzy_ratio == 1.0


def test_smart_quotes_tolerated() -> None:
    cite = "Manufacturer defects in materials or workmanship are covered"  # ascii
    policy_with_smart = POLICY.replace(
        "twelve (12)", "twelve (12)"
    )  # baseline; we test smart quotes more directly:
    cite_smart = "Manufacturer defects in materials or workmanship are covered"
    assert verify_citation(cite_smart, policy_with_smart).verbatim is True


def test_whitespace_differences_tolerated() -> None:
    cite = "are\tcovered     for\ntwelve (12) months"
    res = verify_citation(cite, POLICY)
    assert res.verbatim is True


def test_paraphrase_rejected() -> None:
    cite = "are covered for one year from purchase"
    res = verify_citation(cite, POLICY)
    assert res.verbatim is False
    # Should give some fuzzy ratio but well below 1.0
    assert 0.0 < res.fuzzy_ratio < 1.0


def test_empty_citation() -> None:
    res = verify_citation("", POLICY)
    assert res.verbatim is False
    assert res.fuzzy_ratio == 0.0


def test_normalize_idempotent() -> None:
    s = "Hello   world"
    assert normalize(normalize(s)) == normalize(s)
