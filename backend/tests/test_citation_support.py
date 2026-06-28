"""Unit tests for the decision↔citation consistency check (grounding step 2).

These run offline — the judge LLM call is monkey-patched. They verify the
*seam*: that a verbatim-but-irrelevant citation is caught, that a supporting
citation passes, and that a failed verification degrades to None (best-effort)
rather than raising. The quality of the judge's verdict itself is a smoke /
eval concern (needs a real model).
"""

from __future__ import annotations

import pytest

from app.adjudication import consistency
from app.ai.schemas import CitationSupport, Decision


def _decision(outcome: str, citation: str) -> Decision:
    return Decision(
        outcome=outcome,  # type: ignore[arg-type]
        resolution="replacement" if outcome == "approve" else "none",
        rationale="Test rationale tying the claim to the cited clause for coverage.",
        policy_citation=citation,
        confidence="high",
        missing_info_questions=[],
    )


async def test_supports_verdict_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return CitationSupport(verdict="supports", reasoning="Coverage clause applies to a defect.")

    monkeypatch.setattr(consistency, "chat", fake_chat)
    decision = _decision("approve", "Manufacturer defects are covered for twelve (12) months.")
    result = await consistency.verify_decision_support(decision, "Battery defect at 90 days.")
    assert result is not None
    assert result.verdict == "supports"


async def test_unrelated_citation_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dangerous case: a real policy clause that doesn't justify the outcome."""
    async def fake_chat(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return CitationSupport(
            verdict="unrelated",
            reasoning="Shipping-window clause does not bear on a battery-defect approval.",
        )

    monkeypatch.setattr(consistency, "chat", fake_chat)
    decision = _decision("approve", "Claims for shipping damage must be filed within seven (7) days.")
    result = await consistency.verify_decision_support(decision, "Battery defect at 90 days.")
    assert result is not None
    assert result.verdict == "unrelated"


async def test_contradicting_citation_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return CitationSupport(
            verdict="contradicts",
            reasoning="An exclusion clause cannot justify an approval.",
        )

    monkeypatch.setattr(consistency, "chat", fake_chat)
    decision = _decision("approve", "Wear items are excluded from the standard warranty.")
    result = await consistency.verify_decision_support(decision, "Customer reports worn brake pads.")
    assert result is not None
    assert result.verdict == "contradicts"


async def test_failure_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed verification call returns None (best-effort), never raises."""
    async def boom(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("provider down")

    monkeypatch.setattr(consistency, "chat", boom)
    decision = _decision("approve", "Manufacturer defects are covered for twelve (12) months.")
    result = await consistency.verify_decision_support(decision, "Battery defect at 90 days.")
    assert result is None


def test_prompt_renders_with_all_fields() -> None:
    """The verify_citation prompt template renders with the expected variables."""
    from app.ai.prompt_loader import render_prompt

    text = render_prompt(
        "verify_citation_v1",
        outcome="approve",
        resolution="replacement",
        rationale="r",
        citation="some clause",
        claim_summary="a claim",
    )
    assert "approve" in text
    assert "some clause" in text
    assert "a claim" in text
