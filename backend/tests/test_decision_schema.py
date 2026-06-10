from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai.schemas import Decision


def test_decision_valid() -> None:
    d = Decision(
        outcome="approve",
        resolution="replacement",
        rationale="Battery defect within the 12-month coverage window with strong evidence.",
        policy_citation="covered for twelve (12) months from the date of purchase",
        confidence="high",
    )
    assert d.outcome == "approve"
    assert d.missing_info_questions == []


def test_decision_needs_info_with_questions() -> None:
    d = Decision(
        outcome="needs_info",
        resolution="none",
        rationale="The customer's message is very brief and lacks documentation per Section 3.",
        policy_citation="Claims missing any of the above will be marked",
        confidence="medium",
        missing_info_questions=[
            "Could you share the order number or proof of purchase?",
            "Could you send photos of the issue?",
        ],
    )
    assert len(d.missing_info_questions) == 2


def test_decision_rejects_short_rationale() -> None:
    with pytest.raises(ValidationError):
        Decision(
            outcome="reject",
            resolution="none",
            rationale="too short",
            policy_citation="some valid policy citation here",
            confidence="high",
        )


def test_decision_rejects_short_citation() -> None:
    with pytest.raises(ValidationError):
        Decision(
            outcome="reject",
            resolution="none",
            rationale="A reasonably long rationale describing the decision.",
            policy_citation="too short",
            confidence="high",
        )
