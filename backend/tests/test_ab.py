"""Unit tests for app/evals/ab.py — sticky assignment + split fidelity."""

from __future__ import annotations

import pytest

from app.evals.ab import (
    ab_test,
    choose_arm,
    get_decisions,
    reset_decisions,
    summarize_decisions,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_decisions()


def test_sticky_assignment_same_key_same_arm() -> None:
    arm1 = choose_arm(experiment="exp", arms={"a": 0.5, "b": 0.5}, key="claim-42", record=False)
    arm2 = choose_arm(experiment="exp", arms={"a": 0.5, "b": 0.5}, key="claim-42", record=False)
    assert arm1 == arm2


def test_different_keys_can_get_different_arms() -> None:
    """At 50/50 over many keys, both arms should fire."""
    arms = [
        choose_arm(experiment="exp", arms={"a": 0.5, "b": 0.5}, key=f"claim-{i}", record=False)
        for i in range(200)
    ]
    assert "a" in arms
    assert "b" in arms


def test_split_is_approximately_correct() -> None:
    """50/50 over 1000 keys should give each arm ~500 ± noise."""
    arms = [
        choose_arm(experiment="exp", arms={"a": 0.5, "b": 0.5}, key=f"k{i}", record=False)
        for i in range(1000)
    ]
    n_a = arms.count("a")
    # Allow ±10% deviation
    assert 400 <= n_a <= 600


def test_unequal_weights_respected() -> None:
    """90/10 split should produce ~900 'a' out of 1000."""
    arms = [
        choose_arm(experiment="exp", arms={"a": 0.9, "b": 0.1}, key=f"k{i}", record=False)
        for i in range(1000)
    ]
    n_a = arms.count("a")
    assert 850 <= n_a <= 950


def test_different_experiments_dont_collude() -> None:
    """Same key, different experiment names → independent assignments."""
    different = 0
    for i in range(100):
        a1 = choose_arm(experiment="exp1", arms={"a": 0.5, "b": 0.5}, key=f"k{i}", record=False)
        a2 = choose_arm(experiment="exp2", arms={"a": 0.5, "b": 0.5}, key=f"k{i}", record=False)
        if a1 != a2:
            different += 1
    # ~50% should differ; allow ±15
    assert 35 <= different <= 65


def test_record_appends_to_global_log() -> None:
    choose_arm(experiment="exp", arms={"a": 1.0}, key="k1")
    choose_arm(experiment="exp", arms={"a": 1.0}, key="k2")
    decisions = get_decisions()
    assert len(decisions) == 2
    assert all(d.experiment == "exp" for d in decisions)


def test_record_false_does_not_log() -> None:
    choose_arm(experiment="exp", arms={"a": 1.0}, key="k1", record=False)
    assert get_decisions() == []


def test_summarize_decisions_groups_correctly() -> None:
    for i in range(20):
        choose_arm(experiment="e1", arms={"a": 0.5, "b": 0.5}, key=f"k{i}")
    summary = summarize_decisions()
    assert "e1" in summary
    assert sum(summary["e1"].values()) == 20


def test_decorator_picks_arm_from_kwarg() -> None:
    @ab_test(experiment="exp", arms={"a": 0.5, "b": 0.5}, key_arg="claim_id")
    def pick(claim_id: str) -> str:
        return "ignored"

    assert pick(claim_id="k1") == pick(claim_id="k1")
    # Two different keys may produce different arms (deterministic per key)


def test_decorator_picks_arm_from_positional() -> None:
    @ab_test(experiment="exp", arms={"a": 0.5, "b": 0.5}, key_arg="claim_id")
    def pick(claim_id: str) -> str:
        return "ignored"

    a = pick("k1")
    b = pick(claim_id="k1")
    assert a == b


def test_empty_arms_raises() -> None:
    with pytest.raises(ValueError):
        choose_arm(experiment="e", arms={}, key="k")


def test_zero_weights_raises() -> None:
    with pytest.raises(ValueError):
        choose_arm(experiment="e", arms={"a": 0.0, "b": 0.0}, key="k")
