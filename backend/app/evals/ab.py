"""A/B framework for prompt-version shadow testing (Week 10).

Sticky assignment: the same `key` always picks the same arm, so a single
claim's pipeline always sees one consistent prompt version. Without
stickiness the extract step could fire arm-A while adjudicate fires arm-B
on the same claim — uninterpretable.

The assignment is computed from `sha256(key + experiment_name)` mod weights.
This is deterministic across processes (unlike Python's `hash()` which
salts per process).

Usage:

    @ab_test(
        experiment="adjudicate_prompt",
        arms={"adjudicate_v3": 0.5, "adjudicate_v4": 0.5},
    )
    def pick_adjudicate_prompt(claim_id: str) -> str:
        # The decorator returns the arm name picked for this claim_id.
        # The function body is just for type/IDE — the decorator overrides.
        return "adjudicate_v3"

    # In the pipeline:
    prompt_name = pick_adjudicate_prompt(claim_id="abc-123")
    # → always returns the same arm for "abc-123"

Or as a direct function call:

    arm = choose_arm(
        experiment="adjudicate_prompt",
        arms={"adjudicate_v3": 0.5, "adjudicate_v4": 0.5},
        key="abc-123",
    )

The eval framework records the chosen arm in `EvalRunReport.per_claim[i].ab_arms`
for per-arm metric breakdown post-hoc.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable


@dataclass
class ABDecision:
    """The outcome of a single A/B decision — what was chosen, why, where."""

    experiment: str
    key: str
    arm: str
    arms_seen: dict[str, float]


# Per-process record of every A/B decision made during a run. The eval
# runner can dump this alongside the report for post-hoc arm analysis.
_decisions: list[ABDecision] = []


def get_decisions() -> list[ABDecision]:
    """Return all A/B decisions recorded this process. Used by eval runners."""
    return list(_decisions)


def reset_decisions() -> None:
    """Clear the recorded decisions. Useful at the start of a fresh eval run."""
    _decisions.clear()


def _stable_uniform(key: str, salt: str) -> float:
    """Map `(key, salt)` to a deterministic float in [0.0, 1.0).

    sha256 is overkill for cardinality but cheap and standard. Lower 64 bits
    of the digest interpreted as an integer, normalized to [0,1).
    """
    h = hashlib.sha256(f"{salt}::{key}".encode()).digest()
    n = int.from_bytes(h[:8], "big", signed=False)
    return n / (1 << 64)


def choose_arm(
    *,
    experiment: str,
    arms: dict[str, float],
    key: str,
    record: bool = True,
) -> str:
    """Pick an arm by sticky hash.

    Args:
        experiment: name of the experiment (used as salt so two experiments
            on the same key don't collude).
        arms: mapping from arm name → weight. Weights need not sum to 1; we
            normalize.
        key: the stable identifier (e.g., claim_id). Same key → same arm.
        record: if True, append to the global decision log.

    Returns:
        The chosen arm name.
    """
    if not arms:
        raise ValueError("arms must be non-empty")
    total = sum(arms.values())
    if total <= 0:
        raise ValueError("arm weights must sum to a positive number")
    u = _stable_uniform(key, salt=experiment) * total
    cum = 0.0
    chosen: str | None = None
    for name, weight in arms.items():
        cum += weight
        if u < cum:
            chosen = name
            break
    if chosen is None:
        # Fall through can happen due to floating-point edge cases — pick the
        # last arm.
        chosen = next(reversed(arms))
    if record:
        _decisions.append(
            ABDecision(experiment=experiment, key=key, arm=chosen, arms_seen=dict(arms))
        )
    return chosen


def ab_test(
    *,
    experiment: str,
    arms: dict[str, float],
    key_arg: str = "claim_id",
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Decorator: pick an arm by hashing the value of `key_arg`.

    The decorated function's body is ignored — the decorator returns the
    chosen arm directly. Useful when prompt-picking logic is a thin
    function we want to make explicit at the call site.

    Args:
        experiment: experiment name (salts the hash).
        arms: arm name → weight.
        key_arg: name of the kwarg or positional arg holding the assignment
            key (default: `claim_id`).
    """
    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            if key_arg in kwargs:
                key_val = str(kwargs[key_arg])
            else:
                # Try positional arg by name via __code__.co_varnames
                names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
                if key_arg in names:
                    idx = names.index(key_arg)
                    if idx < len(args):
                        key_val = str(args[idx])
                    else:
                        raise TypeError(
                            f"ab_test: required key_arg {key_arg!r} not provided"
                        )
                else:
                    raise TypeError(
                        f"ab_test: function has no parameter named {key_arg!r}"
                    )
            return choose_arm(experiment=experiment, arms=arms, key=key_val)
        return wrapper
    return decorator


def summarize_decisions(
    decisions: list[ABDecision] | None = None,
) -> dict[str, dict[str, int]]:
    """Group decisions by experiment → arm → count.

    Useful for verifying the actual split matches the configured weights
    after a run.
    """
    decisions = decisions if decisions is not None else _decisions
    out: dict[str, dict[str, int]] = {}
    for d in decisions:
        bucket = out.setdefault(d.experiment, {})
        bucket[d.arm] = bucket.get(d.arm, 0) + 1
    return out
