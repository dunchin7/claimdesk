"""The cost model: turn one automated claim decision into dollars.

Every coefficient here is an **editable assumption with a cited default** — that
is the whole point of the instrument. We don't claim to know what a wrongful
denial costs a given book; we make the assumptions explicit so an operator can
price *their* book by changing the numbers.

Two error types, deliberately asymmetric:

- **Leakage** (auto-approved something that should not have been paid): costs
  roughly the payout that should never have left the building.
- **False denial** (auto-rejected a valid claim): costs *more* than leakage for
  the same claim — because you typically end up paying the claim anyway on
  appeal, **plus** the dispute/complaint handling and the churn of a wronged
  customer. This asymmetry is the insight the ledger exists to surface.

A correct auto-resolution costs nothing and *saves* one human touch (LAE). A
claim routed to a human costs a human touch — the same as the all-manual
baseline — so it nets to zero benefit, neither saving nor leaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Outcome = Literal["approve", "reject", "needs_info"]
ErrorType = Literal["leakage", "false_denial"] | None


@dataclass
class ClaimCost:
    """The priced outcome of a single automated decision."""

    decision: Outcome
    gold: Outcome
    claim_value: float
    auto_resolved: bool          # acted without a human (approve/reject)
    lae_saved: float             # human-review labor avoided vs all-manual
    leakage: float               # $ paid that shouldn't have been
    false_denial: float          # $ exposure from wrongly denying a valid claim
    error_type: ErrorType        # which side of the ledger this error landed on

    @property
    def net_benefit(self) -> float:
        """Net dollars vs an all-manual baseline.

        Manual touches every claim, so the labor cost of a *routed* claim
        cancels against the baseline (net 0). The automation's net benefit is
        the labor it saved on auto-resolved claims, minus any error cost it
        incurred by acting.
        """
        return self.lae_saved - self.leakage - self.false_denial


@dataclass
class CostModel:
    """Editable, cited cost coefficients (all USD)."""

    # A correct auto-resolution avoids one fully-loaded human review touch.
    # Default: a mid-range loss-adjustment-expense per simple claim touch.
    review_labor: float = 12.0

    # Leakage = the payout that should never have gone out, times this factor.
    # 1.0 = you eat the full wrongful payout.
    leakage_multiplier: float = 1.0

    # False denial extras, ON TOP OF eventually paying the claim:
    #   dispute_ev — expected appeal + DOI-complaint handling + bad-faith
    #     exposure per wrongful denial (cited default; UCSPA/NAIC unfair-claims
    #     practices make wrongful denial a regulated, costly event).
    dispute_ev: float = 120.0
    #   churn_cost — expected lost customer lifetime value when you wrongly
    #     deny someone with valid coverage.
    churn_cost: float = 80.0

    # When gold == needs_info, an automated approve/reject is an error of
    # judgment but we don't assume the full payout is owed; weight it down.
    ambiguous_error_weight: float = 0.5

    # provenance for each coefficient, surfaced in the UI / report
    sources: dict[str, str] = field(
        default_factory=lambda: {
            "review_labor": "fully-loaded cost of one manual claim touch (LAE) — editable",
            "leakage_multiplier": "fraction of a wrongful payout the carrier eats — editable",
            "dispute_ev": "expected appeal + DOI-complaint + bad-faith handling per wrongful denial (UCSPA/NAIC) — editable assumption",
            "churn_cost": "expected lost LTV per wrongly-denied valid customer — editable assumption",
        }
    )

    def price_claim(self, decision: Outcome, gold: Outcome, claim_value: float) -> ClaimCost:
        auto_resolved = decision in ("approve", "reject")
        lae_saved = self.review_labor if auto_resolved else 0.0
        leakage = 0.0
        false_denial = 0.0
        error_type: ErrorType = None

        if not auto_resolved:
            # Routed to a human: correct by assumption, no error, no net saving.
            return ClaimCost(decision, gold, claim_value, False, 0.0, 0.0, 0.0, None)

        if decision == gold:
            # Correct auto-resolution: pure labor saving, no error cost.
            return ClaimCost(decision, gold, claim_value, True, lae_saved, 0.0, 0.0, None)

        # Wrong auto-resolution.
        weight = self.ambiguous_error_weight if gold == "needs_info" else 1.0
        if decision == "approve":
            leakage = claim_value * self.leakage_multiplier * weight
            error_type = "leakage"
        else:  # decision == "reject" -> false denial
            owed = claim_value if gold == "approve" else 0.0
            false_denial = (owed + self.dispute_ev + self.churn_cost) * weight
            error_type = "false_denial"

        return ClaimCost(
            decision=decision,
            gold=gold,
            claim_value=claim_value,
            auto_resolved=True,
            lae_saved=lae_saved,
            leakage=leakage,
            false_denial=false_denial,
            error_type=error_type,
        )


# --- claim value estimation (a transparent, editable heuristic) -------------
# What a wrongful payout / owed-on-appeal amount is worth, by device class.
# Deliberately simple and overridable — real deployments join to the contract's
# per-device coverage value.
_DEVICE_VALUES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("laptop", "macbook"), 1200.0),
    (("ipad", "tablet"), 500.0),
    (("headphone", "earbud", "airpod"), 150.0),
    (("watch",), 350.0),
    (("iphone", "galaxy", "phone", "pixel"), 800.0),
)
_DEFAULT_CLAIM_VALUE = 600.0


def estimate_claim_value(text: str, default: float = _DEFAULT_CLAIM_VALUE) -> float:
    """Best-effort device payout value from a claim's free text.

    Transparent on purpose: an operator overrides this with their real
    per-contract coverage values.
    """
    low = text.lower()
    for keywords, value in _DEVICE_VALUES:
        if any(k in low for k in keywords):
            return value
    return default
