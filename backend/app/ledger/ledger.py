"""Price a book of claims into a two-sided P&L.

Takes per-claim decisions (from any decision system / model tier) plus gold
labels and claim values, and aggregates the cost model into the headline
numbers an operator cares about: labor saved, leakage incurred, false-denial
liability incurred, and the net — both absolute and per-1,000 claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ledger.cost_model import ClaimCost, CostModel, Outcome


@dataclass
class ScoredClaim:
    """One claim's decision vs truth, with its dollar exposure.

    `confidence` is the model's self-reported certainty in `decision` (0-1),
    used by the threshold sweep to decide what to auto-resolve vs route. It
    defaults to 1.0 so a book without confidence prices as fully-automated.
    """

    id: str
    decision: Outcome
    gold: Outcome
    claim_value: float
    plan: str = ""
    confidence: float = 1.0


@dataclass
class BookPnL:
    """The two-sided P&L for a book of claims under one decision system."""

    label: str
    n: int
    n_auto_resolved: int
    n_routed: int
    n_leakage_events: int
    n_false_denial_events: int
    lae_saved: float
    leakage: float
    false_denial: float
    per_claim: list[ClaimCost] = field(default_factory=list)

    @property
    def auto_resolve_rate(self) -> float:
        return self.n_auto_resolved / self.n if self.n else 0.0

    @property
    def error_cost(self) -> float:
        return self.leakage + self.false_denial

    @property
    def net(self) -> float:
        return self.lae_saved - self.error_cost

    def per_1000(self, value: float) -> float:
        return value / self.n * 1000 if self.n else 0.0

    def summary(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "auto_resolve_rate": round(self.auto_resolve_rate, 4),
            "n_leakage_events": self.n_leakage_events,
            "n_false_denial_events": self.n_false_denial_events,
            "lae_saved": round(self.lae_saved, 2),
            "leakage": round(self.leakage, 2),
            "false_denial": round(self.false_denial, 2),
            "error_cost": round(self.error_cost, 2),
            "net": round(self.net, 2),
            "per_1000": {
                "lae_saved": round(self.per_1000(self.lae_saved), 2),
                "leakage": round(self.per_1000(self.leakage), 2),
                "false_denial": round(self.per_1000(self.false_denial), 2),
                "net": round(self.per_1000(self.net), 2),
            },
        }


def price_book(
    claims: list[ScoredClaim],
    cost_model: CostModel,
    *,
    label: str = "",
    threshold: float = 0.0,
) -> BookPnL:
    """Price a book. With `threshold > 0`, any claim whose confidence is below
    it is *routed to a human* (treated as needs_info) instead of auto-resolved —
    this is the lever the crossover sweep turns."""
    costs = []
    for c in claims:
        decision = c.decision if c.confidence >= threshold else "needs_info"
        costs.append(cost_model.price_claim(decision, c.gold, c.claim_value))
    return BookPnL(
        label=label,
        n=len(claims),
        n_auto_resolved=sum(1 for c in costs if c.auto_resolved),
        n_routed=sum(1 for c in costs if not c.auto_resolved),
        n_leakage_events=sum(1 for c in costs if c.error_type == "leakage"),
        n_false_denial_events=sum(1 for c in costs if c.error_type == "false_denial"),
        lae_saved=sum(c.lae_saved for c in costs),
        leakage=sum(c.leakage for c in costs),
        false_denial=sum(c.false_denial for c in costs),
        per_claim=costs,
    )


@dataclass
class CurvePoint:
    threshold: float
    auto_resolve_rate: float
    lae_saved: float
    leakage: float
    false_denial: float
    net: float

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 3),
            "auto_resolve_rate": round(self.auto_resolve_rate, 4),
            "lae_saved": round(self.lae_saved, 2),
            "leakage": round(self.leakage, 2),
            "false_denial": round(self.false_denial, 2),
            "net": round(self.net, 2),
        }


def sweep_threshold(
    claims: list[ScoredClaim],
    cost_model: CostModel,
    *,
    thresholds: list[float] | None = None,
) -> list[CurvePoint]:
    """Sweep the auto-resolve confidence threshold and price the book at each.

    Low threshold = automate everything (max labor saved, max error exposure);
    high threshold = route the uncertain ones to humans (less saved, less
    exposure). The crossover — where rising error liability overtakes labor
    saved and `net` turns negative — is the number the instrument exists to find.
    """
    if thresholds is None:
        thresholds = [round(0.50 + 0.05 * i, 2) for i in range(11)]  # 0.50..1.00
    points: list[CurvePoint] = []
    for t in thresholds:
        book = price_book(claims, cost_model, threshold=t)
        points.append(
            CurvePoint(
                threshold=t,
                auto_resolve_rate=book.auto_resolve_rate,
                lae_saved=book.lae_saved,
                leakage=book.leakage,
                false_denial=book.false_denial,
                net=book.net,
            )
        )
    return points
