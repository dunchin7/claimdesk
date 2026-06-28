"""Canonical consumer-electronics protection coverage ontology.

The hard, reusable part of the Coverage Atlas: a single schema that *any*
device-protection plan (AppleCare+, Samsung Care+, a retailer/Asurion plan,
a manufacturer base warranty) can be normalized onto. Once a plan's messy
legal T&C is mapped to this schema, plans become directly comparable and a
claim can be adjudicated against any of them with the same engine.

Grounding is first-class: every normalized fact is a `CoverageItem` that
carries the **verbatim clause** it came from. A normalized "liquid damage is
excluded" with no clause is worthless; with the exact sentence it's an
auditable finding. This mirrors the pipeline's citation discipline — the
model *locates and structures*, it does not *assert*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Whether the plan covers a given peril / addresses a given term.
CoverageStatus = Literal[
    "covered",        # explicitly covered
    "conditional",    # covered subject to a condition / fee / tier / limit
    "excluded",       # explicitly excluded
    "not_addressed",  # the document doesn't speak to this
]


class CoverageItem(BaseModel):
    """One normalized coverage fact + the clause that grounds it."""

    status: CoverageStatus = Field(
        description="Whether this peril/term is covered, conditional, excluded, or unaddressed.",
    )
    detail: str = Field(
        default="",
        max_length=300,
        description=(
            "Short normalized detail: the fee, threshold, condition, or limit "
            "in plain terms (e.g. '$29 service fee, screen only' or 'battery "
            "below 80% of original capacity'). Empty if status is not_addressed."
        ),
    )
    clause: str = Field(
        default="",
        max_length=400,
        description=(
            "Verbatim quote from the policy that grounds this item. Must appear "
            "in the source text. Empty only when status is not_addressed."
        ),
    )


# The canonical dimensions every CE protection plan is normalized onto.
# Perils a customer actually files on:
PERIL_FIELDS = (
    "mechanical_breakdown",     # post-manufacturer-warranty failure
    "accidental_damage",        # drops, handling (ADH)
    "liquid_damage",            # spills, immersion
    "screen_damage",            # often its own fee tier
    "battery_failure",          # degradation / won't hold charge
    "theft",
    "loss",                     # misplaced (rarely covered)
    "power_surge",
    "cosmetic_damage",          # scratches/dents not affecting function
)


class CoverageProfile(BaseModel):
    """A single protection plan, normalized onto the canonical ontology.

    Produced by `app.atlas.extract.extract_coverage_profile` from a plan's
    real T&C text. Comparable across plans; usable as the policy a claim is
    adjudicated against.
    """

    plan_name: str = Field(description="Name of the plan, e.g. 'AppleCare+ for iPhone'.")
    source: str = Field(default="", description="Source URL / document the profile was extracted from.")

    # --- Perils (each a grounded CoverageItem) ---
    mechanical_breakdown: CoverageItem
    accidental_damage: CoverageItem
    liquid_damage: CoverageItem
    screen_damage: CoverageItem
    battery_failure: CoverageItem
    theft: CoverageItem
    loss: CoverageItem
    power_surge: CoverageItem
    cosmetic_damage: CoverageItem

    # --- Commercial terms (each grounded) ---
    term_length: CoverageItem = Field(description="Length of coverage / renewal terms.")
    deductible_or_fee: CoverageItem = Field(
        description="Per-incident deductible / service fee, including damage-type tiers."
    )
    claim_limit: CoverageItem = Field(
        description="Limit on number of claims per term and/or dollar cap."
    )
    transferable: CoverageItem = Field(description="Whether coverage transfers on resale.")

    # --- Exclusions + process (lists of grounded items) ---
    exclusions: list[CoverageItem] = Field(
        default_factory=list,
        description="Notable exclusions, each with its verbatim clause.",
    )
    evidence_required: list[CoverageItem] = Field(
        default_factory=list,
        description="What the claimant must provide / do (proof of purchase, claim window, diagnostics).",
    )
    resolution_types: list[str] = Field(
        default_factory=list,
        description="How claims are resolved: repair, replacement (new/refurb), reimbursement, store_credit.",
    )

    def peril_items(self) -> dict[str, CoverageItem]:
        """The peril fields as a name→item map, for building the comparison matrix."""
        return {name: getattr(self, name) for name in PERIL_FIELDS}
