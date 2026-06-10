"""Pydantic schemas for LLM I/O.

Schema versioning is via the field set on the model. When a breaking change
is needed, evolve the model in place and bump the corresponding prompt's
filename suffix (`extract_v1.md` → `extract_v2.md`). Old eval runs reference
the prompt by filename, which pins the schema-prompt pairing.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

FailureMode = Literal[
    "battery", "motor", "frame", "electrical", "shipping_damage", "other"
]
ClaimType = Literal["defect", "accidental_damage", "shipping", "wear_tear"]
Severity = Literal["cosmetic", "functional", "total_loss"]
EvidenceStrength = Literal["strong", "moderate", "weak"]
CustomerEmotion = Literal["calm", "frustrated", "angry", "polite"]


class ClaimExtraction(BaseModel):
    """Structured extraction from a free-form customer claim.

    Week 2 expansion adds time/serial/dates/emotion/contact-history fields.
    Pairs with `prompts/extract_v2.md`.
    """

    # --- Core (Week 1) ---
    sku: str | None = Field(
        default=None,
        description="SKU referenced by the customer, if any. Null if not mentioned.",
    )
    failure_mode: FailureMode | None = Field(
        default=None,
        description="The primary mode of failure. Null if not determinable.",
    )
    claim_type: ClaimType | None = Field(
        default=None,
        description="Category of the claim. Null if ambiguous.",
    )
    severity: Severity | None = Field(
        default=None,
        description=(
            "Severity of the damage/issue. cosmetic = surface scratches, "
            "functional = doesn't work as expected, total_loss = unrecoverable."
        ),
    )
    evidence_strength: EvidenceStrength = Field(
        description=(
            "How strong is the customer's evidence in this single message? "
            "strong = photos + receipts + specific details (dates, cycle counts, "
            "SKU); moderate = some details; weak = vague or emotional only."
        ),
    )
    customer_summary: str = Field(
        description="One-sentence neutral summary of the customer's claim.",
        min_length=10,
        max_length=300,
    )

    # --- Week 2 expansion ---
    time_since_purchase_days: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Days between purchase and the claim, if the customer states or "
            "implies it (e.g., 'bought 4 months ago' → 120). Null if not stated."
        ),
    )
    mentioned_serial: str | None = Field(
        default=None,
        description=(
            "Serial number explicitly mentioned by the customer. Null otherwise. "
            "Do NOT invent or extract part numbers — only verbatim serials."
        ),
    )
    mentioned_dates: list[date] = Field(
        default_factory=list,
        description=(
            "All ISO dates the customer mentioned (purchase, failure, contact). "
            "Empty list if none stated."
        ),
    )
    customer_emotion: CustomerEmotion = Field(
        description=(
            "Tone of the message. polite = thanks/please present; calm = neutral "
            "and factual; frustrated = mild complaints, sighs; angry = caps, "
            "threats, demands."
        ),
    )
    prior_contact_attempts: bool = Field(
        description=(
            "True iff the customer mentions a prior contact (email/call/ticket). "
            "Default False — do not assume from context."
        ),
    )


# ---------------------------------------------------------------------------
# Week 3: Decision schema
# ---------------------------------------------------------------------------

DecisionOutcome = Literal["approve", "reject", "needs_info"]
Resolution = Literal["refund", "replacement", "repair", "store_credit", "none"]
Confidence = Literal["high", "medium", "low"]


class Decision(BaseModel):
    """Adjudicator's decision on a claim, grounded in policy.

    Pairs with `prompts/adjudicate_v1.md`. The `policy_citation` MUST be a
    verbatim substring of the policy text — this is verified post-hoc by
    `app.adjudication.citation.verify_citation`. If the cited string isn't
    found, `confidence` is automatically downgraded to `low`.
    """

    outcome: DecisionOutcome = Field(
        description="Final decision on the claim.",
    )
    resolution: Resolution = Field(
        default="none",
        description=(
            "Concrete resolution offered. Use `none` when outcome is `reject` "
            "or `needs_info`. For `approve`, pick from the Section 4 framework."
        ),
    )
    rationale: str = Field(
        description=(
            "2-3 sentence explanation tying the claim's facts to the cited "
            "policy clause. Plain English; no boilerplate."
        ),
        min_length=20,
        max_length=800,
    )
    policy_citation: str = Field(
        description=(
            "Verbatim quote from the policy (15–400 chars) that justifies the "
            "decision. Must appear character-for-character in the policy text."
        ),
        min_length=15,
        max_length=400,
    )
    confidence: Confidence = Field(
        description=(
            "Adjudicator's confidence. Use `high` only when the claim's facts "
            "clearly map to the cited clause; use `low` when material info is "
            "missing or the cited clause is ambiguous for this claim."
        ),
    )
    missing_info_questions: list[str] = Field(
        default_factory=list,
        description=(
            "If outcome is `needs_info`, the specific questions to ask the "
            "customer. Empty list otherwise. Each question is one short "
            "sentence, phrased neutrally."
        ),
    )


# ---------------------------------------------------------------------------
# Week 8: Vision / damage assessment
# ---------------------------------------------------------------------------

DamageType = Literal[
    "cosmetic",          # surface scratches, paint blemishes, sticker peel
    "functional",        # affects use but not structural integrity
    "structural",        # frame / fork / hub / weld integrity affected
    "total_loss",        # unrecoverable; replacement required
    "no_damage_visible", # photo doesn't show any clear damage
    "inconclusive",      # photo too blurry/dark/cropped to judge
]
EvidenceQuality = Literal["clear", "partial", "poor"]


class DamageAssessment(BaseModel):
    """Vision-LLM's structured read of a single claim photo.

    Used by `app/vision/classify.py:analyze_photo`. Pairs with
    `prompts/vision_classify_v1.md`. One assessment per photo — a claim
    with multiple photos produces a list of these.
    """

    damage_type: DamageType = Field(
        description="The primary damage category visible in this photo.",
    )
    severity_score: int = Field(
        ge=1,
        le=10,
        description=(
            "1 = trivial scratch; 5 = noticeable functional impact; "
            "10 = total loss / unrideable."
        ),
    )
    affected_components: list[str] = Field(
        default_factory=list,
        description=(
            "Components visibly damaged: e.g., 'frame down tube', 'rear "
            "derailleur', 'battery casing', 'display'. Empty list if no "
            "damage visible. Be specific — 'wheel' is less useful than "
            "'rear wheel rim'."
        ),
    )
    evidence_quality: EvidenceQuality = Field(
        description=(
            "clear = sharp, well-lit, damage is unambiguous; "
            "partial = damage is visible but angle/lighting limit confidence; "
            "poor = blurry, too dark, or cropped at the relevant area."
        ),
    )
    suspicion_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Visual cues that *might* indicate the photo is staged, taken "
            "from the web, or manipulated. Examples: 'lighting inconsistent "
            "between frame and damage', 'object appears placed not used', "
            "'no environmental context', 'damage edges look painted on'. "
            "These are signals not verdicts — the fraud classifier (Week 9) "
            "is what decides."
        ),
    )
    follow_up_photos_needed: list[str] = Field(
        default_factory=list,
        description=(
            "If evidence_quality is partial or poor, specific additional "
            "shots that would resolve the ambiguity. e.g., 'rear hub "
            "close-up at brighter light', 'battery terminal underside'."
        ),
    )
    reasoning: str = Field(
        description=(
            "One-paragraph explanation of the assessment, citing what's "
            "visible. Concrete observations, not editorial."
        ),
        min_length=20,
        max_length=600,
    )

