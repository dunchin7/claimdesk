"""LLM-as-judge scorers (Week 3+).

These judges grade open-ended text the auto-eval can't compare structurally.
Each judge MUST be hand-calibrated against ≥10 human-scored examples before
its scores are trusted in regression metrics.

Calibration workflow:
1. Sample 10 random examples from a recent eval run.
2. Score them by hand on the 1-5 scale.
3. Run the judge on the same 10.
4. If |judge - human| ≤ 1 on ≥8/10, the judge is good enough. Otherwise
   tighten the prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.ai.llm import chat


class EmailQualityScore(BaseModel):
    """1–5 scoring of a customer-facing email."""

    tone: int = Field(ge=1, le=5, description="Warmth/professionalism (1=cold, 5=excellent)")
    clarity: int = Field(ge=1, le=5, description="Easy to understand (1=confusing, 5=crisp)")
    completeness: int = Field(
        ge=1, le=5, description="Covers decision + reason + next steps (1=incomplete, 5=full)"
    )
    notes: str = Field(default="", description="One-line judge comment if score < 5")

    @property
    def average(self) -> float:
        return round((self.tone + self.clarity + self.completeness) / 3, 2)


_EMAIL_JUDGE_PROMPT = """\
You are a senior customer-experience reviewer at PaceLine Cycles. Grade the
following customer-facing email on three axes, each 1-5.

- **tone** (1=cold/robotic, 3=neutral, 5=warm and professional)
- **clarity** (1=confusing or contradictory, 3=understandable, 5=crisp and unambiguous)
- **completeness** (1=missing decision OR reason OR next steps, 3=covers most, 5=fully covers all three)

Be strict. A 5 means there is nothing you would change. A 3 is acceptable but unremarkable. Only use 1 for genuine failures.

Decision context:
- outcome: {outcome}
- resolution: {resolution}

Email:
\"\"\"
{email}
\"\"\"

Output JSON only.
"""


async def judge_email(email: str, outcome: str, resolution: str) -> EmailQualityScore:
    """Score an email on tone/clarity/completeness via LLM-judge."""
    prompt = _EMAIL_JUDGE_PROMPT.format(
        email=email, outcome=outcome, resolution=resolution
    )
    return await chat(
        messages=[{"role": "user", "content": prompt}],
        model_alias="judge",
        response_model=EmailQualityScore,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Decision faithfulness judge
# ---------------------------------------------------------------------------

FaithfulnessVerdict = Literal["faithful", "unfaithful", "partial"]


class FaithfulnessScore(BaseModel):
    verdict: FaithfulnessVerdict = Field(
        description=(
            "faithful = rationale is fully supported by claim+policy facts; "
            "partial = mostly supported with one minor invented claim; "
            "unfaithful = invents facts or contradicts the inputs."
        )
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific invented or contradicted facts. Empty if faithful.",
    )


_FAITHFULNESS_PROMPT = """\
You are checking whether a warranty adjudication's rationale is faithful to
the source material. Faithful means every factual statement in the rationale
is supported by either the claim text or the policy text. Inventing facts
(e.g., a date the customer didn't say) is unfaithful.

<claim_text>
{claim_text}
</claim_text>

<policy_citation>
{policy_citation}
</policy_citation>

<rationale>
{rationale}
</rationale>

Return JSON only.
"""


async def judge_faithfulness(
    claim_text: str, policy_citation: str, rationale: str
) -> FaithfulnessScore:
    prompt = _FAITHFULNESS_PROMPT.format(
        claim_text=claim_text,
        policy_citation=policy_citation,
        rationale=rationale,
    )
    return await chat(
        messages=[{"role": "user", "content": prompt}],
        model_alias="judge",
        response_model=FaithfulnessScore,
        temperature=0.0,
    )
