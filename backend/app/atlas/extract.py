"""policy text → CoverageProfile (the normalization engine).

Turns one plan's real T&C text into the canonical `CoverageProfile`, with
every field grounded in a verbatim clause. This is the reusable IP behind the
Coverage Atlas: run it over any device-protection plan and you get a
comparable, clause-cited profile — and the same policy text then feeds the
adjudication pipeline.

Grounding is verified separately (see `verify_profile_grounding`): the model
*locates and structures*, then code checks each cited clause actually appears
in the source text. A profile's quality is measured by how many of its cells
are grounded, not by how confident the model sounds.
"""

from __future__ import annotations

from app.adjudication.citation import verify_citation
from app.ai.llm import chat
from app.ai.prompt_loader import render_prompt
from app.atlas.schema import CoverageItem, CoverageProfile
from app.core.logging import get_logger

log = get_logger(__name__)

EXTRACT_PROMPT = "extract_coverage_v1"


async def extract_coverage_profile(
    plan_name: str,
    source: str,
    policy_text: str,
    *,
    model_alias: str = "reasoner",
) -> CoverageProfile:
    """Normalize one plan's T&C text into a grounded CoverageProfile.

    Uses the `reasoner` alias so the hard legal-normalization runs on the
    frontier tier when one is configured (falls back to the chat model).
    """
    prompt = render_prompt(
        EXTRACT_PROMPT,
        plan_name=plan_name,
        source=source,
        policy_text=policy_text,
    )
    profile = await chat(
        messages=[{"role": "user", "content": prompt}],
        model_alias=model_alias,
        response_model=CoverageProfile,
        temperature=0.0,
    )
    # The model may not echo these reliably; pin them from the caller.
    profile.plan_name = plan_name
    profile.source = source
    return profile


def verify_profile_grounding(
    profile: CoverageProfile, policy_text: str
) -> tuple[int, int, list[str]]:
    """Check every non-empty clause is a verbatim substring of the source.

    Returns (grounded, total, ungrounded_field_names). This is the integrity
    metric for the atlas — a normalized fact with a clause that isn't actually
    in the policy is a hallucination and shouldn't be trusted.
    """
    grounded = 0
    total = 0
    ungrounded: list[str] = []

    def _check(name: str, item: CoverageItem) -> None:
        nonlocal grounded, total
        if not item.clause:
            return
        total += 1
        if verify_citation(item.clause, policy_text).verbatim:
            grounded += 1
        else:
            ungrounded.append(name)

    for name, item in profile.peril_items().items():
        _check(name, item)
    for name in ("term_length", "deductible_or_fee", "claim_limit", "transferable"):
        _check(name, getattr(profile, name))
    for i, item in enumerate(profile.exclusions):
        _check(f"exclusion[{i}]", item)
    for i, item in enumerate(profile.evidence_required):
        _check(f"evidence[{i}]", item)

    return grounded, total, ungrounded
