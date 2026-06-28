"""Decision↔citation consistency check (grounding step 2).

`app.adjudication.citation.verify_citation` confirms the cited clause is a
real, verbatim substring of the policy. It does NOT confirm the clause
actually *justifies* the decision — a model can cite a real-but-irrelevant,
or even contradictory, clause and still pass the substring check.

This module closes that gap with one narrow specialist call: given the
outcome and the cited clause, classify supports / contradicts / unrelated.
It is the playbook's grounding step 2 ("the value appears in the quote")
applied to a decision rather than a scalar value.

Best-effort: if the verification call fails (after the LLM layer's own
retries), we return None so the pipeline degrades to verbatim-only grounding
rather than blocking every claim during a provider outage.
"""

from __future__ import annotations

from app.ai.llm import chat
from app.ai.prompt_loader import render_prompt
from app.ai.schemas import CitationSupport, Decision
from app.core.logging import get_logger

log = get_logger(__name__)

VERIFY_PROMPT = "verify_citation_v1"


async def verify_decision_support(
    decision: Decision,
    claim_summary: str,
    *,
    judge_alias: str = "judge",
) -> CitationSupport | None:
    """Classify whether `decision.policy_citation` supports `decision.outcome`.

    Returns a `CitationSupport` verdict, or `None` if the verification call
    could not run (best-effort — the caller treats None as "not checked").
    """
    prompt = render_prompt(
        VERIFY_PROMPT,
        outcome=decision.outcome,
        resolution=decision.resolution,
        rationale=decision.rationale,
        citation=decision.policy_citation,
        claim_summary=claim_summary,
    )
    try:
        verdict = await chat(
            messages=[{"role": "user", "content": prompt}],
            model_alias=judge_alias,
            response_model=CitationSupport,
            temperature=0.0,
        )
        return verdict
    except Exception as e:  # noqa: BLE001
        log.warning(
            "citation_support.failed",
            error=str(e)[:120],
            error_type=type(e).__name__,
        )
        return None
