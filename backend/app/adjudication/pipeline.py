"""End-to-end claim processing pipeline (Week 3 + Week 5).

extract → [optionally retrieve] → adjudicate → citation post-validation → draft email.

Two adjudication modes:
- `use_retrieval=False` (Week 3) — `adjudicate_v3` with the entire policy text
  stuffed into the prompt.
- `use_retrieval=True` (Week 5, production default) — `adjudicate_v4` with
  the top-K retrieved policy + safety chunks. Scales to a larger corpus and
  reduces context dilution.

The pipeline is also the function the eval framework runs against the golden
set, so both modes are exposed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.adjudication.citation import CitationResult, verify_citation
from app.adjudication.policy import load_policy
from app.adjudication.retrieve import RetrievedContext, retrieve_for_claim
from app.ai.llm import chat
from app.ai.prompt_loader import render_prompt
from app.ai.schemas import ClaimExtraction, Decision
from app.core.logging import get_logger

log = get_logger(__name__)

EXTRACT_PROMPT = "extract_v2"
# adjudicate_v5 adds the "ambiguous-cause" needs_info trigger
# (electrical+water, structural+impact, range-without-measured-capacity)
# on top of v3's two triggers. Lifted gray-stratum accuracy 12%->65%.
ADJUDICATE_PROMPT_FULL = "adjudicate_v5"
ADJUDICATE_PROMPT_RAG = "adjudicate_v4"
DRAFT_EMAIL_PROMPT = "draft_email_v1"
DEFAULT_POLICY = "policy_v1"

# Production default: whole-policy adjudication (`adjudicate_v3`). Retrieval
# grounding (`adjudicate_v4` + retrieve.py) is a working, tested alternative
# kept available behind this flag — flip to True when corpus growth makes the
# whole policy exceed the context window. As of 2026-05-24, the single 6KB
# policy fits in-prompt and outperforms top-K retrieval by ~7pp accuracy on
# our eval. See DECISIONS.md (2026-05-24 entry).
USE_RETRIEVAL = False

# Week 16: per-claim cost cap. The single-shot pipeline today runs ~$0.001/claim
# (3 LLM calls), so this is a guardrail against runaway retries (an Instructor
# schema-violation chain that burns 10+ calls) or prompt regressions that
# explode tokens. Hit → route to "review" with signal "cost_cap_hit" regardless
# of calibrated_prob. The default is read from settings so ops can tune
# without a code change. ReAct + multi-agent paths have their own per-run cap.
DEFAULT_PIPELINE_COST_CAP_USD = 0.10


@dataclass
class PipelineResult:
    extraction: ClaimExtraction
    decision: Decision
    citation_result: CitationResult
    email: str
    prompt_versions: dict[str, str]
    policy_version: str
    latency_ms: float
    cost_usd: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    retrieved_context: RetrievedContext | None = None
    # Post-Week-14 hardening: calibrated confidence + HITL routing hint
    calibrated_prob: float = 0.5
    route: str = "review"  # auto_resolve / assist / review
    # Week 16: structured reasons attached to the route decision so the
    # operator queue can surface them. e.g. ["cost_cap_hit"], ["injection_signal"]
    route_reasons: list[str] = field(default_factory=list)
    # Phase-4 P1.5: XGBoost fraud P(fraud) ∈ [0,1] when CustomerContext is
    # available (a real customer in the DB, or an eval claim post-load).
    # None when context can't be built — we don't fabricate features.
    fraud_score: float | None = None


async def process_claim(
    raw_input: str,
    photo_descriptions: list[str] | None = None,
    *,
    policy_name: str = DEFAULT_POLICY,
    extractor_alias: str = "extractor",
    reasoner_alias: str = "reasoner",
    extract_temperature: float = 0.0,
    adjudicate_temperature: float = 0.0,
    email_temperature: float = 0.4,
    use_retrieval: bool = USE_RETRIEVAL,
    cost_cap_usd: float = DEFAULT_PIPELINE_COST_CAP_USD,
    # Phase-4 P1.5: identifiers for the fraud lookup. Either is sufficient;
    # claim_id is preferred (used by eval). API endpoint passes email when
    # available. If both are None, fraud scoring is skipped honestly.
    claim_id: str | None = None,
    customer_email: str | None = None,
) -> PipelineResult:
    """Run the full claim-processing chain.

    Steps:
        1. Extract structured claim (`extract_v2`, structured output)
        2. Adjudicate against policy
           - `use_retrieval=False`: full-policy prompt (`adjudicate_v3`)
           - `use_retrieval=True`:  retrieve top-K excerpts, then
             `adjudicate_v4` over those excerpts
        3. Post-validate the citation against source documents
        4. Draft a customer email (`draft_email_v1`, free-form prose)
    """
    photo_descriptions = photo_descriptions or []
    t0 = time.perf_counter()

    # Post-Week-14 hardening: sanitize untrusted customer text before it
    # touches any prompt. Closing prompt tags get replaced with visible
    # markers so an operator can see what was attempted in logs.
    # NOTE: an earlier attempt at structural role-separation (system role
    # for instructions, user role for customer text) regressed accuracy
    # 80% → 52% on the eval because the prompt templates have
    # `<customer_input>{{ customer_text }}</customer_input>` blocks that
    # broke when the value was a placeholder. Escape alone is the
    # load-bearing defense; full role-separation would require a
    # template rewrite which is a bigger lift, deferred to Week 16.
    from app.security.injection import escape_user_input

    safe_raw_input = escape_user_input(raw_input)
    safe_photo_descriptions = [escape_user_input(d) for d in photo_descriptions]

    # ---- Step 1: extract ----
    extract_prompt = render_prompt(
        EXTRACT_PROMPT,
        customer_text=safe_raw_input,
        photo_descriptions=safe_photo_descriptions,
    )
    extraction, extract_cost = await chat(
        messages=[{"role": "user", "content": extract_prompt}],
        model_alias=extractor_alias,
        response_model=ClaimExtraction,
        temperature=extract_temperature,
        return_cost=True,
    )

    # ---- Step 2: adjudicate ----
    policy_text = load_policy(policy_name)

    retrieved_context: RetrievedContext | None = None
    if use_retrieval:
        retrieved_context = await retrieve_for_claim(extraction, safe_raw_input)
        adjudicate_prompt = render_prompt(
            ADJUDICATE_PROMPT_RAG,
            policy_excerpts=retrieved_context.render_excerpts(),
            extraction=extraction,
            customer_text=safe_raw_input,
        )
        adjudicate_prompt_name = ADJUDICATE_PROMPT_RAG
    else:
        adjudicate_prompt = render_prompt(
            ADJUDICATE_PROMPT_FULL,
            policy_text=policy_text,
            extraction=extraction,
            customer_text=safe_raw_input,
            extraction_metadata=None,
        )
        adjudicate_prompt_name = ADJUDICATE_PROMPT_FULL

    decision, adjudicate_cost = await chat(
        messages=[{"role": "user", "content": adjudicate_prompt}],
        model_alias=reasoner_alias,
        response_model=Decision,
        temperature=adjudicate_temperature,
        return_cost=True,
    )

    # ---- Step 3: citation post-validation ----
    # In retrieval mode, validate against the actual source documents the
    # model saw. In full-policy mode, validate against the full policy text.
    # Either way: the model can only legitimately cite what it was shown.
    source_texts: list[str] = [policy_text]
    if retrieved_context is not None:
        seen_paths: set[str] = set()
        for hit in retrieved_context.hits:
            if hit.document_source in seen_paths:
                continue
            seen_paths.add(hit.document_source)
            try:
                source_texts.append(
                    Path(hit.document_source).read_text(encoding="utf-8")
                )
            except OSError as e:
                log.warning(
                    "adjudicate.source_unreadable",
                    path=hit.document_source,
                    error=str(e),
                )
    citation_result = verify_citation(decision.policy_citation, *source_texts)
    if not citation_result.verbatim and decision.confidence != "low":
        log.warning(
            "adjudicate.citation_not_verbatim",
            confidence_was=decision.confidence,
            fuzzy_ratio=citation_result.fuzzy_ratio,
            citation=decision.policy_citation[:120],
        )
        decision = decision.model_copy(update={"confidence": "low"})

    # ---- Step 4: draft customer email ----
    email_prompt = render_prompt(
        DRAFT_EMAIL_PROMPT,
        customer_text=safe_raw_input,
        decision=decision,
    )
    email_resp, email_cost = await chat(
        messages=[{"role": "user", "content": email_prompt}],
        model_alias=reasoner_alias,
        temperature=email_temperature,
        return_cost=True,
    )
    email_text = (
        email_resp["choices"][0]["message"]["content"].strip()
        if isinstance(email_resp, dict)
        else email_resp.choices[0].message.content.strip()
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    total_cost = round(extract_cost + adjudicate_cost + email_cost, 6)
    log.info(
        "pipeline.process_claim",
        outcome=decision.outcome,
        confidence=decision.confidence,
        verbatim=citation_result.verbatim,
        use_retrieval=use_retrieval,
        n_retrieved=len(retrieved_context.hits) if retrieved_context else 0,
        latency_ms=round(latency_ms, 1),
        cost_usd=total_cost,
    )

    # Phase-4 P1.5: XGBoost fraud score. Looked up by claim_id (preferred,
    # used by the eval) or customer_email (API path). If neither is provided
    # OR the customer isn't in the DB, we skip honestly (fraud_score=None)
    # rather than hallucinate features.
    fraud_score: float | None = None
    if claim_id is not None or customer_email is not None:
        try:
            from app.db.session import get_sessionmaker
            from app.fraud.db_adapter import (
                build_context_by_claim_id,
                build_context_by_email,
            )
            from app.fraud.score import score_claim

            sm = get_sessionmaker()
            async with sm() as session:
                if claim_id is not None:
                    ctx = await build_context_by_claim_id(session, claim_id)
                else:
                    ctx = await build_context_by_email(session, customer_email or "")
            if ctx is not None:
                # Build a minimal claim dict — the scorer uses claim_id +
                # claim_date + raw_input + shipping_address. We have id+text;
                # claim_date defaults to today (acceptable, the date is mainly
                # used for window-features that compare against prior claims).
                claim_for_scorer = {
                    "claim_id": claim_id or "",
                    "claim_date": None,  # scorer defaults to date.today()
                    "raw_input": raw_input,
                    "shipping_address": None,
                    "claim_value_usd": None,
                    "days_since_purchase": extraction.time_since_purchase_days,
                }
                fraud_result = await score_claim(
                    claim_for_scorer,
                    ctx,
                    extraction=extraction.model_dump(),
                    include_llm_judge=False,  # save the LLM call in the
                    # single-shot path; the FraudAuditor specialist in the
                    # multi-agent graph keeps the narrative for its own use.
                )
                fraud_score = fraud_result.calibrated_prob
        except Exception as e:  # noqa: BLE001
            log.warning(
                "pipeline.fraud_score_failed",
                error=str(e)[:120],
                error_type=type(e).__name__,
            )

    # Post-Week-14: calibrate the LLM's self-reported confidence into a
    # real probability that the decision is correct. The route hint is
    # what Week-15's HITL router consumes.
    from app.confidence.calibrator import calibrate

    cal = calibrate(
        decision=decision,
        extraction=extraction,
        citation_verbatim=citation_result.verbatim,
        citation_fuzzy=citation_result.fuzzy_ratio,
        fraud_score=fraud_score,
    )

    # Week 16: post-hoc cost cap + injection-signal routing override. Both
    # demote auto_resolve to "review" so a human sees the claim.
    route = cal.route
    route_reasons: list[str] = []
    if total_cost > cost_cap_usd:
        log.warning(
            "pipeline.cost_cap_hit",
            spent=total_cost,
            cap=cost_cap_usd,
            outcome=decision.outcome,
        )
        route = "review"
        route_reasons.append("cost_cap_hit")

    from app.security.injection import detect_injection_signals

    injection_signals = detect_injection_signals(raw_input)
    if injection_signals and route == "auto_resolve":
        log.info(
            "pipeline.injection_signal_demotes_route",
            signals=injection_signals,
        )
        route = "review"
        route_reasons.append("injection_signal")

    # Phase-4 P1.5: fraud-score routing override. Even if the LLM correctly
    # decided this claim, a high fraud probability means an operator should
    # see it before any customer-facing action lands. 0.70 matches the
    # XGBoost calibration band where observed fraud-rate exceeds 50% (per
    # Week-9 model training). Tightening to 0.85 cuts noise but loses the
    # marginal positives; loosening to 0.50 catches more but reviews more.
    if fraud_score is not None and fraud_score >= 0.70 and route == "auto_resolve":
        log.info(
            "pipeline.fraud_score_demotes_route",
            fraud_score=fraud_score,
            outcome=decision.outcome,
        )
        route = "review"
        route_reasons.append("fraud_score_high")

    # Phase-4 P4.2 safety override — applied LAST so it shadows every other
    # routing decision. The XGBoost calibrator (v3) is trained on synthetic
    # data; the 2026-06-06 real-data mini-eval showed it auto-resolves real
    # claims with median confidence 0.994 but is right only 1/8 of the time
    # on operator-labeled rows. Until we retrain on real labels, any
    # production deployment MUST set `DISABLE_AUTO_RESOLVE=true` so every
    # claim goes to operator review. See DECISIONS.md 2026-06-06.
    from app.core.config import get_settings as _get_settings  # local import; avoid cycles at module load
    if _get_settings().disable_auto_resolve and route == "auto_resolve":
        log.warning(
            "pipeline.auto_resolve_disabled",
            outcome=decision.outcome,
            calibrated_prob=cal.calibrated_prob,
            note="DISABLE_AUTO_RESOLVE flag set; demoting to review pending real-data calibrator retrain",
        )
        route = "review"
        route_reasons.append("auto_resolve_disabled_synthetic_calibrator")

    return PipelineResult(
        extraction=extraction,
        decision=decision,
        citation_result=citation_result,
        email=email_text,
        prompt_versions={
            "extract": EXTRACT_PROMPT,
            "adjudicate": adjudicate_prompt_name,
            "draft_email": DRAFT_EMAIL_PROMPT,
        },
        policy_version=policy_name,
        latency_ms=latency_ms,
        cost_usd=total_cost,
        cost_breakdown={
            "extract": round(extract_cost, 6),
            "adjudicate": round(adjudicate_cost, 6),
            "draft_email": round(email_cost, 6),
        },
        retrieved_context=retrieved_context,
        calibrated_prob=cal.calibrated_prob,
        route=route,
        route_reasons=route_reasons,
        fraud_score=fraud_score,
    )
