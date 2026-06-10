"""Multi-agent LangGraph (Week 12).

Architecture:

    START → Orchestrator
             ├─→ Investigator (customer history)        ┐
             ├─→ PolicyInterpreter (policy excerpts)    │ parallel
             ├─→ VisionAnalyst (optional, photos only)  │
             └─→ FraudAuditor (XGBoost + LLM)           ┘
                           │ merge
                           v
                    ContextSynthesizer
                           v
                      Adjudicator
                           v
                        Critic ─┐
                           │    │ (1 replan loop allowed)
                           ├────┘ if disagree
                           │ if agree
                           v
                     Communicator → END

Specialists are functions, not agents — each node is `(state) -> partial
state update`. The graph's job is fan-out, merge, and conditional routing.

Why LangGraph over hand-rolled (different choice than Week 11):
- Parallel fan-out is built in (multiple edges from one node).
- Checkpointing/resumption (Week 15 HITL pause/resume requires it).
- Conditional edges via a router function are clean to express.

State is a TypedDict; LangGraph merges per-field. We mark `cost_usd_accum`
and `step_log` as accumulating fields via `Annotated[..., operator.add]`
so the parallel branches don't clobber each other on merge.
"""

from __future__ import annotations

import operator
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
# AsyncPostgresSaver is the Week-15 durable checkpointer. Imported lazily
# inside `get_default_checkpointer()` because module-import-time it tries
# to resolve `psycopg.AsyncConnection`, which is fine but heavyweight.
from langgraph.graph import END, START, StateGraph

from app.adjudication.citation import verify_citation
from app.adjudication.policy import load_policy
from app.ai.llm import chat
from app.ai.prompt_loader import render_prompt
from app.ai.schemas import Decision
from app.ai.tools.actions import _draft_email, DraftEmailInput
from app.ai.tools.retrieval import RetrievePolicyInput, _retrieve_policy
from app.ai.tools.shopify import (
    LookupCustomerHistoryInput,
    _lookup_customer_history,
)
from app.ai.tools.vision_tool import AnalyzePhotoToolInput, _analyze_photo_tool
from app.core.logging import get_logger

log = get_logger(__name__)

MULTI_AGENT_VERSION = "multi_v3"  # Week 14: + Planner + replan-on-error
DEFAULT_CRITIC_MAX_LOOPS = 1
DEFAULT_REPLAN_MAX_LOOPS = 1  # Week 14

# Week-13 toggle. Set to False as the production default — the A/B on
# 2026-06-02 showed memory injection nets to 0pp overall accuracy lift
# (helps gray cases by +25pp, hurts fraud_suspect by -50pp because per-SKU
# summaries average out per-customer fraud signal). Infrastructure stays
# in tree; flip on when (sku, failure_mode) granularity lands or real
# adjudication history is available. See DECISIONS.md 2026-06-02.
USE_MEMORY = False

# Week-14 toggle. Set to False as the production default — the A/B on
# 2026-06-02 showed Planner-driven dispatch regressed accuracy from 70%
# to 65% (Planner too aggressive about skipping specialists, leaving the
# Adjudicator with less signal). Critic-correction rate jumped to 65%
# (target 5-15%) — the textbook "Adjudicator is bad" signal from the
# design doc. Infrastructure stays in tree; flip on when planning logic
# is sharper or real claims show clear routing opportunities. See
# DECISIONS.md 2026-06-02.
USE_PLANNER = False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class StepEntry(TypedDict, total=False):
    node: str
    elapsed_ms: float
    cost_usd: float
    summary: str


class ClaimState(TypedDict, total=False):
    # --- inputs (set at startup, read by all nodes) ---
    claim_id: str
    customer_id: str | None
    raw_input: str
    photo_urls: list[str]

    # --- inputs (continued) ---
    use_memory: bool  # Week-13 A/B toggle
    use_planner: bool  # Week-14 A/B toggle

    # --- planning + replan (Week 14) ---
    plan: dict[str, Any] | None
    replan_iterations: int
    replan_reason: str | None

    # --- specialist outputs (parallel fan-out branches) ---
    # Each specialist writes to a distinct field; LangGraph's per-field merge
    # handles parallel writes without conflict.
    customer_summary: dict[str, Any] | None
    policy_excerpts: list[dict[str, Any]] | None
    vision_assessment: dict[str, Any] | None
    fraud_score: dict[str, Any] | None
    memory_patterns: list[dict[str, Any]] | None  # Week 13

    # --- downstream outputs (sequential after merge) ---
    synthesized_context: str | None
    draft_decision: dict[str, Any] | None
    critic_verdict: dict[str, Any] | None
    critic_iterations: int
    final_decision: dict[str, Any] | None
    email: str | None
    escalated: bool

    # --- accumulators (annotated with operator.add for parallel safety) ---
    cost_usd_accum: Annotated[float, operator.add]
    step_log: Annotated[list[StepEntry], operator.add]

    # --- meta ---
    trace_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track(node: str):
    """Decorator that times the node and stamps a step_log entry."""
    def deco(fn):
        async def wrapper(state: ClaimState) -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                update = await fn(state)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "multi.node_failed",
                    node=node,
                    trace_id=state.get("trace_id"),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                return {
                    "step_log": [{
                        "node": node,
                        "elapsed_ms": round(elapsed, 1),
                        "cost_usd": 0.0,
                        "summary": f"ERROR: {type(e).__name__}: {e}",
                    }],
                }
            elapsed = (time.perf_counter() - t0) * 1000
            # Pull cost off the update if present; specialists set it explicitly.
            cost = float(update.get("cost_usd_accum", 0.0))
            summary = update.pop("_summary", "")
            step: StepEntry = {
                "node": node,
                "elapsed_ms": round(elapsed, 1),
                "cost_usd": round(cost, 6),
                "summary": summary or "(ok)",
            }
            existing_log = update.get("step_log") or []
            update["step_log"] = [*existing_log, step]
            return update
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@_track("Orchestrator")
async def orchestrator(state: ClaimState) -> dict[str, Any]:
    """Entry node. Classifies the claim and decides which specialists to dispatch.

    Today this is light — it just records the trace ID and notes whether
    photos are present. Future iterations could do an early triage LLM
    call to skip specialists for obvious cases (e.g., empty input → escalate).
    """
    trace_id = state.get("trace_id") or str(uuid.uuid4())
    photos = state.get("photo_urls") or []
    return {
        "trace_id": trace_id,
        "critic_iterations": 0,
        "replan_iterations": 0,
        "escalated": False,
        "_summary": f"trace={trace_id[:8]} photos={len(photos)}",
    }


_PLANNER_PROMPT = """\
You are the Planner in a multi-specialist warranty system. Your job is
to **decide which specialists to dispatch** before they all run, based
on the claim's text. Goal: skip specialists that won't add signal, run
the ones that will. The Adjudicator reads this plan as context.

## Specialists you can dispatch

- **Investigator** — pulls the customer's prior-claim history. Useful for
  any claim from a returning customer or any suspected fraud case.
- **PolicyInterpreter** — retrieves relevant policy + safety excerpts.
  **Always include this** — every claim needs policy grounding.
- **VisionAnalyst** — analyzes attached photos. Only useful if photos
  are present.
- **FraudAuditor** — runs an XGBoost fraud classifier. Only useful for
  claims that show ANY fraud signal (repeat-claim language, address
  inconsistency, unusually high value, very new customer).
- **MemoryConsultant** — retrieves historical adjudication patterns for
  the SKU. Useful for ambiguous cases; can dilute fraud signals so skip
  if fraud_concern is medium/high.

## Routing heuristics

- `clear_approve` claims (clear defect within window, strong evidence)
  → Investigator + PolicyInterpreter. Skip FraudAuditor, MemoryConsultant.
- `clear_reject` claims (wear item, accidental damage without EDP, buyer's
  remorse past 14 days) → PolicyInterpreter only. Cite the exclusion.
- Vague messages with no signal → PolicyInterpreter only; will be needs_info.
- Suspected fraud → Investigator + PolicyInterpreter + FraudAuditor. Skip
  MemoryConsultant (averages out the fraud signal).
- Photos attached → VisionAnalyst always.

## Output format

You'll produce a Plan with: `specialists_to_run` (list, always includes
PolicyInterpreter), `complexity` (clear / moderate / edge_case),
`fraud_concern` (none / low / medium / high), and one paragraph of
reasoning.

{% if replan_reason %}
## Replan context

The previous plan ran but a specialist returned an error. You are being
asked to replan. Original reason: {{ replan_reason }}. Consider whether
to skip the failing specialist or substitute another approach.
{% endif %}

<customer_input>
{{ customer_text }}
</customer_input>

{% if photo_urls %}
Photos attached: {{ photo_urls | length }} URL(s).
{% else %}
No photos attached.
{% endif %}

{% if customer_id %}
customer_id provided: yes (Investigator + FraudAuditor can return real data)
{% else %}
customer_id provided: no (Investigator + FraudAuditor will no-op)
{% endif %}

Produce your Plan now.
"""


@_track("Planner")
async def planner(state: ClaimState) -> dict[str, Any]:
    """Decide which specialists to dispatch for this specific claim.

    Replaces the "always run all four" behavior with a model-driven choice.
    On replan (specialist error), incorporates the error feedback.
    """
    if not state.get("use_planner", USE_PLANNER):
        # Planner disabled: default to "all specialists" so the rest of the
        # graph still works with the planner-driven router.
        default_plan = {
            "specialists_to_run": [
                "Investigator", "PolicyInterpreter", "FraudAuditor"
            ] + (["VisionAnalyst"] if state.get("photo_urls") else []),
            "complexity": "moderate",
            "fraud_concern": "low",
            "reasoning": "(planner disabled — default specialist set)",
        }
        return {"plan": default_plan, "_summary": "planner disabled"}

    from jinja2 import Template
    prompt = Template(_PLANNER_PROMPT).render(
        customer_text=state["raw_input"],
        photo_urls=state.get("photo_urls") or [],
        customer_id=state.get("customer_id"),
        replan_reason=state.get("replan_reason"),
    )
    plan, cost = await chat(
        messages=[{"role": "user", "content": prompt}],
        model_alias="reasoner",
        response_model=_PlanModel,
        temperature=0.0,
        return_cost=True,
        metadata={"trace_id": state.get("trace_id"), "node": "Planner"},
    )
    # Force PolicyInterpreter inclusion — defensive guardrail.
    specialists = list(plan.specialists_to_run)
    if "PolicyInterpreter" not in specialists:
        specialists.append("PolicyInterpreter")
    # VisionAnalyst gated on photo presence regardless of plan
    if "VisionAnalyst" in specialists and not state.get("photo_urls"):
        specialists.remove("VisionAnalyst")
    plan_dict = {
        "specialists_to_run": specialists,
        "complexity": plan.complexity,
        "fraud_concern": plan.fraud_concern,
        "reasoning": plan.reasoning,
    }
    return {
        "plan": plan_dict,
        "cost_usd_accum": cost,
        # If we're being re-entered for a replan, bump the counter
        "replan_iterations": state.get("replan_iterations", 0) + (
            1 if state.get("replan_reason") else 0
        ),
        "_summary": f"specialists={specialists} complexity={plan.complexity} fraud={plan.fraud_concern}",
    }


@_track("Investigator")
async def investigator(state: ClaimState) -> dict[str, Any]:
    """Pulls the customer's prior-claim summary via the Week-11 tool.

    Returns no-op when no customer_id is provided — common for eval flows
    where the claim isn't tied to a DB customer.
    """
    cid = state.get("customer_id")
    if not cid:
        return {
            "customer_summary": None,
            "_summary": "no customer_id provided",
        }
    out = await _lookup_customer_history(LookupCustomerHistoryInput(customer_id=cid))
    summary_text = (
        f"prior_claims={out.n_prior_claims} "
        f"approved={out.approved_count} rejected={out.rejected_count} "
        f"distinct_addrs={out.distinct_shipping_addresses}"
    )
    return {
        "customer_summary": out.model_dump(mode="json"),
        "_summary": summary_text,
    }


@_track("PolicyInterpreter")
async def policy_interpreter(state: ClaimState) -> dict[str, Any]:
    """Retrieves policy + safety excerpts for the claim.

    Uses the customer's raw_input as the query directly. A more sophisticated
    version would extract first then craft a focused query, but our
    structural_800 chunker + ada-002 handles natural-language queries fine.
    """
    raw = state["raw_input"]
    out = await _retrieve_policy(RetrievePolicyInput(query=raw[:500], top_k=6))
    return {
        "policy_excerpts": [h.model_dump(mode="json") for h in out.hits],
        "_summary": f"retrieved {out.n_hits} policy/safety chunks",
    }


@_track("VisionAnalyst")
async def vision_analyst(state: ClaimState) -> dict[str, Any]:
    """Runs the Week-8 vision pipeline on each attached photo.

    Aggregates damage severity across photos by taking the maximum (a
    structural crack in any photo dominates). Skipped entirely when
    photo_urls is empty.
    """
    urls = state.get("photo_urls") or []
    if not urls:
        return {
            "vision_assessment": None,
            "_summary": "no photos to analyze",
        }
    assessments: list[dict[str, Any]] = []
    total_cost = 0.0
    for url in urls[:3]:  # cap at 3 to bound cost
        out = await _analyze_photo_tool(AnalyzePhotoToolInput(image_url=url))
        assessments.append(out.model_dump(mode="json"))
    # Aggregate
    if not assessments:
        return {"vision_assessment": None, "_summary": "vision returned no results"}
    max_severity = max((a.get("severity_score") or 0) for a in assessments)
    max_ai_likelihood = max((a.get("ai_generated_likelihood") or 0.0) for a in assessments)
    damage_types = sorted({a.get("damage_type", "") for a in assessments if a.get("damage_type")})
    aggregate = {
        "n_photos": len(assessments),
        "max_severity": max_severity,
        "max_ai_likelihood": max_ai_likelihood,
        "damage_types": damage_types,
        "per_photo": assessments,
    }
    return {
        "vision_assessment": aggregate,
        "cost_usd_accum": total_cost,
        "_summary": f"{len(assessments)} photos, max_severity={max_severity}, max_ai={max_ai_likelihood:.2f}",
    }


@_track("FraudAuditor")
async def fraud_auditor(state: ClaimState) -> dict[str, Any]:
    """Runs the Week-9 XGBoost fraud classifier when context is available.

    Requires the customer to exist in the training jsonl (so we have a
    CustomerContext to feed the feature extractor). For eval claims whose
    customer_id isn't in the training set, returns a no-signal result.
    The agent can still reason about fraud from text cues via the
    Adjudicator's prompt — it's just not getting an ML score.
    """
    cid = state.get("customer_id")
    if not cid:
        return {
            "fraud_score": None,
            "_summary": "no customer_id; skipping XGBoost",
        }

    # Try to build a CustomerContext from the training jsonl
    import json as _json
    from app.fraud.features import build_customer_contexts

    training_path = Path(__file__).resolve().parents[4] / "data/synthetic/claims_training.jsonl"
    if not training_path.is_file():
        return {
            "fraud_score": None,
            "_summary": "training jsonl unavailable; XGBoost skipped",
        }
    rows = [
        _json.loads(line) for line in training_path.read_text().splitlines() if line.strip()
    ]
    ctxs = build_customer_contexts(rows)
    if cid not in ctxs:
        return {
            "fraud_score": None,
            "_summary": "customer not in training population; XGBoost skipped",
        }

    try:
        from app.fraud.score import score_claim

        # Synthesize a claim row from state for the scorer
        synth_claim = {
            "claim_id": state.get("claim_id") or str(uuid.uuid4()),
            "customer_id": cid,
            "claim_date": datetime.now().date().isoformat(),
            "shipping_address": ctxs[cid].primary_address or "",
            "claim_value_usd": 1000.0,  # placeholder
            "expected_decision": "approve",  # placeholder; not used at scoring time
            "raw_input": state["raw_input"],
            "days_since_purchase": 90,  # placeholder
        }
        score = await score_claim(synth_claim, ctxs[cid], include_llm_judge=False)
        return {
            "fraud_score": {
                "calibrated_prob": score.calibrated_prob,
                "xgboost_score": score.xgboost_score,
                "top_3_features": score.top_3_features,
            },
            "_summary": f"prob={score.calibrated_prob:.2f} top={score.top_3_features[0]['name'] if score.top_3_features else 'n/a'}",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "fraud_score": None,
            "_summary": f"XGBoost error: {type(e).__name__}",
        }


@_track("MemoryConsultant")
async def memory_consultant(state: ClaimState) -> dict[str, Any]:
    """Pull historical adjudication patterns relevant to this claim (Week 13).

    Strategy: try exact SKU match first; if no SKU or no exact pattern,
    fall back to embedding similarity over `summary_embedding`. Returns
    `None` when no signal — ContextSynthesizer omits the block in that case
    so the Adjudicator sees no false context.
    """
    if not state.get("use_memory", USE_MEMORY):
        return {"memory_patterns": None, "_summary": "memory disabled (A/B)"}

    from app.memory.patterns import retrieve_patterns

    # Try to extract SKU from raw_input via a cheap regex — synthetic claims
    # often include "EB-LEVEL-3" etc. inline. A more sophisticated impl
    # would call the extractor first, but the regex catches our SKU shape.
    import re
    raw = state.get("raw_input", "")
    m = re.search(r"\b(EB-[A-Z]+-[A-Z0-9]+|BAT-[A-Z]+(?:-[A-Z])?|CHG-[A-Z]+|ACC-[A-Z]+)\b", raw)
    sku = m.group(1) if m else None

    hits = await retrieve_patterns(sku=sku, query_text=raw, top_k=3)
    if not hits:
        return {"memory_patterns": None, "_summary": "no patterns found"}

    summary_str = (
        f"sku={sku}; exact-match" if sku and hits[0].similarity is None
        else f"top-{len(hits)} by embedding"
    )
    return {
        "memory_patterns": [h.to_dict() for h in hits],
        "_summary": summary_str,
    }


@_track("ContextSynthesizer")
async def context_synthesizer(state: ClaimState) -> dict[str, Any]:
    """Merge the parallel specialist outputs into a single context blob.

    The Adjudicator gets this as a structured prompt section — much smaller
    than re-feeding all the raw specialist outputs.
    """
    parts: list[str] = []

    cust = state.get("customer_summary")
    if cust:
        parts.append(
            f"## Customer history\n"
            f"- prior_claims: {cust.get('n_prior_claims', 0)}\n"
            f"- approved: {cust.get('approved_count', 0)} / "
            f"rejected: {cust.get('rejected_count', 0)}\n"
            f"- distinct_shipping_addresses: {cust.get('distinct_shipping_addresses', 0)}\n"
            f"- first_seen: {cust.get('first_seen_date', 'unknown')}"
        )

    excerpts = state.get("policy_excerpts") or []
    if excerpts:
        chunks_block = "\n\n".join(
            f"<excerpt source=\"{e.get('source','')}\" section=\"{e.get('section','')}\">\n"
            f"{e.get('text','')}\n</excerpt>"
            for e in excerpts
        )
        parts.append(f"## Policy excerpts (retrieved)\n{chunks_block}")

    vision = state.get("vision_assessment")
    if vision:
        parts.append(
            f"## Vision\n"
            f"- max_severity: {vision.get('max_severity')}\n"
            f"- damage_types: {vision.get('damage_types')}\n"
            f"- max_ai_likelihood: {vision.get('max_ai_likelihood'):.2f}"
        )

    # Week 13: historical patterns from the memory store
    patterns = state.get("memory_patterns")
    if patterns:
        from app.memory.patterns import PatternHit, render_patterns_block
        hits = [PatternHit(**p) for p in patterns]
        block = render_patterns_block(hits)
        if block:
            parts.append(block)

    fraud = state.get("fraud_score")
    if fraud:
        top = fraud.get("top_3_features") or []
        top_brief = ", ".join(f"{f['name']}={f.get('value')}" for f in top[:3])
        parts.append(
            f"## Fraud signals (XGBoost)\n"
            f"- calibrated_prob: {fraud.get('calibrated_prob'):.2f}\n"
            f"- top features: {top_brief}"
        )

    synthesized = "\n\n".join(parts) if parts else "(no specialist outputs)"
    return {
        "synthesized_context": synthesized,
        "_summary": f"merged {len(parts)} specialist outputs",
    }


_ADJUDICATE_PROMPT = """\
You are the Adjudicator agent in a multi-specialist warranty system. The
specialists have done their work; you make the decision. Use ONLY the
information in the synthesized context — don't invent facts.

## Decision rules (the same heuristics react_v2 uses)

- Battery "stopped working" / "won't charge" / "died" → §1.1 defect, no BMS
  needed, approve+replacement if within 12mo
- Battery "shorter range" / "capacity dropped" → §1.3 capacity, requires BMS
  export
- Wear items (tires, grips, chains, etc.) → reject, cite §1.5/§2.2
- Accidental damage without Extended Damage Protection → reject, §2.1
- Buyer's remorse >14d or used → reject, §2.5
- Shipping damage <7d with photos → approve+replacement, §3.2/§4.1
- Fraud signals (multi addresses, repeat claims, calibrated_prob ≥ 0.5)
  → reject + escalate, cite §5.3
- Truly vague (no SKU, no failure, no evidence) → needs_info, §3.1

If the previous critic_verdict is present, address its concerns explicitly.

<synthesized_context>
{{ context }}
</synthesized_context>

<customer_input>
{{ customer_text }}
</customer_input>

{% if critic_feedback %}
<critic_feedback>
{{ critic_feedback }}
</critic_feedback>
{% endif %}

Produce a final Decision. policy_citation MUST be a verbatim substring of
one of the policy excerpts.
"""


@_track("Adjudicator")
async def adjudicator(state: ClaimState) -> dict[str, Any]:
    """Final-decision LLM call, structured output."""
    from jinja2 import Template
    critic = state.get("critic_verdict") or {}
    critic_feedback = (
        f"verdict={critic.get('verdict')}; concerns={critic.get('concerns')}"
        if critic else ""
    )
    prompt = Template(_ADJUDICATE_PROMPT).render(
        context=state.get("synthesized_context") or "(empty)",
        customer_text=state["raw_input"],
        critic_feedback=critic_feedback,
    )
    decision, cost = await chat(
        messages=[{"role": "user", "content": prompt}],
        model_alias="reasoner",
        response_model=Decision,
        temperature=0.0,
        return_cost=True,
        metadata={"trace_id": state.get("trace_id"), "node": "Adjudicator"},
    )

    # Verify the citation against the active policy
    try:
        policy_text = load_policy("policy_v1")
        cit = verify_citation(decision.policy_citation, policy_text)
        if not cit.verbatim and decision.confidence != "low":
            decision = decision.model_copy(update={"confidence": "low"})
    except Exception:  # noqa: BLE001
        pass

    return {
        "draft_decision": decision.model_dump(mode="json"),
        "cost_usd_accum": cost,
        "_summary": f"outcome={decision.outcome} resolution={decision.resolution} conf={decision.confidence}",
    }


class CriticVerdict(TypedDict):
    verdict: Literal["agree", "disagree"]
    concerns: list[str]
    confidence: Literal["high", "medium", "low"]


from pydantic import BaseModel, Field

SpecialistName = Literal[
    "Investigator", "PolicyInterpreter", "VisionAnalyst", "FraudAuditor", "MemoryConsultant"
]


class _PlanModel(BaseModel):
    """Planner output. Structured choice of which specialists to dispatch.

    Constrained by Literal enum on `specialists_to_run` so the model can't
    invent specialist names — Instructor + structured outputs enforce this.
    """

    specialists_to_run: list[SpecialistName] = Field(
        description=(
            "Which specialists should fan out for this claim. Always include "
            "PolicyInterpreter; include others when the claim's signals warrant."
        ),
        min_length=1,
    )
    complexity: Literal["clear", "moderate", "edge_case"] = Field(
        description=(
            "`clear` = obvious approve/reject; `moderate` = needs evidence "
            "synthesis; `edge_case` = ambiguous, will likely need critic loop."
        )
    )
    fraud_concern: Literal["none", "low", "medium", "high"] = Field(
        description=(
            "Initial fraud read from the customer message. high → "
            "FraudAuditor must run AND the Adjudicator should weight its "
            "output heavily."
        )
    )
    reasoning: str = Field(
        min_length=20,
        max_length=400,
        description=(
            "One paragraph explaining the routing choice. Helps the "
            "Adjudicator understand WHY specialists were/weren't selected."
        ),
    )


class _CriticVerdictModel(BaseModel):
    verdict: Literal["agree", "disagree"]
    concerns: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


CRITIC_PROMPT_VERSION = "critic_v2"

_CRITIC_PROMPT = """\
You are the Critic in a multi-specialist warranty system. **Your default is
to AGREE.** The Adjudicator has already done the hard work; your job is
ONLY to catch clear, specific, defensible errors — not to second-guess
judgment calls.

Return `verdict="disagree"` ONLY if one of these specific conditions holds:

1. **The cited clause is unrelated to the claim's facts.** e.g., decision
   cites §1.3 (battery capacity) but the customer reported "battery won't
   charge" (which is §1.1 defect, no BMS needed).

2. **A wear item was approved.** e.g., customer reports peeling grips or
   rusted chain and the Adjudicator approved+replacement. Cite §1.5/§2.2.

3. **Accidental damage was approved without Extended Damage Protection.**
   e.g., "I crashed into a curb" → approved with no mention of EDP plan.
   Cite §2.1.

4. **An XGBoost fraud signal was overlooked.** e.g., the synthesized
   context shows `calibrated_prob > 0.5` OR `distinct_shipping_addresses
   >= 3` AND the Adjudicator approved.

5. **`needs_info` was used on a claim with SKU + failure description +
   photos/evidence.** e.g., the customer wrote "my LevelUp 3 battery
   stopped working, serial X, photos attached" and the Adjudicator
   asked for more info.

If NONE of the above apply, return `verdict="agree"` with empty concerns,
regardless of whether you would have made a slightly different call.
**You are not the Adjudicator. Do not relitigate confidence calls,
resolution choices (refund vs replacement), or borderline-window cases.**

## Examples of agreement (DO NOT disagree on these)

- "Adjudicator approved battery defect citing §1.1 within 12mo" → AGREE
  (correct policy interpretation, even if you'd have asked for serial)
- "Adjudicator chose `repair` over `replacement` for motor fault" → AGREE
  (resolution choice is in the Adjudicator's discretion)
- "Adjudicator picked `medium` confidence on a gray case" → AGREE
  (confidence is the Adjudicator's call)
- "Adjudicator declared needs_info on `the bike is broken`" → AGREE
  (genuinely vague message)

<draft_decision>
outcome: {{ outcome }}
resolution: {{ resolution }}
rationale: {{ rationale }}
policy_citation: {{ citation }}
confidence: {{ confidence }}
</draft_decision>

<synthesized_context>
{{ context }}
</synthesized_context>

<customer_input>
{{ customer_text }}
</customer_input>
"""


@_track("Critic")
async def critic(state: ClaimState) -> dict[str, Any]:
    """Second-pass review of the Adjudicator's draft.

    Returns verdict + concerns. The router uses the verdict to decide
    whether to send back to the Adjudicator or proceed to the Communicator.
    """
    from jinja2 import Template
    draft = state.get("draft_decision") or {}
    prompt = Template(_CRITIC_PROMPT).render(
        outcome=draft.get("outcome"),
        resolution=draft.get("resolution"),
        rationale=draft.get("rationale"),
        citation=draft.get("policy_citation"),
        confidence=draft.get("confidence"),
        context=state.get("synthesized_context") or "(empty)",
        customer_text=state["raw_input"],
    )
    verdict, cost = await chat(
        messages=[{"role": "user", "content": prompt}],
        model_alias="judge",
        response_model=_CriticVerdictModel,
        temperature=0.0,
        return_cost=True,
        metadata={"trace_id": state.get("trace_id"), "node": "Critic"},
    )
    return {
        "critic_verdict": verdict.model_dump(mode="json"),
        "critic_iterations": state.get("critic_iterations", 0) + 1,
        "cost_usd_accum": cost,
        "_summary": f"verdict={verdict.verdict} concerns={len(verdict.concerns)}",
    }


@_track("Communicator")
async def communicator(state: ClaimState) -> dict[str, Any]:
    """Final node — draft the customer email, mark decision final."""
    draft = state.get("draft_decision") or {}
    # The decision is final at this point — Critic has either agreed or we've
    # exhausted retries.
    final = dict(draft)

    # Draft the email (skipped if no decision available, e.g., on early
    # escalate paths)
    email_text = ""
    if final.get("outcome"):
        try:
            out = await _draft_email(DraftEmailInput(
                claim_id=state.get("claim_id") or "unknown",
                outcome=final.get("outcome", "needs_info"),
                resolution=final.get("resolution", "none"),
                rationale=final.get("rationale", "(no rationale)"),
                missing_info_questions=final.get("missing_info_questions") or [],
            ))
            email_text = out.email
        except Exception as e:  # noqa: BLE001
            email_text = f"(email drafting failed: {type(e).__name__})"

    return {
        "final_decision": final,
        "email": email_text,
        "_summary": f"final={final.get('outcome', 'none')}",
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _critic_router(state: ClaimState) -> str:
    """After Critic: loop back to Adjudicator if disagree (max 1 retry), else done."""
    verdict = (state.get("critic_verdict") or {}).get("verdict", "agree")
    iters = state.get("critic_iterations", 0)
    if verdict == "disagree" and iters <= DEFAULT_CRITIC_MAX_LOOPS:
        return "Adjudicator"
    return "Communicator"


def _planner_dispatch(state: ClaimState) -> list[str]:
    """Read the Planner's `specialists_to_run` and return that list.

    Defensive: if no plan present (Planner skipped or failed), fall back
    to the Week-12 "run everything" behavior.
    """
    plan = state.get("plan") or {}
    chosen = list(plan.get("specialists_to_run") or [])
    if not chosen:
        # Fallback
        chosen = ["Investigator", "PolicyInterpreter", "FraudAuditor"]
        if state.get("photo_urls"):
            chosen.append("VisionAnalyst")
    # Memory toggle still wins (Week-13 production default is OFF)
    if not state.get("use_memory", USE_MEMORY) and "MemoryConsultant" in chosen:
        chosen.remove("MemoryConsultant")
    return chosen


def _replan_router(state: ClaimState) -> str:
    """After ContextSynthesizer: if any specialist errored AND we have a
    replan budget left, route back to Planner with the error feedback.
    Otherwise proceed to the Adjudicator.
    """
    if state.get("replan_iterations", 0) > DEFAULT_REPLAN_MAX_LOOPS:
        return "Adjudicator"
    # Check step_log for any specialist that returned an ERROR summary
    log = state.get("step_log") or []
    specialist_names = {
        "Investigator", "PolicyInterpreter", "VisionAnalyst",
        "FraudAuditor", "MemoryConsultant",
    }
    errored: list[str] = []
    for entry in log[-12:]:  # only the most recent fan-out window
        if entry.get("node") in specialist_names and "ERROR" in (entry.get("summary") or ""):
            errored.append(entry["node"])
    if errored and state.get("replan_iterations", 0) <= DEFAULT_REPLAN_MAX_LOOPS - 1:
        return "Planner"
    return "Adjudicator"


# ---------------------------------------------------------------------------
# Checkpointers
# ---------------------------------------------------------------------------

# Module-level PostgresSaver — set up once at startup via setup_persistent_checkpointer()
# Calling code reads this through get_default_checkpointer(). If unset (e.g.,
# unit tests, eval runs that don't want DB writes), we fall back to MemorySaver.
_pg_checkpointer: Any | None = None


async def setup_persistent_checkpointer() -> None:
    """Initialize the PostgresSaver. Idempotent — safe to call multiple times.

    Must be called from an async context (the saver opens an asyncpg pool).
    Typically invoked from FastAPI lifespan startup.
    """
    global _pg_checkpointer
    if _pg_checkpointer is not None:
        return
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from app.core.config import get_settings

    # AsyncPostgresSaver uses psycopg internally; convert SQLAlchemy URL to plain
    settings = get_settings()
    pg_url = settings.alembic_database_url.replace(
        "postgresql+psycopg://", "postgresql://"
    )
    # Use the async context manager pattern — the saver manages its own pool.
    # We hold onto the underlying connection across the process lifetime via
    # __aenter__; FastAPI lifespan __aexit__'s it on shutdown.
    saver_cm = AsyncPostgresSaver.from_conn_string(pg_url)
    saver = await saver_cm.__aenter__()
    await saver.setup()  # creates checkpoint tables if missing — idempotent
    _pg_checkpointer = saver
    log.info("multi.checkpointer.postgres_ready")
    # Stash the context manager so shutdown can close cleanly
    setup_persistent_checkpointer._cm = saver_cm  # type: ignore[attr-defined]


async def shutdown_persistent_checkpointer() -> None:
    """Close the PostgresSaver. Called from FastAPI lifespan shutdown."""
    global _pg_checkpointer
    if _pg_checkpointer is None:
        return
    cm = getattr(setup_persistent_checkpointer, "_cm", None)
    if cm is not None:
        await cm.__aexit__(None, None, None)
    _pg_checkpointer = None


def get_default_checkpointer() -> Any:
    """Return the production PostgresSaver if set, else an in-memory fallback.

    Eval scripts and unit tests get MemorySaver — they don't need durability.
    """
    return _pg_checkpointer or MemorySaver()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(checkpointer: Any | None = None) -> Any:
    """Construct and compile the multi-agent StateGraph.

    Caller passes a checkpointer (MemorySaver for stateless eval, PostgresSaver
    for production HITL). Defaults to whatever `get_default_checkpointer()`
    returns — PostgresSaver if `setup_persistent_checkpointer()` ran during
    FastAPI startup, else MemorySaver.
    """
    builder = StateGraph(ClaimState)
    builder.add_node("Orchestrator", orchestrator)
    builder.add_node("Planner", planner)
    builder.add_node("Investigator", investigator)
    builder.add_node("PolicyInterpreter", policy_interpreter)
    builder.add_node("VisionAnalyst", vision_analyst)
    builder.add_node("FraudAuditor", fraud_auditor)
    builder.add_node("MemoryConsultant", memory_consultant)
    builder.add_node("ContextSynthesizer", context_synthesizer)
    builder.add_node("Adjudicator", adjudicator)
    builder.add_node("Critic", critic)
    builder.add_node("Communicator", communicator)

    builder.add_edge(START, "Orchestrator")
    builder.add_edge("Orchestrator", "Planner")

    # Planner decides which specialists fan out. _planner_dispatch reads
    # `state.plan.specialists_to_run`.
    builder.add_conditional_edges(
        "Planner",
        _planner_dispatch,
        {
            "Investigator": "Investigator",
            "PolicyInterpreter": "PolicyInterpreter",
            "VisionAnalyst": "VisionAnalyst",
            "FraudAuditor": "FraudAuditor",
            "MemoryConsultant": "MemoryConsultant",
        },
    )
    # All specialists fan-in to ContextSynthesizer
    builder.add_edge("Investigator", "ContextSynthesizer")
    builder.add_edge("PolicyInterpreter", "ContextSynthesizer")
    builder.add_edge("VisionAnalyst", "ContextSynthesizer")
    builder.add_edge("FraudAuditor", "ContextSynthesizer")
    builder.add_edge("MemoryConsultant", "ContextSynthesizer")

    # Week 14: after synthesis, optionally replan if any specialist errored
    builder.add_conditional_edges(
        "ContextSynthesizer", _replan_router,
        {"Planner": "Planner", "Adjudicator": "Adjudicator"},
    )
    builder.add_edge("Adjudicator", "Critic")
    builder.add_conditional_edges(
        "Critic", _critic_router,
        {"Adjudicator": "Adjudicator", "Communicator": "Communicator"},
    )
    builder.add_edge("Communicator", END)

    return builder.compile(checkpointer=checkpointer or get_default_checkpointer())


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


async def run_multi_agent(
    *,
    claim_id: str,
    raw_input: str,
    photo_urls: list[str] | None = None,
    customer_id: str | None = None,
    use_memory: bool | None = None,
    use_planner: bool | None = None,
) -> dict[str, Any]:
    """Run the compiled graph end-to-end. Returns the final state + metadata.

    Args:
        use_memory: override module-level USE_MEMORY (Week 13 A/B).
        use_planner: override module-level USE_PLANNER (Week 14 A/B).
    """
    graph = build_graph()
    t0 = time.perf_counter()
    initial: ClaimState = {
        "claim_id": claim_id,
        "customer_id": customer_id,
        "raw_input": raw_input,
        "photo_urls": photo_urls or [],
        "use_memory": USE_MEMORY if use_memory is None else use_memory,
        "use_planner": USE_PLANNER if use_planner is None else use_planner,
        "cost_usd_accum": 0.0,
        "step_log": [],
        "trace_id": str(uuid.uuid4()),
        "critic_iterations": 0,
        "replan_iterations": 0,
        "escalated": False,
    }
    config = {"configurable": {"thread_id": claim_id}}
    final_state = await graph.ainvoke(initial, config=config)
    latency_ms = (time.perf_counter() - t0) * 1000

    log.info(
        "multi.run.complete",
        trace_id=final_state.get("trace_id"),
        outcome=(final_state.get("final_decision") or {}).get("outcome"),
        n_steps=len(final_state.get("step_log") or []),
        cost_usd=round(final_state.get("cost_usd_accum") or 0.0, 4),
        latency_ms=round(latency_ms, 1),
    )

    plan = final_state.get("plan") or {}
    return {
        "final_decision": final_state.get("final_decision"),
        "email": final_state.get("email"),
        "cost_usd": round(final_state.get("cost_usd_accum") or 0.0, 6),
        "latency_ms": round(latency_ms, 1),
        "trace_id": final_state.get("trace_id"),
        "step_log": final_state.get("step_log") or [],
        "critic_iterations": final_state.get("critic_iterations", 0),
        "replan_iterations": final_state.get("replan_iterations", 0),
        "fraud_score": final_state.get("fraud_score"),
        "vision_assessment": final_state.get("vision_assessment"),
        "plan": plan,
        "specialists_run": plan.get("specialists_to_run") or [],
        "critic_disagreed": (
            (final_state.get("critic_verdict") or {}).get("verdict") == "disagree"
            or (final_state.get("critic_iterations", 0) > 1)
        ),
        "version": MULTI_AGENT_VERSION,
    }
