"""Side-effecting action tools (Week 11).

These tools mutate state — write to the DB, send emails, or escalate to
human review. All four are designed around **idempotency keys** so re-runs
from a checkpoint don't double-fire:

- The agent is required to pass an idempotency_key per side-effect
  (we use `claim_id` for the standard case)
- The DB has a UNIQUE constraint on (action_type, idempotency_key)
- A repeat call with the same key returns the same result, no new effect

Real RMA / refund integration is Week 16. Today's stubs write to the
`agent_actions` audit table so the eval can verify what the agent
*tried* to do.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.ai.tools.registry import ToolSpec, register_tool
from app.db.models import AgentAction
from app.db.session import get_sessionmaker


# ---------------------------------------------------------------------------
# draft_email
# ---------------------------------------------------------------------------


class DraftEmailInput(BaseModel):
    claim_id: str
    outcome: str = Field(description="One of: approve, reject, needs_info")
    resolution: str = Field(default="none", description="refund / replacement / repair / store_credit / none")
    rationale: str = Field(min_length=20, description="The reasoning to convey to the customer.")
    missing_info_questions: list[str] = Field(default_factory=list)


class DraftEmailOutput(BaseModel):
    status: str = "ok"
    message: str = ""
    email: str = ""


async def _draft_email(inp: DraftEmailInput) -> DraftEmailOutput:
    from app.ai.llm import chat
    from app.ai.prompt_loader import render_prompt
    from app.ai.schemas import Decision

    # Reconstruct a Decision for the existing draft_email_v1 prompt
    decision = Decision(
        outcome=inp.outcome,  # type: ignore[arg-type]
        resolution=inp.resolution,  # type: ignore[arg-type]
        rationale=inp.rationale,
        policy_citation="(supplied by adjudicator, not echoed in email)" * 1,
        confidence="high",
        missing_info_questions=inp.missing_info_questions,
    )
    # We don't have customer_text in this call; pass a placeholder. The
    # email prompt uses customer_text for tone calibration; absence is OK.
    rendered = render_prompt(
        "draft_email_v1",
        customer_text="(customer text not available in this scope)",
        decision=decision,
    )
    resp = await chat(
        messages=[{"role": "user", "content": rendered}],
        model_alias="reasoner",
        temperature=0.4,
    )
    text = (
        resp["choices"][0]["message"]["content"].strip()
        if isinstance(resp, dict)
        else resp.choices[0].message.content.strip()
    )
    return DraftEmailOutput(email=text)


register_tool(ToolSpec(
    name="draft_email",
    description=(
        "Draft the customer-facing email for a decided claim. Pass the "
        "outcome (approve/reject/needs_info), resolution, rationale, and "
        "(if needs_info) the specific questions to ask. Returns the email "
        "body."
    ),
    input_model=DraftEmailInput,
    output_model=DraftEmailOutput,
    handler=_draft_email,
))


# ---------------------------------------------------------------------------
# create_rma
# ---------------------------------------------------------------------------


class CreateRmaInput(BaseModel):
    claim_id: str
    rma_type: str = Field(description="One of: replacement, repair, refund, store_credit")
    idempotency_key: str = Field(
        description="Unique key for this RMA. Re-running with the same key "
        "returns the existing record — does not create a duplicate."
    )
    notes: str = Field(default="", max_length=1000)


class CreateRmaOutput(BaseModel):
    status: str = "ok"
    message: str = ""
    rma_id: str = ""
    duplicate: bool = False


async def _create_rma(inp: CreateRmaInput) -> CreateRmaOutput:
    sm = get_sessionmaker()
    async with sm() as session:
        existing = await session.scalar(
            select(AgentAction).where(
                AgentAction.action_type == "create_rma",
                AgentAction.idempotency_key == inp.idempotency_key,
            )
        )
        if existing is not None:
            return CreateRmaOutput(
                rma_id=str(existing.id),
                duplicate=True,
                message="RMA already exists for this idempotency_key",
            )
        action = AgentAction(
            action_type="create_rma",
            idempotency_key=inp.idempotency_key,
            payload={
                "claim_id": inp.claim_id,
                "rma_type": inp.rma_type,
                "notes": inp.notes,
            },
        )
        session.add(action)
        await session.commit()
        await session.refresh(action)
        return CreateRmaOutput(rma_id=str(action.id))


register_tool(ToolSpec(
    name="create_rma",
    description=(
        "Create a Return Merchandise Authorization. STUB in Phase 2; Phase 3 "
        "Week 16 wires real Shopify RMA. Pass a unique `idempotency_key` (use "
        "the claim_id) so re-runs from a checkpoint don't double-create."
    ),
    input_model=CreateRmaInput,
    output_model=CreateRmaOutput,
    handler=_create_rma,
))


# ---------------------------------------------------------------------------
# escalate_to_human
# ---------------------------------------------------------------------------


class EscalateInput(BaseModel):
    claim_id: str
    reason: str = Field(min_length=10, description="Why this claim needs human review.")
    severity: str = Field(default="medium", description="low / medium / high")
    idempotency_key: str = Field(
        description="Unique key for the escalation. Use the claim_id."
    )


class EscalateOutput(BaseModel):
    status: str = "ok"
    message: str = ""
    escalation_id: str = ""
    duplicate: bool = False


async def _escalate(inp: EscalateInput) -> EscalateOutput:
    sm = get_sessionmaker()
    async with sm() as session:
        existing = await session.scalar(
            select(AgentAction).where(
                AgentAction.action_type == "escalate_to_human",
                AgentAction.idempotency_key == inp.idempotency_key,
            )
        )
        if existing is not None:
            return EscalateOutput(
                escalation_id=str(existing.id),
                duplicate=True,
                message="Escalation already exists for this idempotency_key",
            )
        action = AgentAction(
            action_type="escalate_to_human",
            idempotency_key=inp.idempotency_key,
            payload={
                "claim_id": inp.claim_id,
                "reason": inp.reason,
                "severity": inp.severity,
            },
        )
        session.add(action)
        await session.commit()
        await session.refresh(action)
        return EscalateOutput(escalation_id=str(action.id))


register_tool(ToolSpec(
    name="escalate_to_human",
    description=(
        "Route this claim to human review with a reason. Use when fraud "
        "signals exceed threshold, when the policy excerpts don't cover the "
        "situation, when the customer's message is hostile, or when you've "
        "exhausted other tools without a clear decision. Pass `idempotency_key` "
        "= claim_id."
    ),
    input_model=EscalateInput,
    output_model=EscalateOutput,
    handler=_escalate,
))
