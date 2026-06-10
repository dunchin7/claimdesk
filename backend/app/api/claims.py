"""Claims API."""

from __future__ import annotations

import base64
import binascii
import time
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.adjudication.pipeline import process_claim
from app.ai.agents.react import run_agent
from app.ai.llm import chat
from app.ai.prompt_loader import render_prompt
from app.ai.schemas import ClaimExtraction, DamageAssessment, Decision
from app.core.limits import (
    LIMIT_EXTRACT,
    LIMIT_PROCESS,
    LIMIT_RUN_AGENT,
    LIMIT_VISION,
    limiter,
)
from app.core.logging import get_logger
from app.db.models import AgentRun
from app.db.session import get_sessionmaker
from app.vision.ai_detection import ai_generated_likelihood
from app.vision.classify import analyze_photo
from app.vision.exif import extract_exif

router = APIRouter()
log = get_logger(__name__)


class ExtractRequest(BaseModel):
    raw_input: str = Field(min_length=5, max_length=10_000)
    photo_descriptions: list[str] = Field(default_factory=list, max_length=20)


class ExtractResponse(BaseModel):
    extraction: ClaimExtraction
    prompt_version: str = "extract_v2"


# Single source of truth — eval scripts should import this constant rather
# than hardcoding the string.
EXTRACT_PROMPT = "extract_v2"


@router.post("/extract", response_model=ExtractResponse)
@limiter.limit(LIMIT_EXTRACT)
async def extract_claim(request: Request, req: ExtractRequest) -> ExtractResponse:
    """Extract structured claim data from a free-form customer message.

    Week 2 endpoint: extraction-only with the v2 prompt + expanded schema.
    Week 3 wraps this in `POST /api/claims/process` which adjudicates and
    drafts the customer email.
    """
    prompt = render_prompt(
        EXTRACT_PROMPT,
        customer_text=req.raw_input,
        photo_descriptions=req.photo_descriptions,
    )
    try:
        extraction = await chat(
            messages=[{"role": "user", "content": prompt}],
            model_alias="extractor",
            response_model=ClaimExtraction,
        )
    except Exception as e:
        log.error("extract_claim.failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Extraction failed: {type(e).__name__}",
        ) from e

    log.info(
        "extract_claim.ok",
        sku=extraction.sku,
        failure_mode=extraction.failure_mode,
        evidence_strength=extraction.evidence_strength,
    )
    return ExtractResponse(extraction=extraction)


# ---------------------------------------------------------------------------
# POST /api/claims/process — full pipeline (Week 3)
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    raw_input: str = Field(min_length=5, max_length=10_000)
    photo_descriptions: list[str] = Field(default_factory=list, max_length=20)


class CitationCheck(BaseModel):
    verbatim: bool
    fuzzy_ratio: float


class ProcessResponse(BaseModel):
    extraction: ClaimExtraction
    decision: Decision
    citation_check: CitationCheck
    email: str
    prompt_versions: dict[str, str]
    policy_version: str
    latency_ms: float
    # Week 15: HITL routing
    calibrated_prob: float = 0.5
    route: str = "review"  # auto_resolve / assist / review
    queue_id: str | None = None  # set when enqueued for human review


@router.post("/process", response_model=ProcessResponse)
@limiter.limit(LIMIT_PROCESS)
async def process_claim_endpoint(
    request: Request, req: ProcessRequest
) -> ProcessResponse:
    """End-to-end: extract → adjudicate → citation check → draft email.

    Routes the decision based on `PipelineResult.route`. Auto-resolve
    claims (calibrated_prob ≥ `AUTO_RESOLVE_THRESHOLD`, set to 0.75 in
    `app/confidence/calibrator.py` per the threshold sweep at
    `backend/app/evals/reports/threshold_sweep.md`) return without
    enqueuing; assist and review routes create an `operator_queue` row
    and the customer email is held until the operator confirms.
    """
    try:
        result = await process_claim(
            raw_input=req.raw_input,
            photo_descriptions=req.photo_descriptions,
        )
    except Exception as e:
        log.error(
            "process_claim.failed", error=str(e), error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Processing failed: {type(e).__name__}",
        ) from e

    # Route. Auto-resolve → straight through. Otherwise enqueue + hold email.
    queue_id: str | None = None
    if result.route != "auto_resolve":
        from app.db.session import get_sessionmaker
        from app.hitl.router import enqueue_if_needed
        sm = get_sessionmaker()
        async with sm() as session:
            item = await enqueue_if_needed(
                session,
                claim_id=None,  # API doesn't yet thread a DB claim_id through
                raw_input=req.raw_input,
                result=result,
            )
            await session.commit()
            queue_id = str(item.id) if item else None

    return ProcessResponse(
        extraction=result.extraction,
        decision=result.decision,
        citation_check=CitationCheck(
            verbatim=result.citation_result.verbatim,
            fuzzy_ratio=result.citation_result.fuzzy_ratio,
        ),
        email=result.email,
        prompt_versions=result.prompt_versions,
        policy_version=result.policy_version,
        latency_ms=round(result.latency_ms, 1),
        calibrated_prob=result.calibrated_prob,
        route=result.route,
        queue_id=queue_id,
    )


# ---------------------------------------------------------------------------
# POST /api/claims/analyze-photo — vision (Week 8)
# ---------------------------------------------------------------------------


class AnalyzePhotoRequest(BaseModel):
    # Provide exactly one of these:
    image_url: str | None = Field(
        default=None,
        description="HTTP(S) URL or `data:image/...;base64,...` URL. The model fetches/decodes.",
    )
    image_b64: str | None = Field(
        default=None,
        description="Raw base64-encoded JPEG bytes (no data: prefix). We wrap it for the model.",
    )
    photo_caption: str | None = Field(
        default=None,
        description="Optional one-line caption from the customer.",
        max_length=500,
    )
    claim_context: str | None = Field(
        default=None,
        description="Optional one-paragraph claim summary to focus the model.",
        max_length=2000,
    )
    claim_date: datetime | None = Field(
        default=None,
        description="Date the claim was filed. Drives the timestamp sanity check in AI-detection.",
    )
    detail: Literal["low", "high", "auto"] = "high"


class ExifSummary(BaseModel):
    has_exif: bool
    has_gps: bool
    camera_make: str | None
    camera_model: str | None
    software_field: str | None
    captured_at: datetime | None
    suspicious_software: bool


class AIDetectionSummary(BaseModel):
    score: float
    signals: list[str]
    raw_signals: dict[str, float]


class AnalyzePhotoResponse(BaseModel):
    damage_assessment: DamageAssessment
    exif: ExifSummary
    ai_detection: AIDetectionSummary
    prompt_version: str = "vision_classify_v1"
    latency_ms: float


@router.post("/analyze-photo", response_model=AnalyzePhotoResponse)
@limiter.limit(LIMIT_VISION)
async def analyze_photo_endpoint(
    request: Request, req: AnalyzePhotoRequest
) -> AnalyzePhotoResponse:
    """Analyze a single claim photo: damage classification + EXIF + AI-detection.

    Standalone endpoint for ad-hoc analysis. Integration into the full
    claim pipeline (so /process automatically analyzes attached photos)
    lands in a follow-up.
    """
    if (req.image_url is None) == (req.image_b64 is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of `image_url` or `image_b64`.",
        )

    # Resolve to bytes (for EXIF) and to a URL/data-URL string (for the model)
    image_for_model: str | bytes
    image_bytes: bytes
    if req.image_b64 is not None:
        try:
            image_bytes = base64.b64decode(req.image_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"image_b64 is not valid base64: {e}",
            ) from e
        image_for_model = image_bytes
    else:
        assert req.image_url is not None
        image_for_model = req.image_url
        # For URL inputs, we only get EXIF if the caller passes a data URL.
        # For http(s) URLs the model fetches the file but we don't — EXIF
        # extraction returns "no exif" for those, which is honest.
        if req.image_url.startswith("data:"):
            try:
                comma = req.image_url.index(",")
                image_bytes = base64.b64decode(req.image_url[comma + 1 :], validate=True)
            except (ValueError, binascii.Error):
                image_bytes = b""
        else:
            image_bytes = b""

    t0 = time.perf_counter()
    try:
        assessment = await analyze_photo(
            image_for_model,
            photo_caption=req.photo_caption,
            claim_context=req.claim_context,
            detail=req.detail,
        )
    except Exception as e:
        log.error("analyze_photo.failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Photo analysis failed: {type(e).__name__}",
        ) from e

    exif_signals = extract_exif(image_bytes) if image_bytes else extract_exif(b"")
    ai_result = ai_generated_likelihood(exif_signals, claim_date=req.claim_date)
    latency_ms = (time.perf_counter() - t0) * 1000

    return AnalyzePhotoResponse(
        damage_assessment=assessment,
        exif=ExifSummary(
            has_exif=exif_signals.has_exif,
            has_gps=exif_signals.has_gps,
            camera_make=exif_signals.camera_make,
            camera_model=exif_signals.camera_model,
            software_field=exif_signals.software_field,
            captured_at=exif_signals.captured_at,
            suspicious_software=exif_signals.suspicious_software,
        ),
        ai_detection=AIDetectionSummary(
            score=ai_result.score,
            signals=ai_result.signals,
            raw_signals=ai_result.raw_signals,
        ),
        latency_ms=round(latency_ms, 1),
    )


# ---------------------------------------------------------------------------
# POST /api/claims/run-agent — Week 11 ReAct agent
# ---------------------------------------------------------------------------


class RunAgentRequest(BaseModel):
    claim_id: str = Field(
        description="Stable identifier. Used as the idempotency key for "
        "any RMA / escalation actions the agent takes.",
        min_length=1,
        max_length=128,
    )
    raw_input: str = Field(min_length=5, max_length=10_000)
    photo_urls: list[str] = Field(default_factory=list, max_length=10)
    cost_cap_usd: float = Field(default=0.50, ge=0.01, le=5.0)
    max_iter: int = Field(default=15, ge=1, le=30)


class AgentStep(BaseModel):
    iteration: int
    role: str
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output_brief: str | None = None
    cost_usd: float = 0.0


class RunAgentResponse(BaseModel):
    status: str
    final_decision: dict[str, Any] | None
    cost_usd: float
    n_iterations: int
    n_tool_calls: int
    latency_ms: float
    trace_id: str
    error: str | None = None
    steps: list[AgentStep] = []
    run_id: str | None = None


@router.post("/run-agent", response_model=RunAgentResponse)
@limiter.limit(LIMIT_RUN_AGENT)
async def run_agent_endpoint(
    request: Request, req: RunAgentRequest
) -> RunAgentResponse:
    """Run the ReAct agent on a claim. Persists an AgentRun row."""
    try:
        result = await run_agent(
            claim_id=req.claim_id,
            raw_input=req.raw_input,
            photo_urls=req.photo_urls,
            max_iter=req.max_iter,
            cost_cap_usd=req.cost_cap_usd,
        )
    except Exception as e:
        log.error("run_agent.failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent run failed: {type(e).__name__}",
        ) from e

    # Persist AgentRun. Best-effort — if DB write fails, we still return the
    # result to the caller; the run already happened.
    run_id: str | None = None
    try:
        sm = get_sessionmaker()
        async with sm() as session:
            row = AgentRun(
                claim_id=None,  # claim_id is application-level, not always a DB UUID
                status=result.status,
                cost_usd=result.cost_usd,
                n_iterations=result.n_iterations,
                n_tool_calls=result.n_tool_calls,
                final_decision=result.final_decision,
                error=result.error,
                state=result.messages,
                trace_id=result.trace_id,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            run_id = str(row.id)
    except Exception as e:
        log.warning("run_agent.persist_failed", error=str(e))

    # Convert StepLogEntry → AgentStep with brief tool_output
    steps_brief: list[AgentStep] = []
    for entry in result.step_log:
        tool_output_brief = None
        if entry.tool_output is not None:
            text = (entry.tool_output.get("message", "") or "").strip()
            if not text:
                # Trimmed JSON preview
                from json import dumps as _dumps
                text = _dumps(entry.tool_output)[:200]
            tool_output_brief = text[:200]
        steps_brief.append(AgentStep(
            iteration=entry.iteration,
            role=entry.role,
            content=entry.content,
            tool_name=entry.tool_name,
            tool_args=entry.tool_args,
            tool_output_brief=tool_output_brief,
            cost_usd=entry.cost_usd,
        ))

    return RunAgentResponse(
        status=result.status,
        final_decision=result.final_decision,
        cost_usd=result.cost_usd,
        n_iterations=result.n_iterations,
        n_tool_calls=result.n_tool_calls,
        latency_ms=round(result.latency_ms, 1),
        trace_id=result.trace_id,
        error=result.error,
        steps=steps_brief,
        run_id=run_id,
    )
