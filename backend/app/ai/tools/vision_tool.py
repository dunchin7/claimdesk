"""Vision tool (Week 11) — wraps Week 8's analyze_photo.

The model gets a `damage_type` + `severity_score` + suspicion signals
from a URL or data-URL. AI-image detection (Week 8's EXIF heuristics)
is bundled into the same call so the agent doesn't have to coordinate.
"""

from __future__ import annotations

import base64
import binascii

from pydantic import BaseModel, Field

from app.ai.tools.registry import ToolSpec, register_tool
from app.vision.ai_detection import ai_generated_likelihood
from app.vision.classify import analyze_photo
from app.vision.exif import extract_exif


class AnalyzePhotoToolInput(BaseModel):
    image_url: str | None = Field(
        default=None,
        description="HTTP(S) URL or data: URL of the photo to analyze.",
    )
    image_b64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG bytes (no data: prefix).",
    )
    question: str | None = Field(
        default=None,
        description="Optional one-line question to focus the analysis (e.g., "
        "'is the damage on the frame structural?').",
    )


class AnalyzePhotoToolOutput(BaseModel):
    status: str = "ok"
    message: str = ""
    damage_type: str = ""
    severity_score: int = 0
    affected_components: list[str] = []
    evidence_quality: str = ""
    suspicion_signals: list[str] = []
    reasoning: str = ""
    ai_generated_likelihood: float = 0.0
    ai_signals: list[str] = []


async def _analyze_photo_tool(inp: AnalyzePhotoToolInput) -> AnalyzePhotoToolOutput:
    if (inp.image_url is None) == (inp.image_b64 is None):
        return AnalyzePhotoToolOutput(
            status="error",
            message="Provide exactly one of image_url or image_b64.",
        )

    image_for_model: str | bytes
    image_bytes = b""
    if inp.image_b64:
        try:
            image_bytes = base64.b64decode(inp.image_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            return AnalyzePhotoToolOutput(
                status="error", message=f"Invalid base64: {e}"
            )
        image_for_model = image_bytes
    else:
        assert inp.image_url is not None
        image_for_model = inp.image_url
        # If data: URL, decode for EXIF; otherwise EXIF is N/A (model fetches)
        if inp.image_url.startswith("data:"):
            try:
                comma = inp.image_url.index(",")
                image_bytes = base64.b64decode(inp.image_url[comma + 1 :], validate=True)
            except (ValueError, binascii.Error):
                image_bytes = b""

    try:
        assessment = await analyze_photo(image_for_model, photo_caption=inp.question)
    except Exception as e:  # noqa: BLE001
        return AnalyzePhotoToolOutput(
            status="error",
            message=f"Vision call failed: {type(e).__name__}: {e}",
        )

    exif_signals = extract_exif(image_bytes) if image_bytes else extract_exif(b"")
    ai_result = ai_generated_likelihood(exif_signals)

    return AnalyzePhotoToolOutput(
        damage_type=assessment.damage_type,
        severity_score=assessment.severity_score,
        affected_components=assessment.affected_components,
        evidence_quality=assessment.evidence_quality,
        suspicion_signals=assessment.suspicion_signals,
        reasoning=assessment.reasoning,
        ai_generated_likelihood=ai_result.score,
        ai_signals=ai_result.signals,
    )


register_tool(ToolSpec(
    name="analyze_photo",
    description=(
        "Analyze a claim photo with the vision model and EXIF/AI-image "
        "detection. Returns damage classification (cosmetic / functional / "
        "structural / total_loss), severity 1-10, suspicion signals from the "
        "vision model, and an AI-generation likelihood from EXIF analysis. "
        "Provide either `image_url` (HTTP/data URL) or `image_b64` (raw "
        "base64). The optional `question` focuses the assessment."
    ),
    input_model=AnalyzePhotoToolInput,
    output_model=AnalyzePhotoToolOutput,
    handler=_analyze_photo_tool,
))
