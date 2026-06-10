"""Vision damage classifier (Week 8).

Wraps gpt-4o-mini vision behind the `chat()` abstraction. Accepts either:
- a URL (https://..., http://...) — the model fetches the image
- a base64 data URL (data:image/jpeg;base64,...) — inline payload
- raw image bytes — we encode to a data URL ourselves

The detail level matters for both cost and quality:
- `"low"` — ~85 input tokens per image, model sees 512×512 downscaled
- `"high"` — ~170-2550 tokens per image, model tiles up to 2048×2048

For warranty damage assessment we default to `"high"` because cracks and
surface defects matter at pixel scale. The cost delta is ~30× per image
but absolute cost stays trivial (~$0.001 per photo at high detail).
"""

from __future__ import annotations

import base64
import time
from typing import Literal

from app.ai.llm import chat
from app.ai.prompt_loader import render_prompt
from app.ai.schemas import DamageAssessment
from app.core.logging import get_logger

log = get_logger(__name__)

VISION_PROMPT = "vision_classify_v1"
ImageDetail = Literal["low", "high", "auto"]


def encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> str:
    """Convert raw image bytes to a `data:` URL the model can consume."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _resolve_image_source(image: str | bytes) -> str:
    """Return a string the LLM can use as an image_url value.

    - URL → pass through
    - data URL → pass through
    - bytes → encode to data URL (assume JPEG; callers with PNG should pre-encode)
    """
    if isinstance(image, bytes):
        return encode_image_bytes(image)
    if image.startswith(("data:", "http://", "https://")):
        return image
    raise ValueError(
        "image must be bytes, a data URL, or an http(s) URL "
        f"(got {image[:60]!r}...)"
    )


async def analyze_photo(
    image: str | bytes,
    *,
    photo_caption: str | None = None,
    claim_context: str | None = None,
    detail: ImageDetail = "high",
    model_alias: str = "vision",
) -> DamageAssessment:
    """Run the vision classifier on a single photo.

    Args:
        image: URL, data URL, or raw bytes.
        photo_caption: optional one-line caption the customer attached.
        claim_context: optional one-paragraph summary of the claim (e.g.,
            the extracted `customer_summary`). Helps the model focus.
        detail: "high" for damage classification (default). "low" if you
            just need triage.
        model_alias: defaults to "vision" → currently routes to gpt-4o-mini.
    """
    image_url = _resolve_image_source(image)
    prompt_text = render_prompt(
        VISION_PROMPT,
        photo_caption=photo_caption or "",
        claim_context=claim_context or "",
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": detail},
                },
            ],
        }
    ]
    t0 = time.perf_counter()
    assessment: DamageAssessment = await chat(
        messages=messages,
        model_alias=model_alias,
        response_model=DamageAssessment,
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "vision.analyze_photo",
        damage_type=assessment.damage_type,
        severity=assessment.severity_score,
        evidence=assessment.evidence_quality,
        suspicion_count=len(assessment.suspicion_signals),
        latency_ms=round(latency_ms, 1),
        detail=detail,
    )
    return assessment
