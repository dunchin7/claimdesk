from __future__ import annotations

from app.ai.registry import load_registry, validate_registry


def test_registry_loads() -> None:
    reg = load_registry()
    assert "active" in reg
    assert "extract" in reg["active"]
    assert "adjudicate" in reg["active"]
    assert "draft_email" in reg["active"]


def test_registry_validates_clean() -> None:
    problems = validate_registry()
    assert problems == [], f"registry validation failed: {problems}"


def test_active_prompts_match_pipeline() -> None:
    """The pipeline's prompt names must equal the registry's active entries.

    The pipeline exposes both ADJUDICATE_PROMPT_FULL and ADJUDICATE_PROMPT_RAG;
    `USE_RETRIEVAL` decides which is active in production.
    """
    from app.adjudication.pipeline import (
        ADJUDICATE_PROMPT_FULL,
        ADJUDICATE_PROMPT_RAG,
        DRAFT_EMAIL_PROMPT,
        EXTRACT_PROMPT,
        USE_RETRIEVAL,
    )

    reg = load_registry()
    active_adjudicate = ADJUDICATE_PROMPT_RAG if USE_RETRIEVAL else ADJUDICATE_PROMPT_FULL
    assert reg["active"]["extract"]["prompt"] == EXTRACT_PROMPT
    assert reg["active"]["adjudicate"]["prompt"] == active_adjudicate
    assert reg["active"]["draft_email"]["prompt"] == DRAFT_EMAIL_PROMPT
