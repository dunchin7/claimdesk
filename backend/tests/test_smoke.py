"""End-to-end smoke tests that hit Azure OpenAI.

Skipped automatically if AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT aren't
set (see conftest.py). Mark all tests in this file with `@pytest.mark.smoke`.

Run explicitly:
    uv run pytest tests/test_smoke.py -v -m smoke
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ai.llm import chat, embed
from app.ai.schemas import ClaimExtraction
from app.core.config import get_settings
from app.main import app


@pytest.mark.smoke
async def test_chat_extractor_returns_pydantic() -> None:
    """Single Azure call via the abstraction; expect a typed response."""
    extraction = await chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Customer: My e-bike battery (EB-PACE-500) won't hold a charge "
                    "after 6 months. I have receipts. Photos attached."
                ),
            }
        ],
        model_alias="extractor",
        response_model=ClaimExtraction,
    )
    assert isinstance(extraction, ClaimExtraction)
    assert extraction.customer_summary
    assert extraction.evidence_strength in ("strong", "moderate", "weak")


@pytest.mark.smoke
async def test_embed_returns_vectors_of_expected_dim() -> None:
    """Single Azure embedding call against the embedding resource."""
    vectors = await embed(
        ["My e-bike battery stopped holding a charge.", "Frame cracked at the down tube."]
    )
    assert len(vectors) == 2
    expected_dim = get_settings().azure_openai_embedding_dim
    for v in vectors:
        assert len(v) == expected_dim
        assert all(isinstance(x, float) for x in v[:5])


@pytest.mark.smoke
def test_extract_endpoint_e2e() -> None:
    """POST /api/claims/extract round-trip against real Azure."""
    payload = {
        "raw_input": (
            "Hi, my LevelUp 3 battery is dead. Bought it 4 months ago and only "
            "ridden it on weekends. Less than 30 charge cycles. Receipts attached."
        ),
        "photo_descriptions": ["close-up of battery", "serial number sticker"],
    }
    with TestClient(app) as client:
        r = client.post("/api/claims/extract", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "extraction" in data
        assert data["prompt_version"] == "extract_v1"
        assert data["extraction"]["customer_summary"]
