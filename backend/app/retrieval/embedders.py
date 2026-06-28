"""Embedding helpers built on the LLM abstraction.

Thin layer over `app.ai.llm.embed` for batched corpus ingestion. The actual
chunking + storage pipeline lands in Week 5; this module is the seam.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.ai.llm import embed
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Batch size for embedding requests. Conservative default that stays well
# within provider per-request input limits.
_MAX_BATCH = 16


def expected_dim() -> int:
    """Embedding dimension for the configured model. Drives the pgvector schema."""
    return get_settings().embedding_dim


async def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Embed an iterable of strings with safe batching.

    Returns vectors in the same order as input. Empty input → empty list.
    """
    items = list(texts)
    if not items:
        return []

    out: list[list[float]] = []
    for i in range(0, len(items), _MAX_BATCH):
        batch = items[i : i + _MAX_BATCH]
        vectors = await embed(batch)
        out.extend(vectors)

    expected = expected_dim()
    for j, v in enumerate(out):
        if len(v) != expected:
            log.warning(
                "embedders.dim_mismatch",
                index=j,
                got=len(v),
                expected=expected,
            )
    return out
