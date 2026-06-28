"""Local embedding inference (Week 6).

Runs BGE-large-en-v1.5 (or its smaller siblings) via FastEmbed (ONNX runtime).
The first call downloads ~1.3GB of model weights to the FastEmbed cache
(~/.cache/fastembed by default). Subsequent calls reuse the cache.

We intentionally don't surface this through `app/ai/llm.py:embed()` —
`embed()` is the OpenAI-managed path, and we want the comparison apples-to-
apples: OpenAI for the production model, local for the open-model comparison.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _bge_model() -> "TextEmbedding":  # type: ignore[name-defined]
    """Lazy import + cached singleton so the model loads only when used."""
    from fastembed import TextEmbedding

    model_name = get_settings().bge_model_name
    log.info("local_embedders.load", model=model_name)
    return TextEmbedding(model_name=model_name)


def embed_bge(texts: Iterable[str]) -> list[list[float]]:
    """Embed an iterable of strings with the configured BGE model.

    Returns vectors in the same order as input. Unlike OpenAI embeddings, the
    BGE model has no input-batch cap — we still iterate in chunks of 32 to
    keep memory pressure predictable on the ONNX runtime.
    """
    items = list(texts)
    if not items:
        return []

    model = _bge_model()
    out: list[list[float]] = []
    batch_size = 32
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        # `model.embed` returns a generator of numpy arrays.
        vectors = list(model.embed(batch))
        out.extend(v.tolist() for v in vectors)

    expected = get_settings().bge_embedding_dim
    for j, v in enumerate(out):
        if len(v) != expected:
            log.warning(
                "local_embedders.dim_mismatch",
                index=j,
                got=len(v),
                expected=expected,
            )
    return out
