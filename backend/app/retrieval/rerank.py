"""Cross-encoder reranking (Week 7).

Bi-encoder retrieval (vector + BM25) gives us cheap top-N candidates. A
cross-encoder takes the (query, candidate) pairs and scores them with
query-doc interaction — much higher relevance precision at the cost of
one forward pass per pair. Standard cascade pattern: cheap retriever →
top-20 → cross-encoder → top-5.

Model: `BAAI/bge-reranker-v2-m3` via FastEmbed (ONNX runtime). First
call downloads ~600MB of model weights; subsequent calls reuse the
FastEmbed cache.

We intentionally don't surface this via the LLM abstraction — the
reranker is a small classifier, not a generative LLM, and lives in the
retrieval module alongside the embedders.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.logging import get_logger
from app.retrieval.hybrid import FusedHit
from app.retrieval.search import SearchHit

log = get_logger(__name__)

_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# Note: We start with `Xenova/ms-marco-MiniLM-L-6-v2` — FastEmbed's
# default cross-encoder. Small (~80MB), fast on CPU, strong on
# MSMARCO-style retrieval. The original Week-7 plan calls for
# `bge-reranker-v2-m3`; that's ~600MB and slower but stronger on some
# domains. The `_RERANKER_MODEL` constant is the single swap-point.


@lru_cache(maxsize=1)
def _reranker() -> "TextCrossEncoder":  # type: ignore[name-defined]
    """Lazy import + cached singleton."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    log.info("rerank.load", model=_RERANKER_MODEL)
    return TextCrossEncoder(model_name=_RERANKER_MODEL)


def rerank(
    query: str, candidates: list[SearchHit] | list[FusedHit], *, top_k: int = 5
) -> list[SearchHit]:
    """Rerank candidates by cross-encoder relevance score.

    Accepts either `SearchHit` (from vector/BM25) or `FusedHit` (from
    RRF). Returns `SearchHit` since the reranker score replaces RRF and
    we don't need the intermediate ranks downstream.
    """
    if not candidates:
        return []

    # Normalize to a list of SearchHit + their texts.
    if isinstance(candidates[0], FusedHit):
        hits = [c.hit for c in candidates]  # type: ignore[union-attr]
    else:
        hits = list(candidates)  # type: ignore[arg-type]

    model = _reranker()
    docs = [h.text for h in hits]
    # FastEmbed's API: rerank(query, list[document_text]) -> list[score]
    scores = list(model.rerank(query, docs))
    # Higher score = more relevant
    ranked = sorted(zip(scores, hits, strict=True), key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked[:top_k]]
