"""Hybrid retrieval: vector + BM25 with Reciprocal Rank Fusion (Week 7).

RRF score for a doc d across retrievers R:
    rrf(d) = sum_{r in R, d in r.results} 1 / (k + rank_r(d))

We use k=60 (Cormack et al. 2009 standard). The constant suppresses
the tail (ranks 1-10 dominate the score; ranks 30+ contribute marginally)
and means consistency-across-retrievers beats specialist-in-one.

Why RRF and not weighted score-sum:
- Cosine similarity is on [0, 1]; `ts_rank_cd` can be 0.0001 to 100+.
  Normalizing across runs / queries is fragile.
- RRF uses only rank position. Robust to any underlying scoring scheme.
- Adding a third retriever (e.g., cross-encoder rerank) is one more
  term in the sum — no re-tuning of weights.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.bm25_search import bm25_search
from app.retrieval.search import EmbeddingModel, SearchHit, vector_search

# Standard RRF constant from the original paper. Tuneable but rarely tuned.
RRF_K = 60


@dataclass
class FusedHit:
    """A hit after fusion. Carries the underlying retriever scores so we
    can inspect which retriever surfaced it."""

    hit: SearchHit
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None


def rrf_fusion(
    ranked_lists: Iterable[tuple[str, list[SearchHit]]],
    *,
    k: int = RRF_K,
) -> list[FusedHit]:
    """Fuse multiple ranked SearchHit lists by RRF.

    `ranked_lists` is an iterable of (name, hits) so we can attribute which
    retriever surfaced each fused hit. The `name` is informational; only
    rank matters for the score.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, SearchHit] = {}
    ranks: dict[str, dict[str, int]] = {}

    for name, hits in ranked_lists:
        ranks[name] = {}
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk_id
            ranks[name][cid] = rank
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # Keep the first hit object we saw for each chunk_id — the
            # text/metadata is the same regardless of retriever.
            by_id.setdefault(cid, hit)

    fused = sorted(
        (
            FusedHit(
                hit=by_id[cid],
                rrf_score=score,
                vector_rank=ranks.get("vector", {}).get(cid),
                bm25_rank=ranks.get("bm25", {}).get(cid),
            )
            for cid, score in scores.items()
        ),
        key=lambda f: f.rrf_score,
        reverse=True,
    )
    return fused


async def hybrid_search(
    session: AsyncSession,
    query: str,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    chunker: str | None = None,
    kinds: list[str] | None = None,
    embedding_model: EmbeddingModel = "ada",
) -> list[FusedHit]:
    """Vector top-N ∪ BM25 top-N → RRF → top_k.

    `candidate_k` is the per-retriever pool size (typically 4-10× top_k).
    Larger pools give RRF more material to work with at the cost of a
    bigger BM25/vector query — irrelevant at our scale.
    """
    vec_hits = await vector_search(
        session,
        query,
        top_k=candidate_k,
        chunker=chunker,
        kinds=kinds,
        embedding_model=embedding_model,
    )
    lex_hits = await bm25_search(
        session, query, top_k=candidate_k, chunker=chunker, kinds=kinds
    )
    fused = rrf_fusion([("vector", vec_hits), ("bm25", lex_hits)])
    return fused[:top_k]


RetrievalMode = Literal["vector", "bm25", "hybrid"]
