"""Retrieval search (Week 5: vector-only; Week 6: + BGE column; Week 7: + BM25 + RRF).

The query path:
    1. Embed the query with the same model that produced the chunk vectors.
    2. Order by cosine distance (`embedding <=> query_vec` in pgvector).
    3. Optionally filter by chunker name (so a single ablation run can pick
       which chunker's chunks it sees).
    4. Optionally filter by document kind / source.

Embedding column is selectable: "ada" (OpenAI ada-002, default) or "bge"
(BAAI/bge-large-en-v1.5, Week-6 open-model comparison). The query embedding
and the column MUST come from the same model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm import embed as embed_ada
from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.retrieval.local_embedders import embed_bge

EmbeddingModel = Literal["ada", "bge"]


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    chunker: str
    chunk_index: int
    text: str
    cosine_distance: float
    cosine_similarity: float
    metadata: dict[str, Any]
    document_kind: str
    document_title: str
    document_source: str


async def _embed_query(query: str, model: EmbeddingModel) -> list[float]:
    """Embed the query with the model that matches the column we'll search.

    ada → OpenAI managed call. bge → local FastEmbed inference.
    """
    if model == "ada":
        [vec] = await embed_ada([query])
        return vec
    # bge — synchronous inference; thin wrapper to keep the async signature.
    return embed_bge([query])[0]


async def vector_search(
    session: AsyncSession,
    query: str,
    *,
    top_k: int = 5,
    chunker: str | None = None,
    kinds: list[str] | None = None,
    embedding_model: EmbeddingModel = "ada",
) -> list[SearchHit]:
    """Embed `query` and return the top-K nearest chunks by cosine distance.

    `chunker` filters to chunks produced by a specific chunker (useful for
    ablation; in production we pin one chunker via the Week-5 winner).
    `embedding_model` picks which column to search:
        - "ada" → `chunks.embedding` (OpenAI text-embedding-ada-002, 1536-dim)
        - "bge" → `chunks.embedding_bge` (BAAI/bge-large-en-v1.5, 1024-dim)
    """
    query_vec = await _embed_query(query, embedding_model)
    col = Chunk.embedding if embedding_model == "ada" else Chunk.embedding_bge

    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunker,
            Chunk.chunk_index,
            Chunk.text,
            Chunk.chunk_metadata,
            col.cosine_distance(query_vec).label("cosine_distance"),
            Document.kind,
            Document.title,
            Document.source_path,
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(col.is_not(None))
        .order_by(text("cosine_distance ASC"))
        .limit(top_k)
    )
    if chunker is not None:
        stmt = stmt.where(Chunk.chunker == chunker)
    if kinds:
        stmt = stmt.where(Document.kind.in_(kinds))

    rows = (await session.execute(stmt)).all()
    return [
        SearchHit(
            chunk_id=str(row.id),
            document_id=str(row.document_id),
            chunker=row.chunker,
            chunk_index=row.chunk_index,
            text=row.text,
            cosine_distance=float(row.cosine_distance),
            cosine_similarity=1.0 - float(row.cosine_distance),
            metadata=row.chunk_metadata or {},
            document_kind=row.kind,
            document_title=row.title,
            document_source=row.source_path,
        )
        for row in rows
    ]


# Convenience wrapper that opens its own session — for one-off use from
# scripts / the bench. The pipeline path uses `vector_search` with an
# injected session.
async def search(
    query: str,
    *,
    top_k: int = 5,
    chunker: str | None = None,
    embedding_model: EmbeddingModel = "ada",
) -> list[SearchHit]:
    from app.db.session import get_sessionmaker

    sm = get_sessionmaker()
    async with sm() as session:
        return await vector_search(
            session,
            query,
            top_k=top_k,
            chunker=chunker,
            embedding_model=embedding_model,
        )


_ = get_settings  # keep import non-dead until we wire dim-aware logic
