"""Ingest orchestrator: load → chunk → (optionally) contextualize → embed → persist.

Idempotent at the (document, chunker) level: running the same chunker on the
same document deletes that chunker's old chunks and writes fresh ones. Other
chunkers' chunks for the same document are untouched, which is what the
4-way ablation needs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Chunk, Document
from app.db.session import get_sessionmaker
from app.retrieval.chunkers import Chunker, ChunkOut, get_chunker
from app.retrieval.contextual import add_context
from app.retrieval.embedders import embed_texts, expected_dim
from app.retrieval.loaders import LoadedDocument, load_any

log = get_logger(__name__)


async def _upsert_document(session: AsyncSession, loaded: LoadedDocument) -> Document:
    existing = await session.scalar(
        select(Document).where(Document.source_path == loaded.source_path)
    )
    if existing is not None:
        existing.title = loaded.title
        existing.kind = loaded.kind
        existing.doc_metadata = loaded.metadata
        await session.flush()
        return existing
    doc = Document(
        kind=loaded.kind,
        source_path=loaded.source_path,
        title=loaded.title,
        doc_metadata=loaded.metadata,
    )
    session.add(doc)
    await session.flush()
    return doc


async def _delete_existing_chunks(
    session: AsyncSession, document_id: Any, chunker_name: str
) -> int:
    res = await session.execute(
        delete(Chunk).where(
            Chunk.document_id == document_id, Chunk.chunker == chunker_name
        )
    )
    return res.rowcount or 0


async def _persist_chunks(
    session: AsyncSession,
    document: Document,
    chunker_name: str,
    chunks: list[ChunkOut],
    contexts: dict[int, str],
    embeddings: list[list[float]],
    embedding_model: str,
) -> None:
    # Two-pass insert so children can reference parents by DB id.
    parent_db_ids: dict[int, Any] = {}
    pending_children: list[tuple[ChunkOut, str, list[float]]] = []

    for c, vec in zip(chunks, embeddings, strict=True):
        ctx = contexts.get(c.chunk_index, "")
        text_with_ctx = f"{ctx}\n\n{c.text}".strip() if ctx else c.text
        if c.parent_index is not None:
            pending_children.append((c, text_with_ctx, vec))
            continue
        row = Chunk(
            document_id=document.id,
            chunker=chunker_name,
            chunk_index=c.chunk_index,
            text=c.text,
            text_with_context=text_with_ctx,
            embedding=vec,
            embedding_model=embedding_model,
            chunk_metadata=c.metadata,
            parent_chunk_id=None,
        )
        session.add(row)
        # Don't flush yet; we want the IDs assigned in one round-trip.

    await session.flush()
    # Build parent_index → DB id map for parents we just inserted
    for c in chunks:
        if c.parent_index is None:
            row = await session.scalar(
                select(Chunk).where(
                    Chunk.document_id == document.id,
                    Chunk.chunker == chunker_name,
                    Chunk.chunk_index == c.chunk_index,
                )
            )
            if row is not None:
                parent_db_ids[c.chunk_index] = row.id

    for c, text_with_ctx, vec in pending_children:
        parent_db_id = parent_db_ids.get(c.parent_index) if c.parent_index is not None else None
        row = Chunk(
            document_id=document.id,
            chunker=chunker_name,
            chunk_index=c.chunk_index,
            text=c.text,
            text_with_context=text_with_ctx,
            embedding=vec,
            embedding_model=embedding_model,
            chunk_metadata=c.metadata,
            parent_chunk_id=parent_db_id,
        )
        session.add(row)
    await session.flush()


async def ingest_one(
    path: Path,
    chunker: Chunker | str,
    *,
    with_context: bool = False,
    session: AsyncSession | None = None,
) -> dict[str, Any]:
    """Ingest a single document with one chunker. Returns a summary dict."""
    chunker_obj = chunker if isinstance(chunker, Chunker) else get_chunker(chunker)
    own_session = session is None
    sm = get_sessionmaker() if own_session else None
    # When contextual retrieval is on, store under a distinct tag so a
    # without-context run remains queryable for ablation.
    storage_name = f"{chunker_obj.name}_ctx" if with_context else chunker_obj.name

    async def _do(s: AsyncSession) -> dict[str, Any]:
        loaded = load_any(path)
        document = await _upsert_document(s, loaded)
        deleted = await _delete_existing_chunks(s, document.id, storage_name)

        chunks = chunker_obj.chunk(loaded.text, doc_metadata=loaded.metadata)
        if not chunks:
            log.warning("ingest.empty_chunks", path=str(path), chunker=chunker_obj.name)
            await s.commit()
            return {"path": str(path), "chunker": chunker_obj.name, "n_chunks": 0}

        contexts: dict[int, str] = {}
        if with_context:
            contexts = await add_context(
                chunks, loaded.text, loaded.title, concurrency=8
            )

        # The text we actually embed: with-context if provided, else plain.
        embed_payload = [
            (f"{contexts[c.chunk_index]}\n\n{c.text}" if contexts.get(c.chunk_index) else c.text)
            for c in chunks
        ]
        embeddings = await embed_texts(embed_payload)
        if any(len(v) != expected_dim() for v in embeddings):
            raise RuntimeError(
                "embedding dim mismatch; check AZURE_OPENAI_EMBEDDING_DIM and the model"
            )

        await _persist_chunks(
            s,
            document=document,
            chunker_name=storage_name,
            chunks=chunks,
            contexts=contexts,
            embeddings=embeddings,
            embedding_model=f"azure/{get_settings().azure_openai_embedding_deployment}",
        )
        await s.commit()
        log.info(
            "ingest.done",
            path=str(path),
            chunker=storage_name,
            with_context=with_context,
            n_chunks=len(chunks),
            deleted_old=deleted,
        )
        return {
            "path": str(path),
            "chunker": storage_name,
            "n_chunks": len(chunks),
            "with_context": with_context,
            "deleted_old": deleted,
        }

    if own_session:
        async with sm() as s:  # type: ignore[union-attr]
            return await _do(s)
    return await _do(session)  # type: ignore[arg-type]


async def ingest_corpus(
    paths: Iterable[Path],
    chunkers: list[Chunker | str],
    *,
    with_context: bool = False,
) -> list[dict[str, Any]]:
    """Ingest a list of files with one or more chunkers. Sequential, simple."""
    results: list[dict[str, Any]] = []
    for path in paths:
        for c in chunkers:
            results.append(await ingest_one(path, c, with_context=with_context))
    return results
