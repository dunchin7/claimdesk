"""Backfill BGE embeddings on existing chunks (Week 6).

Reads chunks whose `embedding_bge` column is NULL (or all chunks with
`--force`), runs the BGE model on the chunk text, and writes the vectors.
Idempotent and resumable — re-running fills only the rows that still need it.

Usage:
    uv run python scripts/backfill_bge.py            # only NULL rows
    uv run python scripts/backfill_bge.py --force    # everything
    uv run python scripts/backfill_bge.py --chunker structural_800
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select, update  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.models import Chunk  # noqa: E402
from app.db.session import get_sessionmaker  # noqa: E402
from app.retrieval.local_embedders import embed_bge  # noqa: E402


async def run(force: bool, chunker: str | None, batch_size: int) -> int:
    sm = get_sessionmaker()
    settings = get_settings()
    model_tag = f"local/{settings.bge_model_name}"

    async with sm() as session:
        stmt = select(Chunk.id, Chunk.text, Chunk.chunker)
        if not force:
            stmt = stmt.where(Chunk.embedding_bge.is_(None))
        if chunker is not None:
            stmt = stmt.where(Chunk.chunker == chunker)
        stmt = stmt.order_by(Chunk.id)
        rows = (await session.execute(stmt)).all()

    if not rows:
        print("[backfill] nothing to do — all targeted chunks already have BGE embeddings")
        return 0

    print(f"[backfill] embedding {len(rows)} chunks with {settings.bge_model_name}")
    print(f"[backfill] first call downloads ~1.3GB of model weights (one-time)...")

    t0 = time.perf_counter()
    # Embed in batches we control. embed_bge already batches internally, but
    # we want to commit incrementally so a kill mid-run leaves progress.
    n_written = 0
    for i in range(0, len(rows), batch_size):
        slice_rows = rows[i : i + batch_size]
        texts = [r.text for r in slice_rows]
        vectors = embed_bge(texts)

        async with sm() as session:
            for row, vec in zip(slice_rows, vectors, strict=True):
                await session.execute(
                    update(Chunk)
                    .where(Chunk.id == row.id)
                    .values(embedding_bge=vec)
                )
            await session.commit()
        n_written += len(slice_rows)
        elapsed = time.perf_counter() - t0
        rate = n_written / max(elapsed, 1e-6)
        print(
            f"[backfill] {n_written}/{len(rows)} chunks  "
            f"({elapsed:.1f}s elapsed, {rate:.1f} chunks/s)"
        )

    print(f"[backfill] done — {n_written} chunks tagged with {model_tag}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every chunk, including those that already have a BGE vector.",
    )
    parser.add_argument(
        "--chunker",
        default=None,
        help="Restrict to a specific chunker (e.g., structural_800). Default: all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunks per DB commit. 64 is a reasonable default for our scale.",
    )
    args = parser.parse_args()
    return asyncio.run(
        run(force=args.force, chunker=args.chunker, batch_size=args.batch_size)
    )


if __name__ == "__main__":
    raise SystemExit(main())
