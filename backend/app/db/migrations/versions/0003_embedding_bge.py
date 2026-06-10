"""Week 6: add embedding_bge vector(1024) + HNSW index for BGE comparison

Revision ID: 0003_embedding_bge
Revises: 0002_documents_chunks
Create Date: 2026-05-24

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003_embedding_bge"
down_revision: str | None = "0002_documents_chunks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# BGE-large-en-v1.5 is 1024-dim. Smaller siblings (bge-base 768, bge-small
# 384) would need a different column / migration if we ever ablate them.
BGE_DIM = 1024


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("embedding_bge", Vector(BGE_DIM), nullable=True),
    )
    # HNSW index on the BGE column with the same defaults as the ada column.
    # The 4-way bench reuses the same HNSW params for an apples-to-apples
    # comparison; the Week-6 HNSW sweep operates on the ada column.
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding_bge_hnsw
        ON chunks USING hnsw (embedding_bge vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_bge_hnsw")
    op.drop_column("chunks", "embedding_bge")
