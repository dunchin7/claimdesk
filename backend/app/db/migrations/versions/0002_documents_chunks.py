"""Week 5: documents + chunks (with vector(1536) and tsvector for BM25)

Revision ID: 0002_documents_chunks
Revises: 0001_initial
Create Date: 2026-05-02

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002_documents_chunks"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Embedding dim — keep in sync with `Settings.embedding_dim`.
# Hardcoded here because Alembic migrations should be reproducible across
# environments without runtime settings dependencies.
EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("source_path", sa.String(length=512), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column(
            "doc_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("source_path", name="uq_documents_source_path"),
    )
    op.create_index("ix_documents_kind", "documents", ["kind"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunker", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_with_context", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column(
            "chunk_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("parent_chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("bm25_tsv", postgresql.TSVECTOR(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["parent_chunk_id"], ["chunks.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "document_id", "chunker", "chunk_index", name="uq_chunks_doc_chunker_idx"
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_chunker", "chunks", ["chunker"])

    # HNSW index for cosine similarity on embeddings. We use cosine since
    # ada-002 returns normalized vectors (dot ≈ cosine). m=16, ef_construction=64
    # are pgvector's defaults; the Week-6 ablation will revisit them.
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        """
    )

    # GIN index for BM25 search via tsvector. We populate `bm25_tsv` with a
    # trigger so the application doesn't have to.
    op.execute(
        """
        CREATE INDEX ix_chunks_bm25_tsv ON chunks USING gin (bm25_tsv)
        """
    )

    # Trigger: keep bm25_tsv in sync with text on insert/update.
    op.execute(
        """
        CREATE FUNCTION chunks_bm25_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.bm25_tsv := to_tsvector('english', coalesce(NEW.text, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER chunks_bm25_tsv_update
        BEFORE INSERT OR UPDATE OF text ON chunks
        FOR EACH ROW EXECUTE FUNCTION chunks_bm25_tsv_trigger()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS chunks_bm25_tsv_update ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_bm25_tsv_trigger()")
    op.execute("DROP INDEX IF EXISTS ix_chunks_bm25_tsv")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_index("ix_chunks_chunker", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_kind", table_name="documents")
    op.drop_table("documents")
