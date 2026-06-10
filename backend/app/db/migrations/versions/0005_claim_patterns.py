"""Week 13: claim_patterns table for semantic memory.

Revision ID: 0005_claim_patterns
Revises: 0004_agent_runs
Create Date: 2026-06-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0005_claim_patterns"
down_revision: str | None = "0004_agent_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.create_table(
        "claim_patterns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("sku", sa.String(length=64), nullable=False),
        sa.Column("n_observed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_approved", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_rejected", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_needs_info", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_fraud", sa.Integer(), server_default="0", nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summary_embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "extra",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("sku", name="uq_claim_patterns_sku"),
    )
    op.create_index("ix_claim_patterns_sku", "claim_patterns", ["sku"])
    op.execute(
        "CREATE INDEX ix_claim_patterns_embedding_hnsw "
        "ON claim_patterns USING hnsw (summary_embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_claim_patterns_embedding_hnsw")
    op.drop_index("ix_claim_patterns_sku", table_name="claim_patterns")
    op.drop_table("claim_patterns")
