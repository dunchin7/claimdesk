"""Week 15: operator_queue table for HITL routing.

Revision ID: 0006_operator_queue
Revises: 0005_claim_patterns
Create Date: 2026-06-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_operator_queue"
down_revision: str | None = "0005_claim_patterns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "operator_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("agent_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("route", sa.String(length=32), nullable=False),
        sa.Column("calibrated_prob", sa.Float(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="pending"
        ),
        sa.Column(
            "agent_decision",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "signals",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "enqueued_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("claimed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("operator_id", sa.String(length=128), nullable=True),
        sa.Column(
            "operator_decision",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("operator_notes", sa.Text(), nullable=True),
        sa.Column("raw_input", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["claim_id"], ["claims.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_operator_queue_status", "operator_queue", ["status"])
    op.create_index("ix_operator_queue_route", "operator_queue", ["route"])
    op.create_index(
        "ix_operator_queue_enqueued_at", "operator_queue", ["enqueued_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_operator_queue_enqueued_at", table_name="operator_queue")
    op.drop_index("ix_operator_queue_route", table_name="operator_queue")
    op.drop_index("ix_operator_queue_status", table_name="operator_queue")
    op.drop_table("operator_queue")
