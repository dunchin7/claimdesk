"""Week 11: agent_runs + agent_actions tables.

Revision ID: 0004_agent_runs
Revises: 0003_embedding_bge
Create Date: 2026-06-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_agent_runs"
down_revision: str | None = "0003_embedding_bge"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "started_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "cost_usd",
            sa.DECIMAL(precision=10, scale=6),
            server_default="0",
            nullable=False,
        ),
        sa.Column("n_iterations", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_tool_calls", sa.Integer(), server_default="0", nullable=False),
        sa.Column("final_decision", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "state",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("trace_id", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(["claim_id"], ["claims.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])
    op.create_index("ix_agent_runs_claim_id", "agent_runs", ["claim_id"])

    op.create_table(
        "agent_actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "action_type", "idempotency_key", name="uq_agent_actions_type_key"
        ),
    )
    op.create_index("ix_agent_actions_type", "agent_actions", ["action_type"])


def downgrade() -> None:
    op.drop_index("ix_agent_actions_type", table_name="agent_actions")
    op.drop_table("agent_actions")
    op.drop_index("ix_agent_runs_claim_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_table("agent_runs")
