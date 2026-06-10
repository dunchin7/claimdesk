"""Week 16: webhook_events table for inbound webhook dedupe + audit.

Stores every verified webhook event so replays return 409 instead of
re-processing. The (provider, event_id) UNIQUE constraint is the
dedupe primitive — Shopify retries deliveries on non-2xx, and we MUST
NOT process the same event twice.

Revision ID: 0007_webhook_events
Revises: 0006_operator_queue
Create Date: 2026-06-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_webhook_events"
down_revision: str | None = "0006_operator_queue"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "webhook_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("topic", sa.String(length=128), nullable=False),
        sa.Column("event_id", sa.String(length=256), nullable=False),
        sa.Column(
            "received_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "processed",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "provider", "event_id", name="uq_webhook_events_provider_eventid"
        ),
    )
    op.create_index(
        "ix_webhook_events_topic", "webhook_events", ["topic"]
    )
    op.create_index(
        "ix_webhook_events_received_at", "webhook_events", ["received_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_webhook_events_received_at", table_name="webhook_events")
    op.drop_index("ix_webhook_events_topic", table_name="webhook_events")
    op.drop_table("webhook_events")
