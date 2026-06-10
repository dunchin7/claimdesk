"""initial: customers, products, orders, claims + pgvector extension

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-01

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector lives in the same DB; harmless if already created by docker init.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "customers",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("shopify_customer_id", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "extra",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )
    op.create_index("ix_customers_email", "customers", ["email"])
    op.create_index(
        "ix_customers_shopify_customer_id", "customers", ["shopify_customer_id"]
    )

    op.create_table(
        "products",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("sku", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column(
            "specs",
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
        sa.UniqueConstraint("sku", name="uq_products_sku"),
    )
    op.create_index("ix_products_sku", "products", ["sku"])
    op.create_index("ix_products_category", "products", ["category"])

    op.create_table(
        "orders",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("shopify_order_id", sa.String(length=64), nullable=True),
        sa.Column(
            "items",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("purchased_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("total_usd", sa.DECIMAL(precision=10, scale=2), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_orders_shopify_order_id", "orders", ["shopify_order_id"]
    )
    op.create_index("ix_orders_customer_id", "orders", ["customer_id"])

    op.create_table(
        "claims",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("order_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("sku", sa.String(length=64), nullable=True),
        sa.Column("raw_input", sa.Text(), nullable=False),
        sa.Column(
            "photos",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "extracted",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("decision", sa.String(length=32), nullable=True),
        sa.Column("decision_rationale", sa.Text(), nullable=True),
        sa.Column("citation", sa.Text(), nullable=True),
        sa.Column("resolution", sa.String(length=32), nullable=True),
        sa.Column("fraud_score", sa.Float(), nullable=True),
        sa.Column(
            "cost_usd",
            sa.DECIMAL(precision=10, scale=6),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["order_id"], ["orders.id"], ondelete="SET NULL"
        ),
    )
    op.create_index("ix_claims_sku", "claims", ["sku"])
    op.create_index("ix_claims_decision", "claims", ["decision"])
    op.create_index("ix_claims_customer_id", "claims", ["customer_id"])


def downgrade() -> None:
    op.drop_index("ix_claims_customer_id", table_name="claims")
    op.drop_index("ix_claims_decision", table_name="claims")
    op.drop_index("ix_claims_sku", table_name="claims")
    op.drop_table("claims")

    op.drop_index("ix_orders_customer_id", table_name="orders")
    op.drop_index("ix_orders_shopify_order_id", table_name="orders")
    op.drop_table("orders")

    op.drop_index("ix_products_category", table_name="products")
    op.drop_index("ix_products_sku", table_name="products")
    op.drop_table("products")

    op.drop_index(
        "ix_customers_shopify_customer_id", table_name="customers"
    )
    op.drop_index("ix_customers_email", table_name="customers")
    op.drop_table("customers")
