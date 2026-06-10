"""SQLAlchemy 2.0 models.

Week 1 scope: Customer, Product, Order, Claim. The remaining models from §2.4
of the implementation guide (Document, Chunk, AgentRun, ClaimMessage,
ClaimPattern) are added in their respective weeks via fresh Alembic revisions.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DECIMAL,
    TIMESTAMP,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings

# Embedding dim from settings — drives the pgvector column type and any
# downstream comparators.
_EMBEDDING_DIM = get_settings().azure_openai_embedding_dim


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[UUID]:
    return mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    shopify_customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = _created_at()
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    orders: Mapped[list[Order]] = relationship(back_populates="customer")
    claims: Mapped[list[Claim]] = relationship(back_populates="customer")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[UUID] = _uuid_pk()
    sku: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    specs: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    # manual_doc_id: Mapped[UUID | None]  # added in Week 5 with the Document table
    created_at: Mapped[datetime] = _created_at()


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[UUID] = _uuid_pk()
    customer_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    shopify_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    items: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, server_default="[]")
    purchased_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    total_usd: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    created_at: Mapped[datetime] = _created_at()

    customer: Mapped[Customer] = relationship(back_populates="orders")
    claims: Mapped[list[Claim]] = relationship(back_populates="order")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[UUID] = _uuid_pk()
    customer_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    order_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("orders.id", ondelete="SET NULL"), nullable=True
    )
    sku: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    raw_input: Mapped[str] = mapped_column(Text, nullable=False)
    photos: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, server_default="[]")
    extracted: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    decision: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    decision_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(32), nullable=True)

    fraud_score: Mapped[float | None] = mapped_column(nullable=True)
    cost_usd: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 6), nullable=False, server_default="0"
    )
    # agent_run_id: Mapped[UUID | None]  # added in Week 11 with AgentRun

    created_at: Mapped[datetime] = _created_at()
    resolved_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    customer: Mapped[Customer] = relationship(back_populates="claims")
    order: Mapped[Order | None] = relationship(back_populates="claims")


# ---------------------------------------------------------------------------
# Week 5 — Retrieval corpus
# ---------------------------------------------------------------------------


class Document(Base):
    """Source document (markdown or PDF) ingested into the retrieval corpus."""

    __tablename__ = "documents"

    id: Mapped[UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )  # policy | manual | safety | specs
    source_path: Mapped[str] = mapped_column(
        String(512), nullable=False, unique=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    doc_metadata: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    ingested_at: Mapped[datetime] = _created_at()

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A chunk of a Document.

    Multiple chunker outputs can coexist for the same document, distinguished
    by the `chunker` column. The unique constraint
    `(document_id, chunker, chunk_index)` makes ingestion idempotent: re-running
    the same chunker overwrites that chunker's chunks for that document.
    """

    __tablename__ = "chunks"

    id: Mapped[UUID] = _uuid_pk()
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunker: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Contextually-prefixed text that's actually embedded. Equals `text` when
    # contextual retrieval is disabled.
    text_with_context: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(_EMBEDDING_DIM), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Week 6 — open-model parallel column for the BGE ablation. NULL until
    # the backfill script runs; coexists with `embedding` for direct A/B.
    embedding_bge: Mapped[list[float] | None] = mapped_column(
        Vector(get_settings().bge_embedding_dim), nullable=True
    )
    chunk_metadata: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    parent_chunk_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=True,
    )
    # BM25 column populated as a generated column from `text` in the migration.
    bm25_tsv: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunker", "chunk_index", name="uq_chunks_doc_chunker_idx"
        ),
    )


# ---------------------------------------------------------------------------
# Week 11 — Agent runtime
# ---------------------------------------------------------------------------


class AgentRun(Base):
    """A single ReAct-loop execution against a claim.

    `state` is the full step log: [{role, content, tool_calls, tool_call_id, ...}]
    — the same messages we send to the LLM, so the run is fully replayable.
    """

    __tablename__ = "agent_runs"

    id: Mapped[UUID] = _uuid_pk()
    claim_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("claims.id", ondelete="SET NULL"), nullable=True
    )
    # running | completed | failed | cost_capped | iter_capped | escalated
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    started_at: Mapped[datetime] = _created_at()
    completed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    cost_usd: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 6), nullable=False, server_default="0"
    )
    n_iterations: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    final_decision: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full message log — sized for our LLM context, not for unlimited growth.
    state: Mapped[list[dict]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class AgentAction(Base):
    """Side-effecting tool calls — RMAs, escalations, refunds.

    The UNIQUE (action_type, idempotency_key) constraint is the load-bearing
    safety property: re-running a checkpoint that already created an RMA
    returns the same RMA row, doesn't create a duplicate.
    """

    __tablename__ = "agent_actions"

    id: Mapped[UUID] = _uuid_pk()
    action_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        UniqueConstraint(
            "action_type", "idempotency_key", name="uq_agent_actions_type_key"
        ),
    )


# ---------------------------------------------------------------------------
# Week 13 — Semantic memory (claim patterns)
# ---------------------------------------------------------------------------


class ClaimPattern(Base):
    """Aggregated adjudication pattern keyed by SKU.

    Each row summarizes "how have we historically adjudicated claims for
    this SKU?" — decision counts, top fraud patterns observed, and a
    short LLM-generated narrative. The summary is embedded so the
    MemoryConsultant can also find related SKUs via cosine similarity
    when the new claim's SKU has no direct match.

    Per the design doc, this is regenerated by a nightly mining job. For
    now we run `scripts/mine_patterns.py` on demand.
    """

    __tablename__ = "claim_patterns"

    id: Mapped[UUID] = _uuid_pk()
    sku: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    n_observed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_approved: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_rejected: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_needs_info: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_fraud: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    summary_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(_EMBEDDING_DIM), nullable=True
    )

    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Week 15 — Human-in-the-loop operator queue
# ---------------------------------------------------------------------------


class OperatorQueueItem(Base):
    """A claim awaiting human review.

    Enqueued by `app/hitl/router.py:enqueue_if_needed` when the pipeline
    returns route != "auto_resolve". Operators list these via /api/admin/queue,
    then approve / override / escalate. Each terminal action writes an
    AgentAction audit row keyed by (action_type, queue_id) for idempotency.

    `agent_decision` is the source-of-truth from the pipeline at the moment
    of enqueue — frozen even if upstream prompts change. `operator_decision`
    is the human's call (may equal or differ from agent_decision).
    """

    __tablename__ = "operator_queue"

    id: Mapped[UUID] = _uuid_pk()
    claim_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("claims.id", ondelete="SET NULL"), nullable=True
    )
    agent_run_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
    )

    # Routing context captured at enqueue time
    route: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    calibrated_prob: Mapped[float] = mapped_column(nullable=False)

    # pending → in_review → completed/{approved,overridden,escalated}
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="pending", index=True
    )

    agent_decision: Mapped[dict] = mapped_column(JSONB, nullable=False)
    signals: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    enqueued_at: Mapped[datetime] = _created_at()
    claimed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    operator_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    operator_decision: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    operator_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Raw-input copy so operator UI doesn't need to join claims table for
    # the message text. ~10KB max per claim — bounded by ExtractRequest.
    raw_input: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Week 16 — Inbound webhook events (Shopify, Resend, ...)
# ---------------------------------------------------------------------------


class WebhookEvent(Base):
    """One verified inbound webhook delivery.

    The (provider, event_id) UNIQUE constraint enforces dedupe: Shopify
    retries any non-2xx response, so a second delivery of the same event
    must short-circuit. We store the raw payload for audit + replay.
    """

    __tablename__ = "webhook_events"

    id: Mapped[UUID] = _uuid_pk()
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    topic: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(256), nullable=False)
    received_at: Mapped[datetime] = _created_at()
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    processed: Mapped[bool] = mapped_column(
        nullable=False, server_default="false"
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "provider", "event_id", name="uq_webhook_events_provider_eventid"
        ),
    )
