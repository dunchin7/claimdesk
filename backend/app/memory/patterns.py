"""Semantic memory retrieval (Week 13).

`retrieve_patterns(sku, query_text, top_k)` returns the most relevant
`ClaimPattern` records for a new claim. Two paths:

1. **Exact SKU lookup** (preferred when sku is known and exists in
   `claim_patterns`). Returns the single row — guaranteed-relevant
   historical context for this exact product.

2. **Embedding similarity** (fallback when no SKU or SKU not in patterns).
   Embeds `query_text`, pgvector cosine similarity against
   `summary_embedding`. Returns top_k.

The output is a list of dicts with `summary` + counts that ContextSynthesizer
formats into a "## Past adjudication patterns" prompt block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm import embed
from app.core.logging import get_logger
from app.db.models import ClaimPattern
from app.db.session import get_sessionmaker

log = get_logger(__name__)


@dataclass
class PatternHit:
    sku: str
    n_observed: int
    n_approved: int
    n_rejected: int
    n_needs_info: int
    n_fraud: int
    summary: str
    similarity: float | None  # cosine similarity if found via embedding; None for exact

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "n_observed": self.n_observed,
            "n_approved": self.n_approved,
            "n_rejected": self.n_rejected,
            "n_needs_info": self.n_needs_info,
            "n_fraud": self.n_fraud,
            "summary": self.summary,
            "similarity": self.similarity,
        }


async def _exact_sku(session: AsyncSession, sku: str) -> PatternHit | None:
    row = await session.scalar(
        select(ClaimPattern).where(ClaimPattern.sku == sku)
    )
    if row is None:
        return None
    return PatternHit(
        sku=row.sku,
        n_observed=row.n_observed,
        n_approved=row.n_approved,
        n_rejected=row.n_rejected,
        n_needs_info=row.n_needs_info,
        n_fraud=row.n_fraud,
        summary=row.summary,
        similarity=None,
    )


async def _by_embedding(
    session: AsyncSession, query_text: str, top_k: int
) -> list[PatternHit]:
    [vec] = await embed([query_text[:1000]])
    stmt = (
        select(
            ClaimPattern.sku,
            ClaimPattern.n_observed,
            ClaimPattern.n_approved,
            ClaimPattern.n_rejected,
            ClaimPattern.n_needs_info,
            ClaimPattern.n_fraud,
            ClaimPattern.summary,
            ClaimPattern.summary_embedding.cosine_distance(vec).label("cosine_distance"),
        )
        .where(ClaimPattern.summary_embedding.is_not(None))
        .order_by(text("cosine_distance ASC"))
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        PatternHit(
            sku=r.sku,
            n_observed=r.n_observed,
            n_approved=r.n_approved,
            n_rejected=r.n_rejected,
            n_needs_info=r.n_needs_info,
            n_fraud=r.n_fraud,
            summary=r.summary,
            similarity=round(1.0 - float(r.cosine_distance), 3),
        )
        for r in rows
    ]


async def retrieve_patterns(
    *,
    sku: str | None = None,
    query_text: str | None = None,
    top_k: int = 3,
    session: AsyncSession | None = None,
) -> list[PatternHit]:
    """Retrieve historical patterns relevant to a new claim.

    Strategy:
    - If `sku` is provided and we have an exact-match row, return just that one.
    - If no exact match (or no sku), fall back to embedding search over
      summaries using `query_text`. Returns top_k results.
    - If both `sku` and `query_text` are None, returns [].

    Pass `session` to reuse an open transaction (eval / agent path); else
    we open one ourselves.
    """
    if not sku and not query_text:
        return []

    own_session = session is None
    sm = get_sessionmaker() if own_session else None

    async def _do(s: AsyncSession) -> list[PatternHit]:
        if sku:
            exact = await _exact_sku(s, sku)
            if exact is not None:
                log.debug("memory.exact_sku_hit", sku=sku)
                return [exact]
        if query_text:
            hits = await _by_embedding(s, query_text, top_k)
            log.debug(
                "memory.embedding_hits",
                n=len(hits),
                top_sim=hits[0].similarity if hits else None,
            )
            return hits
        return []

    if own_session:
        async with sm() as s:  # type: ignore[union-attr]
            return await _do(s)
    return await _do(session)  # type: ignore[arg-type]


def render_patterns_block(hits: list[PatternHit]) -> str:
    """Format hits as a prompt-ready '## Past adjudication patterns' block."""
    if not hits:
        return ""
    lines = ["## Past adjudication patterns"]
    for h in hits:
        marker = (
            f"(exact SKU match, n={h.n_observed})"
            if h.similarity is None
            else f"(similarity={h.similarity:.2f}, n={h.n_observed})"
        )
        lines.append(f"### {h.sku} {marker}")
        lines.append(
            f"- counts: approve={h.n_approved}, reject={h.n_rejected}, "
            f"needs_info={h.n_needs_info}, fraud-flagged={h.n_fraud}"
        )
        lines.append(f"- {h.summary}")
        lines.append("")
    return "\n".join(lines).rstrip()
