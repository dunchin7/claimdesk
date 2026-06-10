"""Retrieval for adjudication (Week 5 / RAG).

Given an extracted claim, build a retrieval query and fetch the top-K
relevant chunks from the policy and safety corpus. The adjudicator sees
only these chunks instead of the full policy.

Query construction: we use the extraction's `customer_summary` as the
primary query, augmented with structured signals (failure_mode, claim_type)
when present. This converts the extraction's *concise neutral phrasing*
into a retrieval probe — empirically a better embedding match than the raw
customer text which often has emotional or off-topic content.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.schemas import ClaimExtraction
from app.core.logging import get_logger
from app.db.session import get_sessionmaker
from app.retrieval.search import SearchHit, vector_search
from app.security.injection import escape_user_input

log = get_logger(__name__)

# Active chunker for retrieval. Pinned in registry / pipeline; this constant
# lives here so the retrieve path can stay aligned with the production choice.
RETRIEVAL_CHUNKER = "structural_800"

# Source-doc kinds for two-stage retrieval. Empirically: mixing policy +
# safety with simple top-K caused the safety doc's "needs-info" guidance to
# dominate for clear-defect claims. Splitting per-kind preserves policy
# dominance (the primary source for adjudication) while still allowing
# safety context to break ties on battery-edge cases.
PRIMARY_KINDS = ["policy"]
SECONDARY_KINDS: list[str] = []  # safety doc dominated needs_info — disable
DEFAULT_TOP_K_PRIMARY = 10
DEFAULT_TOP_K_SECONDARY = 0


@dataclass
class RetrievedContext:
    """Aggregated retrieval result handed to the adjudicate prompt."""

    hits: list[SearchHit]
    query: str
    chunker: str

    def render_excerpts(self) -> str:
        """Render top-K hits as XML-tagged excerpts for the prompt.

        Each excerpt is wrapped in <excerpt> with `source` attribute pointing
        to the document so the model can attribute citations correctly.

        Week-16 hardening: chunk text is sanitized through `escape_user_input`
        before rendering. Our policy/manual corpus is trusted today, but
        manufacturer manuals are vendor-supplied content; any future PDF
        could contain structural tags that look like prompt boundaries. The
        sanitizer replaces them with `[REDACTED-…]` markers — harmless to
        an operator reading the trace, but they no longer terminate the
        surrounding `<policy_excerpts>` block.
        """
        if not self.hits:
            return "<excerpt source=\"none\">(no relevant policy excerpts retrieved)</excerpt>"
        out: list[str] = []
        for i, hit in enumerate(self.hits, start=1):
            heading_path = hit.metadata.get("heading_path") or []
            heading = " > ".join(heading_path) if heading_path else hit.document_title
            safe_text = escape_user_input(hit.text)
            out.append(
                f'<excerpt id="{i}" source="{hit.document_title}" '
                f'section="{heading}" similarity="{hit.cosine_similarity:.3f}">\n'
                f"{safe_text}\n"
                f"</excerpt>"
            )
        return "\n\n".join(out)


def _build_query(extraction: ClaimExtraction, raw_input: str) -> str:
    """Construct a retrieval query from the extraction.

    The query is the customer_summary plus key structured signals. We avoid
    the raw_input because it often contains emotional content that dilutes
    the embedding similarity.
    """
    parts: list[str] = [extraction.customer_summary]
    if extraction.failure_mode:
        parts.append(f"failure mode: {extraction.failure_mode}")
    if extraction.claim_type:
        parts.append(f"claim type: {extraction.claim_type}")
    if extraction.severity:
        parts.append(f"severity: {extraction.severity}")
    return " | ".join(parts)


async def retrieve_for_claim(
    extraction: ClaimExtraction,
    raw_input: str,
    *,
    top_k_primary: int = DEFAULT_TOP_K_PRIMARY,
    top_k_secondary: int = DEFAULT_TOP_K_SECONDARY,
    chunker: str = RETRIEVAL_CHUNKER,
) -> RetrievedContext:
    """Two-stage retrieval: top-K primary (policy) + top-K secondary (safety).

    Primary hits are listed first in the excerpts. This biases the
    adjudicator toward the policy's defect-coverage clauses while still
    surfacing safety doc context for battery edge cases.
    """
    query = _build_query(extraction, raw_input)

    sm = get_sessionmaker()
    async with sm() as session:
        primary_hits = await vector_search(
            session, query, top_k=top_k_primary, chunker=chunker, kinds=PRIMARY_KINDS
        )
        secondary_hits: list[SearchHit] = []
        if top_k_secondary > 0 and SECONDARY_KINDS:
            secondary_hits = await vector_search(
                session,
                query,
                top_k=top_k_secondary,
                chunker=chunker,
                kinds=SECONDARY_KINDS,
            )
    hits = primary_hits + secondary_hits

    log.info(
        "adjudicate.retrieve",
        query=query[:200],
        chunker=chunker,
        n_primary=len(primary_hits),
        n_secondary=len(secondary_hits),
        top_similarity=round(hits[0].cosine_similarity, 3) if hits else 0.0,
    )
    return RetrievedContext(hits=hits, query=query, chunker=chunker)
