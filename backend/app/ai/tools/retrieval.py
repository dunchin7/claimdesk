"""Retrieval tools (Week 11).

Thin wrappers over `app/retrieval/search.py:vector_search` filtered by
document kind. Two separate tools (policy vs manual) instead of one
generic retriever — gives the model an explicit choice and helps it
reason about *which corpus to query* before *what to query for*.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.ai.tools.registry import ToolSpec, register_tool
from app.db.session import get_sessionmaker
from app.retrieval.search import vector_search


class RetrievePolicyInput(BaseModel):
    query: str = Field(
        description="Natural-language query about warranty coverage, exclusions, "
        "claim procedures, or any policy-defined rule.",
        min_length=3,
    )
    top_k: int = Field(default=5, ge=1, le=15)


class RetrievedChunk(BaseModel):
    source: str
    section: str
    text: str
    similarity: float


class RetrievePolicyOutput(BaseModel):
    status: str = "ok"
    message: str = ""
    query: str = ""
    n_hits: int = 0
    hits: list[RetrievedChunk] = []


async def _retrieve_from_kinds(
    query: str, top_k: int, kinds: list[str], chunker: str = "structural_800"
) -> RetrievePolicyOutput:
    sm = get_sessionmaker()
    async with sm() as session:
        hits = await vector_search(
            session, query, top_k=top_k, chunker=chunker, kinds=kinds
        )
    out_hits = []
    for h in hits:
        heading = " > ".join(h.metadata.get("heading_path") or []) or h.document_title
        out_hits.append(RetrievedChunk(
            source=h.document_title,
            section=heading,
            text=h.text,
            similarity=round(h.cosine_similarity, 3),
        ))
    return RetrievePolicyOutput(
        query=query, n_hits=len(out_hits), hits=out_hits
    )


async def _retrieve_policy(inp: RetrievePolicyInput) -> RetrievePolicyOutput:
    return await _retrieve_from_kinds(inp.query, inp.top_k, kinds=["policy", "safety"])


register_tool(ToolSpec(
    name="retrieve_policy",
    description=(
        "Retrieve the most relevant warranty-policy and safety-document excerpts "
        "for a natural-language query. Use this before adjudicating to ground "
        "your decision in the actual policy text. Returns up to top_k chunks "
        "with their source document, section heading, and text."
    ),
    input_model=RetrievePolicyInput,
    output_model=RetrievePolicyOutput,
    handler=_retrieve_policy,
))


# ---------------------------------------------------------------------------
# retrieve_manual
# ---------------------------------------------------------------------------


class RetrieveManualInput(BaseModel):
    sku: str | None = Field(
        default=None,
        description="Optional SKU to focus the retrieval (e.g., EB-LEVEL-3). "
        "If provided, the query is augmented with the SKU.",
    )
    query: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=15)


async def _retrieve_manual(inp: RetrieveManualInput) -> RetrievePolicyOutput:
    augmented = f"{inp.sku} {inp.query}" if inp.sku else inp.query
    return await _retrieve_from_kinds(augmented, inp.top_k, kinds=["manual", "specs"])


register_tool(ToolSpec(
    name="retrieve_manual",
    description=(
        "Retrieve relevant excerpts from product manuals and SKU specs sheets. "
        "Useful for technical questions about specific models — error codes, "
        "battery specs, motor wattage, etc. Pass `sku` to focus on a specific "
        "product."
    ),
    input_model=RetrieveManualInput,
    output_model=RetrievePolicyOutput,
    handler=_retrieve_manual,
))
