"""BM25-style lexical search via Postgres tsvector + ts_rank_cd (Week 7).

The infrastructure was set up in migration 0002:
- `chunks.bm25_tsv` is a `tsvector` populated by a trigger that runs
  `to_tsvector('english', text)` on insert/update of the `text` column.
- A GIN index `ix_chunks_bm25_tsv` makes `@@` queries fast.

Query-side tokenization is the subtle bit. The natural choice
`websearch_to_tsquery` produces *phrase queries* with `<->` adjacency
operators, which is too strict for hyphenated SKUs (`EB-LEVEL-3`) — the
chunks have those tokens interspersed with other words, not adjacent.
`plainto_tsquery` is also too strict — it ANDs every term.

The trick: take `plainto_tsquery`'s output, swap `&` → `|`, and cast back
to `tsquery` directly (without re-parsing through `to_tsquery`, which
would re-expand hyphenated forms back to adjacency). This preserves
Postgres's smart tokenization (stopword removal, stemming, hyphen
splitting) while giving us OR semantics that recall any matching term.
`ts_rank_cd` then ranks chunks by how well they cover the OR'd terms.

This is NOT a true BM25 implementation. The Postgres ranking function is
its own beast. For our scale and use case it's "BM25-like enough" — what
matters is that exact-token matches (SKUs, error codes) get retrieved
where vector search would miss them.
"""

from __future__ import annotations

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.retrieval.search import SearchHit


async def bm25_search(
    session: AsyncSession,
    query: str,
    *,
    top_k: int = 5,
    chunker: str | None = None,
    kinds: list[str] | None = None,
) -> list[SearchHit]:
    """Top-K chunks by `ts_rank_cd` against `bm25_tsv`.

    Returns the same `SearchHit` shape as `vector_search` so the two can
    be RRF-fused without translation. The `cosine_similarity` field on
    the hit is **the BM25 rank**, not a cosine — pragmatic reuse, named
    in the data dict to avoid confusion.
    """
    # OR'd lexemes: see module docstring for why we don't use to_tsquery here.
    sql = """
        WITH q AS (
          SELECT regexp_replace(
                   plainto_tsquery('english', :q)::text,
                   ' & ', ' | ', 'g'
                 )::tsquery AS tq
        )
        SELECT
            c.id, c.document_id, c.chunker, c.chunk_index,
            c.text, c.chunk_metadata,
            ts_rank_cd(c.bm25_tsv, q.tq) AS rank,
            d.kind, d.title, d.source_path
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        CROSS JOIN q
        WHERE q.tq IS NOT NULL
          AND q.tq::text <> ''
          AND c.bm25_tsv @@ q.tq
        {chunker_filter}
        {kinds_filter}
        ORDER BY rank DESC
        LIMIT :top_k
    """
    chunker_filter = "AND c.chunker = :chunker" if chunker is not None else ""
    kinds_filter = "AND d.kind = ANY(:kinds)" if kinds else ""
    sql = sql.format(chunker_filter=chunker_filter, kinds_filter=kinds_filter)

    stmt = text(sql).bindparams(
        bindparam("q", query),
        bindparam("top_k", top_k),
    )
    params: dict = {}
    if chunker is not None:
        params["chunker"] = chunker
    if kinds:
        params["kinds"] = kinds

    rows = (await session.execute(stmt, params)).all()
    return [
        SearchHit(
            chunk_id=str(row.id),
            document_id=str(row.document_id),
            chunker=row.chunker,
            chunk_index=row.chunk_index,
            text=row.text,
            cosine_distance=0.0,  # not meaningful for BM25
            cosine_similarity=float(row.rank),  # repurposed: BM25 rank score
            metadata=row.chunk_metadata or {},
            document_kind=row.kind,
            document_title=row.title,
            document_source=row.source_path,
        )
        for row in rows
    ]
