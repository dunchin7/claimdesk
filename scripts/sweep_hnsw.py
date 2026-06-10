"""HNSW parameter sweep (Week 6).

Drops and recreates the chunks HNSW index with each (m, ef_construction)
combination, then runs the 30-query retrieval eval at multiple ef_search
values per build. Reports the recall@5 / latency curve so we can pick an
operating point with conviction.

Three knobs that matter:
- `m`               graph density per layer (build-time). Higher = better recall, larger index, slower build.
- `ef_construction` candidates considered when inserting a vector (build-time). Higher = better-quality graph, slower build.
- `ef_search`       candidates considered at query time (query-time, settable per session). Higher = better recall, slower query.

Build params change the graph structure (need a reindex). Query params don't.

Usage:
    uv run python scripts/sweep_hnsw.py
    uv run python scripts/sweep_hnsw.py --model bge  # uses embedding_bge column
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.retrieval.search import vector_search  # noqa: E402


def _fresh_sessionmaker():
    """Per-script engine with NullPool — sidesteps `lru_cache`'d engine's
    pool state when we DDL the HNSW index between query phases.
    """
    engine = create_async_engine(
        get_settings().database_url, poolclass=NullPool, echo=False
    )
    return async_sessionmaker(engine, expire_on_commit=False)

QUERIES_PATH = ROOT / "backend/app/evals/retrieval_queries.json"

# Build-time configurations to compare. pgvector defaults are (16, 64).
BUILD_CONFIGS: list[tuple[int, int]] = [
    (16, 64),    # pgvector default
    (32, 200),   # high-quality build (paper-recommended for serious corpora)
]

# Query-time ef_search values to sweep. pgvector default at runtime is 40.
EF_SEARCH_VALUES: list[int] = [10, 40, 100, 200]


@dataclass
class CellResult:
    m: int
    ef_construction: int
    ef_search: int
    build_time_s: float
    n_queries: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    p50_latency_ms: float
    p95_latency_ms: float
    per_query: list[dict[str, Any]] = field(default_factory=list)


def _hit_is_relevant(text_str: str, expected_substrings: list[str]) -> bool:
    hay = text_str.lower()
    return any(sub.lower() in hay for sub in expected_substrings)


async def _rebuild_index(
    index_name: str, column: str, m: int, ef_construction: int
) -> float:
    """DROP + CREATE the HNSW index with given params. Returns build seconds."""
    sm = _fresh_sessionmaker()
    async with sm() as session:
        # Drop existing index (CONCURRENTLY = no table lock, but harder to
        # parameterize; use plain DROP here — we're in dev).
        await session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
        await session.commit()
        t0 = time.perf_counter()
        await session.execute(
            text(
                f"CREATE INDEX {index_name} ON chunks USING hnsw "
                f"({column} vector_cosine_ops) "
                f"WITH (m = {m}, ef_construction = {ef_construction})"
            )
        )
        await session.commit()
        return time.perf_counter() - t0


async def _run_queries(
    queries: list[dict],
    embedding_model: str,
    ef_search: int,
    chunker: str,
    top_k: int,
) -> tuple[float, float, float, float, float, list[dict[str, Any]]]:
    """Run the 30-query eval at a given ef_search; return aggregate stats."""
    sm = _fresh_sessionmaker()
    latencies: list[float] = []
    per_query: list[dict[str, Any]] = []
    n_with_relevant = 0
    n_relevant_in_topk_total = 0
    rr_total = 0.0

    async with sm() as session:
        # ef_search is a SESSION GUC in pgvector — set it once per connection.
        await session.execute(text(f"SET hnsw.ef_search = {ef_search}"))
        for q in queries:
            t0 = time.perf_counter()
            hits = await vector_search(
                session,
                q["query"],
                top_k=top_k,
                chunker=chunker,
                embedding_model=embedding_model,
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            relevant_flags = [
                _hit_is_relevant(h.text, q["expected_substrings"]) for h in hits
            ]
            n_relevant_in_topk = sum(relevant_flags)
            if n_relevant_in_topk > 0:
                n_with_relevant += 1
            n_relevant_in_topk_total += n_relevant_in_topk

            rr = 0.0
            for r, flag in enumerate(relevant_flags, start=1):
                if flag:
                    rr = 1.0 / r
                    break
            rr_total += rr
            per_query.append(
                {"id": q["id"], "n_relevant": n_relevant_in_topk, "rr": rr}
            )

    n_q = len(queries)
    precision = n_relevant_in_topk_total / max(n_q * top_k, 1)
    recall = n_with_relevant / max(n_q, 1)
    mrr = rr_total / max(n_q, 1)
    p50 = statistics.median(latencies)
    p95 = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20
        else max(latencies)
    )
    return precision, recall, mrr, p50, p95, per_query


async def run(model: str, chunker: str, top_k: int) -> list[CellResult]:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    index_name, column = (
        ("ix_chunks_embedding_hnsw", "embedding")
        if model == "ada"
        else ("ix_chunks_embedding_bge_hnsw", "embedding_bge")
    )

    cells: list[CellResult] = []
    for m, ef_construction in BUILD_CONFIGS:
        print(f"\n[sweep] rebuilding index {index_name}: m={m}, "
              f"ef_construction={ef_construction} ...")
        build_s = await _rebuild_index(index_name, column, m, ef_construction)
        print(f"[sweep]   built in {build_s:.2f}s")

        for ef_search in EF_SEARCH_VALUES:
            precision, recall, mrr, p50, p95, per_q = await _run_queries(
                queries, model, ef_search, chunker, top_k
            )
            cell = CellResult(
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                build_time_s=build_s,
                n_queries=len(queries),
                precision_at_k=round(precision, 3),
                recall_at_k=round(recall, 3),
                mrr=round(mrr, 3),
                p50_latency_ms=round(p50, 1),
                p95_latency_ms=round(p95, 1),
                per_query=per_q,
            )
            cells.append(cell)
            print(
                f"[sweep]   ef_search={ef_search:>3} | "
                f"P@5={precision:.3f}  R@5={recall:.3f}  MRR={mrr:.3f}  "
                f"p50={p50:>5.1f}ms  p95={p95:>5.1f}ms"
            )

    return cells


def print_table(cells: list[CellResult]) -> None:
    print()
    print("=" * 86)
    print(
        f"{'m':>3} {'ef_construction':>16} {'ef_search':>10} "
        f"{'build':>8} {'P@5':>8} {'R@5':>7} {'MRR':>7} {'p50 ms':>8} {'p95 ms':>8}"
    )
    print("-" * 86)
    for c in cells:
        print(
            f"{c.m:>3} {c.ef_construction:>16} {c.ef_search:>10} "
            f"{c.build_time_s:>7.2f}s {c.precision_at_k:>8.3f} "
            f"{c.recall_at_k:>7.3f} {c.mrr:>7.3f} "
            f"{c.p50_latency_ms:>8.1f} {c.p95_latency_ms:>8.1f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ada",
        choices=("ada", "bge"),
        help="Embedding column to sweep against.",
    )
    parser.add_argument("--chunker", default="structural_800")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "data/synthetic/hnsw_sweep_report.json",
    )
    parser.add_argument(
        "--restore-default",
        action="store_true",
        help="At the end, restore the index to pgvector defaults (16, 64).",
    )
    args = parser.parse_args()

    cells = asyncio.run(run(args.model, args.chunker, args.top_k))
    print_table(cells)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    # Drop per_query detail from on-disk JSON.
    summary = [
        {k: v for k, v in c.__dict__.items() if k != "per_query"} for c in cells
    ]
    args.report_json.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[sweep] wrote {args.report_json}")

    if args.restore_default:
        print("[sweep] restoring index to defaults (m=16, ef_construction=64)")
        index_name, column = (
            ("ix_chunks_embedding_hnsw", "embedding")
            if args.model == "ada"
            else ("ix_chunks_embedding_bge_hnsw", "embedding_bge")
        )
        asyncio.run(_rebuild_index(index_name, column, 16, 64))
        print("[sweep] index restored")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
