"""Ingest the markdown corpus into pgvector.

Usage:
    # Default: all four chunkers, no contextual retrieval (fast)
    uv run python scripts/ingest_corpus.py

    # Single chunker
    uv run python scripts/ingest_corpus.py --chunker structural

    # With contextual retrieval (one extra LLM call per chunk; ~$0.005 total)
    uv run python scripts/ingest_corpus.py --with-context

    # Custom corpus glob
    uv run python scripts/ingest_corpus.py --glob 'data/manuals/*.md'
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.chunkers import DEFAULT_CHUNKERS, get_chunker  # noqa: E402
from app.retrieval.ingest import ingest_corpus  # noqa: E402

DEFAULT_GLOBS = ("data/policies/*.md", "data/manuals/*.md")


def discover(globs: list[str]) -> list[Path]:
    out: list[Path] = []
    for g in globs:
        out.extend(sorted(ROOT.glob(g)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Glob pattern relative to repo root. May be repeated. "
        f"Default: {list(DEFAULT_GLOBS)}",
    )
    parser.add_argument(
        "--chunker",
        action="append",
        default=None,
        help=f"Chunker name; may be repeated. Default: all of {sorted(DEFAULT_CHUNKERS)}",
    )
    parser.add_argument("--with-context", action="store_true")
    args = parser.parse_args()

    globs = args.glob or list(DEFAULT_GLOBS)
    chunker_names = args.chunker or list(DEFAULT_CHUNKERS.keys())
    chunkers = [get_chunker(n) for n in chunker_names]

    paths = discover(globs)
    if not paths:
        print(f"[ingest] no files matched globs={globs}")
        return 1

    print(f"[ingest] {len(paths)} files × {len(chunkers)} chunkers "
          f"(with_context={args.with_context})")
    for p in paths:
        print(f"  - {p.relative_to(ROOT)}")

    results = asyncio.run(
        ingest_corpus(paths, chunkers, with_context=args.with_context)
    )
    print()
    print(f"{'Document':<48} {'Chunker':<28} {'Chunks':>7}")
    print("-" * 86)
    for r in results:
        relpath = Path(r["path"]).resolve().relative_to(ROOT)
        print(f"{str(relpath):<48} {r['chunker']:<28} {r['n_chunks']:>7}")
    print()
    print(f"[ingest] done — {sum(r['n_chunks'] for r in results)} chunks total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
