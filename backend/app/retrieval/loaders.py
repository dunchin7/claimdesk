"""Document loaders: markdown (today) + PDF (when real PDFs land).

The loader's only job is to convert a source file to plain text and produce
a metadata dict. Chunking, embedding, and persistence happen downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedDocument:
    text: str
    title: str
    kind: str
    source_path: str
    metadata: dict[str, Any]


def _infer_title(text: str, fallback: str) -> str:
    """First H1 heading of the markdown, else the filename stem."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _infer_kind(source_path: Path) -> str:
    parent = source_path.parent.name.lower()
    if parent == "policies":
        return "policy"
    if parent == "manuals":
        # Sub-classify manuals by filename pattern
        stem = source_path.stem.lower()
        if "ul2849" in stem or "safety" in stem:
            return "safety"
        if "specs" in stem:
            return "specs"
        return "manual"
    return "other"


def load_markdown(path: Path) -> LoadedDocument:
    text = path.read_text(encoding="utf-8")
    return LoadedDocument(
        text=text,
        title=_infer_title(text, path.stem),
        kind=_infer_kind(path),
        source_path=str(path),
        metadata={"format": "markdown", "byte_size": path.stat().st_size},
    )


def load_pdf(path: Path) -> LoadedDocument:
    """Load a PDF. PyMuPDF for text; pdfplumber re-parses table-dense pages.

    A page is considered table-dense if its text has many short, numeric-or-
    short-token lines (heuristic). For our scale (5 docs) this is overkill,
    but the seam is here so real PDFs drop in without a rewrite.
    """
    try:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "PDF loading requires PyMuPDF. Add `pymupdf` to backend/pyproject.toml "
            "if you need to ingest real PDFs."
        ) from e

    doc = fitz.open(path)
    pages_text: list[str] = []
    table_dense_pages: list[int] = []
    for i, page in enumerate(doc):
        page_text = page.get_text("text")
        pages_text.append(page_text)
        # Heuristic: if more than 30% of non-empty lines are short numeric
        # tokens, treat as table-dense.
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        if not lines:
            continue
        short_numeric = sum(1 for ln in lines if len(ln) <= 8 and any(c.isdigit() for c in ln))
        if short_numeric / len(lines) > 0.3:
            table_dense_pages.append(i)

    # If any pages look table-dense, re-extract them with pdfplumber.
    if table_dense_pages:
        try:
            import pdfplumber  # type: ignore[import-untyped]
        except ImportError:
            pdfplumber = None
        if pdfplumber is not None:
            with pdfplumber.open(path) as pdf:
                for i in table_dense_pages:
                    if i >= len(pdf.pages):
                        continue
                    page = pdf.pages[i]
                    tables = page.extract_tables() or []
                    if not tables:
                        continue
                    table_md_blocks: list[str] = []
                    for tbl in tables:
                        rows = ["| " + " | ".join((cell or "").strip() for cell in row) + " |"
                                for row in tbl if row]
                        if rows:
                            sep = "| " + " | ".join("---" for _ in tbl[0]) + " |"
                            table_md_blocks.append("\n".join([rows[0], sep, *rows[1:]]))
                    if table_md_blocks:
                        pages_text[i] = pages_text[i] + "\n\n" + "\n\n".join(table_md_blocks)

    full_text = "\n\n".join(pages_text)
    return LoadedDocument(
        text=full_text,
        title=_infer_title(full_text, path.stem),
        kind=_infer_kind(path),
        source_path=str(path),
        metadata={
            "format": "pdf",
            "n_pages": len(pages_text),
            "table_dense_pages": table_dense_pages,
        },
    )


def load_any(path: Path) -> LoadedDocument:
    """Dispatch by file extension."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return load_markdown(path)
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file extension: {suffix} ({path})")
