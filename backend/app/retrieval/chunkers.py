"""Document chunkers (Week 5).

Four implementations for retrieval-quality comparison:

- `FixedTokenChunker(size=512, overlap=64)` — token-window slide, ignores semantics
- `SentenceChunker(target_size=512)` — splits on sentence boundaries, packs to ~target
- `StructuralChunker` — splits on markdown headings; one chunk per leaf section
- `HierarchicalChunker` — produces parent (section) + child (paragraph) chunks
  with `parent_index` linking children to parents

Each chunker emits `ChunkOut` objects in document order. The ingest pipeline
consumes these, embeds the (optionally context-prefixed) text, and persists
to the `chunks` table.

Token counts use `tiktoken` (cl100k_base, the OpenAI default for
text-embedding-* and gpt-4o*). For documents that fall outside that
tokenizer's training set (e.g., heavy CJK), the count is approximate but
internally consistent across chunkers.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import tiktoken

# Single shared encoder: cheap to encode/decode, expensive to construct.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _decode(token_ids: list[int]) -> str:
    return _ENCODER.decode(token_ids)


@dataclass
class ChunkOut:
    """Chunker output. The ingest pipeline turns these into `Chunk` rows."""

    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_index: int | None = None  # set by HierarchicalChunker for child chunks


class Chunker(ABC):
    """Abstract base class. Subclasses set `name` and implement `chunk`."""

    name: str

    @abstractmethod
    def chunk(self, text: str, doc_metadata: dict[str, Any] | None = None) -> list[ChunkOut]:
        """Split `text` into chunks. Idempotent and deterministic."""


# ---------------------------------------------------------------------------
# Fixed-token sliding window
# ---------------------------------------------------------------------------


class FixedTokenChunker(Chunker):
    """Slide a fixed-size window with overlap. Ignores content structure.

    This is the strawman / baseline. It's the fastest, requires no parsing,
    and provides a reference floor for the structurally-aware chunkers.
    """

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        if overlap >= size:
            raise ValueError("overlap must be < size")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed_token_{size}_{overlap}"

    def chunk(
        self, text: str, doc_metadata: dict[str, Any] | None = None
    ) -> list[ChunkOut]:
        token_ids = _ENCODER.encode(text)
        chunks: list[ChunkOut] = []
        step = self.size - self.overlap
        idx = 0
        for start in range(0, max(len(token_ids), 1), step):
            window = token_ids[start : start + self.size]
            if not window:
                break
            chunks.append(
                ChunkOut(
                    chunk_index=idx,
                    text=_decode(window).strip(),
                    metadata={
                        "token_start": start,
                        "token_end": start + len(window),
                        "n_tokens": len(window),
                    },
                )
            )
            idx += 1
            if start + self.size >= len(token_ids):
                break
        return chunks


# ---------------------------------------------------------------------------
# Sentence-aware
# ---------------------------------------------------------------------------

# Lightweight English sentence splitter. Tradeoff: correctness on common
# abbreviations vs zero deps. Good enough for our corpus; for production a
# `pysbd` or `spacy` tokenizer would be more robust.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'\(])|(?<=\.)\s*\n+(?=[A-Z\"'\(])"
)


def _split_sentences(text: str) -> list[str]:
    # Preserve paragraph breaks as sentence boundaries.
    pieces = re.split(r"\n{2,}", text.strip())
    sentences: list[str] = []
    for para in pieces:
        para = para.strip()
        if not para:
            continue
        # Within a paragraph, split by punctuation-followed-by-space.
        parts = _SENTENCE_SPLIT_RE.split(para)
        for s in parts:
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


class SentenceChunker(Chunker):
    """Pack sentences greedily until a chunk approaches `target_size` tokens.

    Sentence boundaries preserve more semantic coherence than fixed-window;
    in practice this lifts retrieval recall on prose and hurts it on
    table-heavy content (where sentences are short and noisy).
    """

    def __init__(self, target_size: int = 512, overlap_sentences: int = 1) -> None:
        self.target_size = target_size
        self.overlap_sentences = max(overlap_sentences, 0)
        self.name = f"sentence_{target_size}"

    def chunk(
        self, text: str, doc_metadata: dict[str, Any] | None = None
    ) -> list[ChunkOut]:
        sentences = _split_sentences(text)
        chunks: list[ChunkOut] = []
        current: list[str] = []
        current_tokens = 0
        idx = 0
        for sentence in sentences:
            stoks = _count_tokens(sentence)
            if current and current_tokens + stoks > self.target_size:
                chunks.append(
                    ChunkOut(
                        chunk_index=idx,
                        text=" ".join(current),
                        metadata={
                            "n_sentences": len(current),
                            "n_tokens": current_tokens,
                        },
                    )
                )
                idx += 1
                # Carry over the last N sentences as overlap.
                if self.overlap_sentences:
                    current = current[-self.overlap_sentences :]
                    current_tokens = sum(_count_tokens(s) for s in current)
                else:
                    current = []
                    current_tokens = 0
            current.append(sentence)
            current_tokens += stoks
        if current:
            chunks.append(
                ChunkOut(
                    chunk_index=idx,
                    text=" ".join(current),
                    metadata={
                        "n_sentences": len(current),
                        "n_tokens": current_tokens,
                    },
                )
            )
        return chunks


# ---------------------------------------------------------------------------
# Markdown-structural
# ---------------------------------------------------------------------------

# Heading line: optional whitespace, 1-6 hashes, space, title text.
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")


@dataclass
class _Section:
    level: int
    title: str
    body: list[str]
    heading_path: list[str]


def _parse_markdown_sections(text: str) -> list[_Section]:
    """Walk markdown lines, partitioning into sections by heading.

    Returns leaf sections in document order. Heading path is the breadcrumb
    of titles from H1 down to the section's heading.
    """
    sections: list[_Section] = []
    current_titles: list[str] = []  # stack indexed by level
    current_body: list[str] = []
    current_level = 0
    current_title = ""

    def flush() -> None:
        if current_title or current_body:
            sections.append(
                _Section(
                    level=current_level,
                    title=current_title,
                    body=list(current_body),
                    heading_path=list(current_titles),
                )
            )

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m is None:
            current_body.append(line)
            continue
        # Heading boundary: flush prior section
        flush()
        current_body = []
        new_level = len(m.group("hashes"))
        new_title = m.group("title").strip()
        # Adjust the title stack to this level
        current_titles = current_titles[: new_level - 1] + [new_title]
        current_level = new_level
        current_title = new_title
    flush()
    # Drop empty leading section if present (text before first heading)
    return [s for s in sections if s.title or any(line.strip() for line in s.body)]


class StructuralChunker(Chunker):
    """One chunk per leaf markdown section, with the heading path prepended.

    If a section's body exceeds `max_size_tokens`, it's sub-chunked using a
    sentence packer. This preserves the heading path metadata for every
    sub-chunk so retrieval-time citation can still attribute correctly.
    """

    def __init__(self, max_size_tokens: int = 800) -> None:
        self.max_size_tokens = max_size_tokens
        self.name = f"structural_{max_size_tokens}"
        self._sub_packer = SentenceChunker(target_size=max_size_tokens, overlap_sentences=0)

    def chunk(
        self, text: str, doc_metadata: dict[str, Any] | None = None
    ) -> list[ChunkOut]:
        sections = _parse_markdown_sections(text)
        out: list[ChunkOut] = []
        idx = 0
        for s in sections:
            heading_line = " > ".join(s.heading_path) if s.heading_path else s.title
            body = "\n".join(s.body).strip()
            if not body and not s.title:
                continue
            full = f"# {heading_line}\n\n{body}".strip() if body else f"# {heading_line}"
            if _count_tokens(full) <= self.max_size_tokens:
                out.append(
                    ChunkOut(
                        chunk_index=idx,
                        text=full,
                        metadata={
                            "section_title": s.title,
                            "heading_path": s.heading_path,
                            "section_level": s.level,
                            "n_tokens": _count_tokens(full),
                        },
                    )
                )
                idx += 1
            else:
                # Sub-chunk a long section by sentences, keeping heading_path
                sub_chunks = self._sub_packer.chunk(body)
                for sub in sub_chunks:
                    out.append(
                        ChunkOut(
                            chunk_index=idx,
                            text=f"# {heading_line}\n\n{sub.text}",
                            metadata={
                                "section_title": s.title,
                                "heading_path": s.heading_path,
                                "section_level": s.level,
                                "n_tokens": _count_tokens(sub.text),
                                "is_subchunk": True,
                            },
                        )
                    )
                    idx += 1
        return out


# ---------------------------------------------------------------------------
# Hierarchical (parent + child)
# ---------------------------------------------------------------------------


class HierarchicalChunker(Chunker):
    """Parent (section) + child (paragraph) chunks.

    Children are embedded for precise matching; at retrieval time, the
    matching child's *parent* is returned as context. We emit both parents
    and children in the same `ChunkOut` stream — children carry their
    `parent_index` so the ingest layer can wire `parent_chunk_id` after
    inserting the parent rows.

    The output ordering is: parent_0, child_0_0, child_0_1, ..., parent_1,
    child_1_0, ... — i.e., parent immediately before its children.
    """

    def __init__(
        self,
        parent_max_size: int = 1200,
        child_target_size: int = 256,
    ) -> None:
        self.parent_max_size = parent_max_size
        self.child_target_size = child_target_size
        self.name = f"hierarchical_{parent_max_size}_{child_target_size}"
        self._child_packer = SentenceChunker(
            target_size=child_target_size, overlap_sentences=0
        )

    def chunk(
        self, text: str, doc_metadata: dict[str, Any] | None = None
    ) -> list[ChunkOut]:
        sections = _parse_markdown_sections(text)
        out: list[ChunkOut] = []
        idx = 0
        for s in sections:
            heading_line = " > ".join(s.heading_path) if s.heading_path else s.title
            body = "\n".join(s.body).strip()
            full_parent = f"# {heading_line}\n\n{body}".strip() if body else f"# {heading_line}"
            parent_index = idx
            out.append(
                ChunkOut(
                    chunk_index=parent_index,
                    text=full_parent,
                    metadata={
                        "role": "parent",
                        "section_title": s.title,
                        "heading_path": s.heading_path,
                        "section_level": s.level,
                        "n_tokens": _count_tokens(full_parent),
                    },
                )
            )
            idx += 1
            if not body:
                continue
            children = self._child_packer.chunk(body)
            for sub in children:
                out.append(
                    ChunkOut(
                        chunk_index=idx,
                        text=sub.text,
                        metadata={
                            "role": "child",
                            "section_title": s.title,
                            "heading_path": s.heading_path,
                            "n_tokens": _count_tokens(sub.text),
                        },
                        parent_index=parent_index,
                    )
                )
                idx += 1
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Default chunker presets used by the ingest CLI and the bench script.
DEFAULT_CHUNKERS: dict[str, Chunker] = {
    "fixed_token": FixedTokenChunker(size=512, overlap=64),
    "sentence": SentenceChunker(target_size=512, overlap_sentences=1),
    "structural": StructuralChunker(max_size_tokens=800),
    "hierarchical": HierarchicalChunker(parent_max_size=1200, child_target_size=256),
}


def get_chunker(name: str) -> Chunker:
    if name not in DEFAULT_CHUNKERS:
        raise ValueError(
            f"Unknown chunker {name!r}. Known: {sorted(DEFAULT_CHUNKERS)}"
        )
    return DEFAULT_CHUNKERS[name]
