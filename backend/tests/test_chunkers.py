from __future__ import annotations

import pytest

from app.retrieval.chunkers import (
    DEFAULT_CHUNKERS,
    FixedTokenChunker,
    HierarchicalChunker,
    SentenceChunker,
    StructuralChunker,
    get_chunker,
)

SAMPLE_MARKDOWN = """# Owner's Manual

## 1. Introduction

This is the introduction paragraph. It has multiple sentences. Some are short. Some are longer and provide context for the reader.

## 2. Specifications

| Field | Value |
|---|---|
| Motor | 500 W |
| Battery | 614 Wh |

## 3. Operation

### 3.1 Powering On
Press the power button for 2 seconds.

### 3.2 Pedal Assist
There are five assist levels.

## 4. Safety

Do not submerge the battery in water.
"""


def test_fixed_token_returns_chunks() -> None:
    chunker = FixedTokenChunker(size=64, overlap=8)
    chunks = chunker.chunk(SAMPLE_MARKDOWN)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)
    # Indices are sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_fixed_token_overlap_validation() -> None:
    with pytest.raises(ValueError):
        FixedTokenChunker(size=100, overlap=100)


def test_sentence_chunker_packs_sentences() -> None:
    chunker = SentenceChunker(target_size=64, overlap_sentences=0)
    chunks = chunker.chunk(SAMPLE_MARKDOWN)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata.get("n_sentences", 0) >= 1


def test_structural_one_chunk_per_section() -> None:
    chunker = StructuralChunker(max_size_tokens=2000)
    chunks = chunker.chunk(SAMPLE_MARKDOWN)
    titles = [c.metadata.get("section_title") for c in chunks]
    # We expect at least these section titles present
    for t in ("1. Introduction", "2. Specifications", "3.1 Powering On", "4. Safety"):
        assert t in titles, f"missing section {t!r} in {titles}"


def test_structural_subchunks_long_sections() -> None:
    # Force sub-chunking by setting max_size very low
    chunker = StructuralChunker(max_size_tokens=20)
    chunks = chunker.chunk(SAMPLE_MARKDOWN)
    # Some chunks should be flagged as sub-chunks
    assert any(c.metadata.get("is_subchunk") for c in chunks), \
        "expected sub-chunks when max_size is small"


def test_hierarchical_emits_parent_child_pairs() -> None:
    chunker = HierarchicalChunker(parent_max_size=2000, child_target_size=80)
    chunks = chunker.chunk(SAMPLE_MARKDOWN)
    parents = [c for c in chunks if c.metadata.get("role") == "parent"]
    children = [c for c in chunks if c.metadata.get("role") == "child"]
    assert parents and children
    # Every child has parent_index set to a parent's chunk_index
    parent_indices = {p.chunk_index for p in parents}
    for ch in children:
        assert ch.parent_index in parent_indices


def test_default_chunkers_registry_complete() -> None:
    expected = {"fixed_token", "sentence", "structural", "hierarchical"}
    assert expected.issubset(DEFAULT_CHUNKERS.keys())
    for name in expected:
        c = get_chunker(name)
        # `name` attribute of each chunker is its parameterized identifier
        assert c.name


def test_get_chunker_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown chunker"):
        get_chunker("does_not_exist")


def test_chunkers_deterministic() -> None:
    chunker = StructuralChunker(max_size_tokens=2000)
    a = chunker.chunk(SAMPLE_MARKDOWN)
    b = chunker.chunk(SAMPLE_MARKDOWN)
    assert [(c.chunk_index, c.text) for c in a] == [(c.chunk_index, c.text) for c in b]
