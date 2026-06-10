"""Contextual retrieval (Anthropic, 2024).

For each chunk, prepend a short LLM-generated description of where the chunk
sits in the document. The prefixed text is what we embed; this lifts
retrieval recall by ~15-30% on published benchmarks because the embedding
now captures *position-in-document* signal that bare chunks lack.

Cost: one cheap LLM call per chunk during ingest. With 50 chunks × $0.0001
= $0.005 total for our corpus. Negligible.

Pattern (paraphrased from Anthropic's release):
    "Here is the document: <doc>...</doc>
     Here is a chunk we want to situate: <chunk>...</chunk>
     Give a short, succinct context (1-2 sentences) to situate this chunk
     within the document for retrieval. Answer only with the context."
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.ai.llm import chat
from app.core.logging import get_logger
from app.retrieval.chunkers import ChunkOut

log = get_logger(__name__)

_PROMPT = """\
You are preparing chunks of a document for retrieval indexing. For each
chunk, write a one-sentence context line that situates the chunk in the
document — what section it's from, what topic it covers, and any key
identifiers (model name, error code, section number) a search query might
use to find it.

Do not summarize the chunk's contents. Do not add framing like "This chunk
is...". Output a single sentence under 30 words, in this format:

  From <document title>, <section path>: <topic and identifiers>.

<document_title>{title}</document_title>

<full_document>
{document}
</full_document>

<chunk>
{chunk}
</chunk>

Output the context line only.
"""


async def _context_one(
    chunk: ChunkOut, document_text: str, title: str, max_doc_chars: int
) -> str:
    # Truncate the doc passed in to keep the prompt small. The whole document
    # is given so the LLM can place the chunk; for our 5-page docs the full
    # text fits comfortably.
    doc_payload = document_text[:max_doc_chars]
    prompt = _PROMPT.format(title=title, document=doc_payload, chunk=chunk.text)
    try:
        resp = await chat(
            messages=[{"role": "user", "content": prompt}],
            model_alias="extractor",  # cheap path
            temperature=0.0,
            max_tokens=80,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("contextual.failed", chunk_index=chunk.chunk_index, error=str(e))
        return ""
    text = (
        resp["choices"][0]["message"]["content"].strip()
        if isinstance(resp, dict)
        else resp.choices[0].message.content.strip()
    )
    return text


async def add_context(
    chunks: list[ChunkOut],
    document_text: str,
    title: str,
    *,
    concurrency: int = 8,
    max_doc_chars: int = 32000,
) -> dict[int, str]:
    """Generate context prefixes for each chunk.

    Returns a dict: chunk_index → context line. The ingest pipeline prepends
    `{context}\\n\\n{chunk.text}` before embedding.
    """
    sem = asyncio.Semaphore(concurrency)

    async def gated(c: ChunkOut) -> tuple[int, str]:
        async with sem:
            ctx = await _context_one(c, document_text, title, max_doc_chars)
        return c.chunk_index, ctx

    pairs = await asyncio.gather(*(gated(c) for c in chunks))
    return dict(pairs)
