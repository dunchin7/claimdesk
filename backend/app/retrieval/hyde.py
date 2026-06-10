"""HyDE — Hypothetical Document Embeddings (Gao et al. 2022).

Vector retrieval suffers from the *query-document asymmetry*: queries are
short and interrogative ("are scratches covered?"), docs are long and
declarative ("Normal wear and tear, including cosmetic blemishes, is
excluded..."). Both encode the same meaning but to different regions of
the embedding space.

HyDE: ask a cheap LLM to write a plausible **answer** to the query, then
embed the hypothetical answer instead of the query. The hypothetical
lives in document space, so cosine to real docs is tighter.

Cost per query: ~$0.0001 (one gpt-4o-mini call ~150 tokens) + the embed
call. Latency: ~400ms extra.
"""

from __future__ import annotations

from app.ai.llm import chat
from app.core.logging import get_logger

log = get_logger(__name__)

_HYDE_PROMPT = """\
You are helping retrieve relevant passages from a warranty / e-bike
support corpus. Given the user's question below, write a single
paragraph (3-6 sentences) that *would plausibly be the passage that
answers it*, in the voice of a warranty policy or product manual. Stay
within the domain (e-bikes, warranties, batteries, electrical faults,
shipping damage, claims process). Do not preface with "Sure" or "Here
is". Output only the paragraph.

User question: {query}
"""


async def hyde_expand(query: str) -> str:
    """Return a hypothetical answer to `query`, in document-voice prose."""
    prompt = _HYDE_PROMPT.format(query=query)
    try:
        resp = await chat(
            messages=[{"role": "user", "content": prompt}],
            model_alias="extractor",  # cheap path; quality is fine
            temperature=0.3,
            max_tokens=200,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("hyde.failed", error=str(e), query=query[:100])
        return query  # graceful fallback: original query
    text = (
        resp["choices"][0]["message"]["content"].strip()
        if isinstance(resp, dict)
        else resp.choices[0].message.content.strip()
    )
    return text or query
