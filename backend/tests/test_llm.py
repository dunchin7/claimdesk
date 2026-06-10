from __future__ import annotations

import pytest

from app.ai.llm import _model_aliases, chat, embed


def test_model_aliases_exist() -> None:
    aliases = _model_aliases()
    for required in ("reasoner", "extractor", "vision", "judge", "embedder"):
        assert required in aliases
        assert aliases[required]["model"].startswith("azure/")
        # Each alias carries its own resource config.
        for k in ("api_key", "api_base", "api_version"):
            assert k in aliases[required]


def test_chat_aliases_share_chat_resource() -> None:
    aliases = _model_aliases()
    chat_aliases = ("reasoner", "extractor", "vision", "judge")
    bases = {aliases[a]["api_base"] for a in chat_aliases}
    assert len(bases) == 1, "All chat aliases should hit the same Azure resource"


def test_embedder_uses_distinct_resource() -> None:
    aliases = _model_aliases()
    # Endpoint env vars may be empty in CI; only assert separation when both set.
    chat_base = aliases["extractor"]["api_base"]
    embed_base = aliases["embedder"]["api_base"]
    if chat_base and embed_base:
        assert chat_base != embed_base, (
            "Chat and embedding resources should be distinct Azure endpoints"
        )


async def test_unknown_chat_alias_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model alias"):
        await chat(messages=[{"role": "user", "content": "hi"}], model_alias="nope")


async def test_unknown_embed_alias_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model alias"):
        await embed("hello", model_alias="nope")
