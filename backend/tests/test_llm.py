from __future__ import annotations

import pytest

from app.ai.llm import _model_aliases, chat, embed


def test_model_aliases_exist() -> None:
    aliases = _model_aliases()
    for required in ("reasoner", "extractor", "vision", "judge", "embedder"):
        assert required in aliases
        # Each alias carries a concrete model + api_key.
        assert aliases[required]["model"]
        assert "api_key" in aliases[required]


def test_cheap_tier_shares_chat_model() -> None:
    aliases = _model_aliases()
    assert aliases["extractor"]["model"] == aliases["vision"]["model"]


def test_reasoner_and_judge_share_reasoner_tier() -> None:
    aliases = _model_aliases()
    assert aliases["reasoner"]["model"] == aliases["judge"]["model"]


def test_reasoner_defaults_to_chat_model() -> None:
    # With no reasoner override, the frontier tier falls back to the chat
    # model — the whole pipeline is single-model by default.
    aliases = _model_aliases()
    assert aliases["reasoner"]["model"] == aliases["extractor"]["model"]


def test_reasoner_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_REASONER_MODEL", "frontier-model-x")
    try:
        aliases = _model_aliases()
        assert aliases["reasoner"]["model"] == "frontier-model-x"
        assert aliases["judge"]["model"] == "frontier-model-x"
        # Cheap tier is unaffected.
        assert aliases["extractor"]["model"] != "frontier-model-x"
    finally:
        config.get_settings.cache_clear()


def test_embedder_uses_distinct_model() -> None:
    aliases = _model_aliases()
    # The embedder uses a different model than the chat aliases.
    assert aliases["embedder"]["model"] != aliases["extractor"]["model"]


async def test_unknown_chat_alias_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model alias"):
        await chat(messages=[{"role": "user", "content": "hi"}], model_alias="nope")


async def test_unknown_embed_alias_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model alias"):
        await embed("hello", model_alias="nope")
