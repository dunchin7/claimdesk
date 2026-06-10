"""Tests for `chat_with_fallback` (Week 16).

We monkey-patch the wrapped `chat()` function so we don't hit Azure.
The fallback path is exercised by making the first call raise a
retryable LiteLLM exception and the second call succeed.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest

from app.ai import llm as llm_module


class _FailFirstChat:
    """Stub for `chat()`: raises a retryable exception on the first alias
    in the order it was called, succeeds on subsequent ones."""

    def __init__(
        self, fail_on_aliases: list[str], return_value: Any = "ok"
    ) -> None:
        self.fail_on_aliases = set(fail_on_aliases)
        self.return_value = return_value
        self.calls: list[str] = []

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        model_alias: str = "extractor",
        **kwargs: Any,
    ) -> Any:
        self.calls.append(model_alias)
        if model_alias in self.fail_on_aliases:
            # Simulate Azure transient failure
            raise litellm.exceptions.APIConnectionError(
                message="simulated transient error",
                llm_provider="azure",
                model=model_alias,
            )
        if kwargs.get("return_cost"):
            return self.return_value, 0.001
        return self.return_value


@pytest.mark.asyncio
async def test_fallback_uses_secondary_when_primary_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FailFirstChat(fail_on_aliases=["reasoner"], return_value="from_secondary")
    monkeypatch.setattr(llm_module, "chat", stub)

    result = await llm_module.chat_with_fallback(
        messages=[{"role": "user", "content": "hi"}],
        aliases=["reasoner", "extractor"],
    )
    assert result == "from_secondary"
    assert stub.calls == ["reasoner", "extractor"]


@pytest.mark.asyncio
async def test_fallback_returns_first_success_no_extra_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When primary works, the chain stops there — no waste."""
    stub = _FailFirstChat(fail_on_aliases=[], return_value="primary_ok")
    monkeypatch.setattr(llm_module, "chat", stub)

    result = await llm_module.chat_with_fallback(
        messages=[{"role": "user", "content": "hi"}],
        aliases=["reasoner", "extractor"],
    )
    assert result == "primary_ok"
    assert stub.calls == ["reasoner"]


@pytest.mark.asyncio
async def test_fallback_raises_after_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FailFirstChat(fail_on_aliases=["reasoner", "extractor"])
    monkeypatch.setattr(llm_module, "chat", stub)

    with pytest.raises(litellm.exceptions.APIConnectionError):
        await llm_module.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            aliases=["reasoner", "extractor"],
        )
    assert stub.calls == ["reasoner", "extractor"]


@pytest.mark.asyncio
async def test_fallback_propagates_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-validation or auth error must NOT trigger fallback —
    it'll fail identically on every alias."""

    class _NonRetryableStub:
        async def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("schema validation failed")

    monkeypatch.setattr(llm_module, "chat", _NonRetryableStub())

    with pytest.raises(ValueError):
        await llm_module.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            aliases=["reasoner", "extractor"],
        )


@pytest.mark.asyncio
async def test_fallback_rejects_empty_aliases() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await llm_module.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}], aliases=[]
        )


@pytest.mark.asyncio
async def test_fallback_with_return_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _FailFirstChat(fail_on_aliases=["reasoner"], return_value="paid")
    monkeypatch.setattr(llm_module, "chat", stub)

    result, cost = await llm_module.chat_with_fallback(
        messages=[{"role": "user", "content": "hi"}],
        aliases=["reasoner", "extractor"],
        return_cost=True,
    )
    assert result == "paid"
    assert cost == pytest.approx(0.001)
