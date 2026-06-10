"""LLM provider abstraction.

This module is the **only** place the rest of the app talks to an LLM. Code
elsewhere imports `chat()` or `embed()` and passes a `model_alias`. Swapping
providers later (Azure → Anthropic → Bedrock → local Llama) changes
`MODEL_ALIASES` here and nothing else.

We use TWO Azure resources (chat + embeddings) with different endpoints/keys/
api-versions, so the alias map carries the resource config inline rather than
relying on env-var bridging.

Depends on:
- LiteLLM: provider-agnostic completion/embedding API
- Instructor: Pydantic-typed structured outputs across providers
"""

from __future__ import annotations

from typing import Any, TypedDict, TypeVar

import instructor
import litellm
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class ResourceConfig(TypedDict):
    """Per-call config for LiteLLM. Bound to a model deployment + Azure resource."""

    model: str
    api_key: str
    api_base: str
    api_version: str


def _chat_resource(deployment: str | None = None) -> ResourceConfig:
    s = get_settings()
    return {
        "model": f"azure/{deployment or s.azure_openai_deployment}",
        "api_key": s.azure_openai_api_key,
        "api_base": s.azure_openai_endpoint,
        "api_version": s.azure_openai_api_version,
    }


def _embedding_resource() -> ResourceConfig:
    s = get_settings()
    return {
        "model": f"azure/{s.azure_openai_embedding_deployment}",
        "api_key": s.azure_openai_embedding_api_key,
        "api_base": s.azure_openai_embedding_endpoint,
        "api_version": s.azure_openai_embedding_api_version,
    }


def _model_aliases() -> dict[str, ResourceConfig]:
    """Map of logical alias → resource config.

    Right now `reasoner`, `extractor`, `vision`, and `judge` all collapse to
    the single chat deployment (gpt-4o-mini). When/if a separate gpt-4o is
    provisioned, route the heavier aliases there by passing a different
    deployment name to `_chat_resource()`.
    """
    chat = _chat_resource()
    return {
        "reasoner": chat,
        "extractor": chat,
        "vision": chat,
        "judge": chat,
        "embedder": _embedding_resource(),
    }


_RETRYABLE = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.InternalServerError,
)


def _is_retryable_exception(exc: BaseException) -> bool:
    """True for transient errors worth retrying — direct OR Instructor-wrapped.

    Instructor wraps LiteLLM exceptions in `InstructorRetryException` once
    its own schema-retry budget is exhausted. The wrapped exception bypasses
    our tenacity guard on `_raw_chat` because Instructor calls
    `litellm.acompletion` directly, not via our wrapper. Phase-3 saw 9/10
    errors from this path under concurrency=8 (Azure rate-limit bursts).

    This predicate is used by `_structured_call_with_retry` to recognize
    the wrapped form and retry the whole structured call.
    """
    if isinstance(exc, _RETRYABLE):
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if isinstance(cause, _RETRYABLE):
        return True
    # Instructor wraps with its own type — name-based check avoids importing
    # the symbol (Instructor's public surface moves between minor versions).
    return type(exc).__name__ == "InstructorRetryException"


def cost_of(response: Any) -> float:
    """Best-effort cost estimate for a LiteLLM response, in USD."""
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001
        return 0.0


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
async def _raw_chat(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    return await litellm.acompletion(messages=messages, **kwargs)


async def _structured_call_with_retry(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    max_retries: int,
    max_attempts: int = 5,
    base_wait: float = 4.0,
    max_wait: float = 30.0,
    **call_kwargs: Any,
) -> tuple[Any, Any]:
    """Wrap Instructor's structured-output call with transient-error retry.

    Phase-4 P0c — recovers the 10 rate-limit failures the Phase-3 eval saw.
    The retry on `_raw_chat` doesn't cover this path because Instructor
    bypasses it. We replicate the tenacity behavior here so structured
    calls get the same backoff treatment as raw calls.

    Instructor's own `max_retries` is for SCHEMA violations and stays in
    place — those are forwarded to `create_with_completion`.
    """
    import asyncio
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await client.chat.completions.create_with_completion(
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
                **call_kwargs,
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if not _is_retryable_exception(e) or attempt == max_attempts:
                raise
            wait = min(base_wait * (2 ** (attempt - 1)), max_wait)
            log.warning(
                "llm.structured.retry",
                attempt=attempt,
                max_attempts=max_attempts,
                wait_s=wait,
                error_type=type(e).__name__,
                error=str(e)[:120],
            )
            await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


# Module-level latest-cost tracker. Set by chat() on every call so callers
# that don't want to thread the cost through their return tuple can still
# pick it up. NOT thread-safe; for concurrent eval runs, prefer the
# `return_cost=True` path which threads cost explicitly.
_last_cost_usd: float = 0.0


def get_last_chat_cost_usd() -> float:
    """Return the cost (USD) of the most recent chat() call in this process."""
    return _last_cost_usd


async def chat(
    messages: list[dict[str, Any]],
    model_alias: str = "extractor",
    response_model: type[T] | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_retries: int = 3,
    return_cost: bool = False,
    **kwargs: Any,
) -> T | Any | tuple[T | Any, float]:
    """Send a chat completion.

    If `response_model` is provided, the response is parsed into that Pydantic
    model via Instructor (which retries on schema violations). Otherwise the
    raw LiteLLM response is returned.

    Args:
        messages: list of `{"role": ..., "content": ...}` dicts.
        model_alias: key into MODEL_ALIASES (e.g. "extractor", "reasoner").
        response_model: optional Pydantic model for structured output.
        temperature: default 0.0 (deterministic-ish).
        max_tokens: optional cap.
        max_retries: instructor schema-violation retry budget.
        return_cost: if True, return `(result, cost_usd)`. Otherwise just
            `result` (cost stays available via `get_last_chat_cost_usd()`).
        **kwargs: forwarded to LiteLLM (e.g. `tools`, `tool_choice`).

    Returns:
        Pydantic instance (or raw LiteLLM response) — or a (result, cost) tuple
        when `return_cost=True`.
    """
    global _last_cost_usd
    aliases = _model_aliases()
    if model_alias not in aliases:
        raise ValueError(
            f"Unknown model alias: {model_alias!r}. Known: {sorted(aliases)}"
        )
    config = aliases[model_alias]

    call_kwargs: dict[str, Any] = {**config, "temperature": temperature, **kwargs}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    cost_usd = 0.0
    if response_model is not None:
        # `create_with_completion` returns (parsed_model, raw_litellm_response).
        # The raw response has .usage which `litellm.completion_cost()` reads.
        # We wrap in `_structured_call_with_retry` so Azure rate-limit bursts
        # at high concurrency get backed-off rather than failing the call.
        client = instructor.from_litellm(litellm.acompletion)
        result, raw_completion = await _structured_call_with_retry(
            client,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **call_kwargs,
        )
        cost_usd = cost_of(raw_completion)
        log.debug(
            "llm.chat.structured",
            alias=model_alias,
            schema=response_model.__name__,
            cost_usd=cost_usd,
        )
    else:
        result = await _raw_chat(messages=messages, **call_kwargs)
        cost_usd = cost_of(result)
        log.debug("llm.chat.raw", alias=model_alias, cost_usd=cost_usd)

    _last_cost_usd = cost_usd
    if return_cost:
        return result, cost_usd
    return result


async def chat_with_fallback(
    messages: list[dict[str, Any]],
    *,
    aliases: list[str],
    response_model: type[T] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_retries: int = 3,
    return_cost: bool = False,
    **kwargs: Any,
) -> T | Any | tuple[T | Any, float]:
    """Try each alias in order; fall back on retryable errors.

    Week-16 production hardening. The retryable set is the same as
    `chat()` — Azure rate limits, connection errors, timeouts, 5xx —
    but here we recover by switching aliases instead of just sleeping
    on the same one. Non-retryable errors (auth, bad request, schema
    violation) propagate immediately.

    Today all aliases collapse to the same Azure deployment, so the
    fallback is a no-op in production. Once a separate gpt-4o
    deployment is provisioned, callers can pass `["reasoner-strong",
    "reasoner"]` for a real heavy-then-cheap chain. The infrastructure
    is the load-bearing piece; the model variety lands when quota does.

    Pen-tested via `scripts/verify_fallback.py` which injects a failing
    primary and confirms the secondary serves the request.

    Args:
        aliases: ordered list of model aliases. The first one is tried;
                 on a retryable failure, the next is tried; and so on.
                 If all fail, the last exception is re-raised.
        (others identical to `chat()`)
    """
    if not aliases:
        raise ValueError("aliases must be a non-empty list")

    last_exc: Exception | None = None
    for i, alias in enumerate(aliases):
        try:
            return await chat(
                messages,
                model_alias=alias,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                return_cost=return_cost,
                **kwargs,
            )
        except _RETRYABLE as e:
            last_exc = e
            log.warning(
                "llm.fallback.retryable_failure",
                alias=alias,
                step=i,
                remaining=len(aliases) - i - 1,
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            continue
        except Exception:
            # Non-retryable (auth, schema validation, bad request) — bubble.
            raise

    assert last_exc is not None  # because aliases is non-empty + loop ran
    log.error(
        "llm.fallback.exhausted",
        aliases=aliases,
        error_type=type(last_exc).__name__,
    )
    raise last_exc


async def embed(
    texts: str | list[str], model_alias: str = "embedder"
) -> list[list[float]]:
    """Embed one or more strings.

    Returns a list of vectors. If `texts` is a single string, returns a list
    of length 1. Use `model_alias="embedder"` for the default Azure embedding
    deployment.
    """
    aliases = _model_aliases()
    if model_alias not in aliases:
        raise ValueError(
            f"Unknown model alias: {model_alias!r}. Known: {sorted(aliases)}"
        )
    config = aliases[model_alias]

    if isinstance(texts, str):
        texts = [texts]

    response = await litellm.aembedding(input=texts, **config)
    # LiteLLM normalizes the response shape across providers.
    return [item["embedding"] for item in response["data"]]
