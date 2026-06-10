"""Langfuse tracing (Week 1 stub → Week 10 full wire-up).

Two integration points:

1. **Direct client** — `Langfuse(...)` instance for explicit `flush()` on
   shutdown and for any custom event/span emission we add later.

2. **LiteLLM callback** — registering `langfuse` on LiteLLM's success/failure
   callback list causes every LiteLLM call (including the ones Instructor
   wraps) to emit a generation event automatically. Model, prompt, response,
   usage, cost, latency — all captured without code at the call site.

When Langfuse keys are unset (the dev default), both paths are no-ops —
the LiteLLM callback registration is skipped, the direct client is None.
Production gets observability with zero config beyond setting the keys.
"""

from __future__ import annotations

from typing import Any

import litellm

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_client: Any | None = None
_litellm_callbacks_registered: bool = False


def _ensure_litellm_env_vars() -> None:
    """LiteLLM's Langfuse callback reads keys from env, not from our settings.
    Mirror our settings into env once so the callback finds them.
    """
    import os
    settings = get_settings()
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    if settings.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)


def _register_litellm_callbacks() -> None:
    """Hook LiteLLM into Langfuse. Idempotent — safe to call multiple times."""
    global _litellm_callbacks_registered
    if _litellm_callbacks_registered:
        return
    # `success_callback` and `failure_callback` are LiteLLM's class-level lists.
    # Adding "langfuse" makes LiteLLM dispatch every call to its Langfuse
    # adapter, which reads env vars and POSTs generation events.
    if "langfuse" not in (litellm.success_callback or []):
        litellm.success_callback = list(litellm.success_callback or []) + ["langfuse"]
    if "langfuse" not in (litellm.failure_callback or []):
        litellm.failure_callback = list(litellm.failure_callback or []) + ["langfuse"]
    _litellm_callbacks_registered = True


def init_tracing() -> None:
    """Initialize Langfuse client + LiteLLM callback. Call at app startup.

    No-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't set.
    """
    global _client
    settings = get_settings()
    if not settings.langfuse_enabled:
        log.info("tracing.disabled", reason="langfuse keys not set")
        _client = None
        return

    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError:
        log.warning("tracing.disabled", reason="langfuse package not installed")
        _client = None
        return

    _ensure_litellm_env_vars()

    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    _register_litellm_callbacks()
    log.info(
        "tracing.enabled",
        host=settings.langfuse_host,
        litellm_callbacks=["langfuse"],
    )


def shutdown_tracing() -> None:
    global _client
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as e:  # noqa: BLE001
        log.warning("tracing.flush_failed", error=str(e))
    _client = None


def get_client() -> Any | None:
    return _client


def trace_metadata(**kwargs: Any) -> dict[str, Any]:
    """Build a metadata dict suitable for passing to LiteLLM calls as
    `metadata={...}`. LiteLLM's Langfuse callback attaches metadata to the
    generation event — useful for filtering by claim_id / prompt_version
    in the Langfuse UI.

    Example:
        await chat(messages=[...], metadata=trace_metadata(
            claim_id=claim_id,
            prompt_version="adjudicate_v3",
            eval_run_id=run_id,
        ))
    """
    return {k: v for k, v in kwargs.items() if v is not None}
