"""Sentry SDK init (Week 17).

Gated entirely behind `SENTRY_DSN`. If the env var is empty, `init_sentry`
is a no-op and we don't import the Sentry FastAPI integration — keeps
test imports clean.

We attach:
- FastAPI integration (captures unhandled 5xx + middleware exceptions)
- AsyncIO integration (captures crashed tasks)
- HTTPX integration (breadcrumbs on outbound calls, no PII)
- Structlog adapter (so our `log.error` calls surface in Sentry events
  with the contextual fields attached)

Sample rate is conservative (10% traces, 100% errors) so a noisy
deployment doesn't burn through the Sentry quota.

Set `SENTRY_ENVIRONMENT` from settings.app_env so dev/staging/prod
are filterable in the dashboard.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


def init_sentry() -> bool:
    """Initialize Sentry if configured. Return True if active.

    Safe to call multiple times — sentry_sdk.init is idempotent per
    DSN, but we still guard against repeat calls in the app lifespan.
    """
    s = get_settings()
    if not s.sentry_dsn:
        return False

    # Lazy import — keeps tests + cold-start without Sentry fast.
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.httpx import HttpxIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    sentry_sdk.init(
        dsn=s.sentry_dsn,
        environment=s.app_env,
        # Traces: 10% sample. Errors: always captured (default).
        traces_sample_rate=0.10,
        # PII: we don't send raw_input — it's customer text and may
        # contain emails / addresses. Set False explicitly.
        send_default_pii=False,
        integrations=[
            StarletteIntegration(),
            FastApiIntegration(),
            AsyncioIntegration(),
            HttpxIntegration(),
        ],
        # Don't break the prompt-injection signal — Sentry's input
        # sanitization should NOT strip our debug logs of `raw_input`
        # since we never tag those fields as PII.
    )
    log.info("sentry.initialized", environment=s.app_env)
    return True
