"""Rate limiting (Week 16).

`slowapi` wraps a SlowAPI limiter around FastAPI. We use an in-memory
storage backend (default) — for a single-uvicorn-process deployment that
is fine. If/when we run multiple workers, swap to Redis via
`storage_uri="redis://..."`.

Limits are deliberately generous in dev (200/min) and stricter in prod
(60/min for /process, 30/min for /analyze-photo since vision calls are
3-5× the cost of text-only). The limits are per-IP via
`get_remote_address`. Behind a load balancer / Cloudflare we'd swap that
for an `X-Forwarded-For`-aware extractor.

Admin endpoints are intentionally exempt: those calls come from
authenticated operators on the back-office network, and rate-limiting
the operator queue UI would create a bad UX for triage.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings


def _default_limits() -> list[str]:
    s = get_settings()
    if s.app_env == "prod":
        return ["120/minute"]
    return ["600/minute"]  # dev/test/staging — looser


# One limiter for the whole app; routes opt in via decorator or via
# `@limiter.limit("N/period")`. Endpoint-specific overrides live in the
# route handlers.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=_default_limits(),
    # `headers_enabled=True` requires every rate-limited handler to
    # accept a starlette `Response` arg so slowapi can inject the
    # X-RateLimit-* headers. We keep it off for now — the 429 still
    # carries `Retry-After`, which is the only header clients actually
    # need to back off correctly.
    headers_enabled=False,
    # Strategy: fixed-window is cheaper than moving-window and good enough
    # for abuse prevention.
    strategy="fixed-window",
)

# Specific named limits, referenced from route decorators.
LIMIT_PROCESS = "30/minute"     # full pipeline — Azure quota matters
LIMIT_EXTRACT = "60/minute"     # extraction-only, cheaper
LIMIT_VISION = "20/minute"      # vision is the costliest call
LIMIT_RUN_AGENT = "20/minute"   # multi-call agent loop
LIMIT_WEBHOOK = "300/minute"    # Shopify can burst on backfill; allow it
