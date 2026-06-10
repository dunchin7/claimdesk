from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.ai.agents.multi import (
    setup_persistent_checkpointer,
    shutdown_persistent_checkpointer,
)
from app.ai.registry import validate_registry
from app.api import admin, claims, webhooks
from app.core.config import get_settings
from app.core.limits import limiter
from app.core.logging import configure_logging, get_logger
from app.core.sentry import init_sentry
from app.core.tracing import init_tracing, shutdown_tracing


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    init_tracing()
    sentry_on = init_sentry()
    log = get_logger(__name__)
    log.info("app.observability", sentry=sentry_on, langfuse=settings.langfuse_enabled)
    problems = validate_registry()
    if problems:
        for p in problems:
            log.error("registry.invalid", problem=p)
        raise RuntimeError(
            f"Prompt registry validation failed ({len(problems)} problems). "
            "Fix app/ai/prompts/registry.json or the referenced files."
        )
    # Week 15: bring up the durable checkpointer. Best-effort — if Postgres
    # isn't reachable, the multi-agent graph falls back to MemorySaver and
    # we log a warning. The single-shot pipeline (/api/claims/process)
    # doesn't need a checkpointer at all.
    try:
        await setup_persistent_checkpointer()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "multi.checkpointer.postgres_unavailable",
            error=str(e),
            note="multi-agent will use MemorySaver (no durability)",
        )
    log.info("app.startup", env=settings.app_env)
    try:
        yield
    finally:
        try:
            await shutdown_persistent_checkpointer()
        except Exception as e:  # noqa: BLE001
            log.warning("multi.checkpointer.shutdown_failed", error=str(e))
        shutdown_tracing()
        log.info("app.shutdown")


app = FastAPI(
    title="ClaimDesk",
    description="AI claims co-pilot for self-administered warranty programs.",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting (Week 16). The limiter instance is created in
# `app.core.limits`; here we attach it to `app.state` so per-route
# decorators can find it, register the global 429 handler, and install
# the middleware that consumes Request and enforces the limits.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(claims.router, prefix="/api/claims", tags=["claims"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
