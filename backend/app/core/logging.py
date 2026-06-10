import logging
import sys
from typing import Any

import structlog


# Substring matches against the lowercased key. If any of these appears
# anywhere in a log key, the value is redacted. Substring (not exact) so that
# `azure_openai_embedding_api_key` and `azure_openai_api_key` both match
# `api_key`.
_SENSITIVE_SUBSTRINGS = (
    "api_key",
    "secret",
    "password",
    "token",
    "authorization",
    "dsn",
)


def _redact_sensitive(_: object, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict):
        kl = key.lower()
        if any(s in kl for s in _SENSITIVE_SUBSTRINGS):
            event_dict[key] = "***redacted***"
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog. Call once at process startup."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact_sensitive,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
