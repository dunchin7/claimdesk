from __future__ import annotations

from app.core.logging import _redact_sensitive


def test_redacts_api_key_variants() -> None:
    event = {
        "azure_openai_api_key": "sk-secret",
        "azure_openai_embedding_api_key": "another-secret",
        "api_key": "x",
    }
    out = _redact_sensitive(None, "", event)
    for k in event:
        assert out[k] == "***redacted***"


def test_redacts_secret_password_token_dsn() -> None:
    event = {
        "langfuse_secret_key": "x",
        "user_password": "x",
        "auth_token": "x",
        "sentry_dsn": "x",
        "ok_field": "kept",
    }
    out = _redact_sensitive(None, "", event)
    assert out["langfuse_secret_key"] == "***redacted***"
    assert out["user_password"] == "***redacted***"
    assert out["auth_token"] == "***redacted***"
    assert out["sentry_dsn"] == "***redacted***"
    assert out["ok_field"] == "kept"
