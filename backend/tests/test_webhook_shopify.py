"""Unit tests for Shopify HMAC verification.

The HTTP layer (FastAPI endpoint + DB dedupe) is exercised by
`scripts/verify_shopify_webhook.py` against a live database, since
TestClient + async DB is heavier than these unit tests need to be.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

import pytest

from app.webhooks.shopify import (
    KNOWN_TOPICS,
    extract_event_id,
    verify_shopify_hmac,
)


def _sign(body: bytes, secret: str) -> str:
    """Replica of Shopify's signature: base64(HMAC-SHA256(body, secret))."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def test_verify_accepts_matching_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("SHOPIFY_API_SECRET", "test-secret-1234")

    body = b'{"id":42,"email":"a@b.com"}'
    sig = _sign(body, "test-secret-1234")
    assert verify_shopify_hmac(body, sig) is True


def test_verify_rejects_wrong_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("SHOPIFY_API_SECRET", "test-secret-1234")

    body = b'{"id":42}'
    sig = _sign(body, "wrong-secret")
    assert verify_shopify_hmac(body, sig) is False


def test_verify_rejects_when_body_tampered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-byte change in the body must invalidate the signature."""
    from app.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("SHOPIFY_API_SECRET", "test-secret-1234")

    body = b'{"id":42}'
    sig = _sign(body, "test-secret-1234")
    tampered = b'{"id":43}'
    assert verify_shopify_hmac(tampered, sig) is False


def test_verify_rejects_missing_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("SHOPIFY_API_SECRET", "test-secret-1234")

    assert verify_shopify_hmac(b'{}', None) is False
    assert verify_shopify_hmac(b'{}', "") is False


def test_verify_refuses_unconfigured_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no secret is set, verification MUST fail closed."""
    from app.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("SHOPIFY_API_SECRET", "")

    body = b'{"id":42}'
    sig = _sign(body, "")  # empty key still produces a valid HMAC
    assert verify_shopify_hmac(body, sig) is False


def test_verify_handles_unicode_header_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ASCII header (impossible for a real Shopify delivery, but
    a malicious client might send one) must not crash."""
    from app.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("SHOPIFY_API_SECRET", "test-secret-1234")
    assert verify_shopify_hmac(b"{}", "café") is False


def test_extract_event_id_prefers_header() -> None:
    payload = {"id": 999}
    assert extract_event_id(
        "orders/create", payload, "abc-event-id-from-header"
    ) == "abc-event-id-from-header"


def test_extract_event_id_falls_back_to_payload() -> None:
    payload = {"id": 999}
    eid = extract_event_id("orders/create", payload, None)
    assert eid == "orders/create:999"


def test_extract_event_id_handles_empty_payload() -> None:
    eid = extract_event_id("orders/create", {}, None)
    assert eid == "orders/create:unknown"


def test_known_topics_includes_orders_and_customers() -> None:
    assert "orders/create" in KNOWN_TOPICS
    assert "customers/create" in KNOWN_TOPICS
    assert "refunds/create" in KNOWN_TOPICS
