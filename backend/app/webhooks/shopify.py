"""Shopify webhook verification + handler dispatch.

Shopify signs every webhook delivery with HMAC-SHA256 over the raw request
body, using the shared `Webhooks API secret` from the Shopify admin. The
signature is base64-encoded in the `X-Shopify-Hmac-Sha256` header.

Two security primitives are non-negotiable here:

1. **Verify the HMAC over the RAW body**, not the JSON-decoded body. Any
   re-serialization (e.g., `json.dumps(json.loads(body))`) reorders keys
   and breaks the signature. We accept `bytes` and compute on those bytes.

2. **Use `hmac.compare_digest`**, not `==`. A naive equality check is
   timing-attack vulnerable: it returns as soon as the first mismatched
   byte is found, leaking position info that an attacker can use to brute
   the signature one byte at a time.

If `SHOPIFY_API_SECRET` is empty (dev / not provisioned yet), we refuse
all webhook calls with 503 rather than accepting them. Better to be loud
than to silently accept unsigned requests.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


# Topics we explicitly handle. Anything else is verified, logged, and
# acknowledged (200) — we don't want Shopify to retry forever — but no
# downstream action is taken.
KNOWN_TOPICS = frozenset(
    {
        "orders/create",
        "orders/updated",
        "orders/cancelled",
        "customers/create",
        "customers/update",
        "refunds/create",
    }
)


def verify_shopify_hmac(raw_body: bytes, header_signature: str | None) -> bool:
    """Return True if `header_signature` is a valid Shopify HMAC for `raw_body`.

    Returns False on:
        - missing header
        - missing shared secret (mis-configured server)
        - signature mismatch
        - malformed base64

    Constant-time comparison via `hmac.compare_digest`.
    """
    if not header_signature:
        return False
    secret = get_settings().shopify_api_secret
    if not secret:
        return False

    digest = hmac.new(
        secret.encode("utf-8"), raw_body, hashlib.sha256
    ).digest()
    expected = base64.b64encode(digest)
    try:
        provided = header_signature.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(expected, provided)


def extract_event_id(topic: str, payload: dict[str, Any], header_id: str | None) -> str:
    """Extract a stable dedupe key for the (provider, event_id) UNIQUE.

    Shopify sends `X-Shopify-Webhook-Id` on every delivery — that's the
    canonical event id. We fall back to `<topic>:<payload.id>` for
    payloads we synthesize locally (tests) where the header isn't set.
    """
    if header_id:
        return header_id
    payload_id = payload.get("id") or payload.get("admin_graphql_api_id")
    return f"{topic}:{payload_id}" if payload_id else f"{topic}:unknown"
