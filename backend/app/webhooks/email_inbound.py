"""Resend inbound-email parser (Week 17 scaffolding).

When a customer replies to a held email, Resend (configured for inbound)
posts the parsed message to our webhook. We need to:

  1. **HMAC-verify** the webhook with `RESEND_WEBHOOK_SECRET` (Resend uses
     `svix-signature` — same Stripe-style `v1,whsec` format, but for now
     we treat it as opaque HMAC and `compare_digest`).
  2. **Extract the thread_id** from one of:
     - Subject suffix `[Claim #<thread_id>]` we set when sending
     - `In-Reply-To` header pointing at our outbound message_id
     - Reply-to subaddress `reply+<thread_id>@claimdesk.dev`
  3. **Look up the matching operator_queue** row and append the reply to
     `operator_notes`, set status back to `pending` so an operator sees
     the inbound text on the queue.

This is scaffolding — actually calling `graph.ainvoke(Command(resume=...))`
requires the multi-agent graph to use `interrupt()`, which it doesn't yet.
The endpoint persists the inbound message and surfaces it; the resume hook
is a one-liner away when the graph gains an interrupt step.

We treat `RESEND_WEBHOOK_SECRET` the same way the Shopify HMAC works: the
secret in our `.env` matches what we registered with Resend. The body must
be verified BEFORE parsing.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class InboundEmail:
    """Normalized inbound — what we extract from a Resend payload."""

    from_email: str
    to_email: str
    subject: str
    text: str
    headers: dict[str, str]
    thread_id: str | None     # parsed from subject or headers; None if none found


# Subject pattern we set on outbound emails so we can round-trip.
# Operator email subject: "Update on your warranty claim [Claim #abc123]"
_THREAD_SUBJECT_RE = re.compile(r"\[Claim #([A-Za-z0-9-]{4,64})\]")

# Subaddress fallback: reply+<thread_id>@claimdesk.dev
_THREAD_TO_RE = re.compile(r"reply\+([A-Za-z0-9-]{4,64})@")


def verify_resend_signature(raw_body: bytes, header_signature: str | None) -> bool:
    """Verify the Resend HMAC signature.

    Resend (via Svix) sends `v1,<base64-hmac-sha256>` in `svix-signature`,
    but for the scaffolding we accept the raw base64 form for both. Failure
    modes match `verify_shopify_hmac` — missing header, missing secret,
    bad base64, or signature mismatch all return False. fail-closed.
    """
    if not header_signature:
        return False
    secret = get_settings().resend_webhook_secret
    if not secret:
        return False

    # Strip the optional `v1,` prefix Resend/Svix uses.
    sig = header_signature.split(",", 1)[-1].strip()

    import base64
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(digest)
    try:
        provided = sig.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(expected, provided)


def parse_inbound(payload: dict[str, Any]) -> InboundEmail:
    """Pull the fields we need from Resend's inbound webhook shape.

    Schema today (subject to change as Resend evolves):
        {
          "type": "email.received",
          "data": {
            "from": "alice@example.com",
            "to": "support@claimdesk.dev",
            "subject": "Re: Update on your warranty claim [Claim #abc123]",
            "text": "Thanks, I have the receipt attached.",
            "headers": [{"name": "In-Reply-To", "value": "<msg-id>@..."}]
          }
        }
    """
    data = payload.get("data") or {}
    headers_list = data.get("headers") or []
    headers = {
        (h.get("name") or "").lower(): (h.get("value") or "")
        for h in headers_list
        if isinstance(h, dict)
    }

    from_email = data.get("from") or ""
    to_email = data.get("to") or ""
    subject = data.get("subject") or ""
    text = data.get("text") or ""

    # Try subject pattern first — that's the one we control on outbound.
    thread_id: str | None = None
    m = _THREAD_SUBJECT_RE.search(subject)
    if m:
        thread_id = m.group(1)
    else:
        m = _THREAD_TO_RE.search(to_email)
        if m:
            thread_id = m.group(1)

    return InboundEmail(
        from_email=from_email,
        to_email=to_email,
        subject=subject,
        text=text,
        headers=headers,
        thread_id=thread_id,
    )
