"""Email transport (Week 16).

Two transports, chosen by env config:

- **ResendTransport** (production) — POSTs to api.resend.com if
  `RESEND_API_KEY` is set. Domain verification is a Resend dashboard
  concern, not ours; we trust the configured `from_email` is valid.
- **LogTransport** (dev / when no key) — writes the would-be email to
  the structured log AND to a sent-emails table-equivalent (audit row
  via `AgentAction`). Lets the operator queue flow be exercised end-
  to-end locally without an outbound dependency.

Idempotency: every `send()` is guarded by `idempotency_key`, written to
`agent_actions` with `action_type="email_sent"`. A second call with the
same key returns the existing row instead of re-sending. This is the
same pattern Week 11 established for RMAs and escalations.

The "hold email until approval" wiring lives in
`app/api/admin.py::approve/override` — those endpoints call
`send_decision_email()` from this module. The pipeline only DRAFTS the
email; it never sends.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import AgentAction

log = get_logger(__name__)


@dataclass
class EmailSendResult:
    sent: bool                # True = transport accepted; False = idempotent no-op
    transport: str            # "resend" | "log"
    message_id: str | None    # provider-assigned id (None for log transport)
    idempotency_key: str


class _EmailTransport(abc.ABC):
    name: str

    @abc.abstractmethod
    async def send_raw(
        self, *, to: str, subject: str, body_text: str, from_email: str
    ) -> str:
        """Send the email; return the provider's message_id."""


class LogTransport(_EmailTransport):
    """Dev / fallback transport — logs the email instead of sending it.

    Production never uses this. It exists so the operator-approval flow
    can be exercised on a laptop without buying a domain or signing up
    for Resend.
    """

    name = "log"

    async def send_raw(
        self, *, to: str, subject: str, body_text: str, from_email: str
    ) -> str:
        log.info(
            "email.log_transport.send",
            to=to,
            from_=from_email,
            subject=subject,
            body_preview=body_text[:200],
        )
        return f"log:{to}:{subject[:20]}"


class ResendTransport(_EmailTransport):
    """Send via api.resend.com."""

    name = "resend"
    _ENDPOINT = "https://api.resend.com/emails"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def send_raw(
        self, *, to: str, subject: str, body_text: str, from_email: str
    ) -> str:
        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.post(
                self._ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps({
                    "from": from_email,
                    "to": [to],
                    "subject": subject,
                    "text": body_text,
                }),
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Resend rejected the send: {resp.status_code} {resp.text[:200]}"
            )
        return resp.json().get("id", "resend:unknown")


def _default_transport() -> _EmailTransport:
    s = get_settings()
    if s.resend_api_key:
        return ResendTransport(api_key=s.resend_api_key)
    return LogTransport()


async def send_email(
    session: AsyncSession,
    *,
    to: str,
    subject: str,
    body_text: str,
    idempotency_key: str,
    from_email: str | None = None,
    transport: _EmailTransport | None = None,
) -> EmailSendResult:
    """Send `body_text` to `to` exactly once per `idempotency_key`.

    Side-effect protected by the (action_type='email_sent', idempotency_key)
    UNIQUE constraint on `agent_actions`. The caller is responsible for
    committing the session.

    Args:
        idempotency_key: a stable per-message identifier. For queue-driven
            sends this is `queue:{queue_id}:email`.
    """
    # Pre-check: have we sent this already?
    existing = (
        await session.execute(
            select(AgentAction).where(
                AgentAction.action_type == "email_sent",
                AgentAction.idempotency_key == idempotency_key,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        log.info(
            "email.idempotent_replay",
            idempotency_key=idempotency_key,
            message_id=(existing.payload or {}).get("message_id"),
        )
        return EmailSendResult(
            sent=False,
            transport=(existing.payload or {}).get("transport", "unknown"),
            message_id=(existing.payload or {}).get("message_id"),
            idempotency_key=idempotency_key,
        )

    t = transport or _default_transport()
    from_addr = from_email or "ClaimDesk <noreply@claimdesk.dev>"

    message_id = await t.send_raw(
        to=to, subject=subject, body_text=body_text, from_email=from_addr
    )

    session.add(AgentAction(
        action_type="email_sent",
        idempotency_key=idempotency_key,
        payload={
            "to": to,
            "from": from_addr,
            "subject": subject,
            "transport": t.name,
            "message_id": message_id,
            "body_len": len(body_text),
        },
    ))
    log.info(
        "email.sent",
        to=to,
        transport=t.name,
        message_id=message_id,
        idempotency_key=idempotency_key,
    )
    return EmailSendResult(
        sent=True, transport=t.name, message_id=message_id,
        idempotency_key=idempotency_key,
    )
