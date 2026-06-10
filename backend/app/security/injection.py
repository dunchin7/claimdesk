"""Prompt-injection defense — post-Week-14 hardening.

Defense in depth, two layers:

1. **Escape closing tags + section sentinels** in untrusted text. Our
   prompts wrap customer input in `<customer_input>...</customer_input>`
   blocks. A customer message containing the closing tag could escape
   the block; sanitizing the input neutralizes that.

2. **Role separation at the chat-message level**. The pipeline now sends
   trusted content (instructions, policy, extraction summary) in the
   `system` role and untrusted content (customer text) in the `user`
   role. Modern instruction-tuned models weight `system` over `user`,
   giving us a structural boundary the natural-language "ignore
   injections" instruction can't.

This module provides the escape helper. Role separation is a `pipeline.py`
change (not a property of input text).

What we don't defend against (out of scope here):
- Attacks on retrieved documents (RAG injection — separate hardening
  needed when we ingest customer-provided PDFs in Week 5+'s real-PDF path)
- LLM models that ignore system role (gpt-4o-mini respects it)
- Multi-turn poisoning (we don't have multi-turn agent state at scale yet)
"""

from __future__ import annotations

import re

# The tag names our prompts use. Closing-tag forms are the main attack
# surface; any of these in raw user text gets sanitized.
_PROMPT_TAGS = (
    "policy",
    "policy_excerpts",
    "claim_extraction",
    "customer_input",
    "photo_descriptions",
    "draft_decision",
    "critic_feedback",
    "synthesized_context",
    "system",
    "user",
    "assistant",
)

# Closing-tag pattern, case-insensitive. We replace, not strip, so an
# operator reading logs can SEE what was attempted.
_CLOSING_TAG_RE = re.compile(
    r"</\s*(" + "|".join(_PROMPT_TAGS) + r")\s*>",
    re.IGNORECASE,
)
_OPENING_TAG_RE = re.compile(
    r"<\s*(" + "|".join(_PROMPT_TAGS) + r")\b[^>]*>",
    re.IGNORECASE,
)

# Common natural-language jailbreak prefixes. We don't filter or block;
# we just flag them in the security_signals output so downstream code
# (Week 15 HITL) can route to human review.
_JAILBREAK_PATTERNS = [
    # "ignore [the/all/any/previous/above/prior] [+up to 3 filler words]
    # instructions/prompt/rules". The {0,3} filler-word window catches
    # "ignore all previous instructions", "ignore any of your earlier
    # instructions", etc.
    re.compile(
        r"\b(ignore|disregard)\b(\s+\w+){0,4}\s+(instructions?|prompt|rules)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"\bnew instructions?\b.{0,40}\b(approve|reject|refund|all claims)\b", re.IGNORECASE),
    re.compile(r"\byou are now\b.{0,40}\b(admin|operator|customer service|approver)\b", re.IGNORECASE),
    # Markdown comment / HTML comment hiding instructions
    re.compile(r"<!--.*-->", re.DOTALL),
]


def escape_user_input(text: str) -> str:
    """Sanitize untrusted text to neutralize tag-injection attacks.

    Replaces matching XML-style tags with a visible marker. The marker
    is intentionally readable so an operator scanning a log can see
    what was tried. Tag content is preserved.
    """
    if not text:
        return text
    sanitized = _CLOSING_TAG_RE.sub(lambda m: f"[REDACTED-CLOSE-{m.group(1).lower()}]", text)
    sanitized = _OPENING_TAG_RE.sub(lambda m: f"[REDACTED-OPEN-{m.group(1).lower()}]", sanitized)
    return sanitized


def detect_injection_signals(text: str) -> list[str]:
    """Return human-readable signals worth flagging on this input.

    Used by the Week-15 router to escalate suspicious claims. Note: a
    customer legitimately discussing 'instructions' (e.g., the bike's
    manual instructions) will trip these — they're signals, not blocks.
    """
    if not text:
        return []
    signals: list[str] = []
    if _CLOSING_TAG_RE.search(text):
        signals.append("closing prompt-tag found in customer text")
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(text):
            signals.append(f"injection pattern: {pattern.pattern[:60]}")
            break  # one match is enough — don't multi-tag the same input
    return signals
