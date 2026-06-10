"""Citation post-validation.

The adjudicator must cite a verbatim quote from the policy. We verify that
the cited string actually appears in the policy text. If it does not, we
downgrade the decision's `confidence` to `low` so downstream code can route
to human review.

Two levels of match:
1. Exact substring (after normalizing whitespace + smart quotes)
2. Fuzzy match (≥0.85 similarity on a sliding window) — only used to flag
   "close-but-not-verbatim" so the operator knows what to look at.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

# Smart quotes / dashes that LLMs love to emit. Map to ASCII before comparing.
_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower whitespace + quote variance so verbatim matches survive."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_TRANSLATION)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class CitationResult:
    """Outcome of validating a citation against a policy text."""

    verbatim: bool          # True iff the citation appears in policy after normalization
    fuzzy_ratio: float      # Best fuzzy match ratio in [0, 1]; 1.0 implies verbatim
    matched_window: str     # The closest substring of the policy (for diagnostics)


def verify_citation(citation: str, *source_texts: str) -> CitationResult:
    """Check whether `citation` appears verbatim in any of the `source_texts`.

    Whitespace differences and smart-quote variants are tolerated. Anything
    else (paraphrase, missing words, added qualifiers) is treated as
    not-verbatim.

    Returns the first source it matches verbatim against; for diagnostics
    on misses, reports the best fuzzy ratio across all sources.
    """
    norm_cite = normalize(citation)
    if not norm_cite:
        return CitationResult(verbatim=False, fuzzy_ratio=0.0, matched_window="")

    if not source_texts:
        return CitationResult(verbatim=False, fuzzy_ratio=0.0, matched_window="")

    best_ratio = 0.0
    best_window = ""
    for src in source_texts:
        norm_src = normalize(src)
        if norm_cite in norm_src:
            return CitationResult(
                verbatim=True, fuzzy_ratio=1.0, matched_window=norm_cite
            )
        # Fuzzy fallback (diagnostic only — never auto-passes)
        cite_len = len(norm_cite)
        step = max(cite_len // 4, 1)
        for i in range(0, max(len(norm_src) - cite_len + 1, 1), step):
            window = norm_src[i : i + cite_len]
            if not window:
                break
            ratio = SequenceMatcher(a=norm_cite, b=window, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_window = window
                if best_ratio >= 0.999:
                    break

    return CitationResult(
        verbatim=False, fuzzy_ratio=round(best_ratio, 3), matched_window=best_window
    )
