"""Policy text loader.

For Week 3, the policy is a single hand-written markdown file at
`data/policies/policy_v1.md`. Week 5 swaps this for retrieval-grounded
adjudication where chunks are fetched from pgvector — same `Decision`
contract, same citation post-validation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import WORKSPACE_ROOT

POLICY_DIR = WORKSPACE_ROOT / "data" / "policies"


@lru_cache(maxsize=8)
def load_policy(name: str = "policy_v1") -> str:
    """Load a policy by name (no extension). Cached."""
    path = POLICY_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Policy not found: {path}")
    return path.read_text(encoding="utf-8")
