"""Prompt registry loader and startup validator.

Validates at app startup that:
- Every active prompt file exists
- Every model_alias is known to the LLM abstraction
- Schema names referenced are importable Pydantic models

Wrong combos throw at startup, not in production traffic.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.ai.llm import _model_aliases
from app.ai.prompt_loader import PROMPTS_DIR

REGISTRY_PATH = PROMPTS_DIR / "registry.json"


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def active_prompt(role: str) -> str:
    """Return the active prompt name for a role (e.g., 'extract')."""
    reg = load_registry()
    return reg["active"][role]["prompt"]


def validate_registry() -> list[str]:
    """Return a list of validation problems. Empty list = healthy."""
    problems: list[str] = []
    reg = load_registry()

    aliases = _model_aliases()
    for role, entry in reg["active"].items():
        # Prompt file exists?
        prompt_path = PROMPTS_DIR / f"{entry['prompt']}.md"
        if not prompt_path.is_file():
            problems.append(
                f"role={role}: prompt file missing — {prompt_path}"
            )
        # Model alias is known?
        if entry["model_alias"] not in aliases:
            problems.append(
                f"role={role}: model_alias {entry['model_alias']!r} not in MODEL_ALIASES"
            )
        # Schema importable (best-effort; only check existence in app.ai.schemas)
        if entry.get("schema"):
            try:
                from app.ai import schemas as _schemas

                if not hasattr(_schemas, entry["schema"]):
                    problems.append(
                        f"role={role}: schema {entry['schema']!r} not in app.ai.schemas"
                    )
            except ImportError as e:
                problems.append(f"role={role}: cannot import schemas module — {e}")

    return problems
