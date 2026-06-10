"""Load and render prompt templates from `app/ai/prompts/`.

Prompts are markdown files named `<purpose>_v<n>.md`. They use Jinja2
placeholders. Loading is LRU-cached because prompts don't change at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template

PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=128)
def load_prompt(name: str) -> str:
    """Load a prompt by name, e.g., 'extract_v1'. No file extension."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **context: Any) -> str:
    """Load a prompt and render it with Jinja2.

    Uses `StrictUndefined` so a missing context variable raises immediately
    rather than rendering an empty string into the model.
    """
    raw = load_prompt(name)
    return Template(raw, undefined=StrictUndefined).render(**context)
