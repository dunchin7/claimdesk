from __future__ import annotations

import pytest

from app.ai.prompt_loader import load_prompt, render_prompt


def test_load_extract_v1() -> None:
    text = load_prompt("extract_v1")
    assert "extraction assistant" in text.lower()
    assert "{{ customer_text }}" in text


def test_render_extract_v1_with_photos() -> None:
    rendered = render_prompt(
        "extract_v1",
        customer_text="My battery died.",
        photo_descriptions=["close-up of battery", "charger LED"],
    )
    assert "My battery died." in rendered
    assert "close-up of battery" in rendered
    assert "{{" not in rendered  # no unrendered placeholders


def test_render_extract_v1_no_photos() -> None:
    rendered = render_prompt(
        "extract_v1",
        customer_text="Hello.",
        photo_descriptions=[],
    )
    assert "(none provided)" in rendered


def test_missing_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist_v999")


def test_missing_context_var_raises() -> None:
    # StrictUndefined: missing photo_descriptions should error.
    with pytest.raises(Exception):
        render_prompt("extract_v1", customer_text="hi")
