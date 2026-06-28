from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _force_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default APP_ENV to 'test' for the duration of every test."""
    monkeypatch.setenv("APP_ENV", "test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.smoke tests unless the OpenAI key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        skip_smoke = pytest.mark.skip(
            reason="OPENAI_API_KEY not set; skipping smoke tests"
        )
        for item in items:
            if "smoke" in item.keywords:
                item.add_marker(skip_smoke)
