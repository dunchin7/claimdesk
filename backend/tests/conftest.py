from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _force_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default APP_ENV to 'test' for the duration of every test."""
    monkeypatch.setenv("APP_ENV", "test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.smoke tests unless Azure env is configured."""
    if not (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")):
        skip_smoke = pytest.mark.skip(
            reason="AZURE_OPENAI_API_KEY/ENDPOINT not set; skipping smoke tests"
        )
        for item in items:
            if "smoke" in item.keywords:
                item.add_marker(skip_smoke)
