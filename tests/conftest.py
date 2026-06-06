"""pytest configuration — ensures tests use mock providers by default."""

import os

import pytest


@pytest.fixture(autouse=True)
def _set_mock_llm_provider() -> None:
    """Force mock LLM provider for all tests — no real API key needed."""
    os.environ.setdefault("LLM_PROVIDER", "mock")
