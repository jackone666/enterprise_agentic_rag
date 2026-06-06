"""Provider factory — returns the configured LLM provider based on LLM_PROVIDER env.

Supported values:
    openai-compatible → OpenAIProvider (OpenAI / DeepSeek / vLLM / etc.)
    dashscope         → DashScopeProvider (Alibaba Cloud)
    ollama            → OllamaProvider (local inference via Ollama /api/chat)
    mock              → MockLLMProvider (tests only — no API key needed)

Default: openai-compatible (real LLM, raises if not configured).
"""

from __future__ import annotations

import os

from enterprise_agentic_rag.llm.base import BaseLLMProvider


def get_llm_provider() -> BaseLLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "openai-compatible").lower()

    if provider_name in ("openai", "openai-compatible"):
        from enterprise_agentic_rag.llm.openai_provider import OpenAIProvider
        return OpenAIProvider()

    if provider_name in ("dashscope", "qwen"):
        from enterprise_agentic_rag.llm.dashscope_provider import DashScopeProvider
        return DashScopeProvider()

    if provider_name == "mock":
        from enterprise_agentic_rag.llm.mock_provider import MockLLMProvider
        return MockLLMProvider()

    if provider_name == "ollama":
        from enterprise_agentic_rag.llm.ollama_provider import OllamaProvider
        return OllamaProvider()

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider_name}'. "
        f"Supported values: openai-compatible, dashscope, ollama, mock"
    )
