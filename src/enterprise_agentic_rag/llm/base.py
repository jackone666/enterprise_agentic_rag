"""Base LLM provider interface and response types."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""

    content: str = ""
    success: bool = True
    error: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)  # {prompt_tokens, completion_tokens}


class BaseLLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def structured_generate(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a response conforming to a JSON schema."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    def _timed_generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Synchronous wrapper that adds timing (used by mock)."""
        t0 = time.monotonic()
        try:
            result = self._sync_generate(prompt, **kwargs)
            result.latency_ms = (time.monotonic() - t0) * 1000
            return result
        except Exception as e:
            return LLMResponse(
                success=False,
                error=str(e),
                provider=self.provider_name,
                model=self.model_name,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    def _sync_generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Override in sync providers."""
        raise NotImplementedError
