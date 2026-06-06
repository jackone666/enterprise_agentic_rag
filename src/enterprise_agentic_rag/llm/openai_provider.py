"""OpenAI-compatible LLM provider.

Config via env:
    LLM_API_KEY          (required — no mock fallback)
    LLM_BASE_URL         (default: https://api.openai.com/v1)
    LLM_MODEL            (default: deepseek-chat)
    LLM_TIMEOUT_SECONDS  (default: 30)
    LLM_MAX_RETRIES      (default: 2)

Raises ValueError when API key is missing.  Raises RuntimeError when
the API call fails after retries — no silent mock fallback.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from enterprise_agentic_rag.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible provider — real LLM, no mock fallback."""

    def __init__(self) -> None:
        self._api_key = os.getenv("LLM_API_KEY", "")
        self._base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self._model = os.getenv("LLM_MODEL", "deepseek-chat")
        self._timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self._max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

        if not self._is_configured:
            raise ValueError(
                "LLM_API_KEY is missing or still using placeholder value. "
                "Set LLM_API_KEY in .env or export it in the environment. "
                "For tests, set LLM_PROVIDER=mock."
            )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def _is_configured(self) -> bool:
        """True when a non-placeholder API key is present."""
        return bool(self._api_key and not self._api_key.startswith("sk-xxx"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        return await self._call_openai(prompt, temperature, max_tokens)

    async def structured_generate(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return await self._call_openai(
            prompt, temperature, 2048, structured=True, schema=schema
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _call_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        structured: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from openai import AsyncOpenAI

        t0 = time.monotonic()
        last_error = ""

        for attempt in range(self._max_retries + 1):
            try:
                client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                    max_retries=0,  # we handle retries ourselves
                )

                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if structured and schema:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = await client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""

                return LLMResponse(
                    content=content,
                    success=True,
                    provider="openai",
                    model=self._model,
                    latency_ms=(time.monotonic() - t0) * 1000,
                    usage={
                        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    },
                )

            except Exception as exc:
                last_error = str(exc)
                if attempt < self._max_retries:
                    wait_s = min(2 ** attempt, 8)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, self._max_retries + 1, last_error, wait_s,
                    )
                    await _async_sleep(wait_s)

        # All retries exhausted
        raise RuntimeError(
            f"LLM call failed after {self._max_retries + 1} attempts: {last_error}"
        )


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)
