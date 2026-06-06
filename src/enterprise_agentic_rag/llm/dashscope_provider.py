"""DashScope (Alibaba Cloud) LLM provider.

Config via env:
    LLM_API_KEY   (DASHSCOPE_API_KEY)
    LLM_MODEL     (default: qwen-turbo)
    LLM_BASE_URL  (optional — defaults to DashScope compatible endpoint)

Raises ValueError when API key is missing — no silent mock fallback.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from enterprise_agentic_rag.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

DASHSCOPE_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeProvider(BaseLLMProvider):
    """DashScope provider using OpenAI-compatible endpoint — real LLM, no mock fallback."""

    def __init__(self) -> None:
        self._api_key = os.getenv("LLM_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
        self._base_url = os.getenv("LLM_BASE_URL", "") or DASHSCOPE_DEFAULT_BASE
        self._model = os.getenv("LLM_MODEL", "qwen-turbo")
        self._max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

        if not self._is_configured:
            raise ValueError(
                "LLM_API_KEY / DASHSCOPE_API_KEY is missing or still using placeholder value. "
                "For tests, set LLM_PROVIDER=mock."
            )

    @property
    def provider_name(self) -> str:
        return "dashscope"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def _is_configured(self) -> bool:
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
        return await self._call(prompt, temperature, max_tokens)

    async def structured_generate(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return await self._call(prompt, temperature, 2048)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _call(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        from openai import AsyncOpenAI

        t0 = time.monotonic()
        last_error = ""

        for attempt in range(self._max_retries + 1):
            try:
                client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=30.0,
                    max_retries=0,
                )

                resp = await client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""

                return LLMResponse(
                    content=content,
                    success=True,
                    provider="dashscope",
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
                        "DashScope call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, self._max_retries + 1, last_error, wait_s,
                    )
                    import asyncio
                    await asyncio.sleep(wait_s)

        raise RuntimeError(
            f"DashScope call failed after {self._max_retries + 1} attempts: {last_error}"
        )
