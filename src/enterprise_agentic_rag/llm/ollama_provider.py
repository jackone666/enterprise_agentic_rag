"""Ollama LLM provider — local inference via Ollama /api/chat.

Config via env:
    OLLAMA_BASE_URL        (default: http://localhost:11434)
    OLLAMA_MODEL           (default: qwen3:1.7b)
    OLLAMA_TIMEOUT_SECONDS (default: 60)
    OLLAMA_MAX_RETRIES     (default: 2)

Uses the native /api/chat endpoint (not OpenAI-compatible /v1/chat/completions)
so responses are native JSON with token-count metadata.

Raises RuntimeError on connection refused or model-not-found (404).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from enterprise_agentic_rag.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider — real local inference, no API key required."""

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        import os

        self._base_url: str = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._model: str = model or os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        self._timeout: float = timeout_seconds or float(
            os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")
        )
        self._max_retries: int = max_retries if max_retries is not None else int(
            os.getenv("OLLAMA_MAX_RETRIES", "2")
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

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
        return await self._call(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            structured=False,
            schema=None,
        )

    async def structured_generate(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Append JSON instruction so Ollama (with format=json) returns valid JSON.
        enriched_prompt = (
            prompt
            + "\n\n"
            + "Respond in JSON matching this schema: "
            + json.dumps(schema, ensure_ascii=False)
        )
        return await self._call(
            messages=[{"role": "user", "content": enriched_prompt}],
            temperature=temperature,
            max_tokens=2048,
            structured=True,
            schema=schema,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _call(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        structured: bool,
        schema: dict[str, Any] | None,
    ) -> LLMResponse:
        t0 = time.monotonic()
        last_error = ""

        for attempt in range(self._max_retries + 1):
            try:
                return await self._do_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    structured=structured,
                    t0=t0,
                )
            except httpx.ConnectError:
                # No retry for connection refused — clear error message.
                raise RuntimeError(
                    f"Ollama not running at {self._base_url}. "
                    "Start with: ollama serve"
                )
            except httpx.HTTPStatusError as exc:
                last_error = str(exc)
                if exc.response.status_code == 404:
                    raise RuntimeError(
                        f"Model {self._model!r} not found. "
                        f"Run: ollama pull {self._model}"
                    )
                if exc.response.status_code >= 500 and attempt < self._max_retries:
                    wait_s = min(2**attempt, 8)
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1,
                        self._max_retries + 1,
                        last_error,
                        wait_s,
                    )
                    await _async_sleep(wait_s)
                    continue
                raise
            except Exception as exc:
                last_error = str(exc)
                if attempt < self._max_retries:
                    wait_s = min(2**attempt, 8)
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1,
                        self._max_retries + 1,
                        last_error,
                        wait_s,
                    )
                    await _async_sleep(wait_s)
                    continue
                raise

        # Exhausted retries
        raise RuntimeError(
            f"Ollama call failed after {self._max_retries + 1} attempts: {last_error}"
        )

    async def _do_request(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        structured: bool,
        t0: float,
    ) -> LLMResponse:
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, max_retries=0
        ) as client:
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if structured:
                payload["format"] = "json"

            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()

            data = resp.json()
            content = data.get("message", {}).get("content", "")

            return LLMResponse(
                content=content,
                success=True,
                provider="ollama",
                model=self._model,
                latency_ms=(time.monotonic() - t0) * 1000,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio

    await asyncio.sleep(seconds)
