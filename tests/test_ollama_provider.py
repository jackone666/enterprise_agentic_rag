"""Tests for OllamaProvider.

Run without a real Ollama server — all HTTP interactions are mocked.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from enterprise_agentic_rag.llm.ollama_provider import OllamaProvider


# ===========================================================================
# Helpers
# ===========================================================================
@asynccontextmanager
async def _mock_client(post_fn):
    mock_client = AsyncMock()
    mock_client.post = post_fn
    yield mock_client


# ===========================================================================
# TestProviderName
# ===========================================================================
class TestProviderName:
    def test_provider_name_returns_ollama(self):
        with patch.dict("os.environ", {}, clear=True):
            p = OllamaProvider()
            assert p.provider_name == "ollama"

    def test_model_name_returns_model(self):
        with patch.dict("os.environ", {}, clear=True):
            p = OllamaProvider(model="qwen2:7b")
            assert p.model_name == "qwen2:7b"


# ===========================================================================
# TestInitDefaults
# ===========================================================================
class TestInitDefaults:
    def test_init_uses_default_model(self):
        with patch.dict("os.environ", {}, clear=True):
            p = OllamaProvider()
            assert p._model == "qwen3:1.7b"

    def test_init_uses_default_base_url(self):
        with patch.dict("os.environ", {}, clear=True):
            p = OllamaProvider()
            assert p._base_url == "http://localhost:11434"

    def test_init_uses_default_timeout(self):
        with patch.dict("os.environ", {}, clear=True):
            p = OllamaProvider()
            assert p._timeout == 60.0

    def test_init_uses_default_max_retries(self):
        with patch.dict("os.environ", {}, clear=True):
            p = OllamaProvider()
            assert p._max_retries == 2

    def test_init_reads_env_overrides(self):
        env = {
            "OLLAMA_BASE_URL": "http://custom:9999",
            "OLLAMA_MODEL": "qwen2:7b",
            "OLLAMA_TIMEOUT_SECONDS": "120",
            "OLLAMA_MAX_RETRIES": "5",
        }
        with patch.dict("os.environ", env, clear=True):
            p = OllamaProvider()
            assert p._base_url == "http://custom:9999"
            assert p._model == "qwen2:7b"
            assert p._timeout == 120.0
            assert p._max_retries == 5

    def test_init_constructor_overrides_env(self):
        """Constructor kwargs take precedence over env vars."""
        env = {
            "OLLAMA_MODEL": "env-model",
            "OLLAMA_TIMEOUT_SECONDS": "999",
        }
        with patch.dict("os.environ", env, clear=True):
            p = OllamaProvider(model="constructor-model", timeout_seconds=42.0)
            assert p._model == "constructor-model"
            assert p._timeout == 42.0
            assert p._max_retries == 2  # from env, not overridden


# ===========================================================================
# TestGenerateErrors
# ===========================================================================
class TestGenerateErrors:
    @pytest.mark.asyncio
    async def test_generate_raises_runtimeerror_on_connection_refused(self):
        async def failing_post(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        @asynccontextmanager
        async def ctx_mgr(*args, **kwargs):
            mc = AsyncMock()
            mc.post = failing_post
            yield mc

        with patch("httpx.AsyncClient", ctx_mgr):
            with patch.dict("os.environ", {}, clear=True):
                p = OllamaProvider()
                with pytest.raises(RuntimeError) as exc_info:
                    await p.generate("hello")
                assert "ollama serve" in str(exc_info.value)
                assert "http://localhost:11434" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_raises_runtimeerror_on_404(self):
        async def not_found_post(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 404
            raise httpx.HTTPStatusError(
                "model not found",
                request=MagicMock(),
                response=resp,
            )

        @asynccontextmanager
        async def ctx_mgr(*args, **kwargs):
            mc = AsyncMock()
            mc.post = not_found_post
            yield mc

        with patch("httpx.AsyncClient", ctx_mgr):
            with patch.dict("os.environ", {}, clear=True):
                p = OllamaProvider(model="llama3")
                with pytest.raises(RuntimeError) as exc_info:
                    await p.generate("hello")
                assert "ollama pull llama3" in str(exc_info.value)


# ===========================================================================
# TestGenerateSuccess
# ===========================================================================
class TestGenerateSuccess:
    @pytest.mark.asyncio
    async def test_generate_succeeds_on_200(self):
        ollama_resp = {
            "model": "qwen3:1.7b",
            "message": {"role": "assistant", "content": "Hello from Ollama"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 8,
        }

        async def ok_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = lambda: ollama_resp
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        @asynccontextmanager
        async def ctx_mgr(*args, **kwargs):
            mc = AsyncMock()
            mc.post = ok_post
            yield mc

        with patch("httpx.AsyncClient", ctx_mgr):
            with patch.dict("os.environ", {}, clear=True):
                p = OllamaProvider()
                result = await p.generate("hello", temperature=0.7, max_tokens=512)

        assert result.success is True
        assert result.content == "Hello from Ollama"
        assert result.provider == "ollama"
        assert result.model == "qwen3:1.7b"
        assert result.latency_ms >= 0
        assert result.usage["prompt_tokens"] == 5
        assert result.usage["completion_tokens"] == 8

    @pytest.mark.asyncio
    async def test_generate_extracts_usage_from_ollama_response(self):
        ollama_resp = {
            "message": {"content": "answer"},
            "prompt_eval_count": 123,
            "eval_count": 456,
        }

        async def ok_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = lambda: ollama_resp
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        @asynccontextmanager
        async def ctx_mgr(*args, **kwargs):
            mc = AsyncMock()
            mc.post = ok_post
            yield mc

        with patch("httpx.AsyncClient", ctx_mgr):
            with patch.dict("os.environ", {}, clear=True):
                p = OllamaProvider()
                result = await p.generate("prompt")

        assert result.usage["prompt_tokens"] == 123
        assert result.usage["completion_tokens"] == 456


# ===========================================================================
# TestStructuredGenerate
# ===========================================================================
class TestStructuredGenerate:
    @pytest.mark.asyncio
    async def test_structured_generate_appends_json_instruction(self):
        captured_payload: dict = {}

        async def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json") or {})
            mock_resp = MagicMock()
            mock_resp.json = lambda: {"message": {"content": '{"name": "Alice"}'}}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        @asynccontextmanager
        async def ctx_mgr(*args, **kwargs):
            mc = AsyncMock()
            mc.post = capture_post
            yield mc

        with patch("httpx.AsyncClient", ctx_mgr):
            with patch.dict("os.environ", {}, clear=True):
                p = OllamaProvider()
                schema = {"type": "object", "properties": {"name": {"type": "string"}}}
                await p.structured_generate("Who is Alice?", schema, temperature=0.5)

        assert "Respond in JSON" in captured_payload["messages"][0]["content"]
        assert captured_payload["format"] == "json"


# ===========================================================================
# TestRetryLogic
# ===========================================================================
class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_generate_retries_on_transient_error(self):
        call_count = 0

        async def retry_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.status_code = 500
                raise httpx.HTTPStatusError(
                    "internal error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            mock_resp = MagicMock()
            mock_resp.json = lambda: {"message": {"content": "success after retry"}}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        @asynccontextmanager
        async def ctx_mgr(*args, **kwargs):
            mc = AsyncMock()
            mc.post = retry_then_ok
            yield mc

        with patch("httpx.AsyncClient", ctx_mgr):
            with patch.dict("os.environ", {}, clear=True):
                p = OllamaProvider(max_retries=2)
            with patch(
                "enterprise_agentic_rag.llm.ollama_provider._async_sleep",
                AsyncMock(),
            ):
                result = await p.generate("hello")

        assert result.success is True
        assert result.content == "success after retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_does_not_retry_connection_refused(self):
        """Connection refused is not retried — it raises immediately."""
        call_count = 0

        async def always_refused(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("refused")

        @asynccontextmanager
        async def ctx_mgr(*args, **kwargs):
            mc = AsyncMock()
            mc.post = always_refused
            yield mc

        with patch("httpx.AsyncClient", ctx_mgr):
            with patch.dict("os.environ", {}, clear=True):
                p = OllamaProvider(max_retries=2)
                with pytest.raises(RuntimeError):
                    await p.generate("hello")

        assert call_count == 1  # No retries — immediate failure
