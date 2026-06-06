"""Tests for verifier agent — answer verification with rule and LLM paths."""

import pytest

from enterprise_agentic_rag.agents.verifier_agent import _verify_rules, verify_answer, verify_answer_async
from enterprise_agentic_rag.llm.base import LLMResponse

# ===========================================================================
# TestVerifierRules
# ===========================================================================


class TestVerifierRules:
    """Test rule-based answer verification."""

    def test_verify_empty_answer_returns_false(self):
        verified, reason = _verify_rules("", [], [])
        assert verified is False
        assert "为空" in reason

    def test_verify_whitespace_answer_returns_false(self):
        verified, reason = _verify_rules("   ", [], [])
        assert verified is False

    def test_verify_no_docs_returns_false(self):
        verified, reason = _verify_rules("some answer", [], [])
        assert verified is False
        assert "未检索到" in reason

    def test_verify_low_relevance_noise(self):
        docs = [
            {"score": 0.01, "content": "noise"},
            {"score": 0.01, "content": "more noise"},
            {"score": 0.011, "content": "slightly different noise"},
        ]
        # All near-identical very low scores → noise
        verified, reason = _verify_rules(
            "some answer with [1]",
            [{"source": "x", "index": 1}],
            docs,
        )
        # With citations and scores < 0.02 and max/avg < 1.1, it's treated as noise
        assert "噪音" in reason or verified is False

    def test_verify_missing_citations_with_citation_list(self):
        # Has citation list but no inline markers — triggers a warning
        verified, reason = _verify_rules(
            "a plain answer with no refs at all",
            [{"source": "doc.md", "index": 1}],
            [{"score": 0.5, "content": "doc"}],
        )
        # Missing inline citations triggers a warning but doesn't fail
        # (the citation list is still available for the final answer)
        assert "已自动补充" in reason or "缺少引用标记" in reason

    def test_verify_missing_citations_and_no_list(self):
        # No citation sources at all — triggers the "缺少引用来源" warning
        verified, reason = _verify_rules(
            "a plain answer with no refs",
            [],
            [{"score": 0.5, "content": "doc"}],
        )
        assert "缺少引用来源" in reason

    def test_verify_short_answer(self):
        verified, reason = _verify_rules(
            "短",
            [{"source": "x", "index": 1}],
            [{"score": 0.5, "content": "doc"}],
        )
        assert "过短" in reason

    def test_verify_passes_all_checks(self):
        verified, reason = _verify_rules(
            "这是一个完整且充分的答案，包含 [1] 引用标记，并且内容足够长以达到验证标准。",
            [{"source": "doc.md", "index": 1}],
            [{"score": 0.85, "content": "document content here"}],
        )
        assert verified is True
        assert "通过" in reason


# ===========================================================================
# TestVerifierIntegration
# ===========================================================================


class TestVerifierIntegration:
    """Test verify_answer with mock provider (falls through to rules)."""

    def test_verify_with_mock_provider_empty_answer(self):
        # Mock provider is set by conftest.py
        verified, reason = verify_answer("", [], [])
        assert verified is False
        assert len(reason) > 0

    def test_verify_with_mock_provider_valid_answer(self):
        verified, reason = verify_answer(
            "根据文档 [1]，UIAbility 是 HarmonyOS 应用的基本组件，负责管理应用的生命周期。"
            "开发者可以通过继承 UIAbility 类来创建自己的 Ability。",
            [{"source": "api.md", "index": 1}],
            [{"score": 0.9, "content": "UIAbility 文档内容..."}],
        )
        # Should pass in rule mode
        assert "通过" in reason or verified is True

    def test_verify_returns_tuple(self):
        result = verify_answer("test answer", [], [{"score": 0.8, "content": "doc"}])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    @pytest.mark.asyncio
    async def test_async_verify_uses_llm_provider(self, monkeypatch):
        class FakeProvider:
            provider_name = "fake"
            model_name = "fake-model"

            async def generate(self, prompt, *, temperature=0.0, max_tokens=2048):
                return LLMResponse(
                    content='{"verified": true, "reason": "async llm verified"}',
                    success=True,
                )

        monkeypatch.setattr(
            "enterprise_agentic_rag.llm.provider_factory.get_llm_provider",
            lambda: FakeProvider(),
        )

        verified, reason = await verify_answer_async(
            "根据文档 [1]，UIAbility 是应用组件。",
            [{"source": "api.md", "index": 1}],
            [{"score": 0.9, "content": "UIAbility 是应用组件。"}],
            use_claim_level=False,
        )

        assert verified is True
        assert reason == "async llm verified"
