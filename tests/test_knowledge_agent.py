"""Tests for knowledge agent — answer generation with template fallback."""

import pytest

from enterprise_agentic_rag.agents.knowledge_agent import generate_answer, generate_answer_async
from enterprise_agentic_rag.llm.base import LLMResponse


# ===========================================================================
# TestKnowledgeAgentTemplate
# ===========================================================================


class TestKnowledgeAgentTemplate:
    """Test template-based answer generation."""

    def test_empty_docs_returns_no_info(self):
        answer, citations = generate_answer("test query", [])
        assert "没有找到" in answer
        assert citations == []

    def test_template_with_docs(self):
        docs = [
            {
                "source": "api_doc.md",
                "content": "UIAbility 是 HarmonyOS 应用的基本组件单元。",
                "score": 0.9,
            },
        ]
        answer, citations = generate_answer("什么是UIAbility", docs)
        assert "UIAbility" in answer
        assert len(citations) > 0

    def test_citations_are_deduplicated(self):
        docs = [
            {"source": "api_doc.md", "content": "内容A", "score": 0.9},
            {"source": "api_doc.md", "content": "内容B", "score": 0.8},
            {"source": "faq.md", "content": "内容C", "score": 0.7},
        ]
        _, citations = generate_answer("test", docs)
        # Same source should be deduplicated
        sources = [c["source"] for c in citations]
        assert len(sources) == len(set(sources))
        assert len(citations) <= 2  # api_doc and faq

    def test_answer_includes_citation_markers(self):
        docs = [
            {"source": "doc1.md", "content": "内容A", "score": 0.9},
            {"source": "doc2.md", "content": "内容B", "score": 0.8},
        ]
        answer, citations = generate_answer("test", docs)
        assert "参考来源" in answer
        assert len(citations) == 2

    def test_generate_answer_returns_tuple(self):
        docs = [
            {"source": "doc.md", "content": "content", "score": 0.9},
        ]
        result = generate_answer("test", docs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)

    def test_multiple_docs_concatenated(self):
        docs = [
            {"source": "a.md", "content": "第一段", "score": 0.9},
            {"source": "b.md", "content": "第二段", "score": 0.8},
        ]
        answer, _ = generate_answer("test", docs)
        assert "第一段" in answer
        assert "第二段" in answer

    @pytest.mark.asyncio
    async def test_async_generation_uses_llm_provider(self, monkeypatch):
        class FakeProvider:
            provider_name = "fake"
            model_name = "fake-model"

            async def generate(self, prompt, *, temperature=0.0, max_tokens=2048):
                return LLMResponse(content="LLM answer with [1]", success=True)

        monkeypatch.setattr(
            "enterprise_agentic_rag.llm.provider_factory.get_llm_provider",
            lambda: FakeProvider(),
        )
        docs = [{"source": "doc.md", "content": "content", "score": 0.9}]

        answer, citations = await generate_answer_async("test", docs)

        assert answer == "LLM answer with [1]"
        assert citations[0]["source"] == "doc.md"
