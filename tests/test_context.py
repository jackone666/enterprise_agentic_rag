"""Tests for the Context Manager system.

Covers: TokenBudget, CitationManager, PromptBuilder, and ContextManager.
"""

import pytest

from enterprise_agentic_rag.context.citation_manager import Citation, CitationManager
from enterprise_agentic_rag.context.context_manager import ContextManager
from enterprise_agentic_rag.context.prompt_builder import PromptBuilder
from enterprise_agentic_rag.context.token_budget import BudgetAllocation, TokenBudget

# =========================================================================
# TokenBudget
# =========================================================================


class TestTokenBudget:
    """Tests for priority-based token allocation and truncation."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.budget = TokenBudget(max_tokens=4096)

    def test_estimate_tokens(self) -> None:
        """2 chars ≈ 1 token."""
        assert self.budget.estimate_tokens("hello") == 2       # 5 chars
        assert self.budget.estimate_tokens("") == 0
        assert self.budget.estimate_tokens("a") == 1           # min 1

    def test_allocate_query_always_kept(self) -> None:
        """Query gets priority 1 — always fully kept."""
        alloc = self.budget.allocate(query="测试问题" * 100)  # ~400 chars
        assert alloc.query > 0

    def test_allocate_respects_budget(self) -> None:
        """Total allocation should not exceed max_tokens."""
        alloc = self.budget.allocate(
            query="测试" * 500,
            retrieved_docs=[{"content": "文档内容" * 500} for _ in range(10)],
            tool_results=[{"output": "工具结果" * 300} for _ in range(5)],
            session_summary="摘要" * 200,
            chat_history=[{"content": "历史" * 100} for _ in range(20)],
        )
        total = (
            alloc.query
            + alloc.retrieved_docs
            + alloc.tool_results
            + alloc.session_summary
            + alloc.chat_history
        )
        assert total <= 4096
        assert alloc.remaining >= 0

    def test_low_priority_truncated_first(self) -> None:
        """When budget is tight, chat_history (priority 5) should be truncated."""
        huge_docs = [{"content": "X" * 8000} for _ in range(3)]
        alloc = self.budget.allocate(
            query="短问题",
            retrieved_docs=huge_docs,
            tool_results=[{"output": "Y" * 5000}],
            session_summary="Z" * 5000,
            chat_history=[{"content": "H" * 5000} for _ in range(5)],
        )
        # Chat history should get little or no allocation
        assert alloc.chat_history < 2000  # most budget goes to higher-priority items

    def test_truncate_retrieved_docs(self) -> None:
        docs = [
            {"content": "A" * 100, "source": "a.md"},
            {"content": "B" * 100, "source": "b.md"},
            {"content": "C" * 100, "source": "c.md"},
        ]
        # Allow ~25 tokens = ~50 chars
        truncated = self.budget.truncate_retrieved_docs(docs, max_tokens=25)
        assert len(truncated) <= 3

    def test_truncate_retrieved_docs_partial_last(self) -> None:
        """When a doc doesn't fully fit, it gets partially included with …."""
        docs = [
            {"content": "A" * 200, "source": "a.md"},
            {"content": "B" * 200, "source": "b.md"},
        ]
        # 50 tokens = 100 chars — only half of first doc fits
        truncated = self.budget.truncate_retrieved_docs(docs, max_tokens=50)
        assert len(truncated) == 1
        assert truncated[0]["content"].endswith("…")

    def test_truncate_chat_history_most_recent(self) -> None:
        """Truncation keeps the most recent turns."""
        history = [
            {"role": "user", "content": "msg" + str(i)} for i in range(20)
        ]
        # Allow ~ 10 tokens = 20 chars
        truncated = self.budget.truncate_chat_history(history, max_tokens=10)
        # Should keep only the most recent messages
        if truncated:
            # The last message in truncated should be the last message in original
            assert truncated[-1]["content"] == "msg19"

    def test_truncate_chat_history_empty(self) -> None:
        assert self.budget.truncate_chat_history([], 100) == []

    def test_allocation_attrs(self) -> None:
        alloc = BudgetAllocation(query=100, retrieved_docs=50, remaining=3946)
        assert alloc.query == 100
        assert alloc.remaining == 3946


# =========================================================================
# CitationManager
# =========================================================================


class TestCitationManager:
    """Tests for citation tracking and formatting."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.cm = CitationManager()

    def test_build_citations_from_docs(self) -> None:
        docs = [
            {
                "source": "policy.md",
                "chunk_id": "policy.md_0",
                "score": 0.95,
                "content": "# 访问控制\n企业实行最小权限原则。",
            },
            {
                "source": "api_doc.md",
                "chunk_id": "api_doc.md_2",
                "score": 0.78,
                "content": "## 认证方式\n支持 OAuth2 和 API Key。",
            },
        ]
        citations = self.cm.build_citations(docs)
        assert len(citations) == 2
        assert citations[0].index == 1
        assert citations[0].source == "policy.md"
        assert citations[0].score == 0.95

    def test_extract_section_from_heading(self) -> None:
        docs = [{"source": "f.md", "chunk_id": "f.md_0", "score": 0.9,
                  "content": "# 数据分类\n这是文档内容。\n## 子章节\n更多内容。"}]
        citations = self.cm.build_citations(docs)
        assert citations[0].section == "数据分类"

    def test_extract_section_no_heading(self) -> None:
        docs = [{"source": "f.md", "chunk_id": "f.md_0", "score": 0.5,
                  "content": "这是纯文本，没有标题。"}]
        citations = self.cm.build_citations(docs)
        assert citations[0].section == ""

    def test_excerpt_truncation(self) -> None:
        long_content = "这是非常长的文档内容。" * 30
        docs = [{"source": "long.md", "chunk_id": "long.md_0", "score": 0.6,
                  "content": long_content}]
        citations = self.cm.build_citations(docs)
        assert len(citations[0].excerpt) <= 120 + 3  # +3 for "... "

    def test_format_citation_line(self) -> None:
        c = Citation(index=1, source="test.md", chunk_id="test.md_0",
                     score=0.85, excerpt="测试片段",
                     section="访问控制")
        line = self.cm.format_citation_line(c)
        assert "[1]" in line
        assert "test.md" in line
        assert "访问控制" in line
        assert "0.85" in line

    def test_format_citation_line_no_section(self) -> None:
        c = Citation(index=2, source="doc.md", chunk_id="doc.md_0",
                     score=0.5, excerpt="片段")
        line = self.cm.format_citation_line(c)
        assert "§" not in line

    def test_format_references_section(self) -> None:
        docs = [
            {"source": "a.md", "chunk_id": "a.md_0", "score": 0.9,
             "content": "# 主题A\n内容A。"},
            {"source": "b.md", "chunk_id": "b.md_0", "score": 0.7,
             "content": "# 主题B\n内容B。"},
        ]
        self.cm.build_citations(docs)
        section = self.cm.format_references_section()
        assert "## 📚 参考来源" in section
        assert "[1]" in section
        assert "[2]" in section
        assert "a.md" in section
        assert "b.md" in section

    def test_format_references_section_empty(self) -> None:
        assert self.cm.format_references_section([]) == ""

    def test_citation_data_class(self) -> None:
        c = Citation(index=3, source="s.md", chunk_id="c3", score=0.42)
        assert c.index == 3
        assert c.excerpt == ""
        assert c.section == ""


# =========================================================================
# PromptBuilder
# =========================================================================


class TestPromptBuilder:
    """Tests for role-specific prompt assembly."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.pb = PromptBuilder()

    def test_build_router_prompt_minimal(self) -> None:
        prompt = self.pb.build_router_prompt("如何重置密码？")
        assert "企业级意图分类器" in prompt
        assert "如何重置密码？" in prompt
        assert "policy_question" in prompt

    def test_build_router_prompt_with_user_context(self) -> None:
        prompt = self.pb.build_router_prompt(
            query="API 错误",
            user_context="用户: 张三 | 角色: admin",
        )
        assert "张三" in prompt
        assert "admin" in prompt

    def test_build_router_prompt_with_history(self) -> None:
        history = [
            {"role": "user", "content": "上轮问题"},
            {"role": "assistant", "content": "上轮回答"},
        ]
        prompt = self.pb.build_router_prompt(
            query="新问题",
            chat_history=history,
        )
        assert "历史对话" in prompt
        assert "上轮问题" in prompt

    def test_build_knowledge_prompt_full(self) -> None:
        prompt = self.pb.build_knowledge_prompt(
            query="什么是访问控制？",
            retrieved_docs=[
                {"source": "policy.md", "content": "最小权限原则。"},
            ],
            tool_results=[
                {"tool_name": "get_profile", "output": "用户: 张三", "success": True},
            ],
            chat_history=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
            ],
            user_context="用户: 张三 | 角色: admin",
            session_summary="[摘要] 共 2 轮对话",
        )
        assert "企业知识库问答助手" in prompt
        assert "访问控制" in prompt
        assert "policy.md" in prompt
        assert "get_profile" in prompt
        assert "张三" in prompt
        assert "你好" in prompt
        assert "[摘要]" in prompt

    def test_build_knowledge_prompt_no_docs(self) -> None:
        prompt = self.pb.build_knowledge_prompt(query="问题")
        # The system instruction mentions "参考文档" but there should be
        # no "## 参考文档" section header (which contains actual doc content)
        assert "## 参考文档" not in prompt

    def test_build_knowledge_prompt_no_tools(self) -> None:
        prompt = self.pb.build_knowledge_prompt(query="问题")
        assert "工具执行结果" not in prompt

    def test_build_verifier_prompt(self) -> None:
        prompt = self.pb.build_verifier_prompt(
            draft_answer="密码重置步骤如下：1. 点击忘记密码 2. 输入邮箱",
            retrieved_docs=[
                {"source": "faq.md", "content": "密码重置..."},
            ],
            citations=[{"index": 1, "source": "faq.md"}],
        )
        assert "答案校验器" in prompt
        assert "忘记密码" in prompt
        assert "幻觉" in prompt or "引用" in prompt

    def test_format_history_cjk(self) -> None:
        history = [
            {"role": "user", "content": "中文问题"},
            {"role": "assistant", "content": "中文回答"},
        ]
        formatted = self.pb._format_history(history)
        assert "中文问题" in formatted
        assert "中文回答" in formatted
        assert "👤" in formatted
        assert "🤖" in formatted

    def test_format_history_empty(self) -> None:
        assert self.pb._format_history([]) == ""


# =========================================================================
# ContextManager (integration)
# =========================================================================


class TestContextManager:
    """Integration tests for the ContextManager orchestrator."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.cm = ContextManager(max_tokens=4096)

    def test_build_context_basic(self) -> None:
        """Basic context building with minimal inputs."""
        result = self.cm.build_context(
            query="如何重置密码？",
            user_profile={
                "user_id": "u001",
                "name": "张三",
                "role": "admin",
                "department": "平台工程部",
                "permissions": ["read", "write", "admin"],
                "recent_tickets": ["TKT-001"],
            },
        )

        # Check all expected keys exist
        assert "budget_allocation" in result
        assert "truncated_docs" in result
        assert "truncated_history" in result
        assert "citations" in result
        assert "citations_section" in result
        assert "router_prompt" in result
        assert "knowledge_prompt" in result
        assert "verifier_prompt" in result
        assert "context_window" in result
        assert "token_budget_max" in result
        assert "token_budget_used" in result

    def test_build_context_with_docs(self) -> None:
        """Context with retrieved documents."""
        docs = [
            {
                "source": "policy.md",
                "chunk_id": "policy.md_0",
                "score": 0.95,
                "content": "# 数据分类标准\n机密数据需要特殊处理。",
            },
            {
                "source": "api_doc.md",
                "chunk_id": "api_doc.md_1",
                "score": 0.80,
                "content": "## 认证方式\nAPI Key 放在 Header 中。",
            },
        ]
        result = self.cm.build_context(
            query="数据分类标准是什么？",
            retrieved_docs=docs,
            user_profile={"name": "张三", "role": "admin", "department": "工程部",
                          "permissions": ["read"], "recent_tickets": []},
        )

        # Citations should be built from docs
        assert len(result["citations"]) == 2
        assert result["citations_section"] != ""
        assert "policy.md" in result["citations_section"]
        # Budget should allocate to docs
        assert result["budget_allocation"].retrieved_docs > 0

    def test_build_context_with_tool_results(self) -> None:
        """Context with tool execution results."""
        tool_results = [
            {"tool_name": "get_ticket", "success": True, "output": "TKT-001 详情"},
            {"tool_name": "get_system_status", "success": True, "output": "所有服务正常"},
        ]
        result = self.cm.build_context(
            query="系统状态如何？",
            tool_results=tool_results,
        )
        assert "TKT-001" in result["knowledge_prompt"]
        assert "get_system_status" in result["knowledge_prompt"]

    def test_build_context_with_history(self) -> None:
        """Context with multi-turn chat history."""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            {"role": "user", "content": "什么是 API 密钥？"},
            {"role": "assistant", "content": "API 密钥是用于认证的令牌。"},
        ]
        result = self.cm.build_context(
            query="如何轮换密钥？",
            chat_history=history,
        )
        assert result["truncated_history"] != []
        assert "历史对话" in result["knowledge_prompt"] or "你好" in result["knowledge_prompt"]

    def test_budget_allocation_is_object(self) -> None:
        result = self.cm.build_context(query="测试")
        alloc = result["budget_allocation"]
        assert isinstance(alloc, BudgetAllocation)
        assert alloc.query > 0

    def test_token_budget_used_calculated(self) -> None:
        result = self.cm.build_context(
            query="一个问题" * 50,
            retrieved_docs=[{"content": "文档" * 200, "source": "d.md", "chunk_id": "d_0", "score": 0.9}],
        )
        assert result["token_budget_used"] >= result["token_budget_max"] - result["budget_allocation"].remaining

    def test_context_window_contains_user_info(self) -> None:
        result = self.cm.build_context(
            query="问题",
            user_profile={"name": "张三", "role": "admin", "department": "工程部",
                          "permissions": ["read", "write"], "recent_tickets": ["TKT-001"]},
        )
        assert "张三" in result["context_window"]
        assert "admin" in result["context_window"]

    def test_verifier_prompt_present(self) -> None:
        """Verifier prompt should be generated even with empty draft answer."""
        result = self.cm.build_context(query="测试")
        assert "答案校验器" in result["verifier_prompt"]

    def test_context_truncation_under_pressure(self) -> None:
        """When inputs are huge, context should still be built without error."""
        huge_docs = [
            {"source": f"doc{i}.md", "chunk_id": f"doc{i}.md_0", "score": 0.5,
             "content": "X" * 5000}
            for i in range(20)
        ]
        huge_history = [
            {"role": "user", "content": "Y" * 2000} for _ in range(30)
        ]
        result = self.cm.build_context(
            query="Z" * 1000,
            retrieved_docs=huge_docs,
            chat_history=huge_history,
            tool_results=[{"tool_name": "t", "output": "W" * 3000, "success": True} for _ in range(10)],
            session_summary="S" * 2000,
        )
        # Must not crash
        assert result is not None
        assert "budget_allocation" in result
        # Truncated docs should be fewer than original
        assert len(result["truncated_docs"]) < 20
        # Truncated history should be fewer than original
        assert len(result["truncated_history"]) < 30
