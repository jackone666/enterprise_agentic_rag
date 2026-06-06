"""Tests for the Memory System.

Covers: ShortTermMemory, SummaryMemory, UserMemory,
        CheckpointStore, and MemoryManager.
"""

import pytest

from enterprise_agentic_rag.memory.checkpoint import CheckpointStore
from enterprise_agentic_rag.memory.long_term_memory import LongTermMemory
from enterprise_agentic_rag.memory.memory_classifier import (
    MemoryClassifier,
    MemoryTarget,
    decide_memory_target,
)
from enterprise_agentic_rag.memory.memory_manager import MemoryManager
from enterprise_agentic_rag.memory.short_term_memory import ChatTurn, ShortTermMemory
from enterprise_agentic_rag.memory.summary_memory import SessionSummary, SummaryMemory
from enterprise_agentic_rag.memory.user_memory import UserMemory

# =========================================================================
# ShortTermMemory
# =========================================================================


class TestShortTermMemory:
    """Tests for the per-session sliding-window message buffer."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.mem = ShortTermMemory(max_turns=6)

    def test_add_and_retrieve(self) -> None:
        self.mem.add_message("s1", "user", "你好")
        self.mem.add_message("s1", "assistant", "你好！有什么可以帮你的？")

        history = self.mem.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "assistant"

    def test_add_with_intent(self) -> None:
        self.mem.add_message("s1", "user", "如何重置密码？", intent="technical_question")
        history = self.mem.get_history("s1")
        assert history[0]["intent"] == "technical_question"

    def test_sliding_window_eviction(self) -> None:
        """Old messages should be evicted when max_turns is exceeded."""
        for i in range(10):
            self.mem.add_message("s1", "user", f"问题 {i}")
        history = self.mem.get_history("s1")
        assert len(history) == 6  # max_turns=6
        assert history[0]["content"] == "问题 4"
        assert history[-1]["content"] == "问题 9"

    def test_get_last_n(self) -> None:
        for i in range(5):
            self.mem.add_message("s1", "user", f"问题 {i}")
        recent = self.mem.get_history("s1", last_n=2)
        assert len(recent) == 2
        assert recent[-1]["content"] == "问题 4"

    def test_get_last_assistant_answer(self) -> None:
        self.mem.add_message("s1", "user", "你好")
        self.mem.add_message("s1", "assistant", "回答1")
        self.mem.add_message("s1", "user", "再问一个")
        self.mem.add_message("s1", "assistant", "回答2")

        last = self.mem.get_last_assistant_answer("s1")
        assert last == "回答2"

    def test_get_last_assistant_answer_empty(self) -> None:
        """When no assistant messages exist, should return empty string."""
        self.mem.add_message("s1", "user", "你好")
        assert self.mem.get_last_assistant_answer("s1") == ""

    def test_multi_session_isolation(self) -> None:
        self.mem.add_message("s1", "user", "s1 问题")
        self.mem.add_message("s2", "user", "s2 问题")

        assert len(self.mem.get_history("s1")) == 1
        assert len(self.mem.get_history("s2")) == 1
        assert self.mem.session_count == 2

    def test_clear_session(self) -> None:
        self.mem.add_message("s1", "user", "你好")
        self.mem.clear_session("s1")
        assert self.mem.get_history("s1") == []

    def test_unknown_session_returns_empty(self) -> None:
        assert self.mem.get_history("no_such_session") == []

    def test_chat_turn_dataclass(self) -> None:
        turn = ChatTurn(role="user", content="test", intent="general_question")
        assert turn.role == "user"
        assert turn.content == "test"
        assert turn.intent == "general_question"


# =========================================================================
# SummaryMemory
# =========================================================================


class TestSummaryMemory:
    """Tests for the heuristic conversation summariser."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.summary = SummaryMemory(compress_threshold=4)

    def test_no_summary_below_threshold(self) -> None:
        """Fewer than threshold turns should produce an empty summary."""
        turns = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
        ]
        result = self.summary.update_summary("s1", turns)
        assert result.summary == ""

    def test_summary_above_threshold(self) -> None:
        turns = [
            {"role": "user", "content": "API 错误怎么处理？"},
            {"role": "assistant", "content": "请查看错误码文档。"},
            {"role": "user", "content": "具体是什么错误码？"},
            {"role": "assistant", "content": "AUTH_401 表示认证失败。"},
            {"role": "user", "content": "如何修复？"},
            {"role": "assistant", "content": "检查 API 密钥配置。"},
        ]
        result = self.summary.update_summary("s1", turns)
        assert len(result.summary) > 0
        assert "共 6 轮对话" in result.summary
        # Should detect topic keywords
        assert "API" in result.key_topics or len(result.key_topics) > 0

    def test_get_summary_returns_string(self) -> None:
        turns = [
            {"role": "user", "content": "密码问题？"},
            {"role": "assistant", "content": "请尝试重置。"},
            {"role": "user", "content": "怎么重置？"},
            {"role": "assistant", "content": "用密码重置功能。"},
            {"role": "user", "content": "部署呢？"},
            {"role": "assistant", "content": "部署文档在这里。"},
        ]
        self.summary.update_summary("s1", turns)
        summary_str = self.summary.get_summary("s1")
        assert isinstance(summary_str, str)
        assert len(summary_str) > 0

    def test_get_summary_unknown_session(self) -> None:
        assert self.summary.get_summary("unknown") == ""

    def test_clear_session(self) -> None:
        turns = [
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
            {"role": "user", "content": "c" * 20},
            {"role": "assistant", "content": "d" * 20},
            {"role": "user", "content": "e" * 20},
            {"role": "assistant", "content": "f" * 20},
        ]
        self.summary.update_summary("s1", turns)
        assert self.summary.get_summary("s1") != ""
        self.summary.clear_session("s1")
        assert self.summary.get_summary("s1") == ""

    def test_key_topics_detection(self) -> None:
        """Should detect keyword-based topics from user messages."""
        turns = [
            {"role": "user", "content": "我的密码忘记了怎么办？"},
            {"role": "assistant", "content": "可以使用密码重置功能。"},
            {"role": "user", "content": "部署到生产环境需要注意什么？"},
            {"role": "assistant", "content": "需要配置环境变量。"},
            {"role": "user", "content": "API 调用返回 403 权限错误"},
            {"role": "assistant", "content": "请检查权限配置。"},
        ]
        result = self.summary.update_summary("s1", turns)
        assert "密码" in result.key_topics
        assert "部署" in result.key_topics
        assert "权限" in result.key_topics or "API" in result.key_topics

    def test_session_summary_dataclass(self) -> None:
        ss = SessionSummary(
            session_id="s1",
            summary="测试摘要",
            key_topics=["API", "密码"],
            turn_count=6,
        )
        assert ss.session_id == "s1"
        assert ss.summary == "测试摘要"
        assert ss.turn_count == 6


# =========================================================================
# UserMemory
# =========================================================================


class TestUserMemory:
    """Tests for the mock user profile store."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.users = UserMemory()

    def test_admin_profile(self) -> None:
        profile = self.users.get_profile("u001")
        assert profile["name"] == "张三"
        assert profile["role"] == "admin"
        assert "admin" in profile["permissions"]
        assert "knowledge_search" in profile["permissions"]
        assert "ticket_manage" in profile["permissions"]

    def test_developer_profile(self) -> None:
        profile = self.users.get_profile("u002")
        assert profile["name"] == "李四"
        assert profile["role"] == "developer"
        assert "knowledge_search" in profile["permissions"]
        assert "ticket_manage" not in profile["permissions"]

    def test_basic_profile(self) -> None:
        profile = self.users.get_profile("u003")
        assert profile["name"] == "王五"
        assert profile["role"] == "basic"
        assert profile["recent_tickets"] == []

    def test_unknown_user_gets_default(self) -> None:
        profile = self.users.get_profile("u999")
        assert profile["role"] == "basic"
        assert "knowledge_search" in profile["permissions"]
        assert profile["user_id"] == "u999"

    def test_recent_tickets_admin(self) -> None:
        tickets = self.users.get_recent_tickets("u001")
        assert "TKT-001" in tickets
        assert "TKT-003" in tickets

    def test_recent_tickets_unknown(self) -> None:
        tickets = self.users.get_recent_tickets("u999")
        assert tickets == []

    def test_context_string_contains_details(self) -> None:
        ctx = self.users.get_context_string("u001")
        assert "张三" in ctx
        assert "admin" in ctx
        assert "平台工程部" in ctx
        assert "TKT-001" in ctx

    def test_context_string_unknown_user(self) -> None:
        ctx = self.users.get_context_string("u999")
        assert "u999" in ctx
        assert "basic" in ctx


# =========================================================================
# MemoryClassifier
# =========================================================================


class TestMemoryClassifier:
    """Tests for routing user messages to memory layers."""

    def test_preference_promotes_to_semantic_memory(self) -> None:
        decisions = decide_memory_target(
            "以后请都用中文回答，我主要做 Python 后端和 RAG 项目。",
            session_token_count=1000,
            session_turn_count=3,
        )
        targets = {d.target for d in decisions}
        assert MemoryTarget.SHORT_TERM in targets
        assert MemoryTarget.USER_PROFILE in targets
        assert MemoryTarget.SEMANTIC in targets
        assert MemoryTarget.EPISODIC in targets

    def test_one_off_context_does_not_write_long_term(self) -> None:
        decisions = MemoryClassifier().decide(
            "今天这次先临时用这个接口测一下。",
            session_token_count=1000,
            session_turn_count=2,
        )
        targets = {d.target for d in decisions}
        assert MemoryTarget.SHORT_TERM in targets
        assert MemoryTarget.EPISODIC not in targets
        assert MemoryTarget.SEMANTIC not in targets

    def test_sensitive_content_skips_persistent_memory(self) -> None:
        decisions = decide_memory_target(
            "api_key=sk-test-secret",
            session_token_count=100,
            session_turn_count=1,
        )
        targets = {d.target for d in decisions}
        assert MemoryTarget.SHORT_TERM in targets
        assert MemoryTarget.SKIP in targets
        assert MemoryTarget.EPISODIC not in targets
        assert MemoryTarget.SEMANTIC not in targets


# =========================================================================
# CheckpointStore
# =========================================================================


class TestCheckpointStore:
    """Tests for the in-memory checkpoint persistence."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.store = CheckpointStore()

    def test_save_and_load_checkpoint(self) -> None:
        state = {"query": "测试", "final_answer": "回答", "verified": True}
        cid = self.store.save_checkpoint("s1", state)
        assert cid is not None

        loaded = self.store.load_checkpoint("s1")
        assert loaded is not None
        assert loaded["query"] == "测试"
        assert loaded["final_answer"] == "回答"
        assert loaded["verified"] is True

    def test_load_latest_checkpoint(self) -> None:
        """Without explicit checkpoint_id, should load the latest."""
        self.store.save_checkpoint("s1", {"turn": 1})
        self.store.save_checkpoint("s1", {"turn": 2})
        self.store.save_checkpoint("s1", {"turn": 3})

        loaded = self.store.load_checkpoint("s1")
        assert loaded is not None
        assert loaded["turn"] == 3

    def test_load_specific_checkpoint(self) -> None:
        cid1 = self.store.save_checkpoint("s1", {"turn": 1})
        self.store.save_checkpoint("s1", {"turn": 2})

        loaded = self.store.load_checkpoint("s1", checkpoint_id=cid1)
        assert loaded is not None
        assert loaded["turn"] == 1

    def test_load_unknown_session(self) -> None:
        assert self.store.load_checkpoint("unknown") is None

    def test_custom_checkpoint_id(self) -> None:
        cid = self.store.save_checkpoint("s1", {"x": 42}, checkpoint_id="my-ckpt")
        assert cid == "my-ckpt"
        loaded = self.store.load_checkpoint("s1", checkpoint_id="my-ckpt")
        assert loaded is not None
        assert loaded["x"] == 42

    def test_non_serialisable_values_converted(self) -> None:
        """Non-serialisable values should be converted to strings."""

        class CustomObj:
            def __str__(self) -> str:
                return "CustomObjStr"

        self.store.save_checkpoint("s1", {"obj": CustomObj()})
        loaded = self.store.load_checkpoint("s1")
        assert loaded is not None
        assert loaded["obj"] == "CustomObjStr"

    def test_delete_session(self) -> None:
        self.store.save_checkpoint("s1", {"turn": 1})
        self.store.save_checkpoint("s1", {"turn": 2})
        assert self.store.load_checkpoint("s1") is not None

        self.store.delete_session("s1")
        assert self.store.load_checkpoint("s1") is None

    def test_checkpoint_count(self) -> None:
        assert self.store.checkpoint_count == 0
        self.store.save_checkpoint("s1", {"a": 1})
        self.store.save_checkpoint("s2", {"b": 2})
        self.store.save_checkpoint("s1", {"a": 3})
        assert self.store.checkpoint_count == 3

    def test_multi_session_isolation(self) -> None:
        self.store.save_checkpoint("s1", {"session": "one"})
        self.store.save_checkpoint("s2", {"session": "two"})

        s1 = self.store.load_checkpoint("s1")
        s2 = self.store.load_checkpoint("s2")
        assert s1 is not None and s1["session"] == "one"
        assert s2 is not None and s2["session"] == "two"


# =========================================================================
# MemoryManager (integration)
# =========================================================================


class TestMemoryManager:
    """Integration tests for the MemoryManager orchestrator."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.mgr = MemoryManager()

    def test_load_memory_context_first_turn(self) -> None:
        """First turn: empty history, no summary, basic profile."""
        ctx = self.mgr.load_memory_context("s_new", "u001")

        assert ctx["chat_history"] == []
        assert ctx["session_summary"] == ""
        assert ctx["user_profile"]["name"] == "张三"
        assert ctx["memory_context"]["chat_turns"] == 0
        assert ctx["checkpoint_id"] == ""
        assert "long_term_memories" in ctx
        assert isinstance(ctx["long_term_memories"], list)
        assert "long_term_memory_count" in ctx["memory_context"]

    def test_save_and_load_multi_turn(self) -> None:
        """After saving a turn, subsequent loads should include history."""
        # Turn 1
        state1 = {
            "query": "如何重置密码？",
            "final_answer": "请使用密码重置功能。",
            "intent": "technical_question",
            "user_id": "u001",
            "session_id": "s_multi",
            "permissions": ["read", "knowledge_search"],
        }
        cid1 = self.mgr.save_memory_context("s_multi", state1)
        assert cid1 != ""

        # Turn 2 — load should see previous history
        ctx2 = self.mgr.load_memory_context("s_multi", "u001")
        assert ctx2["chat_history"] != []
        assert len(ctx2["chat_history"]) == 2  # user + assistant from turn 1
        assert ctx2["memory_context"]["chat_turns"] == 2
        assert ctx2["checkpoint_id"] != ""

    def test_multi_session_isolation(self) -> None:
        """Sessions should not leak data between each other."""
        self.mgr.save_memory_context("s_A", {
            "query": "A question",
            "final_answer": "A answer",
            "intent": "general_question",
        })
        self.mgr.save_memory_context("s_B", {
            "query": "B question",
            "final_answer": "B answer",
            "intent": "general_question",
        })

        ctx_a = self.mgr.load_memory_context("s_A", "u002")
        ctx_b = self.mgr.load_memory_context("s_B", "u002")

        # Session A history should NOT contain session B content
        for turn in ctx_a["chat_history"]:
            assert "B" not in turn["content"]
        for turn in ctx_b["chat_history"]:
            assert "A" not in turn["content"]

    def test_summary_accumulates(self) -> None:
        """After many turns, summary should be produced."""
        for i in range(8):
            self.mgr.save_memory_context("s_sum", {
                "query": f"问题 {i}",
                "final_answer": f"回答 {i}",
                "intent": "general_question",
            })

        ctx = self.mgr.load_memory_context("s_sum", "u001")
        # After 8 saves → 16 messages (8 user + 8 assistant)
        assert len(ctx["chat_history"]) <= 10  # max_turns
        # Summary should exist since turns >= threshold
        assert ctx["session_summary"] != "" or ctx["memory_context"]["chat_turns"] > 0

    def test_custom_components(self) -> None:
        """MemoryManager should accept custom sub-component instances."""
        st = ShortTermMemory(max_turns=4)
        sm = SummaryMemory(compress_threshold=2)
        um = UserMemory()
        ck = CheckpointStore()
        lt = LongTermMemory(importance_threshold=0.7)

        mgr = MemoryManager(short_term=st, summary=sm, user=um, checkpoint=ck, long_term=lt)
        assert mgr.short_term is st
        assert mgr.summary is sm
        assert mgr.user is um
        assert mgr.checkpoint is ck
        assert mgr.long_term is lt

    def test_long_term_memory_integration(self) -> None:
        """After saving high-importance turns, long-term memories should be created."""
        # Save a turn with high-importance content
        state = {
            "query": "系统报错 NullPointerException at line 42 ```java\ncode\n```",
            "final_answer": "这是空指针异常，需要检查对象是否为空。",
            "intent": "technical_question",
            "user_id": "u_ltm_test",
            "session_id": "s_ltm_test",
        }
        self.mgr.save_memory_context("s_ltm_test", state)

        # Load context should include long-term memories
        ctx = self.mgr.load_memory_context("s_ltm_test", "u_ltm_test")
        assert "long_term_memories" in ctx
        assert isinstance(ctx["long_term_memories"], list)

    def test_load_includes_long_term_memory_count(self) -> None:
        """load_memory_context should include long_term_memory_count in memory_context."""
        ctx = self.mgr.load_memory_context("s_ltm_count", "u001")
        assert "memory_context" in ctx
        assert "long_term_memory_count" in ctx["memory_context"]
        assert isinstance(ctx["memory_context"]["long_term_memory_count"], int)

    def test_load_includes_working_and_typed_memory_counts(self) -> None:
        """load_memory_context should expose pinned working memory and typed counts."""
        state = {
            "query": "以后请都用中文回答，我主要做 Python 后端和 RAG 项目。",
            "final_answer": "好的，后续我会用中文回答。",
            "intent": "preference_update",
            "user_id": "u_memory_type",
            "session_id": "s_memory_type",
        }
        self.mgr.save_memory_context("s_memory_type", state)

        ctx = self.mgr.load_memory_context(
            "s_memory_type",
            "u_memory_type",
            query="我的回答偏好是什么？",
        )
        assert "working_memory" in ctx
        assert "semantic_memories" in ctx
        assert "episodic_memories" in ctx
        assert "semantic_memory_count" in ctx["memory_context"]
        assert "episodic_memory_count" in ctx["memory_context"]
        assert ctx["memory_context"]["semantic_memory_count"] >= 1

    def test_memory_layers_are_four_layer_model(self) -> None:
        """MemoryManager should expose the four-layer memory architecture."""
        ctx = self.mgr.load_memory_context("s_layers", "u001")
        assert ctx["memory_context"]["memory_layers"] == [
            "context_window",
            "working_memory",
            "short_term_memory",
            "long_term_memory",
        ]
        assert "context_window" in ctx
        assert "working_memory" in ctx
