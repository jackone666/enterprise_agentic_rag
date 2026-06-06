"""Tests for the LongTermMemory component.

Covers: importance scoring, extract & store, retrieval,
        deduplication, multi-user isolation, and delete operations.
"""

import pytest

from enterprise_agentic_rag.memory.long_term_memory import (
    LongTermMemory,
    LongTermMemoryEntry,
)

# =========================================================================
# LongTermMemoryEntry
# =========================================================================


class TestLongTermMemoryEntry:
    """Tests for the LongTermMemoryEntry dataclass."""

    def test_default_values(self) -> None:
        entry = LongTermMemoryEntry(memory_id="m1", user_id="u001", content="测试")
        assert entry.memory_id == "m1"
        assert entry.user_id == "u001"
        assert entry.content == "测试"
        assert entry.importance == 0.0
        assert entry.source_session == ""
        assert entry.source_turn == 0

    def test_full_entry(self) -> None:
        entry = LongTermMemoryEntry(
            memory_id="m2",
            user_id="u001",
            content="重要内容",
            importance=0.85,
            source_session="s_test",
            source_turn=3,
            created_at="2025-01-01T00:00:00Z",
            accessed_at="2025-01-02T00:00:00Z",
        )
        assert entry.importance == 0.85
        assert entry.source_session == "s_test"
        assert entry.source_turn == 3


# =========================================================================
# LongTermMemory — Scoring
# =========================================================================


class TestLongTermMemoryScoring:
    """Tests for the rule-based importance scoring."""

    def test_score_code_block(self) -> None:
        """Turns containing code blocks should score higher."""
        score = LongTermMemory._score_turn(
            {"role": "user", "content": "代码报错：```python\nprint('err')\n```"},
            turn_index=0,
            total_turns=4,
        )
        assert score >= 0.5  # code + error + first turn + user + question-ish

    def test_score_error_message(self) -> None:
        """Turns containing error patterns should score higher."""
        score = LongTermMemory._score_turn(
            {"role": "user", "content": "NullPointerException at line 42，怎么修复？"},
            turn_index=1,
            total_turns=6,
        )
        assert score >= 0.3  # error + user + question

    def test_score_low_importance(self) -> None:
        """Short assistant messages should score low."""
        score = LongTermMemory._score_turn(
            {"role": "assistant", "content": "好的"},
            turn_index=5,
            total_turns=10,
        )
        assert score < 0.3

    def test_score_first_turn_bonus(self) -> None:
        """First turn gets a bonus."""
        score = LongTermMemory._score_turn(
            {"role": "user", "content": "API 返回 500 错误怎么解决？"},
            turn_index=0,
            total_turns=6,
        )
        assert score >= 0.4  # error + first + user + question

    def test_score_long_user_message(self) -> None:
        """Long user messages should score higher."""
        long_msg = "这是一个很长的问题。" * 15  # ~150 chars
        score = LongTermMemory._score_turn(
            {"role": "user", "content": long_msg},
            turn_index=2,
            total_turns=6,
        )
        assert score >= 0.25  # length + user + early

    def test_score_question_mark(self) -> None:
        """Messages with question marks should get a bonus."""
        score = LongTermMemory._score_turn(
            {"role": "user", "content": "这个怎么处理？"},
            turn_index=0,
            total_turns=4,
        )
        assert score >= 0.25  # first + user + question

    def test_score_capped_at_one(self) -> None:
        """Score should never exceed 1.0."""
        score = LongTermMemory._score_turn(
            {
                "role": "user",
                "content": "Traceback error ```code``` 怎么修复这个异常？报错了 "
                + "x" * 100,
            },
            turn_index=0,
            total_turns=4,
        )
        assert score <= 1.0


# =========================================================================
# LongTermMemory — Extract & Store
# =========================================================================


class TestLongTermMemoryExtractStore:
    """Tests for the extract_and_store pipeline."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        # Low threshold to capture more turns for testing
        self.ltm = LongTermMemory(importance_threshold=0.3)

    def test_below_threshold_not_stored(self) -> None:
        """Turns below the importance threshold should be skipped."""
        turns = [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "好"},
        ]
        count = self.ltm.extract_and_store(turns, "u001", "s_test")
        assert count == 0

    def test_high_importance_stored(self) -> None:
        """High-importance turns should be persisted."""
        turns = [
            {
                "role": "user",
                "content": "系统报错：NullPointerException at line 42，怎么修复？```java\nx\n```",
            },
            {"role": "assistant", "content": "这是空指针异常，需要检查..."},
        ]
        count = self.ltm.extract_and_store(turns, "u001", "s_test")
        assert count >= 1

    def test_empty_turns_returns_zero(self) -> None:
        assert self.ltm.extract_and_store([], "u001", "s_test") == 0

    def test_empty_user_id_returns_zero(self) -> None:
        turns = [{"role": "user", "content": "测试"}]
        assert self.ltm.extract_and_store(turns, "", "s_test") == 0

    def test_empty_content_skipped(self) -> None:
        """Turns with empty content should be skipped even if score is high."""
        # The score may be 0 since content is empty (no code, no error, etc.)
        turns = [{"role": "user", "content": ""}]
        count = self.ltm.extract_and_store(turns, "u001", "s_test")
        assert count == 0

    def test_duplicate_not_stored(self) -> None:
        """Exact same content should not be stored twice."""
        turns = [{"role": "user", "content": "NullPointerException at line 42 怎么修复？"}]
        count1 = self.ltm.extract_and_store(turns, "u001", "s1")
        count2 = self.ltm.extract_and_store(turns, "u001", "s2")
        assert count1 >= 1
        assert count2 == 0

    def test_multi_user_isolation(self) -> None:
        """User A's memories should not appear in User B's results."""
        turns_a = [{"role": "user", "content": "NullPointerException at line 42 怎么修复？"}]
        turns_b = [{"role": "user", "content": "数据库连接超时怎么办？"}]
        self.ltm.extract_and_store(turns_a, "u001", "s_a")
        self.ltm.extract_and_store(turns_b, "u002", "s_b")

        recent_a = self.ltm.get_recent("u001", limit=10)
        recent_b = self.ltm.get_recent("u002", limit=10)

        # User A's memories should not contain user B's content
        for m in recent_a:
            assert "数据库连接超时" not in m.get("content", "")
        for m in recent_b:
            assert "NullPointerException" not in m.get("content", "")


# =========================================================================
# LongTermMemory — Retrieve
# =========================================================================


class TestLongTermMemoryRetrieve:
    """Tests for the retrieve and get_recent methods."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.ltm = LongTermMemory(importance_threshold=0.3)

    def test_retrieve_returns_list(self) -> None:
        result = self.ltm.retrieve("u001")
        assert isinstance(result, list)

    def test_get_recent_returns_list(self) -> None:
        result = self.ltm.get_recent("u001")
        assert isinstance(result, list)

    def test_get_recent_after_store(self) -> None:
        """After storing, get_recent should include the stored memory."""
        turns = [{"role": "user", "content": "Traceback error in line 42 ```code```"}]
        self.ltm.extract_and_store(turns, "u001", "s_test")
        recent = self.ltm.get_recent("u001", limit=10)
        assert len(recent) >= 1
        assert any("Traceback" in m.get("content", "") for m in recent)

    def test_retrieve_unknown_user_returns_empty(self) -> None:
        result = self.ltm.retrieve("no_such_user")
        assert result == []

    def test_get_recent_respects_limit(self) -> None:
        for i in range(5):
            turns = [{"role": "user", "content": f"错误 {i} Traceback ```code```"}]
            self.ltm.extract_and_store(turns, "u_limit_test", f"s{i}")
        recent = self.ltm.get_recent("u_limit_test", limit=3)
        assert len(recent) <= 3

    def test_fusion_ranking_prefers_relevant_memory(self) -> None:
        """Relevance should beat a newer but unrelated memory."""
        old_relevant = {
            "memory_id": "old",
            "content": "用户偏好 Python 后端和 RAG 项目回答",
            "importance": 0.9,
            "created_at": "2024-01-01T00:00:00Z",
            "accessed_at": "2024-01-01T00:00:00Z",
        }
        new_unrelated = {
            "memory_id": "new",
            "content": "今天临时测试一个前端按钮样式",
            "importance": 0.2,
            "created_at": "2026-01-01T00:00:00Z",
            "accessed_at": "2026-01-01T00:00:00Z",
        }

        ranked = LongTermMemory._rank_memories(
            "Python RAG 项目",
            [new_unrelated, old_relevant],
            top_k=2,
        )

        assert ranked[0]["memory_id"] == "old"
        assert ranked[0]["final_score"] > ranked[1]["final_score"]
        assert "relevance_score" in ranked[0]
        assert "recency_score" in ranked[0]


# =========================================================================
# LongTermMemory — Delete
# =========================================================================


class TestLongTermMemoryDelete:
    """Tests for delete operations."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.ltm = LongTermMemory(importance_threshold=0.3)

    def test_delete_memory(self) -> None:
        turns = [{"role": "user", "content": "NullPointerException ```code```"}]
        self.ltm.extract_and_store(turns, "u001", "s1")
        recent = self.ltm.get_recent("u001", limit=10)
        assert len(recent) >= 1

        memory_id = recent[0]["memory_id"]
        assert self.ltm.delete_memory(memory_id) is True

        # Should be gone now
        recent2 = self.ltm.get_recent("u001", limit=10)
        assert all(m["memory_id"] != memory_id for m in recent2)

    def test_delete_nonexistent_memory(self) -> None:
        assert self.ltm.delete_memory("nonexistent-id") is False

    def test_delete_user_memories(self) -> None:
        for i in range(3):
            turns = [{"role": "user", "content": f"错误 {i} Traceback ```code```"}]
            self.ltm.extract_and_store(turns, "u_del", f"s{i}")

        count = self.ltm.delete_user_memories("u_del")
        assert count >= 1

        # Should be empty now
        recent = self.ltm.get_recent("u_del", limit=10)
        assert len(recent) == 0
