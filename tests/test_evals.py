"""Tests for the Evaluation system and Data Flywheel.

Covers: EvalDataset, RAGEvaluator, AnswerEvaluator,
        RegressionEvaluator, FeedbackHandler, and auto-capture logic.
"""

import os
import tempfile

import pytest

from enterprise_agentic_rag.evals.answer_eval import AnswerEvaluator
from enterprise_agentic_rag.evals.dataset import EvalCase, EvalDataset, FailedCase
from enterprise_agentic_rag.evals.online_feedback import FeedbackHandler, FeedbackRecord
from enterprise_agentic_rag.evals.rag_eval import RAGEvaluator

# =========================================================================
# EvalDataset
# =========================================================================


class TestEvalDataset:
    """Tests for JSONL dataset loading and failed case saving."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.ds_path = os.path.join(self.tmpdir, "cases.jsonl")
        self.fail_path = os.path.join(self.tmpdir, "failed.jsonl")
        self.ds = EvalDataset(dataset_path=self.ds_path, failed_path=self.fail_path)

    def _write_cases(self, lines: list[str]) -> None:
        with open(self.ds_path, "w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")

    def test_load_empty(self) -> None:
        cases = self.ds.load_cases()
        assert cases == []

    def test_load_cases(self) -> None:
        self._write_cases([
            '{"query": "测试1", "expected_intent": "technical_question", "expected_sources": ["doc.md"], "expected_answer_keywords": ["kw1"], "user_role": "admin", "difficulty": "easy"}',
            '{"query": "测试2", "expected_intent": "policy_question", "expected_sources": [], "expected_answer_keywords": [], "user_role": "basic", "difficulty": "hard"}',
        ])
        cases = self.ds.load_cases()
        assert len(cases) == 2
        assert cases[0].query == "测试1"
        assert cases[1].difficulty == "hard"

    def test_count(self) -> None:
        self._write_cases([
            '{"query": "q1", "expected_intent": "general_question", "expected_sources": [], "expected_answer_keywords": [], "user_role": "basic", "difficulty": "medium"}',
        ])
        assert self.ds.count() == 1

    def test_skips_invalid_json(self) -> None:
        self._write_cases([
            '{"query": "ok", "expected_intent": "general_question", "expected_sources": [], "expected_answer_keywords": [], "user_role": "basic", "difficulty": "medium"}',
            'not valid json',
            '{"query": "also ok", "expected_intent": "general_question", "expected_sources": [], "expected_answer_keywords": [], "user_role": "basic", "difficulty": "medium"}',
        ])
        cases = self.ds.load_cases()
        assert len(cases) == 2

    def test_save_and_load_failed_case(self) -> None:
        fc = FailedCase(
            trace_id="t1", session_id="s1", query="问题",
            intent="troubleshooting", user_id="u001",
            final_answer="答案", fallback_reason="校验失败",
            source="auto",
        )
        ok = self.ds.save_failed_case(fc)
        assert ok is True

        loaded = self.ds.load_failed_cases()
        assert len(loaded) == 1
        assert loaded[0]["trace_id"] == "t1"
        assert loaded[0]["source"] == "auto"

    def test_clear_failed_cases(self) -> None:
        self.ds.save_failed_case(FailedCase(trace_id="t1"))
        self.ds.clear_failed_cases()
        assert self.ds.load_failed_cases() == []

    def test_eval_case_dataclass(self) -> None:
        c = EvalCase(
            query="测试",
            expected_intent="technical_question",
            expected_sources=["doc.md"],
            expected_answer_keywords=["api", "认证"],
            user_role="developer",
            difficulty="medium",
            prompt_version="v2",
        )
        assert c.query == "测试"
        assert c.prompt_version == "v2"

    def test_failed_case_dataclass(self) -> None:
        fc = FailedCase(
            trace_id="t1", session_id="s1", query="q",
            intent="unknown", user_id="u1",
            final_answer="a", fallback_reason="f",
            source="feedback", metadata={"score": 0.5},
        )
        assert fc.metadata["score"] == 0.5

    def test_load_real_dataset(self) -> None:
        """The bundled regression_cases.jsonl should load successfully."""
        ds = EvalDataset()  # uses default path
        cases = ds.load_cases()
        assert len(cases) >= 8, f"Expected >= 8 cases, got {len(cases)}"
        for c in cases:
            assert c.query != ""
            assert c.expected_intent != ""


# =========================================================================
# RAGEvaluator
# =========================================================================


class TestRAGEvaluator:
    """Tests for retrieval quality metrics."""

    def test_hit_at_k_single(self) -> None:
        assert RAGEvaluator.hit_at_k_single(
            ["doc_a.md", "doc_b.md"], ["doc_a.md"], k=5
        ) is True
        assert RAGEvaluator.hit_at_k_single(
            ["doc_c.md"], ["doc_a.md"], k=5
        ) is False

    def test_recall_at_k_single(self) -> None:
        recall = RAGEvaluator.recall_at_k_single(
            ["doc_a.md", "doc_c.md"],
            ["doc_a.md", "doc_b.md"],
        )
        assert recall == 0.5  # 1 out of 2 expected found

    def test_recall_no_expected(self) -> None:
        recall = RAGEvaluator.recall_at_k_single(
            ["doc_a.md"], []
        )
        assert recall == 1.0  # no expectation → trivially satisfied

    def test_recall_empty_retrieved(self) -> None:
        recall = RAGEvaluator.recall_at_k_single(
            [], ["doc_a.md"]
        )
        assert recall == 0.0

    def test_mrr_single_first_position(self) -> None:
        mrr = RAGEvaluator.mrr_single(
            ["doc_a.md", "doc_b.md"], ["doc_a.md"]
        )
        assert mrr == 1.0

    def test_mrr_single_third_position(self) -> None:
        mrr = RAGEvaluator.mrr_single(
            ["doc_x.md", "doc_y.md", "doc_z.md"], ["doc_z.md"]
        )
        assert mrr == 1.0 / 3

    def test_mrr_none_found(self) -> None:
        mrr = RAGEvaluator.mrr_single(
            ["doc_a.md"], ["doc_b.md"]
        )
        assert mrr == 0.0

    def test_evaluate_aggregate(self) -> None:
        evaluator = RAGEvaluator(k=5)
        metrics = evaluator.evaluate(
            queries=["q1", "q2"],
            retrieved_docs_list=[
                [{"source": "doc_a.md", "score": 0.9}, {"source": "doc_b.md", "score": 0.5}],
                [{"source": "doc_c.md", "score": 0.3}],
            ],
            expected_sources_list=[["doc_a.md"], ["doc_a.md"]],
        )
        assert metrics.total_queries == 2
        assert 0 <= metrics.hit_at_k <= 1
        assert 0 <= metrics.mrr <= 1
        assert len(metrics.per_query) == 2


# =========================================================================
# AnswerEvaluator
# =========================================================================


class TestAnswerEvaluator:
    """Tests for answer quality metrics."""

    def test_citation_present(self) -> None:
        evaluator = AnswerEvaluator()
        assert evaluator._citation_present("答案 [1] 来源") > 0
        # Text with no citation markers at all
        assert evaluator._citation_present("这是一个普通的回答文本") == 0.0

    def test_groundedness_with_docs(self) -> None:
        evaluator = AnswerEvaluator()
        docs = [{"content": "API 认证 使用 OAuth2 协议"}]
        grounded = evaluator._groundedness("API 认证方式包括 OAuth2", docs)
        assert grounded > 0

    def test_groundedness_no_docs(self) -> None:
        evaluator = AnswerEvaluator()
        assert evaluator._groundedness("答案", []) == 0.0

    def test_answer_relevance(self) -> None:
        evaluator = AnswerEvaluator()
        rel = evaluator._answer_relevance(
            "密码重置步骤如下", ["密码", "重置"]
        )
        assert rel == 1.0

    def test_answer_relevance_partial(self) -> None:
        evaluator = AnswerEvaluator()
        rel = evaluator._answer_relevance(
            "密码重置", ["密码", "重置", "API"]
        )
        assert rel == 2 / 3

    def test_refusal_correctness_no_docs_proper(self) -> None:
        evaluator = AnswerEvaluator()
        ref = evaluator._refusal_correctness(
            "抱歉，知识库中没有找到相关信息", []
        )
        assert ref == 1.0

    def test_refusal_correctness_no_docs_bad(self) -> None:
        evaluator = AnswerEvaluator()
        # No refusal phrases → should be 0.0
        ref = evaluator._refusal_correctness(
            "这是一个普通的回答，没有拒绝含义", []
        )
        assert ref == 0.0

    def test_refusal_correctness_with_docs(self) -> None:
        evaluator = AnswerEvaluator()
        ref = evaluator._refusal_correctness(
            "正常答案", [{"content": "doc"}]
        )
        assert ref == 1.0  # trivially correct — not a refusal scenario

    def test_evaluate_aggregate(self) -> None:
        evaluator = AnswerEvaluator()
        metrics = evaluator.evaluate(
            queries=["q1", "q2"],
            answers=["答案 [1] 包含引用", "无引用"],
            retrieved_docs_list=[
                [{"content": "密码重置"}],
                [{"content": "认证"}],
            ],
            expected_keywords_list=[["密码"], ["认证"]],
        )
        assert metrics.total_queries == 2
        assert metrics.overall_score >= 0
        assert len(metrics.per_query) == 2


# =========================================================================
# FeedbackHandler
# =========================================================================


class TestFeedbackHandler:
    """Tests for online feedback and auto-capture."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.fail_path = os.path.join(self.tmpdir, "failed.jsonl")
        ds = EvalDataset(failed_path=self.fail_path)
        self.handler = FeedbackHandler(dataset=ds)

    def test_thumbs_down_captures(self) -> None:
        result = self.handler.process_feedback(
            FeedbackRecord(trace_id="t1", session_id="s1",
                           thumbs_up=False, feedback_text="答案不对"),
            result={"query": "测试", "final_answer": "错误"},
        )
        assert result["auto_captured"] is True

    def test_thumbs_up_no_auto_issues(self) -> None:
        result = self.handler.process_feedback(
            FeedbackRecord(trace_id="t2", session_id="s2", thumbs_up=True),
            result={"query": "q", "final_answer": "a", "verified": True},
        )
        assert result["auto_captured"] is False

    def test_auto_capture_need_human(self) -> None:
        reason = FeedbackHandler._auto_capture_reason({
            "need_human": True,
            "verified": True,
            "fallback_reason": "",
        })
        assert "need_human=true" in reason

    def test_auto_capture_not_verified(self) -> None:
        reason = FeedbackHandler._auto_capture_reason({
            "need_human": False,
            "verified": False,
            "fallback_reason": "",
            "verification_reason": "答案过短",
        })
        assert "verified=false" in reason

    def test_auto_capture_fallback_reason(self) -> None:
        reason = FeedbackHandler._auto_capture_reason({
            "need_human": False,
            "verified": True,
            "fallback_reason": "no_relevant_docs",
        })
        assert "no_relevant_docs" in reason

    def test_auto_capture_no_reason_returns_empty(self) -> None:
        reason = FeedbackHandler._auto_capture_reason({
            "need_human": False,
            "verified": True,
            "fallback_reason": "",
        })
        assert reason == ""

    def test_all_three_signals(self) -> None:
        """Multiple signals should all appear in the reason."""
        reason = FeedbackHandler._auto_capture_reason({
            "need_human": True,
            "verified": False,
            "fallback_reason": "tool_failure",
            "verification_reason": "缺少引用",
        })
        assert "need_human" in reason
        assert "verified" in reason
        assert "tool_failure" in reason

    def test_feedback_record_dataclass(self) -> None:
        fr = FeedbackRecord(
            trace_id="t1", session_id="s1",
            thumbs_up=True, feedback_text="很好", user_id="u001",
        )
        assert fr.user_id == "u001"


# =========================================================================
# End-to-end via HTTP-like flow
# =========================================================================


class TestE2EFeedbackFlow:
    """Simulate the full /chat → /feedback cycle."""

    def test_chat_result_stored_and_retrieved(self) -> None:
        """Simulate app-level result storage for feedback lookup."""
        from enterprise_agentic_rag.app.main import _recent_results, _store_result

        # Clear
        _recent_results.clear()

        _store_result("trace-a", {"query": "测试", "final_answer": "答案"})
        assert "trace-a" in _recent_results
        assert _recent_results["trace-a"]["query"] == "测试"

    def test_result_store_cap(self) -> None:
        """Test that the result store doesn't exceed _MAX_RECENT."""
        from enterprise_agentic_rag.app.main import _MAX_RECENT, _recent_results, _store_result

        _recent_results.clear()
        for i in range(_MAX_RECENT + 10):
            _store_result(f"trace-{i}", {"query": str(i)})
        assert len(_recent_results) <= _MAX_RECENT
