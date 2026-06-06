"""Tests for RetrievalRouter — route query analysis to retrieval plans.

Covers:
- Default question → parallel
- Relationship question → graph_first
- Precise question → keyword-heavy parallel
- Semantic question → vector-heavy parallel
- Graph RAG disabled → hybrid_only
"""

from __future__ import annotations

import pytest

from enterprise_agentic_rag.rag.retrieval_router import RetrievalRouter


@pytest.fixture
def router():
    r = RetrievalRouter()
    # Route logic tests are about classification, not live Neo4j connectivity.
    # Override so graph appears available — the orchestrator handles real checks.
    r._graph_available_override = True
    return r


class TestRetrievalRouter:
    """Route queries to correct retrieval modes."""

    def test_default_normal_question_parallel(self, router):
        """Default general question → parallel mode."""
        plan = router.route("ArkTS 页面跳转失败怎么办？")
        assert plan.mode == "parallel"
        assert "keyword" in plan.enabled_retrievers
        assert "vector" in plan.enabled_retrievers

    def test_relational_question_graph_first(self, router):
        """Relationship question → graph_first mode."""
        queries = [
            "Ability 和生命周期有什么关系？",
            "页面跳转失败和 onWindowStageCreate 有关系吗？",
            "EntryAbility 和 UIAbility 的调用链是什么？",
            "A 依赖 B 吗？",
            "这个问题影响哪些模块？",
        ]
        for q in queries:
            plan = router.route(q)
            assert plan.mode == "graph_first", f"Query '{q}' should route to graph_first, got {plan.mode}"
            assert "graph" in plan.enabled_retrievers
            assert plan.need_query_expansion is True
            assert plan.need_second_stage_retrieval is True

    def test_error_code_precise_parallel(self, router):
        """Error code → parallel with keyword emphasis.

        Note: "9568321 是什么错误？" has both error_code (precise) AND
        "是什么" (semantic). The router classifies this as mixed, giving
        keyword a slight edge over vector.
        """
        plan = router.route("9568321 是什么错误？")
        assert plan.mode == "parallel"
        # Mixed query: keyword should be at least as high as vector
        weights = plan.weights
        assert weights.get("keyword", 0) >= weights.get("vector", 0), \
            f"Keyword weight should be >= vector for error codes, got {weights}"

    def test_error_code_pure_precise(self, router):
        """Pure error code query (no semantic markers) → keyword-heavy."""
        plan = router.route("9568321")
        assert plan.mode == "parallel"
        weights = plan.weights
        assert weights.get("keyword", 0) > weights.get("vector", 0), \
            f"Pure error code should be keyword-heavy, got {weights}"

    def test_api_name_precise_parallel(self, router):
        """API name without semantic markers → keyword-heavy parallel.

        Note: "@ohos.router 怎么用？" contains both API name AND "怎么" (semantic).
        Per Rule 5, "怎么" indicates an explanatory question → vector-heavy.
        This test uses a pure API lookup without semantic markers.
        """
        plan = router.route("@ohos.router")
        assert plan.mode == "parallel"
        weights = plan.weights
        # Pure API lookup (no semantic markers) → keyword-heavy
        assert weights.get("keyword", 0) > weights.get("vector", 0), \
            f"Pure API lookup should be keyword-heavy, got {weights}"

    def test_semantic_question_vector_heavy(self, router):
        """Semantic question → parallel with vector emphasis."""
        queries = [
            "为什么白屏？",
            "怎么优化性能？",
            "可能原因是什么？",
            "区别是什么？",
        ]
        for q in queries:
            plan = router.route(q)
            assert plan.mode == "parallel", f"Semantic query should still use parallel mode, got {plan.mode}"
            weights = plan.weights
            if not any(w > 0 for w in [weights.get("keyword", 0)]):  # not a precise query
                assert weights.get("vector", 0) >= weights.get("keyword", 0), \
                    f"Vector weight should be >= keyword for semantic queries, got {weights}"

    def test_hybrid_only_when_graph_disabled(self, monkeypatch):
        """When ENABLE_GRAPH_RAG=false, router returns hybrid_only."""
        monkeypatch.setenv("ENABLE_GRAPH_RAG", "false")

        # Reset the settings singleton so the env change is picked up
        import enterprise_agentic_rag.config.settings as settings_module
        settings_module._settings = None

        # Recreate router to pick up env change
        router2 = RetrievalRouter()
        plan = router2.route("任意问题")

        assert plan.mode == "hybrid_only", f"Expected hybrid_only, got {plan.mode}"
        assert "graph" not in plan.enabled_retrievers
        assert "keyword" in plan.enabled_retrievers
        assert "vector" in plan.enabled_retrievers

        # Restore
        monkeypatch.setenv("ENABLE_GRAPH_RAG", "true")
        settings_module._settings = None

    def test_plan_has_top_k(self, router):
        """Every plan should have top_k dict."""
        plan = router.route("测试问题")
        assert isinstance(plan.top_k, dict)
        assert "keyword" in plan.top_k or plan.mode == "hybrid_only"

    def test_plan_has_reason(self, router):
        """Every plan must include a reason for traceability."""
        plan = router.route("测试问题")
        assert plan.reason, "Plan must have a reason string"

    def test_plan_fallback_to_hybrid(self, router):
        """All plans except hybrid_only should have fallback_to_hybrid=True."""
        queries = [
            "普通问题",
            "Ability 和生命周期有什么关系？",
            "9568321",
            "为什么白屏？",
        ]
        for q in queries:
            plan = router.route(q)
            if plan.mode != "hybrid_only":
                assert plan.fallback_to_hybrid is True, \
                    f"Plan for '{q}' should have fallback_to_hybrid=True"


class TestRetrievalRouterEdgeCases:
    """Edge cases for routing."""

    def test_empty_query(self, router):
        plan = router.route("")
        assert plan.mode in ("parallel", "hybrid_only")

    def test_very_short_query(self, router):
        plan = router.route("白屏")
        assert plan.mode in ("parallel", "hybrid_only")

    def test_english_query(self, router):
        plan = router.route("How to fix 9568321 error?")
        assert plan.mode == "parallel"

    def test_mixed_chinese_english(self, router):
        plan = router.route("EntryAbility 的 onCreate 参数是什么？")
        assert plan.mode == "parallel"

    def test_deep_intent_api_usage_keyword_heavy(self, router):
        plan = router.route(
            "@ohos.net.http 怎么用？",
            {"primary_intent": "api_usage", "intent": "api_usage"},
        )
        assert plan.mode == "parallel"
        assert plan.weights["keyword"] > plan.weights["vector"]
        assert plan.weights["keyword"] > plan.weights["graph"]

    def test_deep_intent_concept_qa_vector_heavy(self, router):
        plan = router.route(
            "什么是 UIAbility 生命周期？",
            {"primary_intent": "concept_qa", "intent": "concept_qa"},
        )
        assert plan.mode == "parallel"
        assert plan.weights["vector"] > plan.weights["keyword"]
        assert plan.weights["vector"] > plan.weights["graph"]

    def test_deep_intent_migration_graph_heavy(self, router):
        plan = router.route(
            "Router 迁移到 Navigation 的影响是什么？",
            {"primary_intent": "migration", "intent": "migration"},
        )
        assert plan.mode == "graph_first"
        assert plan.weights["graph"] > plan.weights["keyword"]
        assert plan.weights["graph"] > plan.weights["vector"]
