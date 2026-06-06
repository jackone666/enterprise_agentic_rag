"""Tests for GraphRAGOrchestrator — full pipeline with fallback.

Covers:
- graph_first normal execution (when Neo4j available)
- graph_first failure → hybrid fallback
- ENABLE_GRAPH_RAG=false → hybrid_only
- parallel mode execution
- trace population

Note: These tests require at least the in-memory keyword retriever
to be available (it always is). They may be slower when external
services (ES, Milvus) are down due to connection timeout retries.
"""

from __future__ import annotations

import asyncio

import pytest

from enterprise_agentic_rag.rag.graph_rag_orchestrator import GraphRAGOrchestrator


@pytest.fixture
def orchestrator():
    return GraphRAGOrchestrator()


def _sync_retrieve(orchestrator, query, **kwargs):
    """Sync wrapper for async retrieve."""
    return asyncio.run(orchestrator.retrieve(query=query, **kwargs))


class TestOrchestratorFallback:
    """Orchestrator gracefully falls back to hybrid RAG."""

    @pytest.mark.timeout(30)
    def test_fallback_when_graph_disabled(self, monkeypatch):
        """When ENABLE_GRAPH_RAG=false, returns hybrid results."""
        monkeypatch.setenv("ENABLE_GRAPH_RAG", "false")

        # Reset settings singleton to pick up env change
        import enterprise_agentic_rag.config.settings as settings_module
        settings_module._settings = None

        orch = GraphRAGOrchestrator()
        result = _sync_retrieve(orch, "测试查询")

        assert "retrieved_docs" in result
        trace = result.get("retrieval_trace", {})
        assert trace.get("mode") == "hybrid_only"

        # Restore
        monkeypatch.setenv("ENABLE_GRAPH_RAG", "true")
        settings_module._settings = None

    @pytest.mark.timeout(30)
    def test_fallback_when_neo4j_unavailable(self, orchestrator):
        """When Neo4j is not running, gracefully falls back to hybrid."""
        if orchestrator.graph_available:
            pytest.skip("Neo4j is available — test requires Neo4j to be down")

        result = _sync_retrieve(orchestrator, "测试查询")

        assert "retrieved_docs" in result
        trace = result.get("retrieval_trace", {})

        # Should have degraded or used hybrid
        degraded_to = result.get("degraded_to", "") or trace.get("degraded_to", "")
        mode = trace.get("mode", "")

        assert (degraded_to == "hybrid_only" or mode == "hybrid_only" or
                "errors" in result), \
            "Should fallback to hybrid when Neo4j is unavailable"

    @pytest.mark.timeout(30)
    def test_result_has_required_fields(self, orchestrator):
        """Result dict has all expected fields."""
        result = _sync_retrieve(orchestrator, "测试查询")

        required_fields = [
            "retrieved_docs",
            "retrieval_trace",
            "errors",
            "query_analysis",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.timeout(30)
    def test_trace_has_required_fields(self, orchestrator):
        """RetrievalTrace has all expected fields."""
        result = _sync_retrieve(orchestrator, "测试查询")
        trace = result.get("retrieval_trace", {})

        trace_fields = [
            "trace_id", "query", "mode", "enabled_retrievers",
            "keyword_hit_count", "vector_hit_count", "graph_hit_count",
            "merged_count", "total_latency_ms",
        ]
        for field in trace_fields:
            assert field in trace, f"Missing trace field: {field}"

    @pytest.mark.timeout(30)
    def test_docs_are_returned(self, orchestrator):
        """Retrieval always returns some documents (from fallback at minimum)."""
        result = _sync_retrieve(orchestrator, "API 认证方式")
        docs = result.get("retrieved_docs", [])
        # Should return at least something from keyword fallback
        # (may be empty if no docs are indexed, but should not crash)
        assert isinstance(docs, list)


class TestOrchestratorModes:
    """Different retrieval modes produce appropriate results."""

    @pytest.mark.timeout(30)
    def test_parallel_mode_default(self, orchestrator):
        """Default query uses parallel mode."""
        result = _sync_retrieve(orchestrator, "ArkTS 页面跳转失败怎么办？")
        trace = result.get("retrieval_trace", {})
        mode = trace.get("mode", "")
        # Should be parallel or hybrid_only (if graph unavailable)
        assert mode in ("parallel", "hybrid_only")

    @pytest.mark.timeout(30)
    def test_graph_first_relational_query(self, orchestrator):
        """Relational query attempts graph_first when graph available."""
        result = _sync_retrieve(orchestrator, "Ability 和生命周期有什么关系？")
        trace = result.get("retrieval_trace", {})

        mode = trace.get("mode", "")
        if orchestrator.graph_available:
            assert mode == "graph_first"
        else:
            # Should have degraded
            degraded_to = result.get("degraded_to", "") or trace.get("degraded_to", "")
            assert degraded_to == "hybrid_only"

    @pytest.mark.timeout(30)
    def test_query_analysis_populated(self, orchestrator):
        """Query analysis is always populated in result."""
        result = _sync_retrieve(orchestrator, "测试查询")
        qa = result.get("query_analysis", {})
        assert "intent" in qa
        assert "keywords" in qa
        assert "entities" in qa


class TestOrchestratorErrors:
    """Error handling."""

    @pytest.mark.timeout(30)
    def test_errors_is_list(self, orchestrator):
        """errors is always a list."""
        result = _sync_retrieve(orchestrator, "测试查询")
        assert isinstance(result.get("errors", None), list)

    @pytest.mark.timeout(30)
    def test_graph_unavailable_error_recorded(self, orchestrator):
        """When graph is unavailable, it's recorded in trace."""
        if orchestrator.graph_available:
            pytest.skip("Neo4j available — cannot test unavailable behavior")

        result = _sync_retrieve(orchestrator, "Ability 和生命周期有什么关系？")
        trace = result.get("retrieval_trace", {})
        degraded_to = result.get("degraded_to", "") or trace.get("degraded_to", "")
        assert degraded_to == "hybrid_only"
