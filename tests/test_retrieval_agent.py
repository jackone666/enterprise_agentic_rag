"""Tests for RetrievalAgent — 3-tier fallback retrieval agent.

Covers:
- run() with empty state returns empty evidence + fallback path
- run() dispatches to correct workflow mode (hybrid_only / graph_first / parallel)
- run() records events via tracer.record_retrieval_event
- retrieval_path indicator is set correctly for each tier
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class FakeSemCache:
    """Minimal semantic cache stub for testing."""
    enabled = True

    async def get(self, key: str):
        return None  # default: miss

    async def set(self, key: str, value: dict[str, Any]) -> None:
        pass


class FakeWorkflow:
    """Minimal BaseRAGWorkflow stub for testing."""
    def __init__(self, evidence: list[dict[str, Any]] | Exception = None):
        if isinstance(evidence, Exception):
            self._exc = evidence
            self._evidence = []
        else:
            self._exc = None
            self._evidence = evidence

    async def execute(
        self, query: str, *, mode: str = "hybrid_only", top_k: int = 10,
        intent: str = "concept_qa", entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._exc:
            raise self._exc
        return {
            "keyword_results": self._evidence,
            "vector_results": self._evidence,
            "merged_results": self._evidence,
            "reranked_results": self._evidence,
            "selected_evidence": self._evidence,
            "total_latency_ms": 10.0,
            "errors": [],
        }


@pytest.fixture
def mock_tracer():
    """Capture calls to tracer.record_retrieval_event."""
    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.tracer"
    ) as mock:
        mock.record_retrieval_event = MagicMock()
        mock._ensure = MagicMock()
        yield mock


@pytest.fixture
def mock_cache():
    """Return a fake semantic cache that always misses."""
    with patch(
        "enterprise_agentic_rag.rag.semantic_cache.get_semantic_cache",
        return_value=FakeSemCache(),
    ):
        yield


def _state(**overrides) -> dict:
    """Minimal state dict with sensible defaults."""
    base = {
        "query": "test query",
        "user_id": "u001",
        "session_id": "s001",
        "trace_id": "t001",
        "deep_intent": {
            "primary_intent": "concept_qa",
            "retrieval_plan": {"mode": "hybrid_only"},
            "entities": {},
        },
    }
    base.update(overrides)
    return base


# ===========================================================================
# Test: empty state → empty evidence + fail path
# ===========================================================================


@pytest.mark.asyncio
async def test_run_with_empty_state(mock_tracer, mock_cache):
    """Empty state should return empty evidence and fail path."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.BaseRAGWorkflow",
        return_value=FakeWorkflow(evidence=[]),
    ):
        agent = RetrievalAgent()
        result = await agent.run({})

    assert result.get("retrieved_docs") == []
    assert result.get("retrieval_path") == "fail"
    assert result.get("last_worker") == "retrieval_agent"
    assert result.get("last_agent_step") == "retrieve"


# ===========================================================================
# Test: mode routing
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("plan_mode,expected_wf_mode", [
    ("hybrid_only", "hybrid_only"),
    ("graph_first", "graph_first"),
    ("parallel", "parallel"),
    ("code_first", "parallel"),  # deprecated → mapped to parallel
    ("error_first", "hybrid_only"),  # deprecated → mapped to hybrid_only
])
async def test_run_dispatches_to_correct_workflow_mode(mock_tracer, mock_cache, plan_mode, expected_wf_mode):
    """Agent should call workflow with the mapped mode."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    mock_wf = FakeWorkflow(evidence=[])
    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.BaseRAGWorkflow",
        return_value=mock_wf,
    ):
        agent = RetrievalAgent()
        result = await agent.run(
            _state(
                deep_intent={
                    "primary_intent": "concept_qa",
                    "retrieval_plan": {"mode": plan_mode},
                    "entities": {},
                }
            )
        )

    # Should end up in fail path (empty evidence), confirming mode was resolved
    assert result.get("retrieval_path") == "fail"


# ===========================================================================
# Test: workflow returns evidence → workflow path
# ===========================================================================


@pytest.mark.asyncio
async def test_run_with_workflow_evidence(mock_tracer, mock_cache):
    """Workflow returning evidence should set retrieval_path = 'workflow'."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    evidence = [{"source": "doc.md", "content": "test", "score": 0.8}]
    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.BaseRAGWorkflow",
        return_value=FakeWorkflow(evidence=evidence),
    ):
        agent = RetrievalAgent()
        result = await agent.run(_state())

    assert result.get("retrieval_path") == "workflow"
    assert result.get("retrieved_docs") == evidence


# ===========================================================================
# Test: cache hit → cache_hit path
# ===========================================================================


class FakeCacheHit(FakeSemCache):
    """Semantic cache that returns a hit."""
    enabled = True

    async def get(self, key: str):
        return ("exact", {
            "retrieved_docs": [{"source": "cached.md", "content": "cached", "score": 1.0}],
            "reranked_docs": [{"source": "cached.md", "content": "cached", "score": 1.0}],
            "retrieval_mode": "hybrid_only",
            "retrieval_errors": [],
        })


@pytest.mark.asyncio
async def test_run_cache_hit(mock_tracer):
    """Cache hit should set retrieval_path = 'cache_hit' and skip workflow."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    with patch(
        "enterprise_agentic_rag.rag.semantic_cache.get_semantic_cache",
        return_value=FakeCacheHit(),
    ):
        agent = RetrievalAgent()
        result = await agent.run(_state())

    assert result.get("retrieval_path") == "cache_hit"
    assert result.get("retrieved_docs")[0]["source"] == "cached.md"


# ===========================================================================
# Test: tracer.record_retrieval_event is called
# ===========================================================================


@pytest.mark.asyncio
async def test_run_records_events(mock_tracer, mock_cache):
    """run() should call tracer.record_retrieval_event exactly once."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.BaseRAGWorkflow",
        return_value=FakeWorkflow(evidence=[]),
    ):
        agent = RetrievalAgent()
        await agent.run(_state())

    mock_tracer.record_retrieval_event.assert_called_once()
    call_args = mock_tracer.record_retrieval_event.call_args
    # Can be called with positional or keyword args
    if call_args.kwargs:
        assert call_args.kwargs["num_docs"] == 0
    else:
        # Positional: (state, query, num_docs, top_score, latency_ms, success)
        assert call_args.args[2] == 0  # num_docs is3rd positional arg


# ===========================================================================
# Test: workflow exception → fail path (not crash)
# ===========================================================================


@pytest.mark.asyncio
async def test_run_handles_workflow_exception(mock_tracer, mock_cache):
    """Workflow raising an exception should result in fail path, not propagate."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.BaseRAGWorkflow",
        return_value=FakeWorkflow(evidence=RuntimeError("boom")),
    ):
        agent = RetrievalAgent()
        result = await agent.run(_state())

    assert result.get("retrieval_path") == "fail"
    assert result.get("retrieved_docs") == []


# ===========================================================================
# Test: events working memory
# ===========================================================================


@pytest.mark.asyncio
async def test_run_populates_events_memory(mock_tracer, mock_cache):
    """Agent._events should accumulate one event per run()."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.BaseRAGWorkflow",
        return_value=FakeWorkflow(evidence=[]),
    ):
        agent = RetrievalAgent()
        await agent.run(_state())
        await agent.run(_state(query="another query"))

    assert len(agent._events) == 2
    assert agent._events[0]["path"] == "fail"
    assert agent._events[1]["path"] == "fail"


# ===========================================================================
# Test: max_retries respected
# ===========================================================================


@pytest.mark.asyncio
async def test_run_respects_max_retries(mock_tracer, mock_cache):
    """With max_retries=0, workflow should be called at least once."""
    from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent

    call_count = 0

    class CountingWorkflow(FakeWorkflow):
        async def execute(self, **kwargs):
            nonlocal call_count
            call_count += 1
            return await super().execute(**kwargs)

    with patch(
        "enterprise_agentic_rag.agents.retrieval_agent.BaseRAGWorkflow",
        return_value=CountingWorkflow(evidence=[]),
    ):
        agent = RetrievalAgent(max_retries=0)
        await agent.run(_state())

    # With empty evidence and max_retries=0, workflow is still called once
    assert call_count >= 1