"""Retrieval and query-rewrite nodes.

The ``retrieve_knowledge`` node implements the 3-tier fallback chain:
- Tier 0: semantic cache
- Tier 1: BaseRAGWorkflow dispatch (hybrid_only / graph_first / parallel)
- Tier 2: write-back to cache

No Tier 2 (GraphRAG Orchestrator) or Tier 3 (legacy Retriever) fallback —
BaseRAGWorkflow handles all modes internally with graceful degradation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enterprise_agentic_rag.graph.cache import cache_scope
from enterprise_agentic_rag.graph.dependencies import tracer
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)


# Mode mapping: schema RetrievalMode → BaseRAGWorkflow mode
_MODE_MAP = {
    "hybrid_only": "hybrid_only",
    "graph_first": "graph_first",
    "parallel": "parallel",
    # Deprecated schema values mapped to nearest supported mode
    "code_first": "parallel",
    "error_first": "hybrid_only",  # error content handled by keyword weights inside workflow
}


async def retrieve_knowledge(state: AgentState) -> dict[str, Any]:
    """Retrieve knowledge — dispatches to BaseRAGWorkflow by mode.

    Dispatch logic (based on deep_intent RetrievalMode):
    - graph_first  → BaseRAGWorkflow(mode="graph_first")
    - parallel      → BaseRAGWorkflow(mode="parallel")  [code gen, API usage]
    - hybrid_only / default → BaseRAGWorkflow(mode="hybrid_only")

    Also integrates:
    - Semantic cache: skip retrieval on cache hit
    - Cross-Encoder reranking: used inside the workflow

    Falls back gracefully on any failure (returns empty evidence + logs).
    """
    query = state.get("query", "")
    permission_scope = cache_scope(state)
    cache_query = f"{permission_scope}\n{query}"
    t0 = time.time()

    # ── Tier 0: Semantic cache check ──
    try:
        from enterprise_agentic_rag.rag.semantic_cache import get_semantic_cache

        cache = get_semantic_cache()
        if cache.enabled:
            cache_result = await cache.get(cache_query)
            if cache_result is not None:
                hit_type, cached_value = cache_result
                latency_ms = (time.time() - t0) * 1000
                logger.info("Semantic cache %s hit, latency=%.0fms", hit_type, latency_ms)
                tracer.record_retrieval_event(
                    dict(state),
                    query=query,
                    num_docs=len(cached_value.get("retrieved_docs", [])),
                    top_score=1.0,
                    latency_ms=latency_ms,
                    success=True,
                )
                return {
                    **cached_value,
                    "retrieval_mode": f"cache_{hit_type}",
                    "last_worker": "retrieval_service",
                    "last_agent_step": "retrieve",
                }
    except Exception as exc:
        logger.debug("Semantic cache lookup skipped: %s", exc)

    # ── Determine retrieval mode from deep_intent ──
    deep_intent = state.get("deep_intent", {})
    retrieval_plan = deep_intent.get("retrieval_plan", {})
    mode = retrieval_plan.get("mode", "hybrid_only") if isinstance(retrieval_plan, dict) else "hybrid_only"
    primary_intent = deep_intent.get("primary_intent", "concept_qa")
    entities = deep_intent.get("entities", {})
    top_k = 10

    # Map to supported workflow mode
    workflow_mode = _MODE_MAP.get(mode, "hybrid_only")

    # ── Tier 1: BaseRAGWorkflow dispatch ──
    errors: list[str] = []
    workflow_result: dict[str, Any] = {}

    try:
        from enterprise_agentic_rag.workflows import BaseRAGWorkflow

        wf = BaseRAGWorkflow()
        workflow_result = await wf.execute(
            query=query,
            mode=workflow_mode,
            top_k=top_k,
            intent=primary_intent,
            entities=entities,
        )
        errors.extend(workflow_result.get("errors", []))

    except Exception as exc:
        logger.warning("BaseRAGWorkflow failed: %s — returning empty evidence", exc)
        errors.append(f"BaseRAGWorkflow failed: {exc}")
        workflow_result = {"selected_evidence": [], "errors": errors}

    # ── Assemble results ──
    evidence = workflow_result.get("selected_evidence", [])
    reranked = workflow_result.get("reranked_results", evidence)

    latency_ms = (time.time() - t0) * 1000
    top_score = max((r.get("score", 0) for r in evidence), default=0.0)

    tracer.record_retrieval_event(
        dict(state),
        query=query,
        num_docs=len(evidence),
        top_score=top_score,
        latency_ms=latency_ms,
        success=len(evidence) > 0,
    )

    result_state: dict[str, Any] = {
        "retrieved_docs": evidence,
        "reranked_docs": reranked,
        "query_analysis": state.get("query_analysis", {}),
        "retrieval_plan": retrieval_plan,
        "retrieval_mode": mode,
        "retrieval_errors": errors,
        "last_worker": "retrieval_service",
        "last_agent_step": "retrieve",
    }

    # ── Tier 2: Store in semantic cache ──
    if evidence:
        try:
            from enterprise_agentic_rag.rag.semantic_cache import get_semantic_cache

            cache = get_semantic_cache()
            if cache.enabled:
                cacheable = {
                    "retrieved_docs": evidence,
                    "reranked_docs": reranked,
                    "retrieval_mode": mode,
                    "retrieval_errors": errors,
                }
                asyncio.ensure_future(cache.set(cache_query, cacheable))
        except Exception:
            pass  # Cache write is non-critical

    # Check for no results / low quality
    if not evidence or all(r.get("score", 0) <= 0 for r in evidence):
        fb = _evaluate_failure(dict(state), fallback_type="no_relevant_docs")
        return {**result_state, **fb}

    if all(r.get("score", 0) < 0.1 for r in evidence):
        fb = _evaluate_failure(dict(state), fallback_type="low_retrieval_score")
        return {**result_state, **fb}

    return result_state


def _evaluate_failure(state: dict[str, Any], fallback_type: str) -> dict[str, Any]:
    """Evaluate failure and return recovery actions.

    Imports recovery lazily to avoid circular dependencies.
    """
    try:
        from enterprise_agentic_rag.graph.dependencies import recovery

        return recovery.evaluate_failure(state, fallback_type=fallback_type)
    except Exception:
        return {}


async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrite the query for a second retrieval attempt.

    Records the retry and produces a reformulated query.
    """
    original = state.get("query", "")

    try:
        from enterprise_agentic_rag.graph.dependencies import recovery

        rewritten = recovery.rewrite_query(original)
        retry_updates = recovery.record_retry(
            dict(state),
            node_key="retrieve",
            reason=f"原始查询无结果，改写为: {rewritten}",
        )
    except Exception:
        # Recovery unavailable — use simple rewrite
        rewritten = original
        retry_updates = {}

    return {
        **retry_updates,
        "query": rewritten,
        "fallback_reason": "",
        "recovery_action": "retry",
        "recoverable": True,
        "last_worker": "retrieval_service",
        "last_agent_step": "rewrite_query",
    }
