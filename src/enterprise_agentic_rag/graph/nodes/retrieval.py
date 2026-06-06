"""Retrieval and query-rewrite nodes.

The ``retrieve_knowledge`` node implements the 5-tier fallback chain:
- Tier 0: semantic cache
- Tier 1: intent-aware workflow dispatch (hybrid / graph / error / code)
- Tier 2: GraphRAG Orchestrator
- Tier 3: legacy Retriever
- Tier 4: external search augmentation
- Tier 5: write-back to cache
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enterprise_agentic_rag.graph.cache import cache_scope
from enterprise_agentic_rag.graph.dependencies import recovery, retriever, tracer
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)


async def retrieve_knowledge(state: AgentState) -> dict[str, Any]:
    """Retrieve knowledge — dispatches to intent-aware retrieval workflow.

    Dispatch logic (based on deep_intent RetrievalMode):
    - graph_first → GraphFirstWorkflow (migration, compatibility, API relationships)
    - error_first → ErrorFirstWorkflow (error diagnosis, crash analysis)
    - code_first → CodeGenerationWorkflow (code gen, API usage examples)
    - hybrid_only / parallel / default → HybridRAGWorkflow

    Also integrates:
    - Semantic cache: skip retrieval on cache hit
    - External search: augment with GitHub/StackOverflow/Web results
    - Cross-Encoder reranking: used inside each workflow

    Falls back gracefully on any failure.
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

    # ── Tier 1: Dispatch to intent-aware workflow ──
    errors: list[str] = []
    workflow_result: dict[str, Any] = {}

    try:
        if mode == "graph_first":
            from enterprise_agentic_rag.workflows.graph_first_workflow import GraphFirstWorkflow

            wf = GraphFirstWorkflow()
            workflow_result = await wf.execute(
                query=query, top_k=top_k, intent=primary_intent, entities=entities,
            )
        elif mode == "error_first":
            from enterprise_agentic_rag.workflows.error_first_workflow import ErrorFirstWorkflow

            wf = ErrorFirstWorkflow()
            workflow_result = await wf.execute(
                query=query, top_k=top_k, intent=primary_intent, entities=entities,
            )
        elif mode == "code_first":
            from enterprise_agentic_rag.workflows.code_generation_workflow import CodeGenerationWorkflow

            wf = CodeGenerationWorkflow()
            workflow_result = await wf.execute(
                query=query, top_k=top_k, intent=primary_intent, entities=entities,
            )
        else:
            from enterprise_agentic_rag.workflows.hybrid_rag_workflow import HybridRAGWorkflow

            wf = HybridRAGWorkflow()
            workflow_result = await wf.execute(
                query=query, top_k=top_k, intent=primary_intent, entities=entities,
            )

        errors.extend(workflow_result.get("errors", []))

    except Exception as exc:
        logger.warning("Intent-aware workflow failed: %s — falling back to original RAG", exc)
        errors.append(f"Workflow dispatch failed: {exc}")
        workflow_result = {}

    # ── Tier 2: Fallback to original GraphRAG Orchestrator ──
    if not workflow_result.get("selected_evidence"):
        try:
            from enterprise_agentic_rag.rag.graph_rag_orchestrator import GraphRAGOrchestrator

            orchestrator = GraphRAGOrchestrator()
            rag_result = await orchestrator.retrieve(
                query=query,
                query_analysis=state.get("query_analysis"),
                top_k=5,
            )
            workflow_result["selected_evidence"] = rag_result.get("retrieved_docs", [])
            if rag_result.get("errors"):
                errors.extend(rag_result["errors"])
        except Exception as exc:
            logger.warning("GraphRAG orchestrator fallback failed: %s", exc)
            errors.append(f"GraphRAG fallback failed: {exc}")

    # ── Tier 3: Ultimate fallback to old Retriever ──
    if not workflow_result.get("selected_evidence"):
        try:
            results = retriever.search(query)
            workflow_result["selected_evidence"] = results
        except Exception as exc:
            logger.warning("Ultimate retriever fallback failed: %s", exc)
            errors.append(f"Ultimate retriever fallback failed: {exc}")
            workflow_result["selected_evidence"] = []

    # ── Tier 4: External search augmentation ──
    # (external search disabled — moved to retrieval agent layer in v3.2)

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

    # ── Tier 5: Store in semantic cache ──
    if evidence:
        try:
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
        fb = recovery.evaluate_failure(dict(state), fallback_type="no_relevant_docs")
        return {**result_state, **fb}

    if all(r.get("score", 0) < 0.1 for r in evidence):
        fb = recovery.evaluate_failure(dict(state), fallback_type="low_retrieval_score")
        return {**result_state, **fb}

    return result_state


async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrite the query for a second retrieval attempt.

    Records the retry and produces a reformulated query.
    """
    original = state.get("query", "")
    rewritten = recovery.rewrite_query(original)

    retry_updates = recovery.record_retry(
        dict(state),
        node_key="retrieve",
        reason=f"原始查询无结果，改写为: {rewritten}",
    )

    return {
        **retry_updates,
        "query": rewritten,
        "fallback_reason": "",
        "recovery_action": "retry",
        "recoverable": True,
        "last_worker": "retrieval_service",
        "last_agent_step": "rewrite_query",
    }
