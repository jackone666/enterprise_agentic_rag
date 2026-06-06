"""RetrievalAgent — Agent-abstracted RAG retrieval with 3-tier fallback.

Design
~~~~~~~
``RetrievalAgent`` is the internal agent responsible for all knowledge-base
retrieval.  It owns the 3-tier fallback chain:

* **Tier 0 — cache_hit**: semantic cache check; no retrieval needed.
* **Tier 1 — workflow**: delegate to ``BaseRAGWorkflow`` dispatching on mode.
* **Tier 2 — fail**: no recoverable evidence; return empty evidence.

Each invocation records its events in the tracer (for observability) and
updates the shared state dict with the retrieval result patch.

The graph node ``retrieve_knowledge`` (in ``graph/nodes/retrieval.py``) is
now a thin wrapper that instantiates ``RetrievalAgent`` and calls
``run(state)``. The node signature stays unchanged so ``graph/builder.py``
needs no modification.

v3.2 simplification
~~~~~~~~~~~~~~~~~~~
Replaces the 5-tier fallback chain (cache → GraphRAG Orchestrator →
hybrid workflow → legacy retriever → fail) with the 3 tiers above.
``BaseRAGWorkflow`` handles all mode-specific retrieval internally with
graceful degradation, so no separate orchestrator tier is needed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enterprise_agentic_rag.graph.cache import cache_scope
from enterprise_agentic_rag.graph.dependencies import tracer
from enterprise_agentic_rag.workflows import BaseRAGWorkflow

logger = logging.getLogger(__name__)

# Mode mapping: schema RetrievalMode → BaseRAGWorkflow mode
_MODE_MAP = {
    "hybrid_only": "hybrid_only",
    "graph_first": "graph_first",
    "parallel": "parallel",
    # Deprecated schema values mapped to nearest supported mode
    "code_first": "parallel",
    "error_first": "hybrid_only",
}

# Error patterns that warrant a query rewrite before retry
_RETRYABLE_PATTERNS = frozenset({
    "timeout", "connection refused", "temporary failure",
    "rate limit", "429", "503", "504",
})


class RetrievalAgent:
    """Agent-abstracted retrieval with 3-tier fallback and mode dispatch.

    Attributes
    ----------
    retrieval_path : str
        One of ``"cache_hit"``, ``"workflow"``, ``"fail"`` — set after
        each ``run()`` call so callers can inspect how the result was
        obtained.
    events : list[dict]
        Working memory: all events recorded during the current session.
    """

    def __init__(
        self,
        workflow: BaseRAGWorkflow | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize the RetrievalAgent.

        Parameters
        ----------
        workflow : BaseRAGWorkflow | None
            Optional workflow instance. If omitted the agent builds its own.
        max_retries : int
            Number of retries on transient retrieval failures before giving up.
        """
        self._workflow = workflow or BaseRAGWorkflow()
        self._max_retries = max_retries
        self._events: list[dict[str, Any]] = []
        self.retrieval_path: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute retrieval and return a state patch dict.

        Parameters
        ----------
        state : dict[str, Any]
            Current LangGraph state.  Required keys: ``query``,
            ``deep_intent`` (may be empty dict).

        Returns
        -------
        dict[str, Any]
            State patch to be merged upstream.  Keys:
            ``retrieved_docs``, ``reranked_docs``, ``retrieval_plan``,
            ``retrieval_mode``, ``retrieval_errors``, ``retrieval_path``,
            ``last_worker``, ``last_agent_step``.
        """
        t0 = time.time()
        query = state.get("query", "")
        permission_scope = cache_scope(state)
        cache_key = f"{permission_scope}\n{query}"

        # ── Tier 0: Semantic cache ──────────────────────────────────────
        cached = await self._try_cache(cache_key)
        if cached is not None:
            hit_type, cached_value = cached
            latency_ms = (time.time() - t0) * 1000
            self.retrieval_path = "cache_hit"
            self._record_event(
                state,
                path=self.retrieval_path,
                query=query,
                num_docs=len(cached_value.get("retrieved_docs", [])),
                top_score=1.0,
                latency_ms=latency_ms,
                success=True,
            )
            return {
                **cached_value,
                "retrieval_path": self.retrieval_path,
                "last_worker": "retrieval_agent",
                "last_agent_step": "retrieve",
            }

        # ── Tier 1: BaseRAGWorkflow ─────────────────────────────────────
        deep_intent = state.get("deep_intent", {})
        mode = self._resolve_mode(deep_intent)
        primary_intent = deep_intent.get("primary_intent", "concept_qa")
        entities = deep_intent.get("entities", {})
        top_k = 10

        evidence, errors = await self._retrieve_with_retry(
            query=query,
            mode=mode,
            top_k=top_k,
            intent=primary_intent,
            entities=entities,
        )

        latency_ms = (time.time() - t0) * 1000
        top_score = max((r.get("score", 0) for r in evidence), default=0.0)

        if evidence:
            self.retrieval_path = "workflow"
            result_state = self._assemble_result(
                evidence=evidence,
                reranked=evidence,
                mode=mode,
                errors=errors,
                retrieval_plan=deep_intent.get("retrieval_plan", {}),
            )
            # Cache the result for next time
            await self._cache_result(cache_key, result_state)
        else:
            self.retrieval_path = "fail"
            result_state = self._assemble_result(
                evidence=[],
                reranked=[],
                mode=mode,
                errors=errors or ["No relevant documents retrieved"],
                retrieval_plan=deep_intent.get("retrieval_plan", {}),
            )
            fb = self._evaluate_failure(state, fallback_type="no_relevant_docs")
            result_state = {**result_state, **fb}

        self._record_event(
            state,
            path=self.retrieval_path,
            query=query,
            num_docs=len(evidence),
            top_score=top_score,
            latency_ms=latency_ms,
            success=len(evidence) > 0,
        )

        return {
            **result_state,
            "retrieval_path": self.retrieval_path,
            "last_worker": "retrieval_agent",
            "last_agent_step": "retrieve",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _try_cache(
        self, cache_key: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Return (hit_type, cached_value) on cache hit, else None."""
        try:
            from enterprise_agentic_rag.rag.semantic_cache import get_semantic_cache

            cache = get_semantic_cache()
            if cache.enabled:
                result = await cache.get(cache_key)
                if result is not None:
                    logger.info("Semantic cache %s hit", result[0])
                    return result
        except Exception as exc:
            logger.debug("Semantic cache lookup skipped: %s", exc)
        return None

    async def _cache_result(
        self, cache_key: str, result_state: dict[str, Any]
    ) -> None:
        """Write retrieval result to semantic cache (non-critical)."""
        try:
            from enterprise_agentic_rag.rag.semantic_cache import get_semantic_cache

            cache = get_semantic_cache()
            if cache.enabled:
                evidence = result_state.get("retrieved_docs", [])
                if evidence:
                    asyncio.ensure_future(
                        cache.set(
                            cache_key,
                            {
                                "retrieved_docs": evidence,
                                "reranked_docs": result_state.get("reranked_docs", []),
                                "retrieval_mode": result_state.get("retrieval_mode", ""),
                                "retrieval_errors": result_state.get("retrieval_errors", []),
                            },
                        )
                    )
        except Exception:
            pass  # Non-critical

    async def _retrieve_with_retry(
        self,
        query: str,
        mode: str,
        top_k: int,
        intent: str,
        entities: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Call BaseRAGWorkflow with retry on transient errors."""
        errors: list[str] = []
        attempt = 0

        while attempt <= self._max_retries:
            attempt += 1
            try:
                result = await self._workflow.execute(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    intent=intent,
                    entities=entities,
                )
                errors.extend(result.get("errors", []))
                evidence = result.get("selected_evidence", [])

                # Check for low-quality evidence
                if evidence and all(r.get("score", 0) < 0.1 for r in evidence):
                    if attempt <= self._max_retries:
                        logger.warning(
                            "Retrieval attempt %d/%d returned low scores, retrying",
                            attempt, self._max_retries,
                        )
                        continue

                return evidence, errors

            except Exception as exc:
                err_str = str(exc)
                errors.append(f"workflow: {exc}")
                if self._is_retryable(err_str) and attempt <= self._max_retries:
                    logger.warning(
                        "Retrieval attempt %d/%d failed with retryable error: %s",
                        attempt, self._max_retries, err_str,
                    )
                    continue
                # Non-retryable or exhausted — return empty
                return [], errors

        return [], errors

    async def _cache_result_on_result(
        self, cache_key: str, evidence: list[dict[str, Any]], mode: str, errors: list[str]
    ) -> None:
        """Cache write helper (kept for explicit call sites)."""
        if evidence:
            await self._cache_result(
                cache_key,
                {
                    "retrieved_docs": evidence,
                    "reranked_docs": evidence,
                    "retrieval_mode": mode,
                    "retrieval_errors": errors,
                },
            )

    @staticmethod
    def _resolve_mode(deep_intent: dict[str, Any]) -> str:
        """Resolve the retrieval mode from deep_intent.retrieval_plan."""
        retrieval_plan = deep_intent.get("retrieval_plan", {})
        raw_mode = (
            retrieval_plan.get("mode", "hybrid_only")
            if isinstance(retrieval_plan, dict)
            else "hybrid_only"
        )
        return _MODE_MAP.get(raw_mode, "hybrid_only")

    @staticmethod
    def _assemble_result(
        evidence: list[dict[str, Any]],
        reranked: list[dict[str, Any]],
        mode: str,
        errors: list[str],
        retrieval_plan: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "retrieved_docs": evidence,
            "reranked_docs": reranked,
            "retrieval_plan": retrieval_plan,
            "retrieval_mode": mode,
            "retrieval_errors": errors,
        }

    @staticmethod
    def _is_retryable(error: str) -> bool:
        if not error:
            return False
        lower = error.lower()
        return any(pat.lower() in lower for pat in _RETRYABLE_PATTERNS)

    @staticmethod
    def _evaluate_failure(state: dict[str, Any], fallback_type: str) -> dict[str, Any]:
        """Lazily import recovery to avoid circular dependencies."""
        try:
            from enterprise_agentic_rag.graph.dependencies import recovery
            return recovery.evaluate_failure(state, fallback_type=fallback_type)
        except Exception:
            return {}

    def _record_event(
        self,
        state: dict[str, Any],
        path: str,
        query: str,
        num_docs: int,
        top_score: float,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record a retrieval event in the tracer and working memory."""
        evt = {
            "path": path,
            "query": query,
            "num_docs": num_docs,
            "top_score": top_score,
            "latency_ms": latency_ms,
            "success": success,
        }
        self._events.append(evt)
        tracer.record_retrieval_event(
            state,
            query=query,
            num_docs=num_docs,
            top_score=top_score,
            latency_ms=latency_ms,
            success=success,
        )