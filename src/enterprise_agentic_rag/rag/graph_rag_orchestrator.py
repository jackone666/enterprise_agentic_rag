"""Graph-Augmented Hybrid RAG Orchestrator.

Ties together:
- RetrievalRouter → determines mode
- GraphRetriever → graph candidates
- KeywordRetriever / VectorRetriever → existing retrievers
- Three-way fusion → combined results
- Reranker → final ranking

Key guarantees:
1. When ENABLE_GRAPH_RAG=false, falls back to existing Hybrid RAG.
2. When Neo4j is unavailable, auto-degrades to keyword + vector.
3. Single retriever failure does not crash the pipeline.
4. Graph indexing failure does not affect existing ingestion.
5. All errors and degradation events are recorded in RetrievalTrace.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from enterprise_agentic_rag.config.settings import get_settings
from enterprise_agentic_rag.rag.fusion import (
    normalize_weights_for_fallback,
    three_way_rrf_fusion,
    weighted_rrf_fusion,
)
from enterprise_agentic_rag.rag.graph.graph_retriever import GraphRetriever
from enterprise_agentic_rag.rag.graph.graph_schema import Candidate, RetrievalPlan
from enterprise_agentic_rag.rag.observability.retrieval_trace import (
    RetrievalTrace,
    get_retrieval_tracer,
)
from enterprise_agentic_rag.rag.retrieval_router import RetrievalRouter

logger = logging.getLogger(__name__)

# Lazy singleton for external retriever
_external_retriever = None


def _get_external_retriever():
    """Lazy init of the external retriever singleton."""
    global _external_retriever
    if _external_retriever is None:
        from enterprise_agentic_rag.rag.external.external_retriever import ExternalRetriever
        _external_retriever = ExternalRetriever()
    return _external_retriever


class GraphRAGOrchestrator:
    """Orchestrate Graph-Augmented Hybrid RAG retrieval.

    Reuses existing keyword and vector retrievers.
    Adds graph retrieval with dynamic routing and fallback.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._router = RetrievalRouter()
        self._graph_retriever = GraphRetriever()
        self._tracer = get_retrieval_tracer()

    @property
    def graph_enabled(self) -> bool:
        self._settings = get_settings()
        return self._settings.graph_rag.enabled

    @property
    def graph_available(self) -> bool:
        return self.graph_enabled and self._graph_retriever.available

    # ------------------------------------------------------------------
    # Main orchestration entry point
    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        query_analysis: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Execute Graph-Augmented Hybrid RAG retrieval.

        Args:
            query: Raw user query.
            query_analysis: Optional pre-computed analysis.
            top_k: Number of final results.

        Returns:
            Dict with:
            - retrieved_docs: final ranked documents (list of dicts)
            - retrieval_trace: RetrievalTrace dict
            - degraded_from: degradation source (or "")
            - degraded_to: degradation target (or "")
            - errors: list of error strings
            - query_analysis: analysis dict
        """
        t0 = time.time()
        trace_id = uuid.uuid4().hex[:12]
        self._settings = get_settings()
        query_analysis = query_analysis or self._analyze_query(query)
        errors: list[str] = []

        # Create trace
        trace = self._tracer.create_trace(
            trace_id=trace_id,
            query=query,
            query_analysis=query_analysis,
        )

        # 1. Route: determine retrieval plan
        plan = self._router.route(query, query_analysis)
        self._tracer.populate_from_plan(trace, plan)

        # 2. If graph RAG disabled globally, fast-path to hybrid
        if not self.graph_enabled:
            logger.info("Graph RAG disabled — using hybrid_only")
            result = await self._execute_hybrid_fallback(
                query, top_k, trace, plan,
                degraded_from=plan.mode,
                degraded_to="hybrid_only",
            )
        else:
            # 3. Execute based on plan mode
            try:
                if plan.mode == "hybrid_only":
                    degraded_from = "graph_unavailable" if not self.graph_available else ""
                    degraded_to = "hybrid_only" if degraded_from else ""
                    result = await self._execute_hybrid_fallback(
                        query, top_k, trace, plan,
                        degraded_from=degraded_from,
                        degraded_to=degraded_to,
                    )
                elif plan.mode == "graph_first":
                    result = await self._execute_graph_first(
                        query, top_k, trace, plan, query_analysis, errors,
                    )
                elif plan.mode in ("parallel", "keyword_first", "vector_first"):
                    result = await self._execute_parallel(
                        query, top_k, trace, plan, query_analysis, errors,
                    )
                else:
                    # Unknown mode → fallback
                    logger.warning("Unknown retrieval mode: %s — falling back to hybrid", plan.mode)
                    result = await self._execute_hybrid_fallback(
                        query, top_k, trace, plan,
                        degraded_from=plan.mode,
                        degraded_to="hybrid_only",
                    )
            except Exception as exc:
                logger.error("Graph RAG orchestration failed: %s — falling back to hybrid", exc)
                errors.append(f"Orchestration error: {exc}")
                result = await self._execute_hybrid_fallback(
                    query, top_k, trace, plan,
                    degraded_from=plan.mode,
                    degraded_to="hybrid_only",
                )

        # 4. External search — triggered when internal results are poor
        external_docs: list[dict] = []
        if plan.enable_external:
            try:
                external_t0 = time.time()
                ext_retriever = _get_external_retriever()
                if ext_retriever.available:
                    external_docs = await ext_retriever.search(
                        query=query,
                        sources=plan.external_sources or None,
                        top_k=3,
                    )
                    trace.external_search_latency_ms = round((time.time() - external_t0) * 1000, 2)
                    trace.external_hit_count = len(external_docs)
                    logger.info("External search returned %d results", len(external_docs))
            except Exception as exc:
                logger.warning("External search failed (non-fatal): %s", exc)
                errors.append(f"External search: {exc}")

        # Merge external results into retrieved docs
        if external_docs:
            existing = result.get("retrieved_docs", [])
            result["retrieved_docs"] = existing + external_docs

        # Finalize trace — sync degradation info from plan back to trace
        trace.total_latency_ms = round((time.time() - t0) * 1000, 2)
        trace.errors = errors

        # Propagate plan degradation fields to trace (set by _execute_* methods)
        if plan.degraded_from:
            trace.degraded_from = plan.degraded_from
            trace.degraded_to = plan.degraded_to
        if plan.graph_failed:
            trace.graph_failed = True

        # Log trace summary
        trace.log_summary()

        result["retrieval_trace"] = trace.to_dict()
        result["query_analysis"] = query_analysis
        result["errors"] = errors

        return result

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------

    async def _execute_parallel(
        self,
        query: str,
        top_k: int,
        trace: RetrievalTrace,
        plan: RetrievalPlan,
        query_analysis: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any]:
        """Execute parallel retrieval: keyword + vector + graph concurrently.

        Rule 7: Three tasks launched together via asyncio.ensure_future.
        Graph failure is non-fatal:
        - graph_candidates = []
        - graph weight → 0
        - keyword / vector weights re-normalized
        - trace.graph_failed = True
        - Other retrievers continue unaffected.
        """
        degraded_from = ""
        degraded_to = ""

        keyword_t0 = time.time()
        vector_t0 = time.time()
        graph_t0 = time.time()

        # Rule 7: launch all three tasks simultaneously
        kw_task = asyncio.ensure_future(
            self._keyword_retrieve_async(query, plan.top_k.get("keyword", top_k))
        )
        vec_task = asyncio.ensure_future(
            self._vector_retrieve_async(query, plan.top_k.get("vector", top_k))
        )

        # Graph retrieval (optional, parallel — Rule 7)
        graph_task = None
        graph_results: list[dict] = []
        if "graph" in plan.enabled_retrievers and self.graph_available:
            graph_task = asyncio.ensure_future(
                self._graph_retrieve_async(query_analysis, plan)
            )

        # Gather keyword + vector
        kw_results, vec_results = await asyncio.gather(kw_task, vec_task, return_exceptions=True)
        if isinstance(kw_results, Exception):
            errors.append(f"Keyword retrieval failed: {kw_results}")
            kw_results = []
        if isinstance(vec_results, Exception):
            errors.append(f"Vector retrieval failed: {vec_results}")
            vec_results = []

        trace.keyword_latency_ms = round((time.time() - keyword_t0) * 1000, 2)
        trace.vector_latency_ms = round((time.time() - vector_t0) * 1000, 2)
        trace.keyword_hit_count = len(kw_results)
        trace.vector_hit_count = len(vec_results)

        # Gather graph results (Rule 7: non-fatal)
        if graph_task:
            try:
                graph_results = await graph_task
            except Exception as exc:
                errors.append(f"Graph retrieval failed: {exc}")
                graph_results = []
                degraded_from = plan.mode
                degraded_to = "parallel_keyword_vector_only"
                trace.graph_failed = True
                logger.warning(
                    "parallel mode: graph retrieval failed — "
                    "graph weight → 0, continuing with keyword+vector"
                )

        trace.graph_latency_ms = round((time.time() - graph_t0) * 1000, 2)
        trace.graph_hit_count = len(graph_results)

        # Determine available retrievers for fusion
        available = ["keyword", "vector"]
        if graph_results:
            available.append("graph")

        # Rule 7: re-normalize weights when graph fails
        weights = normalize_weights_for_fallback(plan.weights, available)
        trace.fusion_weights = weights

        fusion_t0 = time.time()

        if "graph" in available and weights.get("graph", 0) > 0:
            # Three-way fusion
            fused = three_way_rrf_fusion(
                keyword_candidates=kw_results,
                vector_candidates=vec_results,
                graph_candidates=graph_results,
                keyword_weight=weights.get("keyword", 0.3),
                vector_weight=weights.get("vector", 0.5),
                graph_weight=weights.get("graph", 0.2),
                k=self._settings.fusion.rrf_k,
                top_n=top_k * 2,  # Get more for reranker
            )
            trace.fusion_method = "rrf_three_way"
        else:
            # Two-way fusion (keyword + vector, graph failed or not in plan)
            fused = weighted_rrf_fusion(
                vector_results=vec_results,
                keyword_results=kw_results,
                vector_weight=weights.get("vector", 0.6),
                keyword_weight=weights.get("keyword", 0.4),
                k=self._settings.fusion.rrf_k,
                top_n=top_k * 2,
            )
            trace.fusion_method = "rrf_two_way"

        trace.fusion_latency_ms = round((time.time() - fusion_t0) * 1000, 2)
        trace.merged_count = len(fused)

        # Rerank
        reranked = self._rerank(query, fused, top_k)

        # Count graph paths
        graph_paths_count = 0
        for doc in reranked:
            paths = doc.get("graph_paths", [])
            graph_paths_count += len(paths)
        trace.graph_paths_count = graph_paths_count
        trace.reranked_count = len(reranked)

        # Rule 7: sync degradation back to plan for full observability
        if degraded_from:
            plan.degraded_from = degraded_from
            plan.degraded_to = degraded_to
            plan.graph_failed = True

        return {
            "retrieved_docs": reranked,
            "degraded_from": degraded_from,
            "degraded_to": degraded_to,
        }

    async def _execute_graph_first(
        self,
        query: str,
        top_k: int,
        trace: RetrievalTrace,
        plan: RetrievalPlan,
        query_analysis: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any]:
        """Execute graph_first mode: graph → query expansion → keyword + vector.

        Rule 8:
        Stage 1: Graph retrieval to find entities, paths, evidence chunks.
        Stage 2: Expand query with graph_paths + related_entities, then
                 keyword+vector with expanded_query in parallel.
        Fallback: If graph fails or returns empty → degraded_to=hybrid_only,
                  falls straight through to keyword+vector hybrid search.
        """
        graph_t0 = time.time()

        # Stage 1: Graph retrieval (Rule 8)
        try:
            graph_results = await self._graph_retrieve_async(query_analysis, plan)
        except Exception as exc:
            errors.append(f"Graph retrieval failed in graph_first: {exc}")
            trace.graph_failed = True
            trace.graph_latency_ms = round((time.time() - graph_t0) * 1000, 2)
            logger.warning("graph_first: graph failed → degraded_from=graph_first "
                           "degraded_to=hybrid_only")
            plan.degraded_from = "graph_first"
            plan.degraded_to = "hybrid_only"
            plan.graph_failed = True
            return await self._execute_hybrid_fallback(
                query, top_k, trace, plan,
                degraded_from="graph_first",
                degraded_to="hybrid_only",
            )

        trace.graph_latency_ms = round((time.time() - graph_t0) * 1000, 2)
        trace.graph_hit_count = len(graph_results)

        if not graph_results:
            trace.graph_failed = True
            logger.info("graph_first: no graph results → degraded_from=graph_first "
                        "degraded_to=hybrid_only")
            plan.degraded_from = "graph_first"
            plan.degraded_to = "hybrid_only"
            plan.graph_failed = True
            return await self._execute_hybrid_fallback(
                query, top_k, trace, plan,
                degraded_from="graph_first",
                degraded_to="hybrid_only",
            )

        # Stage 2: Query expansion
        from enterprise_agentic_rag.rag.graph.query_expander import QueryExpander

        # Convert graph_results (dicts) to Candidates for QueryExpander
        graph_candidates = [_dict_to_candidate(d) for d in graph_results]

        expander = QueryExpander()
        expansion = expander.expand(query, graph_candidates)

        expanded_query = expansion["expanded_query"]
        trace.original_query = query
        trace.expanded_query = expanded_query
        trace.expansion_terms = expansion["expansion_terms"]

        logger.info("graph_first: expanded query with %d terms", len(expansion["expansion_terms"]))

        # Stage 3: Keyword + Vector retrieval with expanded query
        kw_t0 = time.time()
        vec_t0 = time.time()

        kw_task = asyncio.ensure_future(
            self._keyword_retrieve_async(expanded_query, plan.top_k.get("keyword", top_k))
        )
        vec_task = asyncio.ensure_future(
            self._vector_retrieve_async(expanded_query, plan.top_k.get("vector", top_k))
        )

        kw_results, vec_results = await asyncio.gather(kw_task, vec_task, return_exceptions=True)
        if isinstance(kw_results, Exception):
            errors.append(f"Keyword retrieval failed: {kw_results}")
            kw_results = []
        if isinstance(vec_results, Exception):
            errors.append(f"Vector retrieval failed: {vec_results}")
            vec_results = []

        trace.keyword_latency_ms = round((time.time() - kw_t0) * 1000, 2)
        trace.vector_latency_ms = round((time.time() - vec_t0) * 1000, 2)
        trace.keyword_hit_count = len(kw_results)
        trace.vector_hit_count = len(vec_results)

        # Stage 4: Three-way fusion
        fusion_t0 = time.time()

        available = ["keyword", "vector", "graph"]
        weights = normalize_weights_for_fallback(plan.weights, available)
        trace.fusion_weights = weights

        fused = three_way_rrf_fusion(
            keyword_candidates=kw_results,
            vector_candidates=vec_results,
            graph_candidates=graph_results,
            keyword_weight=weights.get("keyword", 0.2),
            vector_weight=weights.get("vector", 0.3),
            graph_weight=weights.get("graph", 0.5),
            k=self._settings.fusion.rrf_k,
            top_n=top_k * 2,
        )
        trace.fusion_method = "rrf_three_way_graph_first"
        trace.fusion_latency_ms = round((time.time() - fusion_t0) * 1000, 2)
        trace.merged_count = len(fused)

        # Rerank
        reranked = self._rerank(query, fused, top_k)

        # Count graph paths
        graph_paths_count = 0
        for doc in reranked:
            paths = doc.get("graph_paths", [])
            graph_paths_count += len(paths)
        trace.graph_paths_count = graph_paths_count
        trace.reranked_count = len(reranked)

        return {
            "retrieved_docs": reranked,
            "degraded_from": "",
            "degraded_to": "",
        }

    async def _execute_hybrid_fallback(
        self,
        query: str,
        top_k: int,
        trace: RetrievalTrace,
        plan: RetrievalPlan,
        degraded_from: str = "",
        degraded_to: str = "",
    ) -> dict[str, Any]:
        """Fallback to original Hybrid RAG (keyword + vector only).

        Rule 8 (graph_first failure) + Rule 9 (hybrid_only):
        Completely reuses the original keyword + vector hybrid search.
        No graph retriever called. No graph_paths generated.
        """
        logger.info("Executing hybrid fallback: %s → %s",
                     degraded_from or "none", degraded_to or "hybrid_only")

        # Rule 9: hybrid_only — zero graph involvement
        trace.degraded_from = degraded_from
        trace.degraded_to = degraded_to
        trace.mode = "hybrid_only"
        trace.enabled_retrievers = ["keyword", "vector"]
        trace.fusion_weights = {"keyword": 0.4, "vector": 0.6, "graph": 0.0}

        # Sync degradation back to plan
        if degraded_from or degraded_to:
            plan.degraded_from = degraded_from
            plan.degraded_to = degraded_to

        fusion_t0 = time.time()

        try:
            from enterprise_agentic_rag.rag.fusion import fusion_retrieve
            results = fusion_retrieve(query, top_k=top_k)
        except Exception:
            # Ultimate fallback: in-memory keyword retriever
            from enterprise_agentic_rag.rag.retriever import KeywordRetriever
            kr = KeywordRetriever(top_k=top_k)
            results = kr.search(query)

        trace.fusion_latency_ms = round((time.time() - fusion_t0) * 1000, 2)
        trace.fusion_method = "rrf_two_way"
        trace.fusion_weights = {"keyword": 0.4, "vector": 0.6, "graph": 0.0}
        trace.merged_count = len(results)

        # Rerank
        reranked = self._rerank(query, results, top_k)
        trace.reranked_count = len(reranked)

        # Update hit counts from results
        trace.keyword_hit_count = len(results)  # Best-effort estimate
        trace.vector_hit_count = len(results)

        return {
            "retrieved_docs": reranked,
            "degraded_from": degraded_from,
            "degraded_to": degraded_to,
        }

    # ------------------------------------------------------------------
    # Async retriever wrappers
    # ------------------------------------------------------------------

    async def _keyword_retrieve_async(self, query: str, top_k: int) -> list[dict]:
        """Async wrapper for keyword retrieval (ES or memory Jaccard)."""
        from enterprise_agentic_rag.rag.es_keyword_store import ESKeywordStore
        from enterprise_agentic_rag.rag.retriever import KeywordRetriever

        es_store = ESKeywordStore()
        if es_store.available:
            return es_store.search(query, top_k=top_k)

        kr = KeywordRetriever(top_k=top_k)
        return kr.search(query)

    async def _vector_retrieve_async(self, query: str, top_k: int) -> list[dict]:
        """Async wrapper for vector retrieval (Milvus)."""
        from enterprise_agentic_rag.rag.embedding_provider import get_embedding_provider
        from enterprise_agentic_rag.rag.milvus_store import MilvusStore

        ms = MilvusStore()
        if not ms.available:
            return []

        ep = get_embedding_provider()
        vec = ep.embed_query(query)
        return ms.search(vec, top_k=top_k)

    async def _graph_retrieve_async(
        self,
        query_analysis: dict[str, Any],
        plan: RetrievalPlan,
    ) -> list[dict]:
        """Async wrapper for graph retrieval.

        Converts Candidate objects to dicts for compatibility.
        """
        if not self.graph_available:
            return []

        candidates = self._graph_retriever.retrieve(
            query_analysis=query_analysis,
            top_k=plan.top_k.get("graph", 10),
            graph_depth=plan.graph_depth,
        )
        return [c.to_dict() for c in candidates]

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_query(query: str) -> dict[str, Any]:
        """Quick query analysis for retrieval routing.

        Extracts entities and keywords without running a separate intent classifier.
        The main workflow should pass deep-intent-enriched query_analysis when
        available; this fallback only supplies routing hints.
        """
        from enterprise_agentic_rag.rag.query_rewriter import classify_query_type

        query_type = classify_query_type(query)

        # Extract simple keywords
        import re
        keywords = re.findall(r'[一-鿿\w]+', query)
        keywords = [k for k in keywords if len(k) >= 2][:10]

        # Extract entities using the entity extractor
        entities: list[str] = []
        try:
            from enterprise_agentic_rag.rag.graph.entity_extractor import extract_entities_from_chunk
            ents = extract_entities_from_chunk(query)
            entities = [e.name for e in ents]
        except Exception:
            pass

        return {
            "intent": query_type,
            "query_type": query_type,
            "keywords": keywords,
            "entities": entities,
            "original_query": query,
        }

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    @staticmethod
    def _rerank(query: str, docs: list[dict], top_n: int) -> list[dict]:
        """Apply reranker to fused results."""
        if not docs:
            return []
        try:
            from enterprise_agentic_rag.rag.reranker import Reranker
            reranker = Reranker()
            return reranker.rerank(query, docs, top_n=top_n)
        except Exception:
            return docs[:top_n]


# ===========================================================================
# Helpers
# ===========================================================================


def _dict_to_candidate(d: dict[str, Any]) -> Candidate:
    """Convert a dict back to a Candidate for QueryExpander compatibility."""
    from enterprise_agentic_rag.rag.graph.graph_schema import GraphPath

    graph_paths = []
    for gp_dict in d.get("graph_paths", []):
        graph_paths.append(GraphPath(
            path_entities=gp_dict.get("path_entities", []),
            path_relations=gp_dict.get("path_relations", []),
            evidence_chunk_id=gp_dict.get("evidence_chunk_id", ""),
            relation_weight=gp_dict.get("relation_weight", 1.0),
            path_score=gp_dict.get("path_score", 0.0),
            path_length=gp_dict.get("path_length", 0),
        ))

    return Candidate(
        chunk_id=d.get("chunk_id", ""),
        doc_id=d.get("doc_id", ""),
        content=d.get("content", ""),
        source_path=d.get("source_path", d.get("source", "")),
        graph_score=d.get("graph_score", 0.0),
        raw_scores=d.get("raw_scores", {}),
        matched_sources=d.get("matched_sources", []),
        graph_paths=graph_paths,
        metadata=d.get("metadata", {}),
    )
