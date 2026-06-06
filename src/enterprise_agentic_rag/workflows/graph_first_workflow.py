"""Graph-first retrieval workflow — graph → expand → keyword + vector.

Suitable for:
- Migration queries (Router→Navigation, FA→Stage, JS→ArkTS)
- Compatibility checks (API Level, HarmonyOS NEXT)
- Permission chain analysis
- API relationship / multi-hop reasoning
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enterprise_agentic_rag.rag.evidence_selector import EvidenceSelector
from enterprise_agentic_rag.rag.graph_search_tool import graph_search
from enterprise_agentic_rag.rag.keyword_search_tool import keyword_search
from enterprise_agentic_rag.rag.merger import Merger
from enterprise_agentic_rag.rag.reranker_wrapper import RerankerWrapper
from enterprise_agentic_rag.rag.retrieval_router import DynamicWeights
from enterprise_agentic_rag.rag.vector_search_tool import vector_search

logger = logging.getLogger(__name__)


class GraphFirstWorkflow:
    """Graph-first retrieval workflow.

    Execution:
        1. graph_search → entity relations + graph paths
        2. expand query from graph results
        3. keyword_search + vector_search (parallel, with expanded query)
        4. three-way merge
        5. rerank
        6. evidence selection

    Fallback: If graph fails → hybrid_only (keyword + vector)
    """

    def __init__(self) -> None:
        self._merger = Merger()
        self._reranker = RerankerWrapper()
        self._evidence_selector = EvidenceSelector()

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        intent: str = "migration",
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute graph-first workflow.

        Args:
            query: User query string.
            top_k: Number of results per retriever.
            intent: Primary intent.
            entities: Extracted entities (used as graph seeds).

        Returns:
            Dict with all retrieval results and evidence.
        """
        t0 = time.time()
        entities = entities or {}
        errors: list[str] = []
        degraded = False

        # Stage 1: Graph retrieval
        graph_result = await graph_search(
            query=query, top_k=15, intent=intent, entities=entities,
        )

        graph_results = graph_result.results
        graph_meta = graph_result.metadata
        entity_relations = graph_meta.get("entity_relations", [])
        expanded_query = graph_meta.get("expanded_query")

        if graph_result.error:
            errors.append(f"Graph search: {graph_result.error}")

        # Fallback: if graph failed → hybrid_only
        if not graph_results and not entity_relations:
            logger.warning("GraphFirstWorkflow: graph returned no results → degrading to hybrid")
            degraded = True
            expanded_query = query  # Use original query

        # Stage 2: Expand query (use expanded_query from graph or derive)
        search_query = expanded_query or query
        if not expanded_query and entity_relations:
            # Build expanded query from entity relations
            entities_in_graph = set()
            for rel in entity_relations:
                entities_in_graph.add(rel.get("source", ""))
                entities_in_graph.add(rel.get("target", ""))
            search_query = f"{query} {' '.join(entities_in_graph)}"

        # Stage 3: Parallel keyword + vector with expanded query
        kw_task = keyword_search(
            query=search_query, top_k=top_k, intent=intent, entities=entities,
        )
        vec_task = vector_search(
            query=search_query, top_k=top_k, intent=intent, entities=entities,
        )

        kw_result, vec_result = await asyncio.gather(kw_task, vec_task, return_exceptions=True)

        kw_results = kw_result.results if not isinstance(kw_result, Exception) else []
        vec_results = vec_result.results if not isinstance(vec_result, Exception) else []

        # Stage 4: Three-way merge with graph-heavy weights
        weights = DynamicWeights.for_intent(intent)
        if degraded:
            weights.graph_weight = 0.0  # No graph contribution when degraded

        source_results = {
            "keyword_search": kw_results,
            "vector_search": vec_results,
        }
        if graph_results:
            source_results["graph_search"] = graph_results

        merged = self._merger.merge(source_results, weights=weights, top_n=20)

        # Stage 5: Rerank
        reranked = self._reranker.rerank(query, merged, primary_intent=intent, top_n=10)

        # Stage 6: Evidence selection
        evidence = self._evidence_selector.select(reranked, primary_intent=intent, max_chunks=5)

        total_latency_ms = (time.time() - t0) * 1000

        logger.info(
            "GraphFirst: graph=%d kw=%d vec=%d merged=%d evidence=%d latency=%.0fms degraded=%s",
            len(graph_results), len(kw_results), len(vec_results),
            len(merged), len(evidence), total_latency_ms, degraded,
        )

        return {
            "keyword_results": kw_results,
            "vector_results": vec_results,
            "graph_results": graph_results,
            "entity_relations": entity_relations,
            "expanded_query": search_query,
            "merged_results": merged,
            "reranked_results": reranked,
            "selected_evidence": evidence,
            "total_latency_ms": round(total_latency_ms, 2),
            "degraded": degraded,
            "errors": errors,
        }
