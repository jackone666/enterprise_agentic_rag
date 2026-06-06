"""Hybrid RAG workflow — keyword + vector search with RRF fusion.

The default and fallback mode. Suitable for:
- Simple concept Q&A
- General API explanations
- Straightforward knowledge lookups
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enterprise_agentic_rag.rag.keyword_search_tool import keyword_search
from enterprise_agentic_rag.rag.vector_search_tool import vector_search
from enterprise_agentic_rag.rag.merger import Merger
from enterprise_agentic_rag.rag.reranker_wrapper import RerankerWrapper
from enterprise_agentic_rag.rag.evidence_selector import EvidenceSelector
from enterprise_agentic_rag.rag.retrieval_router import DynamicWeights

logger = logging.getLogger(__name__)


class HybridRAGWorkflow:
    """Keyword + vector hybrid retrieval workflow.

    Execution:
        1. keyword_search + vector_search (parallel via asyncio.gather)
        2. merge (weighted RRF fusion)
        3. rerank
        4. evidence selection
    """

    def __init__(self) -> None:
        self._merger = Merger()
        self._reranker = RerankerWrapper()
        self._evidence_selector = EvidenceSelector()

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        intent: str = "concept_qa",
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid RAG workflow.

        Args:
            query: User query string.
            top_k: Number of results per retriever.
            intent: Primary intent for weight adjustment.
            entities: Extracted entities.

        Returns:
            Dict with keyword_results, vector_results, merged_results,
            reranked_results, selected_evidence.
        """
        t0 = time.time()
        entities = entities or {}

        # Stage 1: Parallel keyword + vector
        kw_task = keyword_search(
            query=query, top_k=top_k, intent=intent, entities=entities,
        )
        vec_task = vector_search(
            query=query, top_k=top_k, intent=intent, entities=entities,
        )

        kw_result, vec_result = await asyncio.gather(kw_task, vec_task, return_exceptions=True)

        kw_results = kw_result.results if not isinstance(kw_result, Exception) else []
        vec_results = vec_result.results if not isinstance(vec_result, Exception) else []

        # Stage 2: Merge with intent-aware weights
        weights = DynamicWeights.for_intent(intent)
        source_results = {
            "keyword_search": kw_results,
            "vector_search": vec_results,
        }
        merged = self._merger.merge(source_results, weights=weights, top_n=15)

        # Stage 3: Rerank
        reranked = self._reranker.rerank(query, merged, primary_intent=intent, top_n=10)

        # Stage 4: Evidence selection
        evidence = self._evidence_selector.select(reranked, primary_intent=intent, max_chunks=5)

        total_latency_ms = (time.time() - t0) * 1000

        logger.info(
            "HybridRAG: kw=%d vec=%d merged=%d reranked=%d evidence=%d latency=%.0fms",
            len(kw_results), len(vec_results), len(merged),
            len(reranked), len(evidence), total_latency_ms,
        )

        return {
            "keyword_results": kw_results,
            "vector_results": vec_results,
            "merged_results": merged,
            "reranked_results": reranked,
            "selected_evidence": evidence,
            "total_latency_ms": round(total_latency_ms, 2),
            "errors": (
                ([str(kw_result)] if isinstance(kw_result, Exception) else []) +
                ([str(vec_result)] if isinstance(vec_result, Exception) else [])
            ),
        }
