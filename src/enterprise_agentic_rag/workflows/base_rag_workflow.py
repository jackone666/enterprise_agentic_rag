"""BaseRAGWorkflow — unified retrieval workflow for all modes.

Replaces 4 separate workflow classes (hybrid, graph-first, error-first,
code-generation) with a single class that dispatches by ``mode``.

Available modes (maps to ``RetrievalMode`` in schema):
    hybrid_only  — keyword + vector parallel (default / fallback)
    graph_first  — graph → expanded query → keyword + vector
    parallel     — sample code + API ref → official doc → keyword + vector

Execution pattern (shared by all modes):
    1. Parallel source retrievals (mode-specific)
    2. Weighted RRF merge via Merger
    3. Rerank via RerankerWrapper
    4. Evidence selection via EvidenceSelector
    5. Optional semantic cache write-back
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enterprise_agentic_rag.rag.evidence_selector import EvidenceSelector
from enterprise_agentic_rag.rag.keyword_search_tool import keyword_search
from enterprise_agentic_rag.rag.merger import Merger
from enterprise_agentic_rag.rag.reranker_wrapper import RerankerWrapper
from enterprise_agentic_rag.rag.retrieval_router import DynamicWeights
from enterprise_agentic_rag.rag.vector_search_tool import vector_search

logger = logging.getLogger(__name__)


class BaseRAGWorkflow:
    """Unified retrieval workflow with mode-based dispatch.

    Modes:
        hybrid_only  — keyword + vector (default)
        graph_first  — graph search → expanded query → keyword + vector
        parallel     — multi-source parallel (code/API/official) → keyword + vector

    All modes return the same dict shape for downstream compatibility.
    """

    def __init__(self) -> None:
        self._merger = Merger()
        self._reranker = RerankerWrapper()
        self._evidence_selector = EvidenceSelector()

    async def execute(
        self,
        query: str,
        mode: str = "hybrid_only",
        top_k: int = 10,
        intent: str = "concept_qa",
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute retrieval for the given mode.

        Args:
            query:       User query string.
            mode:        One of "hybrid_only", "graph_first", "parallel".
            top_k:       Results per retriever.
            intent:      Primary intent for weight adjustment.
            entities:    Extracted entities (apis, components, errors, etc.).

        Returns:
            Dict with keyword_results, vector_results, merged_results,
            reranked_results, selected_evidence, total_latency_ms, errors.
        """
        entities = entities or {}
        t0 = time.time()
        errors: list[str] = []

        if mode == "graph_first":
            results, errors = await self._execute_graph_first(query, top_k, intent, entities)
        elif mode == "parallel":
            results, errors = await self._execute_parallel(query, top_k, intent, entities)
        else:
            results, errors = await self._execute_hybrid(query, top_k, intent, entities)

        # Always apply: merge → rerank → evidence selection
        weights = DynamicWeights.for_intent(intent)
        merged = self._merger.merge(results, weights=weights, top_n=20)
        reranked = self._reranker.rerank(query, merged, primary_intent=intent, top_n=10)
        evidence = self._evidence_selector.select(reranked, primary_intent=intent, max_chunks=5)

        total_latency_ms = (time.time() - t0) * 1000

        # Collect keyword + vector results for the standard return shape
        kw = results.get("keyword_search", [])
        vec = results.get("vector_search", [])

        logger.info(
            "BaseRAG[%s]: kw=%d vec=%d merged=%d evidence=%d latency=%.0fms",
            mode, len(kw), len(vec), len(merged), len(evidence), total_latency_ms,
        )

        return {
            "keyword_results": kw,
            "vector_results": vec,
            "merged_results": merged,
            "reranked_results": reranked,
            "selected_evidence": evidence,
            "total_latency_ms": round(total_latency_ms, 2),
            "errors": errors,
        }

    # -------------------------------------------------------------------------
    # Mode implementations
    # -------------------------------------------------------------------------

    async def _execute_hybrid(
        self,
        query: str,
        top_k: int,
        intent: str,
        entities: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Keyword + vector parallel (default)."""
        errors: list[str] = []

        kw_task = keyword_search(query=query, top_k=top_k, intent=intent, entities=entities)
        vec_task = vector_search(query=query, top_k=top_k, intent=intent, entities=entities)

        kw_result, vec_result = await asyncio.gather(kw_task, vec_task, return_exceptions=True)

        kw = kw_result.results if not isinstance(kw_result, Exception) else []
        vec = vec_result.results if not isinstance(vec_result, Exception) else []

        for name, r in [("keyword", kw_result), ("vector", vec_result)]:
            if isinstance(r, Exception):
                errors.append(f"{name}: {r}")

        return {"keyword_search": kw, "vector_search": vec}, errors

    async def _execute_graph_first(
        self,
        query: str,
        top_k: int,
        intent: str,
        entities: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Graph search → expand query → keyword + vector."""
        errors: list[str] = []
        search_query = query

        try:
            from enterprise_agentic_rag.rag.graph_search_tool import graph_search

            graph_result = await graph_search(
                query=query, top_k=15, intent=intent, entities=entities,
            )
            graph_results = graph_result.results
            graph_meta = graph_result.metadata

            entity_relations = graph_meta.get("entity_relations", [])
            expanded_query = graph_meta.get("expanded_query")

            if graph_result.error:
                errors.append(f"Graph search: {graph_result.error}")

            # Build expanded query from entity relations
            if not expanded_query and entity_relations:
                entities_in_graph = set()
                for rel in entity_relations:
                    entities_in_graph.add(rel.get("source", ""))
                    entities_in_graph.add(rel.get("target", ""))
                search_query = f"{query} {' '.join(entities_in_graph)}"
            elif expanded_query:
                search_query = expanded_query

            kw_task = keyword_search(query=search_query, top_k=top_k, intent=intent, entities=entities)
            vec_task = vector_search(query=search_query, top_k=top_k, intent=intent, entities=entities)

            kw_result, vec_result = await asyncio.gather(kw_task, vec_task, return_exceptions=True)

            kw = kw_result.results if not isinstance(kw_result, Exception) else []
            vec = vec_result.results if not isinstance(vec_result, Exception) else []

            for name, r in [("keyword", kw_result), ("vector", vec_result)]:
                if isinstance(r, Exception):
                    errors.append(f"{name}: {r}")

            source_results: dict[str, list[dict[str, Any]]] = {"keyword_search": kw, "vector_search": vec}
            if graph_results:
                source_results["graph_search"] = graph_results
            return source_results, errors

        except Exception as exc:
            logger.warning("Graph-first workflow failed: %s — falling back to hybrid", exc)
            errors.append(f"Graph-first: {exc}")
            kw_task = keyword_search(query=query, top_k=top_k, intent=intent, entities=entities)
            vec_task = vector_search(query=query, top_k=top_k, intent=intent, entities=entities)
            kw_result, vec_result = await asyncio.gather(kw_task, vec_task, return_exceptions=True)
            kw = kw_result.results if not isinstance(kw_result, Exception) else []
            vec = vec_result.results if not isinstance(vec_result, Exception) else []
            return {"keyword_search": kw, "vector_search": vec}, errors

    async def _execute_parallel(
        self,
        query: str,
        top_k: int,
        intent: str,
        entities: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Multi-source parallel: sample code + API ref → official doc → keyword + vector.

        Used for code_generation and api_usage intent modes.
        """
        errors: list[str] = []
        code_apis = entities.get("apis", [])
        code_components = entities.get("components", [])
        code_query_terms = " ".join(code_apis + code_components)

        # Stage 1: Parallel sample code + API reference
        sample_task = keyword_search(
            query=f"{query} example code sample {code_query_terms}".strip(),
            top_k=top_k, intent=intent, entities=entities,
        )
        api_task = keyword_search(
            query=f"{query} API reference {code_query_terms}".strip(),
            top_k=top_k, intent=intent, entities=entities,
        )

        sample_result, api_result = await asyncio.gather(
            sample_task, api_task, return_exceptions=True,
        )

        sample_results = sample_result.results if not isinstance(sample_result, Exception) else []
        api_results = api_result.results if not isinstance(api_result, Exception) else []

        # Stage 2: Official doc search
        doc_task = keyword_search(
            query=f"{query} 官方文档 official documentation",
            top_k=top_k, intent=intent, entities=entities,
        )
        doc_result = await doc_task
        doc_results = doc_result.results if not isinstance(doc_result, Exception) else []

        # Stage 3: Parallel keyword + vector (supplemental)
        kw_task = keyword_search(query=query, top_k=top_k, intent=intent, entities=entities)
        vec_task = vector_search(query=query, top_k=top_k, intent=intent, entities=entities)

        kw_result, vec_result = await asyncio.gather(kw_task, vec_task, return_exceptions=True)
        kw = kw_result.results if not isinstance(kw_result, Exception) else []
        vec = vec_result.results if not isinstance(vec_result, Exception) else []

        for name, r in [
            ("sample_code", sample_result), ("api_ref", api_result),
            ("official_doc", doc_result), ("keyword", kw_result), ("vector", vec_result),
        ]:
            if isinstance(r, Exception):
                errors.append(f"{name}: {r}")

        # Boost sample code weight for code generation
        weights = DynamicWeights.for_intent(intent)
        weights.sample_code_weight = 0.40

        return {
            "sample_code": sample_results,
            "api_reference": api_results,
            "official_doc": doc_results,
            "keyword_search": kw,
            "vector_search": vec,
        }, errors
