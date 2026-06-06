"""Error-first retrieval workflow — error KB → parallel faq/ticket/official/keyword.

Suitable for:
- Error diagnosis (BusinessError, permission denied, etc.)
- Build failures (hvigor ERROR, compile failed)
- Crash analysis (SIGABRT, white screen, black screen)
- Runtime errors (TypeError, ReferenceError)
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


class ErrorFirstWorkflow:
    """Error-first retrieval workflow.

    Execution:
        1. error_diagnosis_search (keyword with error-focused weights)
        2. parallel: keyword_search + vector_search + official_doc_search + ticket_search
        3. merge with error-heavy weights
        4. rerank
        5. evidence selection (prioritize diagnosis-oriented chunks)

    Fallback: If no error results → hybrid_only
    """

    def __init__(self) -> None:
        self._merger = Merger()
        self._reranker = RerankerWrapper()
        self._evidence_selector = EvidenceSelector()

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        intent: str = "error_diagnosis",
        scenario: str = "",
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute error-first workflow.

        Args:
            query: User query string (should contain error info).
            top_k: Number of results per retriever.
            intent: Primary intent (error_diagnosis or project_debug).
            scenario: Detected scenario (permission_error, build_error, etc.).
            entities: Extracted entities.

        Returns:
            Dict with all retrieval results and diagnosis-oriented evidence.
        """
        t0 = time.time()
        entities = entities or {}
        errors_list: list[str] = []

        # Build error-focused query
        errors_from_entities = entities.get("errors", [])
        error_terms = " ".join(errors_from_entities) if errors_from_entities else ""
        error_query = f"{query} {error_terms}".strip()

        # Stage 1: Error-focused keyword search (higher weight for error content)
        err_task = keyword_search(
            query=error_query, top_k=top_k, intent=intent,
            scenario=scenario, entities=entities,
        )

        # Stage 2: Parallel auxiliary searches
        kw_task = keyword_search(
            query=query, top_k=top_k, intent=intent, entities=entities,
        )
        vec_task = vector_search(
            query=query, top_k=top_k, intent=intent, entities=entities,
        )

        err_result, kw_result, vec_result = await asyncio.gather(
            err_task, kw_task, vec_task, return_exceptions=True,
        )

        err_results = err_result.results if not isinstance(err_result, Exception) else []
        kw_results = kw_result.results if not isinstance(kw_result, Exception) else []
        vec_results = vec_result.results if not isinstance(vec_result, Exception) else []

        if isinstance(err_result, Exception):
            errors_list.append(f"error_diagnosis: {err_result}")
        if isinstance(kw_result, Exception):
            errors_list.append(f"keyword: {kw_result}")
        if isinstance(vec_result, Exception):
            errors_list.append(f"vector: {vec_result}")

        # Stage 3: Merge with error-heavy weights
        weights = DynamicWeights.for_intent(intent, scenario)

        source_results = {
            "error_diagnosis": err_results,
            "keyword_search": kw_results,
            "vector_search": vec_results,
        }

        merged = self._merger.merge(source_results, weights=weights, top_n=15)

        # Stage 4: Rerank (error-focused boost)
        reranked = self._reranker.rerank(query, merged, primary_intent=intent, top_n=10)

        # Stage 5: Diagnosis-oriented evidence selection
        evidence = self._evidence_selector.select(reranked, primary_intent=intent, max_chunks=5)

        total_latency_ms = (time.time() - t0) * 1000

        logger.info(
            "ErrorFirst: err=%d kw=%d vec=%d merged=%d evidence=%d latency=%.0fms scenario=%s",
            len(err_results), len(kw_results), len(vec_results),
            len(merged), len(evidence), total_latency_ms, scenario,
        )

        # Build diagnosis context
        diagnosis_context = self._build_diagnosis_context(
            query, scenario, evidence, errors_from_entities,
        )

        return {
            "keyword_results": kw_results,
            "vector_results": vec_results,
            "error_results": err_results,
            "merged_results": merged,
            "reranked_results": reranked,
            "selected_evidence": evidence,
            "diagnosis_context": diagnosis_context,
            "total_latency_ms": round(total_latency_ms, 2),
            "errors": errors_list,
        }

    @staticmethod
    def _build_diagnosis_context(
        query: str,
        scenario: str,
        evidence: list[dict[str, Any]],
        error_terms: list[str],
    ) -> dict[str, Any]:
        """Build structured diagnosis context from evidence."""
        possible_causes: list[str] = []
        fix_suggestions: list[str] = []

        for doc in evidence[:3]:
            content = doc.get("content", "").lower()
            if any(kw in content for kw in ("原因", "cause", "由于", "because")):
                possible_causes.append(doc.get("content", "")[:300])
            if any(kw in content for kw in ("解决", "修复", "fix", "方案", "修改")):
                fix_suggestions.append(doc.get("content", "")[:300])

        return {
            "scenario": scenario,
            "error_terms": error_terms,
            "possible_causes_count": len(possible_causes),
            "fix_suggestions_count": len(fix_suggestions),
        }
