"""Code generation workflow — sample → api_ref → official_doc → code_review.

Suitable for:
- Code generation requests
- Project-level debugging with code analysis
- API usage examples
- Code review and best practice checks
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


class CodeGenerationWorkflow:
    """Code-first retrieval workflow.

    Execution:
        1. sample_code_search + api_reference_search (parallel)
        2. official_doc_search (for validation)
        3. keyword_search + vector_search (parallel, supplemental)
        4. merge with code-heavy weights
        5. rerank (code snippet boost)
        6. evidence selection (prioritize code examples)

    Note: code_review is a future tool (Phase 2).
    """

    def __init__(self) -> None:
        self._merger = Merger()
        self._reranker = RerankerWrapper()
        self._evidence_selector = EvidenceSelector()

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        intent: str = "code_generation",
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute code generation workflow.

        Args:
            query: User query string.
            top_k: Number of results per retriever.
            intent: Primary intent.
            entities: Extracted entities.

        Returns:
            Dict with all retrieval results and code-oriented evidence.
        """
        t0 = time.time()
        entities = entities or {}
        errors_list: list[str] = []

        # Build code-focused query variations
        code_apis = entities.get("apis", [])
        code_components = entities.get("components", [])
        code_query_terms = " ".join(code_apis + code_components)

        # Stage 1: Parallel sample code + API reference search
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

        # Stage 2: Official doc search (for validation)
        doc_task = keyword_search(
            query=f"{query} 官方文档 official documentation",
            top_k=top_k, intent=intent, entities=entities,
        )
        doc_result = await doc_task
        doc_results = doc_result.results if not isinstance(doc_result, Exception) else []

        # Stage 3: Parallel keyword + vector (supplemental)
        kw_task = keyword_search(
            query=query, top_k=top_k, intent=intent, entities=entities,
        )
        vec_task = vector_search(
            query=query, top_k=top_k, intent=intent, entities=entities,
        )

        kw_result, vec_result = await asyncio.gather(
            kw_task, vec_task, return_exceptions=True,
        )

        kw_results = kw_result.results if not isinstance(kw_result, Exception) else []
        vec_results = vec_result.results if not isinstance(vec_result, Exception) else []

        # Collect errors
        for name, r in [("sample_code", sample_result), ("api_ref", api_result),
                         ("official_doc", doc_result), ("keyword", kw_result), ("vector", vec_result)]:
            if isinstance(r, Exception):
                errors_list.append(f"{name}: {r}")

        # Stage 4: Merge with code-heavy weights
        weights = DynamicWeights.for_intent(intent)
        weights.sample_code_weight = 0.40  # Increase code priority

        source_results = {
            "sample_code": sample_results,
            "api_reference": api_results,
            "official_doc": doc_results,
            "keyword_search": kw_results,
            "vector_search": vec_results,
        }

        merged = self._merger.merge(source_results, weights=weights, top_n=20)

        # Stage 5: Rerank (code snippet boost)
        reranked = self._reranker.rerank(query, merged, primary_intent=intent, top_n=10)

        # Stage 6: Evidence selection (prioritize code examples)
        evidence = self._evidence_selector.select(reranked, primary_intent=intent, max_chunks=5)

        total_latency_ms = (time.time() - t0) * 1000

        # Identify which evidence chunks contain code
        code_evidence = [
            doc for doc in evidence if "```" in doc.get("content", "")
        ]
        non_code_evidence = [
            doc for doc in evidence if "```" not in doc.get("content", "")
        ]

        logger.info(
            "CodeGen: sample=%d api=%d doc=%d kw=%d vec=%d merged=%d evidence=%d (code=%d) latency=%.0fms",
            len(sample_results), len(api_results), len(doc_results),
            len(kw_results), len(vec_results),
            len(merged), len(evidence), len(code_evidence), total_latency_ms,
        )

        return {
            "keyword_results": kw_results,
            "vector_results": vec_results,
            "sample_code_results": sample_results,
            "api_reference_results": api_results,
            "official_doc_results": doc_results,
            "merged_results": merged,
            "reranked_results": reranked,
            "selected_evidence": evidence,
            "code_evidence_count": len(code_evidence),
            "non_code_evidence_count": len(non_code_evidence),
            "total_latency_ms": round(total_latency_ms, 2),
            "errors": errors_list,
        }
