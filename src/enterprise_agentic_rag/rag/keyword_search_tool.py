"""Keyword retriever — unified tool wrapper for ES keyword search + Jaccard fallback.

Reuses existing ESKeywordStore and KeywordRetriever from rag/.
"""

from __future__ import annotations

import time
from typing import Any

from enterprise_agentic_rag.rag.unified_schemas import UnifiedToolInput, UnifiedToolOutput


class KeywordRetrieverTool:
    """Keyword search tool wrapping ES + in-memory Jaccard fallback.

    Input: UnifiedToolInput
    Output: UnifiedToolOutput with results containing:
        - id, title, content, source, doc_type, score, metadata
    """

    TOOL_NAME = "keyword_search"

    def __init__(self) -> None:
        self._es_available: bool | None = None

    async def execute(self, **kwargs: Any) -> UnifiedToolOutput:
        """Execute keyword search."""
        t0 = time.time()
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 10)
        filters = kwargs.get("filters", {})
        intent = kwargs.get("intent", "")
        scenario = kwargs.get("scenario", "")
        entities = kwargs.get("entities", {})

        results: list[dict[str, Any]] = []
        error: str | None = None

        # Try ES first
        try:
            from enterprise_agentic_rag.rag.es_keyword_store import ESKeywordStore

            es_store = ESKeywordStore()
            if es_store.available:
                raw_results = es_store.search(query, top_k=top_k)
                results = self._normalize_results(raw_results, "elasticsearch")
        except Exception as exc:
            error = f"ES keyword search failed: {exc}"

        # Fallback to in-memory Jaccard
        if not results:
            try:
                from enterprise_agentic_rag.rag.retriever import KeywordRetriever

                kr = KeywordRetriever(top_k=top_k)
                raw_results = kr.search(query)
                results = self._normalize_results(raw_results, "memory_jaccard")
                error = None  # Fallback succeeded
            except Exception as exc2:
                if not error:
                    error = f"Keyword search failed: {exc2}"

        latency_ms = (time.time() - t0) * 1000
        confidence = self._calc_confidence(results)

        return UnifiedToolOutput(
            tool_name=self.TOOL_NAME,
            results=results,
            confidence=confidence,
            metadata={
                "intent": intent,
                "scenario": scenario,
                "top_k": top_k,
                "backend": "elasticsearch" if error is None else "memory_jaccard",
            },
            error=error,
            latency_ms=round(latency_ms, 2),
        )

    def _normalize_results(
        self, raw: list[dict[str, Any]], source_prefix: str
    ) -> list[dict[str, Any]]:
        """Normalize raw results to unified output format."""
        normalized = []
        for i, doc in enumerate(raw):
            normalized.append({
                "id": doc.get("chunk_id", f"{source_prefix}_{i}"),
                "title": doc.get("source", "未知文档"),
                "content": doc.get("content", ""),
                "source": doc.get("source", source_prefix),
                "doc_type": doc.get("doc_type", "documentation"),
                "score": float(doc.get("score", 0)),
                "metadata": {
                    "api_level": doc.get("api_level", ""),
                    "version": doc.get("version", ""),
                    "module": doc.get("module", ""),
                    "url": doc.get("url", ""),
                    "updated_at": doc.get("updated_at", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                },
            })
        return normalized

    @staticmethod
    def _calc_confidence(results: list[dict[str, Any]]) -> float:
        """Calculate confidence based on result count and scores."""
        if not results:
            return 0.0
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        count_factor = min(1.0, len(results) / 10)
        return round(avg_score * count_factor, 4)


async def keyword_search(
    query: str,
    top_k: int = 10,
    intent: str = "",
    scenario: str = "",
    entities: dict[str, Any] | None = None,
    filters: dict[str, Any] | None = None,
) -> UnifiedToolOutput:
    """Convenience function for keyword search."""
    tool = KeywordRetrieverTool()
    return await tool.execute(
        query=query,
        top_k=top_k,
        intent=intent,
        scenario=scenario,
        entities=entities or {},
        filters=filters or {},
    )
