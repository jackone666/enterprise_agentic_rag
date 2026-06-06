"""RerankerWrapper — cross-encoder priority with intent-aware boosting.

Reranking chain (priority order):
1. Cross-Encoder (Ollama qwen3-reranker-0.6b)
2. API reranker (configurable endpoint)
3. Rule-based keyword overlap + source diversity

Wraps cross_encoder_reranker.py and rag/reranker.py with:
- Intent-aware score adjustments
- Source diversity enforcement
- Minimum relevance threshold
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RerankerWrapper:
    """Intelligent reranker with cross-encoder priority and intent-aware adjustments.

    Execution:
        1. Cross-Encoder reranking (async, Ollama qwen3)
        2. Fallback: API reranker or rule-based
        3. Intent-based score boosts
        4. Source diversity (prevent single-source dominance)
        5. Minimum relevance filtering
    """

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        primary_intent: str = "",
        top_n: int = 5,
        min_score: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Rerank documents with cross-encoder priority."""
        if not documents:
            return []

        # Stage 1: Cross-Encoder reranking (async)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — use synchronous fallback
                reranked = self._sync_rerank_chain(query, documents, top_n)
            else:
                reranked = loop.run_until_complete(
                    self._cross_encoder_rerank(query, documents, top_n)
                )
        except RuntimeError:
            # No event loop — use synchronous fallback
            reranked = self._sync_rerank_chain(query, documents, top_n)
        except Exception:
            reranked = self._sync_rerank_chain(query, documents, top_n)

        # Stage 2: Apply intent-aware boosts
        if primary_intent:
            reranked = self._apply_intent_boost(reranked, primary_intent)

        # Stage 3: Enforce source diversity
        diverse = self._enforce_diversity(reranked, top_n)

        # Stage 4: Filter by minimum score
        filtered = [d for d in diverse if d.get("score", 0) >= min_score]

        if not filtered and diverse:
            filtered = diverse[:top_n]

        return filtered[:top_n]

    async def rerank_async(
        self,
        query: str,
        documents: list[dict[str, Any]],
        primary_intent: str = "",
        top_n: int = 5,
        min_score: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Async version of rerank for use in async contexts."""
        if not documents:
            return []

        # Stage 1: Cross-Encoder reranking
        reranked = await self._cross_encoder_rerank(query, documents, top_n)

        # Stage 2-4: Same as sync version
        if primary_intent:
            reranked = self._apply_intent_boost(reranked, primary_intent)

        diverse = self._enforce_diversity(reranked, top_n)
        filtered = [d for d in diverse if d.get("score", 0) >= min_score]

        if not filtered and diverse:
            filtered = diverse[:top_n]

        return filtered[:top_n]

    async def _cross_encoder_rerank(
        self, query: str, docs: list[dict[str, Any]], top_n: int,
    ) -> list[dict[str, Any]]:
        """Try cross-encoder reranking, fall back to other methods."""
        try:
            from enterprise_agentic_rag.rag.cross_encoder_reranker import get_cross_encoder

            ce = get_cross_encoder()
            if ce.available:
                return await ce.rerank(query, docs, top_n)
        except Exception as exc:
            logger.debug("Cross-encoder not available: %s", exc)

        # Fallback to API/rule-based reranker
        return self._sync_rerank_chain(query, docs, top_n)

    def _sync_rerank_chain(
        self, query: str, docs: list[dict[str, Any]], top_n: int,
    ) -> list[dict[str, Any]]:
        """Synchronous fallback rerank chain (API → rule)."""
        try:
            from enterprise_agentic_rag.rag.reranker import Reranker

            base_reranker = Reranker()
            reranked = base_reranker.rerank(query, docs, top_n=top_n * 2)
            return reranked
        except Exception:
            pass

        # Ultimate fallback: simple keyword overlap
        return self._simple_rerank(query, docs)[: top_n * 2]

    def _simple_rerank(
        self, query: str, docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simple keyword overlap reranking fallback."""
        qtokens = set(query.lower().split())
        for d in docs:
            content = d.get("content", "").lower()
            ctokens = set(content.split())
            overlap = len(qtokens & ctokens)
            base = d.get("score", 0)
            d["rerank_score"] = base * 0.5 + overlap * 0.5
        docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
        return docs

    def _apply_intent_boost(
        self, docs: list[dict[str, Any]], intent: str,
    ) -> list[dict[str, Any]]:
        """Apply intent-specific score boosts."""
        for doc in docs:
            boost = 1.0
            content = doc.get("content", "").lower()
            source = str(doc.get("source", "")).lower()
            doc_type = str(doc.get("doc_type", "")).lower()

            if intent == "api_usage" and ("api" in doc_type or "@ohos" in content):
                boost = 1.2
            elif intent == "code_generation" and ("code" in source or "```" in content):
                boost = 1.3
            elif intent == "error_diagnosis" and any(
                kw in content for kw in ("error", "报错", "错误", "failed", "exception")
            ):
                boost = 1.25
            elif intent == "migration" and any(
                kw in content for kw in ("migration", "迁移", "升级", "废弃")
            ):
                boost = 1.3
            elif intent == "compatibility" and any(
                kw in content for kw in ("version", "兼容", "api level", "harmonyos")
            ):
                boost = 1.2
            elif intent == "concept_qa" and ("official" in source or "doc" in doc_type):
                boost = 1.15

            current = doc.get("rerank_score", doc.get("cross_encoder_score", doc.get("score", 0)))
            doc["rerank_score"] = current * boost

        docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
        return docs

    @staticmethod
    def _enforce_diversity(
        docs: list[dict[str, Any]], top_n: int,
    ) -> list[dict[str, Any]]:
        """Enforce source diversity: prefer unique sources."""
        seen_sources: set[str] = set()
        diverse: list[dict[str, Any]] = []
        others: list[dict[str, Any]] = []

        for doc in docs:
            source = str(doc.get("source", ""))
            if source and source not in seen_sources:
                seen_sources.add(source)
                diverse.append(doc)
            else:
                others.append(doc)

        while len(diverse) < top_n and others:
            diverse.append(others.pop(0))

        return diverse[:top_n]


def rerank_results(
    query: str,
    documents: list[dict[str, Any]],
    primary_intent: str = "",
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Convenience function for reranking."""
    wrapper = RerankerWrapper()
    return wrapper.rerank(query, documents, primary_intent, top_n=top_n)
