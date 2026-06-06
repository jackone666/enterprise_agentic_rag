"""Reranker — Cross-Encoder priority with API and rule-based fallback.

Priority chain:
1. Cross-Encoder (Ollama qwen3-reranker-0.6b) — highest quality
2. API-based reranker (configurable endpoint) — medium quality
3. Rule-based keyword overlap + position — always available
"""

from __future__ import annotations

import os
from typing import Any


class Reranker:
    """Rerank retrieved documents by relevance to query.

    Priority: Cross-Encoder > API-based reranker > rule-based fallback
    """

    def __init__(self) -> None:
        self._api_url = os.getenv("RERANK_API_URL", "")
        self._api_key = os.getenv("RERANK_API_KEY", "")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """Rerank documents and return top_n results.

        Tries cross-encoder first (if available), then API reranker,
        then rule-based fallback.
        """
        if not documents:
            return []

        # Stage 1: Try Cross-Encoder (async → sync wrapper for compat)
        try:
            result = self._try_cross_encoder(query, documents, top_n)
            if result:
                return result
        except Exception:
            pass

        # Stage 2: Try API reranker
        if self._api_url and self._api_key:
            result = self._api_rerank(query, documents, top_n)
            if result:
                return result

        # Stage 3: Rule-based fallback
        return self._rule_rerank(query, documents, top_n)

    @staticmethod
    def _try_cross_encoder(
        query: str, docs: list[dict], top_n: int,
    ) -> list[dict] | None:
        """Try the Ollama cross-encoder reranker synchronously."""
        import asyncio

        try:
            from enterprise_agentic_rag.rag.cross_encoder_reranker import get_cross_encoder
            ce = get_cross_encoder()
            if not ce.available:
                return None

            # Run async rerank in a synchronous context
            try:
                loop = asyncio.get_running_loop()
                # Already in async context — can't block
                return None
            except RuntimeError:
                # No running loop — safe to run
                return asyncio.run(ce.rerank(query, docs, top_n))
        except Exception:
            return None

    def _api_rerank(self, query: str, docs: list[dict], top_n: int) -> list[dict] | None:
        import httpx
        try:
            texts = [d.get("content", "")[:500] for d in docs]
            resp = httpx.post(
                self._api_url,
                json={"query": query, "documents": texts, "top_n": top_n},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=15.0,
            )
            if resp.status_code != 200:
                return None
            scores = resp.json().get("scores", [])
            for i, s in enumerate(scores):
                if i < len(docs):
                    docs[i]["rerank_score"] = s
            docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
            return docs[:top_n]
        except Exception:
            return None

    def _rule_rerank(self, query: str, docs: list[dict], top_n: int) -> list[dict]:
        """Rule-based: keyword density + source diversity boost."""
        qtokens = set(query.lower().split())
        for d in docs:
            content = d.get("content", "").lower()
            ctokens = set(content.split())
            overlap = len(qtokens & ctokens)
            # Position boost: earlier chunks score higher
            pos = int(d.get("chunk_index", 0))
            d["rerank_score"] = overlap + max(0, 3 - pos) * 0.5

        docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
        # Source diversity: prefer unique sources
        seen: set[str] = set()
        result = []
        for d in docs:
            src = d.get("source", "")
            if src not in seen or len(result) >= top_n:
                seen.add(src)
                result.append(d)
            if len(result) >= top_n:
                break
        return result[:top_n]
