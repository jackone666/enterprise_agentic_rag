"""Retriever — Milvus vector search + ES keyword search with fallback chain.

Priority cascade:
1. Fusion: Milvus vector + ES keyword → Weighted RRF
2. Milvus-only vector search (if ES down)
3. ES keyword search (if Milvus down)
4. In-memory Jaccard keyword retriever (ultimate fail-safe, always available)
"""

from __future__ import annotations

from enterprise_agentic_rag.rag.document_loader import load_markdown_files
from enterprise_agentic_rag.rag.embedding_provider import get_embedding_provider
from enterprise_agentic_rag.rag.es_keyword_store import ESKeywordStore
from enterprise_agentic_rag.rag.milvus_store import MilvusStore
from enterprise_agentic_rag.rag.splitter import split_documents


class KeywordRetriever:
    """Fallback retriever — keyword overlap (Jaccard-like).

    Always available — zero external dependencies.  Used as the ultimate
    safety net when both Milvus and Elasticsearch are unreachable.
    """

    def __init__(self, chunk_size: int = 500, top_k: int = 5) -> None:
        self.top_k = top_k
        raw_docs = load_markdown_files()
        self._chunks = split_documents(raw_docs, chunk_size=chunk_size)

    def _tokenize(self, text: str) -> set[str]:
        return set(text.lower().split())

    def _score(self, query_tokens: set[str], chunk_text: str) -> float:
        chunk_tokens = self._tokenize(chunk_text)
        if not query_tokens or not chunk_tokens:
            return 0.0
        intersection = query_tokens & chunk_tokens
        return len(intersection) / len(query_tokens)

    def search(self, query: str) -> list[dict[str, object]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[float, int, dict[str, str]]] = []
        for idx, chunk in enumerate(self._chunks):
            score = self._score(query_tokens, chunk["content"])
            if score > 0:
                scored.append((score, idx, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, object]] = []
        for score, _idx, chunk in scored[: self.top_k]:
            results.append({
                "content": chunk["content"],
                "source": chunk["source"],
                "score": score,
                "chunk_id": f"{chunk['source']}_{chunk.get('chunk_index', '0')}",
            })
        return results


class Retriever:
    """Unified retriever — Milvus + ES + in-memory fallback.

    Retrieval priority::

        ① Fusion (Milvus vector + ES keyword → Weighted RRF)
        ② Milvus-only vector search
        ③ ES keyword search
        ④ In-memory Jaccard (always available)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self._embedder = get_embedding_provider()
        self._milvus = MilvusStore(vector_size=self._embedder.vector_size)
        self._es_keyword = ESKeywordStore()
        self._mem_keyword = KeywordRetriever(chunk_size=chunk_size, top_k=top_k)

    def search(self, query: str, use_fusion: bool = True) -> list[dict[str, object]]:
        """Search with progressive fallback.

        Args:
            query: Raw user query.
            use_fusion: Whether to attempt RRF fusion when both backends
                        are available.

        Returns:
            List of result dicts with ``content``, ``source``, ``score``,
            and ``chunk_id``.
        """
        # Rewrite query for better recall
        from enterprise_agentic_rag.rag.query_rewriter import rewrite_query
        rewritten = rewrite_query(query)
        search_query = rewritten["rewritten"]

        # ① Fusion: Milvus vector + ES keyword → Weighted RRF
        if use_fusion and self._milvus.available:
            try:
                from enterprise_agentic_rag.rag.fusion import fusion_retrieve
                results = fusion_retrieve(search_query, top_k=self.top_k)
                if results:
                    return results
            except Exception:
                pass

        # ② Milvus-only vector search
        if self._milvus.available:
            try:
                vec = self._embedder.embed_query(search_query)
                results = self._milvus.search(vec, top_k=self.top_k)
                if results:
                    return results
            except Exception:
                pass

        # ③ ES keyword search
        if self._es_keyword.available:
            try:
                results = self._es_keyword.search(search_query, top_k=self.top_k)
                if results:
                    return results
            except Exception:
                pass

        # ④ Ultimate fallback: in-memory Jaccard
        from enterprise_agentic_rag.config.settings import get_settings
        if not get_settings().runtime.allow_in_memory_fallback:
            raise RuntimeError("retrieval backends unavailable and in-memory fallback is disabled")
        return self._mem_keyword.search(search_query)

    @property
    def backend(self) -> str:
        if self._milvus.available and self._es_keyword.available:
            return "milvus+es"
        if self._milvus.available:
            return "milvus"
        if self._es_keyword.available:
            return "elasticsearch"
        return "memory_jaccard"
