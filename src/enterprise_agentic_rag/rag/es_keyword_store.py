"""Elasticsearch keyword store — IK Analyzer-powered full-text search.

Provides Chinese-aware keyword retrieval as a replacement for the
in-memory Jaccard-based KeywordRetriever.

Tokenizer strategy:
  - Index time:  ik_max_word  (fine-grained — "认证方式" → ["认证","方式"])
  - Search time: ik_smart     (coarse-grained — "认证方式" → ["认证方式"])

Gracefully falls back when Elasticsearch is unreachable.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.config.settings import get_settings

logger = logging.getLogger(__name__)

_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "ik_index_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                },
                "ik_search_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                },
            }
        },
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "ik_index_analyzer",
                "search_analyzer": "ik_search_analyzer",
            },
            "source": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "ik_index_analyzer",
                "search_analyzer": "ik_search_analyzer",
            },
            "tags": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
        }
    },
}


class ESKeywordStore:
    """Elasticsearch-based keyword search with IK Analyzer tokenizer.

    Usage::

        store = ESKeywordStore()
        if store.available:
            store.ensure_index()
            store.index_chunks(chunks)
            results = store.search("API 认证方式", top_k=5)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        index_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._host = host or settings.elasticsearch.host
        self._port = port or settings.elasticsearch.port
        self._index = index_name or settings.elasticsearch.index
        self._client: Any = None
        self._available: bool | None = None

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------
    @property
    def client(self):
        """Lazily initialised Elasticsearch client."""
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch

                self._client = Elasticsearch(
                    f"http://{self._host}:{self._port}",
                    request_timeout=10,
                    max_retries=2,
                    retry_on_timeout=True,
                )
                # Quick connectivity check
                if not self._client.ping():
                    self._client = None
            except Exception:
                logger.debug("Elasticsearch client init failed — keyword search degraded")
                self._client = None
        return self._client

    @property
    def available(self) -> bool:
        """Whether Elasticsearch is reachable."""
        if self._available is None:
            self._available = self._check_health()
        return self._available

    def _check_health(self) -> bool:
        try:
            c = self.client
            if c is None:
                return False
            return bool(c.ping())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def ensure_index(self) -> bool:
        """Create the index with IK Analyzer mapping if it does not exist.

        Returns True if the index is ready after the call.
        """
        c = self.client
        if c is None:
            return False

        try:
            if not c.indices.exists(index=self._index):
                c.indices.create(index=self._index, body=_INDEX_SETTINGS)
                logger.info("Created ES index '%s' with IK Analyzer mapping", self._index)
            return True
        except Exception as exc:
            logger.warning("Failed to create ES index '%s': %s", self._index, exc)
            return False

    def delete_index(self) -> bool:
        """Drop the index (for full re-ingestion)."""
        c = self.client
        if c is None:
            return False
        try:
            if c.indices.exists(index=self._index):
                c.indices.delete(index=self._index)
            return True
        except Exception as exc:
            logger.warning("Failed to delete ES index: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Bulk-index chunks into Elasticsearch.

        Each chunk dict should have: ``chunk_id``, ``source``, ``content``,
        and optionally ``title``, ``tags``, ``tenant_id``.

        Returns the number of successfully indexed chunks.
        """
        c = self.client
        if c is None:
            return 0

        if not self.ensure_index():
            return 0

        from elasticsearch.helpers import bulk

        actions = []
        for ch in chunks:
            doc = {
                "_index": self._index,
                "_id": ch.get("chunk_id", ""),
                "_source": {
                    "content": ch.get("content", ""),
                    "source": ch.get("source", ""),
                    "chunk_id": ch.get("chunk_id", ""),
                    "title": ch.get("title", ch.get("source", "")),
                    "tags": ch.get("tags", []),
                    "tenant_id": ch.get("tenant_id", "default"),
                },
            }
            actions.append(doc)

        try:
            success, errors = bulk(c, actions, raise_on_error=False, refresh=True)
            if errors:
                logger.warning("ES bulk index: %d errors out of %d docs", len(errors), len(actions))
            return success
        except Exception as exc:
            logger.warning("ES bulk index failed: %s", exc)
            return 0

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks belonging to *source* (for update-then-reindex).

        Returns the number of deleted documents.
        """
        c = self.client
        if c is None:
            return 0
        try:
            result = c.delete_by_query(
                index=self._index,
                body={"query": {"term": {"source": source}}},
                refresh=True,
            )
            return result.get("deleted", 0)
        except Exception as exc:
            logger.warning("ES delete_by_source failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search via Elasticsearch with BM25 scoring.

        Args:
            query: Raw user query (Chinese or English).
            top_k: Maximum results to return.
            filters: Optional ``{field: value}`` term filters
                     (e.g. ``{"tenant_id": "default"}``).

        Returns:
            List of result dicts with ``content``, ``source``, ``score``,
            ``chunk_id``, and ``title``.
        """
        c = self.client
        if c is None:
            return []

        must_clauses: list[dict] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "title"],
                    "type": "best_fields",
                }
            }
        ]

        filter_clauses: list[dict] = []
        if filters:
            for field, value in filters.items():
                filter_clauses.append({"term": {field: value}})

        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": must_clauses,
                }
            },
            "size": top_k,
        }
        if filter_clauses:
            body["query"]["bool"]["filter"] = filter_clauses

        try:
            resp = c.search(index=self._index, body=body)
            hits = resp["hits"]["hits"]
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
            for hit in hits:
                src = hit["_source"]
                source_name = src.get("source", "")
                # Deduplicate by source to avoid returning multiple chunks
                # from the same document dominating results
                if source_name in seen:
                    continue
                seen.add(source_name)
                results.append({
                    "content": src.get("content", ""),
                    "source": source_name,
                    "score": round(hit["_score"] or 0.0, 4),
                    "chunk_id": src.get("chunk_id", hit["_id"]),
                    "title": src.get("title", ""),
                })
            return results
        except Exception as exc:
            logger.warning("ES search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        c = self.client
        if c is None:
            return {"available": False}
        try:
            index_exists = bool(c.indices.exists(index=self._index))
            doc_count = c.count(index=self._index).get("count", 0) if index_exists else 0
            return {
                "available": True,
                "index": self._index,
                "index_exists": index_exists,
                "doc_count": doc_count,
                "host": f"{self._host}:{self._port}",
            }
        except Exception:
            return {"available": False}
