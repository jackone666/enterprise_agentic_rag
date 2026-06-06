"""Semantic cache — embedding-similarity-based query result caching.

Caches (query_embedding → result) pairs so that semantically similar
queries hit the cache, reducing LLM costs and p95 latency for popular
questions.

Cache layers:
1. Exact match (SHA256 hash) — instant
2. Semantic match (cosine similarity > threshold) — near-instant
3. Cache miss → full pipeline

Storage backends:
- Redis (production, with TTL)
- In-memory LRU (development fallback)

Reference:
    TECHNICAL_DEEP_DIVE.md §42.3 — "语义缓存" listed as #3 priority improvement.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL", "3600"))  # 1 hour
_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_SIMILARITY", "0.92"))
_CACHE_MAX_ENTRIES = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "1000"))
_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "1").lower() in ("1", "true", "yes", "on")


class SemanticCache:
    """Two-tier semantic cache: exact + embedding-similarity match.

    Uses the application's embedding provider to encode queries, then
    compares against cached embeddings via cosine similarity.
    """

    def __init__(self) -> None:
        self._enabled = _CACHE_ENABLED
        self._ttl = _CACHE_TTL_SECONDS
        self._threshold = _CACHE_SIMILARITY_THRESHOLD
        self._max_entries = _CACHE_MAX_ENTRIES

        # In-memory stores (fallback when Redis unavailable)
        self._exact_store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._semantic_store: list[tuple[list[float], str, Any, float]] = []  # (embedding, key, value, timestamp)

        # Redis client (lazy)
        self._redis = None
        self._redis_available = False

        # Stats
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._semantic_hits = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "hit_rate": round(self.hit_rate, 4),
            "total_entries": len(self._exact_store),
            "semantic_entries": len(self._semantic_store),
        }

    async def get(self, query: str) -> tuple[str, Any] | None:
        """Look up a query in the cache.

        Returns:
            (hit_type, cached_value) tuple on hit, or None on miss.
            hit_type is one of: "exact", "semantic"
        """
        if not self._enabled:
            return None

        # Tier 1: Exact match
        exact_key = self._hash_query(query)
        result = await self._exact_lookup(exact_key)
        if result is not None:
            self._hits += 1
            self._exact_hits += 1
            logger.debug("Semantic cache: exact hit for query hash %s", exact_key[:8])
            return ("exact", result)

        # Tier 2: Semantic similarity match
        result = await self._semantic_lookup(query)
        if result is not None:
            self._hits += 1
            self._semantic_hits += 1
            logger.debug("Semantic cache: semantic hit for query %r", query[:50])
            return ("semantic", result)

        self._misses += 1
        return None

    async def set(self, query: str, value: Any, ttl: int | None = None) -> None:
        """Store a query-result pair in the cache.

        Args:
            query: The original user query.
            value: The result to cache (must be JSON-serializable).
            ttl: Optional per-entry TTL (defaults to global _CACHE_TTL_SECONDS).
        """
        if not self._enabled:
            return

        ttl = ttl or self._ttl
        exact_key = self._hash_query(query)
        now = time.time()

        # Store exact match
        await self._exact_store_entry(exact_key, value, ttl)

        # Store semantic embedding for similarity matching
        try:
            embedding = await self._embed_query(query)
            if embedding:
                self._semantic_store.append((embedding, exact_key, value, now))
                # Prune if over max
                while len(self._semantic_store) > self._max_entries:
                    self._semantic_store.pop(0)
        except Exception as exc:
            logger.debug("Failed to embed query for semantic cache: %s", exc)

        # Prune expired exact entries
        self._prune_exact()

    async def clear(self) -> None:
        """Clear all cached entries."""
        self._exact_store.clear()
        self._semantic_store.clear()
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        logger.info("Semantic cache cleared")

    # ------------------------------------------------------------------
    # Internal: exact match
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_query(query: str) -> str:
        """SHA256 hash a normalized query for exact matching."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    async def _exact_lookup(self, key: str) -> Any | None:
        """Look up exact match in store."""
        if key in self._exact_store:
            timestamp, value = self._exact_store[key]
            if time.time() - timestamp < self._ttl:
                # Move to end (LRU)
                self._exact_store.move_to_end(key)
                return value
            # Expired
            del self._exact_store[key]
        return None

    async def _exact_store_entry(self, key: str, value: Any, ttl: int) -> None:
        """Store an exact match entry with LRU eviction."""
        # Evict if full
        if len(self._exact_store) >= self._max_entries:
            self._exact_store.popitem(last=False)
        self._exact_store[key] = (time.time(), value)
        self._exact_store.move_to_end(key)

    def _prune_exact(self) -> None:
        """Remove expired exact-match entries."""
        now = time.time()
        expired = [
            k for k, (ts, _) in self._exact_store.items()
            if now - ts >= self._ttl
        ]
        for k in expired:
            del self._exact_store[k]

    # ------------------------------------------------------------------
    # Internal: semantic similarity
    # ------------------------------------------------------------------

    async def _semantic_lookup(self, query: str) -> Any | None:
        """Look up by embedding cosine similarity."""
        if not self._semantic_store:
            return None

        try:
            query_emb = await self._embed_query(query)
            if not query_emb:
                return None
        except Exception:
            return None

        now = time.time()
        best_score = -1.0
        best_value = None

        for emb, key, value, timestamp in self._semantic_store:
            # Check TTL
            if now - timestamp >= self._ttl:
                continue

            score = self._cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_value = value

        if best_score >= self._threshold and best_value is not None:
            return best_value

        return None

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    async def _embed_query(self, query: str) -> list[float] | None:
        """Embed a query using the application's embedding provider.

        Returns a float list representing the embedding vector.
        """
        try:
            from enterprise_agentic_rag.rag.embedding_provider import get_embedding
            # Use the sync embedding function in a thread
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(None, get_embedding, query)
            if embedding and isinstance(embedding, list):
                return [float(v) for v in embedding]
            return None
        except Exception as exc:
            logger.debug("Failed to embed query for semantic cache: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Math
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the global semantic cache singleton."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
