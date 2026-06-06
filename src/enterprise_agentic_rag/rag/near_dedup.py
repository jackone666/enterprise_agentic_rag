"""Near-deduplication — SHA256 exact match + embedding cosine similarity.

Two-layer dedup:
1. Fast path: SHA256 content hash exact match (skip duplicate)
2. Slow path: embedding cosine similarity against registered fingerprints

Persistence: Redis primary, in-memory dict fallback.
Redis keys:
  dedup:doc:{doc_id}       → Hash {content_hash, embedding, metadata}
  dedup:hash:{content_hash} → Set of doc_ids (for O(1) exact-match lookup)
  dedup:all                 → Set of all doc_ids (for bulk reload)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Redis key prefixes
REDIS_KEY_PREFIX = "dedup:doc"
REDIS_HASH_PREFIX = "dedup:hash"
REDIS_ALL_KEY = "dedup:all"

# TTL for dedup keys (30 days — re-ingestion cadence friendly)
DEFAULT_TTL = 60 * 60 * 24 * 30


class NearDedupIndex:
    """Document-level semantic fingerprint index for near-duplicate detection.

    Redis-first with in-memory fallback.  Cosine similarity is always
    computed in Python/NumPy — Redis is used only for persistence and
    O(1) exact-hash lookups.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        index_path: str | None = None,
    ) -> None:
        self.threshold = threshold

        # In-memory fingerprint cache (for cosine similarity)
        self._fingerprints: dict[str, dict[str, Any]] = {}

        # Redis client (lazy)
        self._redis: Any = None
        self._redis_available: bool | None = None

        # Legacy JSON fallback path (kept for migration / cold-start)
        if index_path is None:
            from pathlib import Path
            index_path = str(
                Path(__file__).resolve().parents[3] / "data" / "near_dedup_index.json"
            )
        self._index_path = index_path

        # Load existing fingerprints into memory
        self._load()

    # ------------------------------------------------------------------
    # Redis client (lazy)
    # ------------------------------------------------------------------
    @property
    def redis(self):
        """Lazy Redis connection, shared across the process."""
        if self._redis is None and self._redis_available is not False:
            try:
                import redis as redis_lib

                from enterprise_agentic_rag.config.settings import get_settings

                s = get_settings()
                self._redis = redis_lib.from_url(
                    s.redis.connection_url, decode_responses=True
                )
                self._redis.ping()
                self._redis_available = True
            except Exception:
                self._redis_available = False
                self._redis = None
                logger.debug("Redis unavailable for dedup — using in-memory fallback")
        return self._redis

    @property
    def redis_available(self) -> bool:
        if self._redis_available is None:
            _ = self.redis  # trigger lazy init
        return self._redis_available or False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(
        self,
        doc_id: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Register a document fingerprint. Returns True if new, False if duplicate."""
        import hashlib

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # 1. Exact hash check — Redis O(1) + in-memory fallback
        if self.redis_available:
            try:
                rkey = self._hash_key(content_hash)
                if self._redis.exists(rkey):
                    return False  # exact duplicate
            except Exception:
                pass
        else:
            for fid, fp in self._fingerprints.items():
                if fp.get("content_hash") == content_hash:
                    return False

        # 2. Near-duplicate check via embedding (NumPy, always in-memory)
        if embedding and self._fingerprints:
            vec = np.array(embedding, dtype=np.float32)
            for fid, fp in self._fingerprints.items():
                existing_vec = np.array(fp.get("embedding", []), dtype=np.float32)
                if len(existing_vec) == 0:
                    continue
                sim = self._cosine_sim(vec, existing_vec)
                if sim >= self.threshold:
                    return False  # near-duplicate

        # Register new fingerprint — memory
        self._fingerprints[doc_id] = {
            "content_hash": content_hash,
            "embedding": embedding,
            "metadata": metadata or {},
        }

        # Register new fingerprint — Redis
        if self.redis_available:
            try:
                self._save_redis(doc_id, content_hash, embedding, metadata or {})
            except Exception:
                logger.debug("Redis save failed for doc_id=%s", doc_id)

        return True

    def detect(self, embedding: list[float]) -> tuple[bool, str | None, float]:
        """Check if an embedding matches any registered document.

        Returns (is_duplicate, matched_doc_id, max_similarity).
        """
        if not self._fingerprints:
            return False, None, 0.0

        vec = np.array(embedding, dtype=np.float32)
        max_sim = 0.0
        matched = None
        for fid, fp in self._fingerprints.items():
            existing = np.array(fp.get("embedding", []), dtype=np.float32)
            if len(existing) == 0:
                continue
            sim = self._cosine_sim(vec, existing)
            if sim > max_sim:
                max_sim = sim
                matched = fid

        if max_sim >= self.threshold:
            return True, matched, max_sim
        return False, None, max_sim

    def remove(self, doc_id: str) -> bool:
        if doc_id in self._fingerprints:
            content_hash = self._fingerprints[doc_id].get("content_hash", "")
            del self._fingerprints[doc_id]

            # Remove from Redis
            if self.redis_available:
                try:
                    pipe = self._redis.pipeline()
                    pipe.delete(self._doc_key(doc_id))
                    pipe.srem(self._hash_key(content_hash), doc_id)
                    pipe.srem(REDIS_ALL_KEY, doc_id)
                    pipe.execute()
                except Exception:
                    pass
            return True
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "total_docs": len(self._fingerprints),
            "threshold": self.threshold,
            "doc_ids": list(self._fingerprints.keys()),
            "redis_available": self.redis_available,
        }

    # ------------------------------------------------------------------
    # Persistence: Redis primary, JSON file fallback
    # ------------------------------------------------------------------
    def _save_redis(
        self,
        doc_id: str,
        content_hash: str,
        embedding: list[float] | None,
        metadata: dict[str, Any],
    ) -> None:
        """Write a single fingerprint to Redis (atomic pipeline)."""
        r = self.redis
        if r is None:
            return

        emb_json = json.dumps(embedding or [], ensure_ascii=False)

        pipe = r.pipeline()
        # Store the fingerprint
        pipe.hset(
            self._doc_key(doc_id),
            mapping={
                "content_hash": content_hash,
                "embedding": emb_json,
                "metadata": json.dumps(metadata, ensure_ascii=False),
            },
        )
        pipe.expire(self._doc_key(doc_id), DEFAULT_TTL)
        # Index by content hash
        pipe.sadd(self._hash_key(content_hash), doc_id)
        pipe.expire(self._hash_key(content_hash), DEFAULT_TTL)
        # Global index
        pipe.sadd(REDIS_ALL_KEY, doc_id)
        pipe.expire(REDIS_ALL_KEY, DEFAULT_TTL)
        pipe.execute()

    def _load(self) -> None:
        """Load fingerprints: Redis first, JSON file fallback."""
        # Try Redis
        if self.redis_available:
            try:
                r = self.redis
                doc_ids = r.smembers(REDIS_ALL_KEY)
                if doc_ids:
                    loaded: dict[str, dict[str, Any]] = {}
                    for doc_id in doc_ids:
                        data = r.hgetall(self._doc_key(doc_id))
                        if not data:
                            continue
                        embedding = []
                        try:
                            embedding = json.loads(data.get("embedding", "[]"))
                        except (json.JSONDecodeError, TypeError):
                            pass
                        metadata = {}
                        try:
                            metadata = json.loads(data.get("metadata", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            pass
                        loaded[doc_id] = {
                            "content_hash": data.get("content_hash", ""),
                            "embedding": embedding,
                            "metadata": metadata,
                        }
                    if loaded:
                        self._fingerprints = loaded
                        logger.debug(
                            "Loaded %d dedup fingerprints from Redis", len(loaded)
                        )
                        return
            except Exception:
                logger.debug("Redis load failed, falling back to JSON file")

        # Fallback to JSON file (cold-start / migration)
        self._load_json()

    def _load_json(self) -> None:
        """Load fingerprints from legacy JSON file."""
        import os

        if not os.path.exists(self._index_path):
            return
        try:
            with open(self._index_path, encoding="utf-8") as f:
                data = json.load(f)
            self._fingerprints = {
                fid: {
                    "content_hash": fp.get("content_hash", ""),
                    "embedding": fp.get("embedding", []),
                    "metadata": fp.get("metadata", {}),
                }
                for fid, fp in data.items()
            }
            logger.debug(
                "Loaded %d dedup fingerprints from JSON fallback",
                len(self._fingerprints),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _doc_key(doc_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:{doc_id}"

    @staticmethod
    def _hash_key(content_hash: str) -> str:
        return f"{REDIS_HASH_PREFIX}:{content_hash}"


# Create sampling embedding for long documents
def sample_doc_embedding(
    content: str,
    embed_fn: Any,
    head_chars: int = 500,
    tail_chars: int = 300,
) -> list[float]:
    """Generate a document-level fingerprint by embedding head+tail."""
    head = content[:head_chars]
    tail = content[-tail_chars:] if len(content) > tail_chars else ""
    sample = head + ("\n...\n" if tail else "") + tail
    vecs = embed_fn.embed([sample])
    return vecs[0] if vecs else []
