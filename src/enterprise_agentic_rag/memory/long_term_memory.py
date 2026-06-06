"""Long-term memory — cross-session persistent user and task memories.

Memory types:
- episodic: what happened before, e.g. past questions, task outcomes, project events.
- semantic: stable preferences, project facts, domain knowledge, and business rules.

Scoring: rule-based signals (zero LLM cost).
Storage: PostgreSQL (canonical) + Milvus vector DB + Redis cache.
Dedup:  SHA256 exact match + embedding cosine similarity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

VECTOR_COLLECTION = "long_term_memories"
REDIS_CACHE_TTL = 86400  # 24h
DEFAULT_DEDUP_THRESHOLD = 0.92


@dataclass
class LongTermMemoryEntry:
    memory_id: str
    user_id: str
    content: str
    importance: float = 0.0
    memory_type: str = "episodic"
    source_session: str = ""
    source_turn: int = 0
    created_at: str = ""
    accessed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LongTermMemory:
    """Cross-session episodic/semantic memory with fusion ranking."""

    def __init__(
        self,
        importance_threshold: float = 0.5,
        max_memories_per_user: int = 100,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ) -> None:
        self.importance_threshold = importance_threshold
        self.max_memories_per_user = max_memories_per_user
        self.dedup_threshold = dedup_threshold

        # In-memory fallback store: {user_id: [LongTermMemoryEntry, ...]}
        self._store: dict[str, list[LongTermMemoryEntry]] = {}
        # In-memory embedding cache for dedup: {user_id: [(content_hash, embedding), ...]}
        self._embed_cache: dict[str, list[tuple[str, list[float]]]] = {}

        # Lazy clients
        self._redis = None
        self._redis_available: bool | None = None
        self._repo = None
        self._embedder = None
        self._vector_store = None

    # ------------------------------------------------------------------
    # Lazy clients
    # ------------------------------------------------------------------
    @property
    def redis(self):
        if self._redis is None and self._redis_available is not False:
            try:
                import redis.asyncio as aioredis

                from enterprise_agentic_rag.config.settings import get_settings

                s = get_settings()
                self._redis = aioredis.from_url(
                    s.redis.connection_url, decode_responses=True
                )
                self._redis_available = True
            except Exception:
                self._redis_available = False
                self._redis = None
                logger.warning("Redis unavailable for long-term memory")
        return self._redis

    @property
    def redis_available(self) -> bool:
        if self._redis_available is None:
            _ = self.redis
        return self._redis_available or False

    @property
    def repo(self):
        if self._repo is None:
            try:
                from enterprise_agentic_rag.storage.repositories import Repository

                self._repo = Repository()
            except Exception:
                self._repo = False
        return self._repo if self._repo is not False else None

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from enterprise_agentic_rag.rag.embedding_provider import (
                    get_embedding_provider,
                )

                self._embedder = get_embedding_provider()
            except Exception:
                self._embedder = False
        return self._embedder if self._embedder is not False else None

    @property
    def vector_store(self):
        if self._vector_store is None:
            try:
                from enterprise_agentic_rag.rag.milvus_store import MilvusStore

                vector_size = self.embedder.vector_size if self.embedder is not None else 768
                vs = MilvusStore(collection_name=VECTOR_COLLECTION, vector_size=vector_size)
                if vs.available:
                    vs.ensure_collection(VECTOR_COLLECTION)
                    self._vector_store = vs
                else:
                    self._vector_store = False
            except Exception:
                self._vector_store = False
        return self._vector_store if self._vector_store is not False else None

    # ------------------------------------------------------------------
    # Public API — Extract & Store
    # ------------------------------------------------------------------
    def extract_and_store(
        self,
        turns: list[dict],
        user_id: str,
        session_id: str,
        memory_type: str = "episodic",
    ) -> int:
        """Score each turn against the importance threshold and persist high-scoring ones.

        Returns the number of new memories stored (after dedup).
        """
        if not turns or not user_id:
            return 0

        total_turns = len(turns)
        stored = 0

        for idx, turn in enumerate(turns):
            score = self._score_turn(turn, turn_index=idx, total_turns=total_turns)
            if score < self.importance_threshold:
                continue

            content = turn.get("content", "").strip()
            if not content:
                continue

            # Dedup check
            if self._is_duplicate(content, user_id):
                continue

            entry = LongTermMemoryEntry(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                content=content,
                importance=score,
                memory_type=turn.get("memory_type", memory_type),
                source_session=session_id,
                source_turn=idx,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                accessed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metadata=turn.get("metadata", {}),
            )

            # In-memory fallback
            self._store.setdefault(user_id, []).append(entry)
            # Enforce per-user cap
            if len(self._store[user_id]) > self.max_memories_per_user:
                self._store[user_id] = self._store[user_id][-self.max_memories_per_user:]

            # Persist to PostgreSQL
            self._save_pg(entry)

            # Persist to vector DB
            self._save_vector(entry)

            # Cache in Redis
            self._cache_redis(user_id, entry)

            stored += 1

        return stored

    # ------------------------------------------------------------------
    # Public API — Retrieve
    # ------------------------------------------------------------------
    def retrieve(
        self,
        user_id: str,
        query: str | None = None,
        top_k: int = 5,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Semantic search over the user's long-term memories.

        When ``query`` is None, returns the most recent memories.
        """
        results: list[dict] = []

        # Try vector search when query is provided
        if query and self.embedder and self.vector_store:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    vec = loop.run_until_complete(
                        asyncio.to_thread(self.embedder.embed_query, query)
                    )
                    results = self._search_vector(vec, user_id, top_k)
            except Exception:
                logger.warning("Vector search failed for long-term memory", exc_info=True)

        # Fallback: PG query sorted by recency + importance
        if not results:
            results = self._get_pg_memories(user_id, limit=top_k)

        # Last resort: in-memory fallback
        if not results:
            results = self._get_fallback_memories(user_id, limit=top_k)

        if memory_type:
            results = [
                r for r in results
                if r.get("memory_type", "episodic") == memory_type
            ][:top_k]

        results = self._rank_memories(query or "", results, top_k)

        # Touch accessed_at for returned memories
        self._touch_accessed(user_id, results)

        return results

    def get_recent(self, user_id: str, limit: int = 10) -> list[dict]:
        """Return most recent long-term memories without embedding search."""
        results = self._get_pg_memories(user_id, limit=limit)
        if not results:
            results = self._get_fallback_memories(user_id, limit=limit)
        return results

    def store_entry(
        self,
        user_id: str,
        content: str,
        *,
        session_id: str = "",
        source_turn: int = 0,
        importance: float = 0.0,
        memory_type: str = "episodic",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a pre-classified long-term memory candidate."""
        content = (content or "").strip()
        if not user_id or not content:
            return False
        if self._is_duplicate(content, user_id):
            return False

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry = LongTermMemoryEntry(
            memory_id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            source_session=session_id,
            source_turn=source_turn,
            created_at=now,
            accessed_at=now,
            metadata=metadata or {},
        )
        self._store.setdefault(user_id, []).append(entry)
        if len(self._store[user_id]) > self.max_memories_per_user:
            self._store[user_id] = self._store[user_id][-self.max_memories_per_user:]
        self._save_pg(entry)
        self._save_vector(entry)
        self._cache_redis(user_id, entry)
        return True

    # ------------------------------------------------------------------
    # Public API — Delete
    # ------------------------------------------------------------------
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by its ID."""
        deleted = False

        # In-memory
        for uid, entries in self._store.items():
            self._store[uid] = [e for e in entries if e.memory_id != memory_id]
            if len(self._store[uid]) < len(entries):
                deleted = True

        # PG
        self._delete_pg(memory_id)

        # Vector DB
        self._delete_vector(memory_id)

        return deleted

    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user. Returns count deleted."""
        count = len(self._store.pop(user_id, []))

        # PG
        pg_count = self._delete_user_pg(user_id)
        count = max(count, pg_count)

        # Clear Redis cache
        self._clear_redis_user(user_id)

        return count

    # ------------------------------------------------------------------
    # Importance scoring (rule-based, zero LLM cost)
    # ------------------------------------------------------------------
    @staticmethod
    @staticmethod
    def _score_turn(turn: dict, turn_index: int, total_turns: int) -> float:
        """Score a single turn 0.0–1.0 with semantic importance assessment.

        Upgraded scoring (Technical Deep Dive §34.4):
        Base rule-based signals + semantic importance + user feedback.

        Signals (additive, capped at 1.0):
        - Contains code block (```...```)       +0.25
        - Contains error/exception text          +0.20
        - First turn in session                  +0.15
        - Message length > 100 chars             +0.15
        - User role (vs assistant)              +0.10
        - Contains question mark                 +0.10
        - Semantic density (info-rich content)   +0.10  ← NEW
        - API/technical reference density         +0.10  ← NEW
        - Turn in first half of session          +0.05
        - User feedback boost (if available)     +0.15  ← NEW
        - Time decay penalty (older = less)      -0.05–0.20 ← NEW
        """
        content = turn.get("content", "")
        role = turn.get("role", "user")
        score = 0.0

        # ── Code block detection ──
        if "```" in content:
            score += 0.25

        # ── Error / exception patterns ──
        _err_patterns = (
            r"(?i)(\b(error|exception|traceback|stack\s?trace)\b|"
            r"nullpointer|NullPointer|"
            r"失败|错误|异常|报错|堆栈|"
            r"500|502|503|404|401|403|timeout|超时)"
        )
        if re.search(_err_patterns, content):
            score += 0.20

        # ── First turn bonus ──
        if turn_index == 0:
            score += 0.15

        # ── Message length ──
        if len(content) > 100:
            score += 0.15

        # ── User role ──
        if role == "user":
            score += 0.10

        # ── Contains question mark ──
        if "?" in content or "？" in content:
            score += 0.10

        # ── NEW: Semantic density (information richness) ──
        # Detect API references, version numbers, configuration patterns
        tech_patterns = (
            r"(@ohos\.[\w.]+|"
            r"import\s+\{.*?\}\s+from|"
            r"API\s*\d{1,2}|"
            r"HarmonyOS\s*(?:NEXT\s*)?\d+\.\d+|"
            r"https?://[^\s]+|"
            r"[A-Z][a-z]+(?:[A-Z][a-z]+)+)"  # PascalCase identifiers
        )
        tech_matches = len(re.findall(tech_patterns, content))
        if tech_matches >= 2:
            score += 0.10  # Technically dense content

        # ── NEW: API/technical reference density ──
        api_refs = re.findall(r"@ohos\.[\w.]+", content)
        if len(api_refs) >= 3:
            score += 0.10  # Rich API documentation content

        # ── Early in session ──
        if total_turns > 0 and turn_index < total_turns / 2:
            score += 0.05

        # ── NEW: User feedback boost ──
        feedback_score = turn.get("feedback_score", 0)
        if feedback_score > 0:
            score += min(0.15, feedback_score * 0.15)

        # ── NEW: Time decay penalty ──
        created_at = turn.get("created_at", "")
        if created_at:
            try:
                import time as _time
                from datetime import datetime
                if isinstance(created_at, str):
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    created_dt = created_at
                age_hours = (_time.time() - created_dt.timestamp()) / 3600
                if age_hours > 168:  # 1 week
                    score -= 0.10
                elif age_hours > 720:  # 30 days
                    score -= 0.20
            except Exception:
                pass  # Time decay is best-effort

        return max(0.0, min(score, 1.0))

    # ------------------------------------------------------------------
    # Fusion ranking
    # ------------------------------------------------------------------
    @classmethod
    def _rank_memories(
        cls,
        query: str,
        memories: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Rank memories by relevance, time decay, and importance.

        The formula is intentionally simple and explainable:
        final = relevance * 0.5 + importance * 0.3 + recency_decay * 0.2
        """
        if not memories:
            return []

        ranked = []
        for memory in memories:
            item = dict(memory)
            relevance = cls._calculate_relevance(query, item)
            importance = cls._normalise_float(item.get("importance", 0.0))
            recency = cls._calculate_recency_decay(item)
            final_score = relevance * 0.5 + importance * 0.3 + recency * 0.2

            item["relevance_score"] = relevance
            item["recency_score"] = recency
            item["final_score"] = final_score
            ranked.append(item)

        ranked.sort(
            key=lambda m: (
                m.get("final_score", 0.0),
                m.get("importance", 0.0),
                m.get("created_at", ""),
            ),
            reverse=True,
        )
        return ranked[:top_k]

    @staticmethod
    def _calculate_relevance(query: str, memory: dict) -> float:
        score = memory.get("score")
        if isinstance(score, (int, float)):
            return max(0.0, min(float(score), 1.0))

        query_terms = LongTermMemory._tokenise(query)
        content_terms = LongTermMemory._tokenise(memory.get("content", ""))
        if not query_terms or not content_terms:
            return 0.0
        overlap = len(query_terms & content_terms)
        return min(1.0, overlap / max(1, len(query_terms)))

    @staticmethod
    def _calculate_recency_decay(memory: dict) -> float:
        raw = memory.get("accessed_at") or memory.get("created_at")
        if not raw:
            return 0.0
        try:
            if isinstance(raw, datetime):
                dt = raw
            else:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400)
            return float(pow(2.718281828, -age_days / 30.0))
        except Exception:
            return 0.0

    @staticmethod
    def _normalise_float(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        lowered = (text or "").lower()
        terms = set(re.findall(r"[a-z0-9_]{2,}", lowered))
        terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
        return terms

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def _is_duplicate(self, content: str, user_id: str) -> bool:
        """Check if content is a duplicate of an existing memory.

        Layer 1: SHA256 exact-match hash (fast, O(1))
        Layer 2: Cosine similarity of embeddings (slower, only if embedder available)
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Layer 1 — exact hash
        cache = self._embed_cache.get(user_id, [])
        for cached_hash, _ in cache:
            if cached_hash == content_hash:
                return True

        # Also check in-memory store
        for entry in self._store.get(user_id, []):
            entry_hash = hashlib.sha256(entry.content.encode()).hexdigest()[:16]
            if entry_hash == content_hash:
                return True

        # Layer 2 — cosine similarity (only if embedder available)
        emb = self.embedder
        if emb and cache:
            try:
                import asyncio

                import numpy as np

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    new_vec = loop.run_until_complete(
                        asyncio.to_thread(emb.embed_query, content)
                    )
                    new_arr = np.asarray(new_vec, dtype=np.float32)
                    new_norm = float(np.linalg.norm(new_arr))
                    if new_norm == 0:
                        return False

                    for _, cached_vec in cache:
                        cached_arr = np.asarray(cached_vec, dtype=np.float32)
                        cached_norm = float(np.linalg.norm(cached_arr))
                        if cached_norm == 0:
                            continue
                        sim = float(np.dot(new_arr, cached_arr) / (new_norm * cached_norm))
                        if sim >= self.dedup_threshold:
                            return True
            except Exception:
                pass

        # Cache this embedding for future dedup checks
        if emb:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    vec = loop.run_until_complete(
                        asyncio.to_thread(emb.embed_query, content)
                    )
                    self._embed_cache.setdefault(user_id, []).append((content_hash, vec))
                    # Cap cache size
                    if len(self._embed_cache[user_id]) > self.max_memories_per_user * 2:
                        self._embed_cache[user_id] = self._embed_cache[user_id][
                            -self.max_memories_per_user:
                        ]
            except Exception:
                pass

        return False

    # ------------------------------------------------------------------
    # PostgreSQL persistence
    # ------------------------------------------------------------------
    def _save_pg(self, entry: LongTermMemoryEntry) -> None:
        repo = self.repo
        if repo is None:
            return
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            coro = repo.upsert_long_term_memory(
                memory_id=entry.memory_id,
                user_id=entry.user_id,
                content=entry.content,
                importance=entry.importance,
                memory_type=entry.memory_type,
                source_session=entry.source_session,
                source_turn=entry.source_turn,
                metadata=entry.metadata,
            )
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except Exception:
            pass

    def _get_pg_memories(self, user_id: str, limit: int = 10) -> list[dict]:
        repo = self.repo
        if repo is None:
            return []
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(
                    repo.get_long_term_memories(user_id, limit)
                )
        except Exception:
            pass
        return []

    def _delete_pg(self, memory_id: str) -> None:
        repo = self.repo
        if repo is None:
            return
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(repo.delete_long_term_memory(memory_id))
        except Exception:
            pass

    def _delete_user_pg(self, user_id: str) -> int:
        repo = self.repo
        if repo is None:
            return 0
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(
                    repo.delete_user_long_term_memories(user_id)
                )
        except Exception:
            pass
        return 0

    def _touch_accessed(self, user_id: str, results: list[dict]) -> None:
        """Update accessed_at for retrieved memories (best-effort, only in-memory)."""
        store_entries = self._store.get(user_id, [])
        if not store_entries:
            return
        result_ids = {r.get("memory_id") for r in results if r.get("memory_id")}
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for entry in store_entries:
            if entry.memory_id in result_ids:
                entry.accessed_at = now

    # ------------------------------------------------------------------
    # Vector DB persistence
    # ------------------------------------------------------------------
    def _save_vector(self, entry: LongTermMemoryEntry) -> None:
        vs = self.vector_store
        emb = self.embedder
        if vs is None or emb is None:
            return
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return

            vec = loop.run_until_complete(
                asyncio.to_thread(emb.embed_query, entry.content)
            )
            chunks = [
                {
                    "chunk_id": entry.memory_id,
                    "source": f"ltm:{entry.user_id}",
                    "content": entry.content,
                    "title": f"memory_{entry.importance:.2f}",
                    "tags": [entry.user_id, entry.source_session],
                    "metadata": {
                        "memory_type": entry.memory_type,
                        **entry.metadata,
                    },
                    "created_at": entry.created_at,
                }
            ]
            vs.upsert_chunks(chunks, [vec], collection_name=VECTOR_COLLECTION)
        except Exception:
            pass

    def _search_vector(
        self, query_vector: list[float], user_id: str, top_k: int
    ) -> list[dict]:
        vs = self.vector_store
        if vs is None:
            return []
        try:
            raw = vs.search(
                query_vector,
                top_k=top_k,
                collection_name=VECTOR_COLLECTION,
                filters={"source": f"ltm:{user_id}"},
            )
            # Keep a client-side guard as a defense-in-depth check.
            results = []
            for r in raw:
                tags = r.get("metadata", {}).get("tags", r.get("tags", []))
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                if user_id in (tags or []):
                    results.append({
                        "memory_id": r.get("chunk_id", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                        "importance": 0.0,
                        "memory_type": r.get("metadata", {}).get("memory_type", "episodic"),
                    })
            return results[:top_k]
        except Exception:
            return []

    def _delete_vector(self, memory_id: str) -> None:
        vs = self.vector_store
        if vs is None:
            return
        try:
            vs.delete_by_source(f"ltm:{memory_id}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Redis cache
    # ------------------------------------------------------------------
    def _cache_redis(self, user_id: str, entry: LongTermMemoryEntry) -> None:
        if not self.redis_available:
            return
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                r = self.redis
                key = self._redis_key(user_id)
                val = json.dumps(
                    {
                        "memory_id": entry.memory_id,
                        "content": entry.content,
                        "importance": entry.importance,
                        "memory_type": entry.memory_type,
                        "source_session": entry.source_session,
                        "created_at": entry.created_at,
                        "metadata": entry.metadata,
                    },
                    ensure_ascii=False,
                )
                loop.run_until_complete(self._redis_push(r, key, val))
        except Exception:
            pass

    async def _redis_push(self, r, key: str, val: str) -> None:
        if r:
            await r.lpush(key, val)
            await r.ltrim(key, 0, self.max_memories_per_user - 1)
            await r.expire(key, REDIS_CACHE_TTL)

    def _clear_redis_user(self, user_id: str) -> None:
        if not self.redis_available:
            return
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self.redis.delete(self._redis_key(user_id)))
        except Exception:
            pass

    @staticmethod
    def _redis_key(user_id: str) -> str:
        return f"ltm:{user_id}:recent"

    # ------------------------------------------------------------------
    # In-memory fallback
    # ------------------------------------------------------------------
    def _get_fallback_memories(self, user_id: str, limit: int = 10) -> list[dict]:
        entries = self._store.get(user_id, [])
        # Sort by importance desc, then recency
        entries = sorted(entries, key=lambda e: (e.importance, e.created_at), reverse=True)
        return [
            {
                "memory_id": e.memory_id,
                "content": e.content,
                "importance": e.importance,
                "memory_type": e.memory_type,
                "source_session": e.source_session,
                "created_at": e.created_at,
                "metadata": e.metadata,
            }
            for e in entries[:limit]
        ]
