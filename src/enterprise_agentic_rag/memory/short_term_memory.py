"""Per-session short-term memory — Redis-backed with in-memory fallback.

Redis key:  chat:{session_id}:history (LIST, TTL 24h)
Also writes to PostgreSQL messages table when available.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "chat"
DEFAULT_TTL_SECONDS = 86400  # 24 hours


@dataclass
class ChatTurn:
    role: str
    content: str
    intent: str = ""


class ShortTermMemory:
    """Redis-first sliding-window memory with in-memory fallback."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        # In-memory fallback store
        self._fallback: dict[str, deque[ChatTurn]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )
        self._redis = None
        self._redis_available: bool | None = None
        self._repo = None  # lazy PostgreSQL repo

    # ------------------------------------------------------------------
    # Redis client (lazy)
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
                logger.warning("Redis unavailable — using in-memory short-term memory")
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

    # ------------------------------------------------------------------
    # Public API (synchronous — wraps async internally)
    # ------------------------------------------------------------------
    def add_message(
        self, session_id: str, role: str, content: str, intent: str = ""
    ) -> None:
        """Add a message to Redis + in-memory fallback + PostgreSQL."""
        turn = ChatTurn(role=role, content=content, intent=intent)

        # Always write to in-memory fallback
        self._fallback[session_id].append(turn)

        # Try Redis
        if self.redis_available:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._add_redis(session_id, turn))
                else:
                    loop.run_until_complete(self._add_redis(session_id, turn))
            except Exception:
                pass

        # Try PostgreSQL
        self._add_pg_sync(session_id, role, content, intent)

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        """Get history from Redis, falling back to in-memory."""
        if self.redis_available:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't block in async context; use fallback
                    return self._get_fallback(session_id, last_n)
                return loop.run_until_complete(self._get_redis(session_id, last_n))
            except Exception:
                pass
        return self._get_fallback(session_id, last_n)

    def get_last_assistant_answer(self, session_id: str) -> str:
        history = self.get_history(session_id)
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                return turn.get("content", "")
        return ""

    def clear_session(self, session_id: str) -> None:
        self._fallback.pop(session_id, None)
        if self.redis_available:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(self._clear_redis(session_id))
            except Exception:
                pass

    @property
    def session_count(self) -> int:
        return len(self._fallback)

    # ------------------------------------------------------------------
    # Redis async helpers
    # ------------------------------------------------------------------
    async def _add_redis(self, session_id: str, turn: ChatTurn) -> None:
        key = self._redis_key(session_id)
        entry = json.dumps({
            "role": turn.role, "content": turn.content, "intent": turn.intent
        }, ensure_ascii=False)
        try:
            r = self.redis
            if r:
                await r.rpush(key, entry)
                await r.ltrim(key, -self.max_turns, -1)
                await r.expire(key, DEFAULT_TTL_SECONDS)
        except Exception:
            pass

    async def _get_redis(self, session_id: str, last_n: int | None = None) -> list[dict]:
        key = self._redis_key(session_id)
        try:
            r = self.redis
            if r:
                items = await r.lrange(key, 0, -1)
                if last_n:
                    items = items[-last_n:]
                return [json.loads(it) for it in items]
        except Exception:
            pass
        return self._get_fallback(session_id, last_n)

    async def _clear_redis(self, session_id: str) -> None:
        try:
            r = self.redis
            if r:
                await r.delete(self._redis_key(session_id))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    def _add_pg_sync(self, session_id: str, role: str, content: str, intent: str) -> None:
        repo = self.repo
        if repo is None:
            return
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            coro = repo.insert_message(
                session_id=session_id, role=role, content=content, intent=intent
            )
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # In-memory fallback
    # ------------------------------------------------------------------
    def _get_fallback(self, session_id: str, last_n: int | None = None) -> list[dict]:
        buf = self._fallback.get(session_id, deque())
        items = list(buf) if last_n is None else list(buf)[-last_n:]
        return [{"role": t.role, "content": t.content, "intent": t.intent} for t in items]

    @staticmethod
    def _redis_key(session_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:{session_id}:history"
