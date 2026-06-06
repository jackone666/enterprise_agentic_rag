"""Checkpoint persistence — Redis-backed with in-memory fallback.

Redis key format: checkpoint:{session_id}:{checkpoint_id}
TTL: 24 hours

Also stores an ordered list per session for retrieval of "latest".
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "checkpoint"
DEFAULT_TTL = 86400  # 24 hours


class CheckpointStore:
    """Redis-first checkpoint store with in-memory fallback."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}  # fallback
        self._session_order: dict[str, list[str]] = {}
        self._redis = None
        self._redis_available: bool | None = None

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
                self._redis = aioredis.from_url(s.redis.connection_url, decode_responses=True)
                self._redis_available = True
            except Exception:
                self._redis_available = False
                self._redis = None
        return self._redis

    @property
    def redis_available(self) -> bool:
        if self._redis_available is None:
            _ = self.redis
        return self._redis_available or False

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_checkpoint(
        self, session_id: str, state: dict[str, Any], checkpoint_id: str | None = None
    ) -> str:
        cid = checkpoint_id or str(uuid.uuid4())

        # Clean state
        clean: dict[str, Any] = {}
        for k, v in state.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                clean[k] = v
            else:
                clean[k] = str(v)

        # In-memory fallback
        key = f"{session_id}::{cid}"
        self._store[key] = clean
        self._session_order.setdefault(session_id, []).append(cid)

        # Redis
        self._save_redis(session_id, cid, clean)

        return cid

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_checkpoint(
        self, session_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any] | None:
        # Try Redis first
        if checkpoint_id:
            val = self._load_redis(session_id, checkpoint_id)
            if val:
                return val

        # Latest from Redis
        latest = self._load_latest_redis(session_id)
        if latest:
            return latest

        # Fallback to in-memory
        if checkpoint_id:
            return self._store.get(f"{session_id}::{checkpoint_id}")
        order = self._session_order.get(session_id, [])
        if not order:
            return None
        return self._store.get(f"{session_id}::{order[-1]}")

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_session(self, session_id: str) -> None:
        self._session_order.pop(session_id, None)
        keys_to_del = [k for k in self._store if k.startswith(f"{session_id}::")]
        for k in keys_to_del:
            self._store.pop(k, None)

        if self.redis_available:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(self._delete_redis_session(session_id))
            except Exception:
                pass

    @property
    def checkpoint_count(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------
    def _save_redis(self, session_id: str, cid: str, state: dict) -> None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running() and self.redis:
                rkey = self._redis_key(session_id, cid)
                okey = self._order_key(session_id)
                val = json.dumps(state, ensure_ascii=False, default=str)
                loop.run_until_complete(self._redis_pipeline(rkey, okey, cid, val))
        except Exception:
            pass

    async def _redis_pipeline(self, rkey: str, okey: str, cid: str, val: str) -> None:
        r = self.redis
        if r:
            async with r.pipeline() as pipe:
                pipe.set(rkey, val, ex=DEFAULT_TTL)
                pipe.rpush(okey, cid)
                pipe.expire(okey, DEFAULT_TTL)
                await pipe.execute()

    def _load_redis(self, session_id: str, checkpoint_id: str) -> dict | None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running() and self.redis:
                val = loop.run_until_complete(
                    self.redis.get(self._redis_key(session_id, checkpoint_id))
                )
                if val:
                    return json.loads(val)
        except Exception:
            pass
        return None

    def _load_latest_redis(self, session_id: str) -> dict | None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running() and self.redis:
                r = self.redis
                okey = self._order_key(session_id)
                items = loop.run_until_complete(r.lrange(okey, -1, -1))
                if items:
                    val = loop.run_until_complete(
                        r.get(self._redis_key(session_id, items[0]))
                    )
                    if val:
                        return json.loads(val)
        except Exception:
            pass
        return None

    async def _delete_redis_session(self, session_id: str) -> None:
        r = self.redis
        if r:
            okey = self._order_key(session_id)
            cids = await r.lrange(okey, 0, -1)
            keys = [self._redis_key(session_id, c) for c in cids] + [okey]
            if keys:
                await r.delete(*keys)

    @staticmethod
    def _redis_key(session_id: str, checkpoint_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:{session_id}:{checkpoint_id}"

    @staticmethod
    def _order_key(session_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:{session_id}:order"
