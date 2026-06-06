"""Distributed rate limiter — Redis sliding window + fail-open fallback."""

from __future__ import annotations

import time

# Lua script for atomic sliding window rate limit
_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
    return 0
end
redis.call('ZADD', key, now, now .. '-' .. count)
redis.call('EXPIRE', key, math.ceil(window))
return 1
"""


class RateLimiter:
    """Redis sliding window rate limiter with fail-open fallback."""

    def __init__(self, max_per_minute: int = 60, window_seconds: int = 60) -> None:
        self.limit = max_per_minute
        self.window = window_seconds
        self._redis = None
        self._available: bool | None = None
        self._lua_sha: str | None = None
        # In-memory fallback
        self._fallback: dict[str, list[float]] = {}

    @property
    def redis(self):
        if self._redis is None and self._available is not False:
            try:
                import redis

                from enterprise_agentic_rag.config.settings import get_settings
                s = get_settings()
                pool = redis.ConnectionPool.from_url(
                    s.redis.connection_url, max_connections=5,
                    socket_keepalive=True, health_check_interval=30,
                )
                self._redis = redis.Redis(connection_pool=pool)
                self._lua_sha = self._redis.script_load(_LUA_SLIDING_WINDOW)
                self._available = True
            except Exception:
                self._available = False
        return self._redis

    def is_allowed(self, key: str = "global") -> bool:
        """Check if request is allowed. Returns True if allowed, False if rate limited."""
        r = self.redis
        if r and self._lua_sha:
            try:
                result = r.evalsha(self._lua_sha, 1, f"ratelimit:{key}", time.time(), self.window, self.limit)
                return result == 1
            except Exception:
                pass

        from enterprise_agentic_rag.config.settings import get_settings
        if not get_settings().runtime.fail_open_rate_limiter:
            return False

        # Development fallback only: per-process in-memory limiter.
        return self._fallback_check(key)

    def _fallback_check(self, key: str) -> bool:
        now = time.time()
        fk = f"fb:{key}"
        timestamps = self._fallback.get(fk, [])
        timestamps = [t for t in timestamps if now - t < self.window]
        if len(timestamps) >= self.limit:
            self._fallback[fk] = timestamps
            return False
        timestamps.append(now)
        self._fallback[fk] = timestamps
        return True


# Singleton
_limiter: RateLimiter | None = None

def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
