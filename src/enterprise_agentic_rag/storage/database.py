"""Database connection manager — async SQLAlchemy engine + session factory.

Reads DATABASE_URL from environment.  Falls back to in-memory
when the database is unreachable.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from enterprise_agentic_rag.config.settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages the async SQLAlchemy engine and provides session factories."""

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        self._url = database_url or settings.postgres.async_url
        self._engine = create_async_engine(
            self._url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._available: bool | None = None  # None = not checked yet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield an async session (context-manager style)."""
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    async def check_connection(self) -> bool:
        """Verify the database is reachable.  Caches result."""
        if self._available is not None:
            return self._available
        parsed = urlparse(self._url)
        host = parsed.hostname
        port = parsed.port or 5432
        if host:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    pass
            except OSError:
                self._available = False
                if not get_settings().runtime.allow_in_memory_fallback:
                    raise RuntimeError("PostgreSQL unavailable and in-memory fallback is disabled")
                logger.warning("PostgreSQL unavailable — falling back to in-memory store")
                return self._available
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            self._available = True
        except Exception:
            self._available = False
            if not get_settings().runtime.allow_in_memory_fallback:
                raise RuntimeError("PostgreSQL unavailable and in-memory fallback is disabled")
            logger.warning("PostgreSQL unavailable — falling back to in-memory store")
        return self._available

    @property
    def available(self) -> bool:
        return self._available or False

    async def init_tables(self) -> None:
        """Create all tables defined in the models module."""
        from enterprise_agentic_rag.storage.models import Base

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
