"""Permission checker — PostgreSQL users table with mock fallback."""

from __future__ import annotations

from typing import Any


# Mock fallback (used only when PostgreSQL is unreachable)
PERMISSION_MAP: dict[str, dict[str, Any]] = {
    "u001": {"role": "admin", "permissions": ["read", "write", "admin", "knowledge_search", "ticket_manage"]},
}


def check_permission_sync(user_id: str) -> dict[str, Any]:
    """Check user permissions — PostgreSQL first, mock fallback."""
    from enterprise_agentic_rag.config.settings import get_settings
    settings = get_settings()
    try:
        profile = _get_pg_profile(user_id)
        if profile:
            return {"role": profile.get("role", "basic"), "permissions": profile.get("permissions", ["read", "knowledge_search"])}
    except Exception:
        if not settings.runtime.allow_in_memory_fallback:
            return {"role": "unknown", "permissions": []}

    # Mock fallback
    if not settings.runtime.allow_in_memory_fallback:
        return {"role": "unknown", "permissions": []}
    if user_id in PERMISSION_MAP:
        return PERMISSION_MAP[user_id]
    return {"role": "basic", "permissions": ["read", "knowledge_search"]}


def _get_pg_profile(user_id: str) -> dict[str, Any] | None:
    import asyncio
    try:
        from enterprise_agentic_rag.storage.database import get_db_manager
        dbm = get_db_manager()
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return None
        if not loop.run_until_complete(dbm.check_connection()):
            return None
        from enterprise_agentic_rag.storage.repositories import get_user as pg_get_user
        async def _get():
            async with dbm.session() as sess:
                return await pg_get_user(sess, user_id)
        return loop.run_until_complete(_get())
    except Exception:
        return None
