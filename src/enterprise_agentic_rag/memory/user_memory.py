"""User memory — PostgreSQL users table with in-memory fallback.

Reads user profiles from PostgreSQL.
Falls back to pre-defined mock profiles when PostgreSQL is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# In-memory fallback profiles (used when PostgreSQL is unavailable)
_MOCK_PROFILES: dict[str, dict[str, Any]] = {
    "u001": {
        "user_id": "u001", "name": "张三", "role": "admin",
        "department": "平台工程部", "email": "zhangsan@company.com",
        "permissions": ["read", "write", "admin", "knowledge_search", "ticket_manage"],
        "recent_tickets": ["TKT-001", "TKT-003"],
        "preferred_language": "zh-CN",
    },
    "u002": {
        "user_id": "u002", "name": "李四", "role": "developer",
        "department": "产品研发部", "email": "lisi@company.com",
        "permissions": ["read", "knowledge_search"],
        "recent_tickets": ["TKT-002"],
        "preferred_language": "zh-CN",
    },
    "u003": {
        "user_id": "u003", "name": "王五", "role": "basic",
        "department": "市场部", "email": "wangwu@company.com",
        "permissions": ["read", "knowledge_search"],
        "recent_tickets": [],
        "preferred_language": "zh-CN",
    },
}

_DEFAULT_PROFILE: dict[str, Any] = {
    "role": "basic", "department": "未知",
    "permissions": ["read", "knowledge_search"],
    "recent_tickets": [], "preferred_language": "zh-CN",
}


class UserMemory:
    """User profile store — PostgreSQL primary, in-memory fallback."""

    def __init__(self) -> None:
        self._repo = None
        self._pg_available: bool | None = None

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
    # Public API
    # ------------------------------------------------------------------
    def get_profile(self, user_id: str) -> dict[str, Any]:
        """Read user profile from PostgreSQL, falling back to mock."""
        pg_profile = self._get_pg_profile(user_id)
        if pg_profile:
            return pg_profile
        # Fallback to mock
        if user_id in _MOCK_PROFILES:
            return dict(_MOCK_PROFILES[user_id])
        return {"user_id": user_id, "name": f"用户{user_id}", **_DEFAULT_PROFILE}

    def get_recent_tickets(self, user_id: str) -> list[str]:
        return self.get_profile(user_id).get("recent_tickets", [])

    def get_context_string(self, user_id: str) -> str:
        p = self.get_profile(user_id)
        tickets = ", ".join(p.get("recent_tickets", [])) or "无"
        return (
            f"用户: {p.get('name', user_id)} | "
            f"角色: {p.get('role')} | "
            f"部门: {p.get('department')} | "
            f"最近工单: {tickets}"
        )

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    def _get_pg_profile(self, user_id: str) -> dict[str, Any] | None:
        repo = self.repo
        if repo is None:
            return None
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(repo.get_user(user_id))
        except Exception:
            pass
        return None
