"""User profile adapter — API → PostgreSQL → mock fallback chain.

Config:
    USER_PROFILE_API_BASE_URL
    USER_PROFILE_API_TOKEN
"""

from __future__ import annotations

import os
from typing import Any

from enterprise_agentic_rag.tools.adapters.http_client import ProductionHTTPClient


class UserProfileAdapter:
    """Unified user profile lookup with cascading fallback."""

    def __init__(self) -> None:
        self._base_url = os.getenv("USER_PROFILE_API_BASE_URL", "")
        self._token = os.getenv("USER_PROFILE_API_TOKEN", "")
        self._client: ProductionHTTPClient | None = None

    @property
    def client(self) -> ProductionHTTPClient | None:
        if self._client is None and self._base_url:
            self._client = ProductionHTTPClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
            )
        return self._client

    async def get_user_profile(self, user_id: str, trace_id: str = "") -> dict[str, Any]:
        # 1. Try real API
        c = self.client
        if c is not None:
            r = await c.get(f"/users/{user_id}", trace_id=trace_id)
            if r.success:
                return {"success": True, "data": r.data, "error": "", "source": "api"}
            # API failed — fall through

        # 2. Try PostgreSQL
        try:
            from enterprise_agentic_rag.memory.user_memory import UserMemory
            um = UserMemory()
            profile = um.get_profile(user_id)
            if profile.get("user_id"):
                return {"success": True, "data": profile, "error": "", "source": "postgresql"}
        except Exception:
            pass

        # 3. Mock fallback
        from enterprise_agentic_rag.tools.user_profile_tool import _MOCK_PROFILES
        if user_id in _MOCK_PROFILES:
            return {"success": True, "data": _MOCK_PROFILES[user_id], "error": "", "source": "mock"}
        return {
            "success": True,
            "data": {"user_id": user_id, "name": f"用户{user_id}", "role": "basic"},
            "error": "",
            "source": "mock_default",
        }
