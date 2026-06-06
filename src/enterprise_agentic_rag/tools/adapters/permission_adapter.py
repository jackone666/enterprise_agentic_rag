"""Permission adapter — API → PostgreSQL → mock fallback chain.

Config:
    PERMISSION_API_BASE_URL
    PERMISSION_API_TOKEN
"""

from __future__ import annotations

import os
from typing import Any

from enterprise_agentic_rag.tools.adapters.http_client import ProductionHTTPClient


class PermissionAdapter:
    """Unified permission check with cascading fallback."""

    def __init__(self) -> None:
        self._base_url = os.getenv("PERMISSION_API_BASE_URL", "")
        self._token = os.getenv("PERMISSION_API_TOKEN", "")
        self._client: ProductionHTTPClient | None = None

    @property
    def client(self) -> ProductionHTTPClient | None:
        if self._client is None and self._base_url:
            self._client = ProductionHTTPClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
            )
        return self._client

    async def check_user_permission(
        self, user_id: str, resource: str, action: str, trace_id: str = ""
    ) -> dict[str, Any]:
        """Check if user has permission for resource+action.

        Returns {'allowed': bool, 'permissions': [...], 'source': str}.
        """
        # 1. Try real API
        c = self.client
        if c is not None:
            r = await c.get(
                "/check",
                params={"user_id": user_id, "resource": resource, "action": action},
                trace_id=trace_id,
            )
            if r.success:
                return {"allowed": True, "permissions": r.data if isinstance(r.data, list) else [], "source": "api"}
            # API failed — fall through

        # 2. Try PostgreSQL users.permissions
        try:
            from enterprise_agentic_rag.memory.user_memory import UserMemory
            um = UserMemory()
            profile = um.get_profile(user_id)
            perms = profile.get("permissions", [])
            allowed = action in perms or "admin" in perms or "read" in perms
            return {"allowed": allowed, "permissions": perms, "source": "postgresql"}
        except Exception:
            pass

        # 3. Mock fallback
        from enterprise_agentic_rag.tools.permission_tool import check_permission_sync
        info = check_permission_sync(user_id)
        return {
            "allowed": "knowledge_search" in info.get("permissions", []),
            "permissions": info.get("permissions", []),
            "source": "mock",
        }
