"""System status adapter — API → mock fallback.

Config:
    SYSTEM_STATUS_API_BASE_URL
    SYSTEM_STATUS_API_TOKEN
"""

from __future__ import annotations

import os
from typing import Any

from enterprise_agentic_rag.tools.adapters.http_client import ProductionHTTPClient


class SystemStatusAdapter:
    """Unified system status with mock fallback."""

    def __init__(self) -> None:
        self._base_url = os.getenv("SYSTEM_STATUS_API_BASE_URL", "")
        self._token = os.getenv("SYSTEM_STATUS_API_TOKEN", "")
        self._client: ProductionHTTPClient | None = None

    @property
    def client(self) -> ProductionHTTPClient | None:
        if self._client is None and self._base_url:
            self._client = ProductionHTTPClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
            )
        return self._client

    async def get_system_status(self, trace_id: str = "") -> dict[str, Any]:
        c = self.client
        if c is not None:
            r = await c.get("/status", trace_id=trace_id)
            if r.success:
                return {"success": True, "data": r.data, "error": "", "source": "api"}

        # Mock fallback
        mock_data = {
            "services": {
                "api_gateway": "healthy",
                "auth_service": "healthy",
                "database": "healthy",
                "storage": "degraded",
            },
            "overall": "degraded",
        }
        return {"success": True, "data": mock_data, "error": "", "source": "mock"}

    async def get_error_code_detail(self, error_code: str, trace_id: str = "") -> dict[str, Any]:
        c = self.client
        if c is not None:
            r = await c.get(f"/errors/{error_code}", trace_id=trace_id)
            if r.success:
                return {"success": True, "data": r.data, "error": "", "source": "api"}

        # Mock fallback
        codes = {
            "AUTH_401": {"code": "AUTH_401", "severity": "high", "description": "认证失败", "fix": "检查 API Key"},
            "RATE_429": {"code": "RATE_429", "severity": "medium", "description": "速率限制", "fix": "降低请求频率"},
            "SYS_500": {"code": "SYS_500", "severity": "critical", "description": "内部服务错误", "fix": "联系运维"},
        }
        data = codes.get(error_code.upper(), {"code": error_code, "description": "未知错误码"})
        return {"success": True, "data": data, "error": "", "source": "mock"}
