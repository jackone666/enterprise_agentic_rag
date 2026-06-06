"""Ticket system adapter — production API with mock fallback.

Config:
    TICKET_API_BASE_URL  — e.g. https://ticket.internal.company.com/api/v1
    TICKET_API_TOKEN     — Bearer token

Fallback chain: Real API → Mock ticket_tool
"""

from __future__ import annotations

import os
from typing import Any

from enterprise_agentic_rag.tools.adapters.http_client import ProductionHTTPClient


class TicketAdapter:
    """Unified ticket operations — real API or mock."""

    def __init__(self) -> None:
        self._base_url = os.getenv("TICKET_API_BASE_URL", "")
        self._token = os.getenv("TICKET_API_TOKEN", "")
        self._client: ProductionHTTPClient | None = None
        self._mock = None

    @property
    def client(self) -> ProductionHTTPClient | None:
        if self._client is None and self._base_url:
            self._client = ProductionHTTPClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
            )
        return self._client

    @property
    def mock(self):
        if self._mock is None:
            from enterprise_agentic_rag.tools.ticket_tool import _TICKET_STORE
            # Return a simple lookup interface
            class _Mock:
                @staticmethod
                def get(ticket_id: str) -> dict | None:
                    return _TICKET_STORE.get(ticket_id)
                @staticmethod
                def create(user_id: str, issue: str) -> dict:
                    tid = f"TKT-{len(_TICKET_STORE) + 1:03d}"
                    _TICKET_STORE[tid] = {
                        "id": tid, "user_id": user_id, "issue": issue, "status": "open"
                    }
                    return _TICKET_STORE[tid]
            self._mock = _Mock()
        return self._mock

    async def query_ticket(self, ticket_id: str, trace_id: str = "") -> dict[str, Any]:
        c = self.client
        if c is None:
            # Fallback to mock
            result = self.mock.get(ticket_id)
            return {
                "success": result is not None,
                "data": result,
                "error": "" if result else f"工单 {ticket_id} 不存在",
                "source": "mock",
            }
        r = await c.get(f"/tickets/{ticket_id}", trace_id=trace_id)
        return {
            "success": r.success,
            "data": r.data,
            "error": r.error,
            "latency_ms": r.latency_ms,
            "source": "api",
        }

    async def create_ticket(self, user_id: str, issue: str, trace_id: str = "") -> dict[str, Any]:
        c = self.client
        if c is None:
            result = self.mock.create(user_id, issue)
            return {
                "success": True,
                "data": result,
                "error": "",
                "source": "mock",
            }
        r = await c.post("/tickets", json_body={"user_id": user_id, "issue": issue}, trace_id=trace_id)
        return {
            "success": r.success,
            "data": r.data,
            "error": r.error,
            "latency_ms": r.latency_ms,
            "source": "api",
        }
