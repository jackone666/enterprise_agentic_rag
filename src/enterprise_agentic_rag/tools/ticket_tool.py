"""Mock ticket management tools.

Provides:
- ``query_ticket`` — look up a ticket by ID.
- ``create_ticket`` — create a new support ticket (sensitive).
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# In-memory mock ticket store
# ---------------------------------------------------------------------------
_MOCK_TICKETS: dict[str, dict[str, Any]] = {
    "TKT-001": {
        "ticket_id": "TKT-001",
        "user_id": "u001",
        "title": "SDK 接入 AUTH_401 错误",
        "status": "open",
        "priority": "high",
        "created_at": "2026-05-28T10:30:00",
        "assigned_to": "engineer-zhang",
    },
    "TKT-002": {
        "ticket_id": "TKT-002",
        "user_id": "u002",
        "title": "API 限流问题",
        "status": "in_progress",
        "priority": "normal",
        "created_at": "2026-05-29T14:00:00",
        "assigned_to": "engineer-li",
    },
    "TKT-003": {
        "ticket_id": "TKT-003",
        "user_id": "u001",
        "title": "密码重置申请",
        "status": "closed",
        "priority": "low",
        "created_at": "2026-05-20T09:00:00",
        "assigned_to": "engineer-wang",
    },
}

_next_id = 4


def _next_ticket_id() -> str:
    global _next_id
    tid = f"TKT-{_next_id:03d}"
    _next_id += 1
    return tid


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class QueryTicketTool(BaseTool):
    """Look up a support ticket by its ID."""

    name: str = "query_ticket"
    description: str = "根据工单 ID 查询工单的详细信息，包括状态、优先级、处理人等。"
    is_sensitive: bool = False
    required_permissions: list[str] = ["read"]
    input_schema: dict[str, Any] = {"ticket_id": "str — 工单编号，例如 TKT-001"}
    output_schema: dict[str, Any] = {"ticket": "dict — 工单详情或 None"}

    async def execute(self, ticket_id: str = "", **_: Any) -> ToolResult:
        """Execute ticket lookup."""
        if not ticket_id:
            return ToolResult(success=False, error="缺少参数: ticket_id")

        ticket = _MOCK_TICKETS.get(ticket_id)
        if ticket is None:
            return ToolResult(
                success=True,
                output={"found": False, "message": f"工单 {ticket_id} 不存在"},
            )

        return ToolResult(success=True, output={"found": True, "ticket": dict(ticket)})


class CreateTicketTool(BaseTool):
    """Create a new support ticket (sensitive operation)."""

    name: str = "create_ticket"
    description: str = "为用户创建新的支持工单。需要 write 权限，且需要用户确认。"
    is_sensitive: bool = True
    tier: str = "sensitive"
    required_permissions: list[str] = ["write"]
    max_retries: int = 1
    input_schema: dict[str, Any] = {
        "user_id": "str — 用户 ID",
        "issue": "str — 问题描述",
    }
    output_schema: dict[str, Any] = {"ticket_id": "str", "status": "str"}

    async def execute(self, user_id: str = "", issue: str = "", **_: Any) -> ToolResult:
        """Create a new ticket in the mock store."""
        if not user_id or not issue:
            return ToolResult(success=False, error="缺少参数: user_id 和 issue 均为必填")

        tid = _next_ticket_id()
        ticket = {
            "ticket_id": tid,
            "user_id": user_id,
            "title": issue[:80],
            "status": "open",
            "priority": "normal",
            "created_at": "2026-06-01T12:00:00",
            "assigned_to": "auto-assign",
        }
        _MOCK_TICKETS[tid] = ticket

        return ToolResult(
            success=True,
            output={"ticket_id": tid, "status": "open", "message": f"工单 {tid} 已创建"},
        )
