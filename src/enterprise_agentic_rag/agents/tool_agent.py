"""Tool orchestration agent.

Decides which tool(s) to call based on intent and query content,
then executes them via :class:`ToolExecutor`.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.tools.base import ToolResult
from enterprise_agentic_rag.tools.code_execution_tool import get_code_execution_tool
from enterprise_agentic_rag.tools.executor import ToolExecutor
from enterprise_agentic_rag.tools.registry import ToolRegistry
from enterprise_agentic_rag.tools.system_status_tool import GetErrorCodeDetailTool, GetSystemStatusTool
from enterprise_agentic_rag.tools.ticket_tool import CreateTicketTool, QueryTicketTool
from enterprise_agentic_rag.tools.user_profile_tool import GetUserProfileTool

# ---------------------------------------------------------------------------
# Singleton registry — built once
# ---------------------------------------------------------------------------
_registry = ToolRegistry()
_registry.register_many([
    QueryTicketTool(),
    CreateTicketTool(),
    GetUserProfileTool(),
    GetSystemStatusTool(),
    GetErrorCodeDetailTool(),
    get_code_execution_tool(),
])

_executor = ToolExecutor(_registry)


# ---------------------------------------------------------------------------
# Intent → tool mapping
# ---------------------------------------------------------------------------

# Keywords that trigger specific tools
_TOOL_TRIGGERS: list[tuple[list[str], str]] = [
    (["系统状态", "服务状态", "健康检查", "运行状态"], "get_system_status"),
    (["错误码", "error code", "401", "429", "500", "AUTH"], "get_error_code_detail"),
    (["工单", "ticket", "Ticket", "TKT"], "query_ticket"),
    (["用户信息", "用户档案", "我的信息", "个人信息"], "get_user_profile"),
    (["创建工单", "提交工单", "新建工单"], "create_ticket"),
    (["执行代码", "运行代码", "验证代码", "执行示例", "运行示例"], "execute_code"),
]


def _select_tools(query: str, intent: str) -> list[tuple[str, dict[str, Any]]]:
    """Select tool calls based on query keywords and intent.

    Args:
        query: Raw user query.
        intent: Classified intent label.

    Returns:
        List of ``(tool_name, params)`` tuples.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    query_lower = query.lower()

    # Extract ticket_id if present (e.g. "TKT-001")
    import re
    ticket_match = re.search(r"TKT-\d{3}", query, re.IGNORECASE)

    # Extract error code if present (e.g. "AUTH_401")
    error_match = re.search(r"(AUTH_\d{3}|RATE_\d{3}|SYS_\d{3})", query, re.IGNORECASE)

    if intent == "troubleshooting":
        # Always check system status + error code detail for troubleshooting
        calls.append(("get_system_status", {}))
        if error_match:
            calls.append(("get_error_code_detail", {"error_code": error_match.group(1).upper()}))

    elif intent == "ticket_query":
        if ticket_match:
            calls.append(("query_ticket", {"ticket_id": ticket_match.group(0).upper()}))
        else:
            calls.append(("query_ticket", {"ticket_id": ""}))  # will return "缺少参数"

    # Keyword-based triggers (allow overlap with intent-based)
    seen = {c[0] for c in calls}
    for keywords, tool_name in _TOOL_TRIGGERS:
        if tool_name in seen:
            continue
        if any(kw.lower() in query_lower for kw in keywords):
            if tool_name == "get_error_code_detail" and error_match:
                calls.append((tool_name, {"error_code": error_match.group(1).upper()}))
            elif tool_name == "query_ticket" and ticket_match:
                calls.append((tool_name, {"ticket_id": ticket_match.group(0).upper()}))
            elif tool_name not in ("get_error_code_detail", "query_ticket", "create_ticket"):
                calls.append((tool_name, {}))
            elif tool_name == "create_ticket":
                # Don't auto-execute create_ticket — needs confirmation
                pass
            seen.add(tool_name)

    return calls


async def call_tools(
    query: str,
    intent: str,
    user_id: str,
    user_permissions: list[str],
) -> tuple[list[ToolResult], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Orchestrate tool execution for a user query.

    Args:
        query: Raw user query.
        intent: Classified intent label.
        user_id: Caller's user ID.
        user_permissions: Caller's permission set.

    Returns:
        Tuple of:
        - ``tool_results``: Results of executed tools.
        - ``tool_calls``: Audit log entries (name + params).
        - ``tool_errors``: Error strings for failed tools.
        - ``pending_tool_confirmations``: Sensitive tools that need approval.
    """
    selected = _select_tools(query, intent)

    results: list[ToolResult] = []
    tool_calls: list[dict[str, Any]] = []
    tool_errors: list[str] = []
    pending: list[dict[str, Any]] = []

    for tool_name, params in selected:
        # Inject user_id when relevant
        if tool_name in ("query_ticket", "get_user_profile"):
            params.setdefault("user_id", user_id)

        # --- Try safe execution first ---
        result = await _executor.execute(
            tool_name=tool_name,
            params=params,
            user_permissions=user_permissions,
            skip_confirmation=False,
        )

        tool_calls.append({"tool_name": tool_name, "params": params})
        results.append(result)

        if result.success:
            continue

        # --- Handle pending confirmation ---
        if "需要确认" in result.error:
            pending.append({
                "tool_name": tool_name,
                "params": params,
                "reason": result.error,
            })
            tool_errors.append(f"{tool_name}: 等待用户确认")
            continue

        # --- Handle denial / failure ---
        tool_errors.append(f"{tool_name}: {result.error}")

    return results, tool_calls, tool_errors, pending


def get_tool_registry() -> ToolRegistry:
    """Return the singleton tool registry."""
    return _registry
