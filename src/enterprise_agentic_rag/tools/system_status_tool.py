"""Mock system status tool.

Simulates querying a live system for operational status and error-code details.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# Mock system data
# ---------------------------------------------------------------------------
_MOCK_SERVICES: dict[str, dict[str, Any]] = {
    "api_gateway": {"status": "healthy", "uptime_hours": 720, "latency_p99_ms": 45},
    "auth_service": {"status": "healthy", "uptime_hours": 720, "latency_p99_ms": 12},
    "ticket_service": {"status": "degraded", "uptime_hours": 680, "latency_p99_ms": 320},
    "knowledge_base": {"status": "healthy", "uptime_hours": 720, "latency_p99_ms": 5},
}

_MOCK_ERROR_CODES: dict[str, dict[str, Any]] = {
    "AUTH_401": {
        "code": "AUTH_401",
        "title": "认证失败",
        "description": "API Key 无效或已过期",
        "common_causes": [
            "API Key 已过期",
            "请求头中未正确传递 Authorization",
            "Key 无访问目标资源的权限",
            "IP 地址不在白名单中",
        ],
        "suggested_actions": [
            "登录管理控制台重新生成 API Key",
            "检查请求头格式: Authorization: Bearer <key>",
            "确认 Key 已授权访问目标资源",
            "联系管理员将 IP 加入白名单",
        ],
    },
    "RATE_429": {
        "code": "RATE_429",
        "title": "请求频率超限",
        "description": "短时间内请求次数超过限制",
        "common_causes": [
            "客户端未实现退避重试",
            "并发请求数过高",
        ],
        "suggested_actions": [
            "实现指数退避 (exponential backoff)",
            "降低并发请求数",
            "联系管理员提升配额",
        ],
    },
    "SYS_500": {
        "code": "SYS_500",
        "title": "服务器内部错误",
        "description": "后端服务出现未预期的异常",
        "common_causes": [
            "数据库连接池耗尽",
            "下游服务超时",
            "代码未捕获的异常",
        ],
        "suggested_actions": [
            "查看服务日志定位根因",
            "检查数据库连接状态",
            "如持续出现，请提紧急工单",
        ],
    },
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class GetSystemStatusTool(BaseTool):
    """Return the overall system health status."""

    name: str = "get_system_status"
    description: str = "查询系统各服务的运行状态，返回健康状态、运行时长、延迟等指标。"
    is_sensitive: bool = False
    required_permissions: list[str] = ["read"]
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {"services": "dict — 各服务状态"}

    async def execute(self, **_: Any) -> ToolResult:
        """Return mock system status."""
        services = {
            name: dict(info) for name, info in _MOCK_SERVICES.items()
        }
        overall = (
            "degraded"
            if any(s["status"] == "degraded" for s in services.values())
            else "healthy"
        )
        return ToolResult(
            success=True,
            output={"overall": overall, "services": services},
        )


class GetErrorCodeDetailTool(BaseTool):
    """Return detailed information for a given error code."""

    name: str = "get_error_code_detail"
    description: str = "查询指定错误码的详细信息，包括原因分析和建议操作。"
    is_sensitive: bool = False
    required_permissions: list[str] = ["read"]
    input_schema: dict[str, Any] = {"error_code": "str — 错误码，例如 AUTH_401"}
    output_schema: dict[str, Any] = {"error_detail": "dict — 错误码详情"}

    async def execute(self, error_code: str = "", **_: Any) -> ToolResult:
        """Look up an error code."""
        if not error_code:
            return ToolResult(success=False, error="缺少参数: error_code")

        detail = _MOCK_ERROR_CODES.get(error_code.upper())
        if detail is None:
            return ToolResult(
                success=True,
                output={"found": False, "message": f"未找到错误码 {error_code} 的详细信息"},
            )

        return ToolResult(success=True, output={"found": True, "error_detail": dict(detail)})
