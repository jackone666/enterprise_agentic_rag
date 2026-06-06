"""Mock user profile tool.

Returns a canned profile for known user IDs.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# In-memory mock user store
# ---------------------------------------------------------------------------
_MOCK_USERS: dict[str, dict[str, Any]] = {
    "u001": {
        "user_id": "u001",
        "name": "张三",
        "role": "admin",
        "department": "平台工程部",
        "email": "zhangsan@company.com",
        "permissions": ["read", "write", "admin", "knowledge_search", "ticket_manage"],
    },
    "u002": {
        "user_id": "u002",
        "name": "李四",
        "role": "developer",
        "department": "产品研发部",
        "email": "lisi@company.com",
        "permissions": ["read", "knowledge_search"],
    },
    "u003": {
        "user_id": "u003",
        "name": "王五",
        "role": "basic",
        "department": "市场部",
        "email": "wangwu@company.com",
        "permissions": ["read", "knowledge_search"],
    },
}


class GetUserProfileTool(BaseTool):
    """Fetch a user profile by ID."""

    name: str = "get_user_profile"
    description: str = "根据用户 ID 查询用户档案信息，包括姓名、角色、部门、权限等。"
    is_sensitive: bool = False
    required_permissions: list[str] = ["read"]
    input_schema: dict[str, Any] = {"user_id": "str — 用户唯一标识"}
    output_schema: dict[str, Any] = {"profile": "dict — 用户档案"}

    async def execute(self, user_id: str = "", **_: Any) -> ToolResult:
        """Look up user profile."""
        if not user_id:
            return ToolResult(success=False, error="缺少参数: user_id")

        profile = _MOCK_USERS.get(user_id)
        if profile is None:
            return ToolResult(
                success=True,
                output={"found": False, "message": f"用户 {user_id} 不存在"},
            )

        return ToolResult(success=True, output={"found": True, "profile": dict(profile)})
