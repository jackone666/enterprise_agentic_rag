"""Permission gate and refusal nodes.

The permission check runs immediately after memory load — a denied
user skips the entire retrieval/generation pipeline and is sent to
``final_refusal`` instead.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.graph.dependencies import recovery
from enterprise_agentic_rag.graph.state import AgentState
from enterprise_agentic_rag.tools.permission_tool import check_permission_sync


async def check_permission(state: AgentState) -> dict[str, Any]:
    user_id = state.get("user_id", "")
    info = check_permission_sync(user_id)
    perms = info["permissions"]

    # If permission denied, record fallback reason
    if "knowledge_search" not in perms:
        fb = recovery.evaluate_failure(dict(state), fallback_type="permission_denied")
        return {**fb, "user_role": info["role"], "permissions": perms}

    return {"user_role": info["role"], "permissions": perms, "error": ""}


async def final_refusal_node(state: AgentState) -> dict[str, Any]:
    """Polite refusal when the user lacks required permissions."""
    return {
        "final_answer": (
            "抱歉，您当前的账号权限不足以访问知识库系统。\n\n"
            "如需获取权限，请联系您的部门管理员或提交权限申请工单。\n"
            "我们为此带来的不便深表歉意。"
        ),
        "need_human": False,
        "recoverable": False,
    }
