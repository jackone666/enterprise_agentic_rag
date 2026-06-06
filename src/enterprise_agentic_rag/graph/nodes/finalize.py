"""Final-answer and human-escalation nodes.

``finalize_answer_node`` produces the user-facing string. If no draft
answer exists (e.g. code-only request) it builds one from the
code-snippet state. ``human_fallback_node`` packages the full context
so a human agent can pick up the request.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.graph.dependencies import recovery
from enterprise_agentic_rag.graph.state import AgentState


async def finalize_answer_node(state: AgentState) -> dict[str, Any]:
    tool_errors = state.get("tool_errors", [])
    draft = state.get("draft_answer", "")

    code_snippet = state.get("code_snippet", "")
    if code_snippet and not draft:
        code_lang = state.get("code_language", "typescript")
        code_verified = state.get("code_verified", False)
        exec_result = state.get("code_execution_result", {})

        parts = ["根据您的需求，生成以下代码示例：\n"]
        parts.append(f"```{code_lang}")
        parts.append(code_snippet)
        parts.append("```")

        if code_verified:
            parts.append("\n✅ 代码已在沙箱中成功执行")
        elif exec_result:
            err = exec_result.get("stderr", "执行失败")[:200]
            parts.append(f"\n⚠️ 代码执行验证未通过: {err}")
            parts.append("\n请根据实际环境调整代码后再使用。")
        else:
            parts.append("\n⚠️ 代码未经过执行验证，仅供参考。请在实际环境中测试后再使用。")

        draft = "\n".join(parts)

    if tool_errors:
        note = "\n\n---\n⚠️ 部分工具执行出现错误，以上回答可能不完整。"
        draft += note

    return {"final_answer": draft, "need_human": False}


async def human_fallback_node(state: AgentState) -> dict[str, Any]:
    """Build a detailed human escalation message with full context."""
    fb_reason = state.get("fallback_reason", "unknown")
    tool_errors = state.get("tool_errors", [])
    intent = state.get("intent", "unknown")
    reason = state.get("verification_reason", "")

    payload = recovery._build_human_payload(dict(state))

    parts = ["您的问题需要人工协助处理。\n"]
    parts.append(f"**失败原因**: {fb_reason}")
    parts.append(f"**意图**: {intent}")
    if reason:
        parts.append(f"**校验结果**: {reason}")
    if tool_errors:
        parts.append(f"**工具错误**: {', '.join(tool_errors)}")

    retry_history = state.get("retry_history", [])
    if retry_history:
        parts.append(f"**重试记录**: 共 {len(retry_history)} 次")
        for entry in retry_history[-3:]:
            parts.append(f"  - {entry.get('node')} (第{entry.get('attempt')}次): {entry.get('reason', '')}")

    parts.append("\n请稍候，客服人员将尽快与您联系。")

    return {
        "final_answer": "\n".join(parts),
        "need_human": True,
        "human_fallback_payload": payload,
    }
