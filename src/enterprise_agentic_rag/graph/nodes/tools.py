"""Tool invocation node.

Executes the tool agent's tool calls, captures errors, and writes the
audit trail back to the state. Tool failures never crash the workflow.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.agents.master_agent import MasterAgent
from enterprise_agentic_rag.agents.tool_agent import call_tools
from enterprise_agentic_rag.graph.dependencies import master_agent, tracer
from enterprise_agentic_rag.graph.state import AgentState


async def call_tools_node(state: AgentState) -> dict[str, Any]:
    """Execute relevant tools based on intent and query content."""
    query = state.get("query", "")
    intent = master_agent.tool_intent(dict(state))
    user_id = state.get("user_id", "")
    perms = state.get("permissions", [])

    results, tool_calls, tool_errors, pending = await call_tools(
        query=query,
        intent=intent,
        user_id=user_id,
        user_permissions=perms,
    )

    serialised_results = [
        {
            "tool_name": r.tool_name,
            "success": r.success,
            "output": r.output,
            "error": r.error,
            "latency_ms": round(r.latency_ms, 2),
        }
        for r in results
    ]

    audit_logs = [
        {
            "tool_name": tc["tool_name"],
            "params": tc["params"],
            "user_id": user_id,
            "success": results[i].success if i < len(results) else False,
        }
        for i, tc in enumerate(tool_calls)
    ]

    for i, r in enumerate(results):
        params = tool_calls[i]["params"] if i < len(tool_calls) else {}
        tracer.record_tool_event(
            dict(state),
            tool_name=r.tool_name,
            params=params,
            output=str(r.output)[:200],
            latency_ms=r.latency_ms,
            success=r.success,
            error=r.error or "",
        )

    return {
        "tool_calls": tool_calls,
        "tool_results": serialised_results,
        "tool_errors": tool_errors,
        "pending_tool_confirmations": pending,
        "tool_audit_logs": audit_logs,
        "last_worker": "tool_agent",
        "last_agent_step": "call_tools",
    }
