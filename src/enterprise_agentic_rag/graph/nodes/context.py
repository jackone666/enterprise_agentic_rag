"""Context builder node.

Combines retrieved docs, tool results, chat history, and user profile
into a token-budgeted context window that downstream generation nodes
can consume directly.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.graph.dependencies import context_manager
from enterprise_agentic_rag.graph.state import AgentState


async def build_context(state: AgentState) -> dict[str, Any]:
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    session_summary = state.get("session_summary", "")
    retrieved_docs = state.get("retrieved_docs", [])
    tool_results = state.get("tool_results", [])
    user_profile = state.get("user_profile", {})

    if retrieved_docs:
        retrieved_docs = context_manager.enrich_docs_with_graph_paths(retrieved_docs)

    structured = context_manager.build_context(
        query=query,
        chat_history=chat_history,
        session_summary=session_summary,
        retrieved_docs=retrieved_docs,
        tool_results=tool_results,
        user_profile=user_profile,
    )

    return {
        "structured_context": structured,
        "context_window": structured["context_window"],
        "prompt_context": {
            "router_prompt": structured["router_prompt"],
            "knowledge_prompt": structured["knowledge_prompt"],
            "verifier_prompt": structured["verifier_prompt"],
        },
        "token_budget": {
            "max": structured["token_budget_max"],
            "used": structured["token_budget_used"],
            "remaining": (
                structured["token_budget_max"] - structured["token_budget_used"]
            ),
        },
        "last_worker": "context_manager",
        "last_agent_step": "build_context",
    }
