"""Memory-related graph nodes: load + save.

These are the entry and exit nodes of the workflow:
- ``load_memory`` populates the state with chat history, session summary,
  and user profile before any routing decision is made.
- ``save_memory`` records final metrics and persists the QA turn to
  PostgreSQL (fire-and-forget) before the workflow ends.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.graph.dependencies import memory, tracer
from enterprise_agentic_rag.graph.persistence import persist_qa_log
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)


async def load_memory(state: AgentState) -> dict[str, Any]:
    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "anonymous")

    ctx = memory.load_memory_context(session_id, user_id, query=state.get("query", ""))
    return {
        "chat_history": ctx["chat_history"],
        "session_summary": ctx["session_summary"],
        "user_profile": ctx["user_profile"],
        "memory_context": ctx["memory_context"],
        "memory_ckpt_id": ctx["checkpoint_id"],
        # Initialise recovery fields (preserve pre-set values for tests)
        "retry_count": state.get("retry_count", {}),
        "retry_history": state.get("retry_history", []),
        "recoverable": True,
    }


async def save_memory(state: AgentState) -> dict[str, Any]:
    session_id = state.get("session_id", "default")

    # Record final request metrics
    intent = state.get("intent", "unknown")
    need_human = state.get("need_human", False)
    has_fallback = bool(state.get("fallback_reason", ""))

    tracer.metrics.record_request(
        session_id=session_id,
        intent=intent,
        latency_ms=0.0,
        success=not need_human,
        need_human=need_human,
        has_fallback=has_fallback,
    )

    cid = memory.save_memory_context(session_id, dict(state))
    persist_qa_log(dict(state))

    return {"memory_ckpt_id": cid}
