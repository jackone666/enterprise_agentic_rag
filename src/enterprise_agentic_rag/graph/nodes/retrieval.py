"""Retrieval and query-rewrite nodes.

``retrieve_knowledge`` delegates to ``RetrievalAgent.run(state)``.
The 3-tier fallback (cache → workflow → fail) lives inside the agent.
``rewrite_query`` is kept here as it is still used for query改写 retries.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.agents.retrieval_agent import RetrievalAgent
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)

# Module-level singleton so node instances are reused across invocations
_retrieval_agent: RetrievalAgent | None = None


def _get_retrieval_agent() -> RetrievalAgent:
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = RetrievalAgent()
    return _retrieval_agent


async def retrieve_knowledge(state: AgentState) -> dict[str, Any]:
    """Retrieve knowledge — delegates to RetrievalAgent.

    The node signature is unchanged so graph/builder.py needs no updates.
    """
    agent = _get_retrieval_agent()
    return await agent.run(dict(state))


async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrite the query for a second retrieval attempt."""
    original = state.get("query", "")

    try:
        from enterprise_agentic_rag.graph.dependencies import recovery
        rewritten = recovery.rewrite_query(original)
        retry_updates = recovery.record_retry(
            dict(state),
            node_key="retrieve",
            reason=f"原始查询无结果，改写为: {rewritten}",
        )
    except Exception:
        rewritten = original
        retry_updates = {}

    return {
        **retry_updates,
        "query": rewritten,
        "fallback_reason": "",
        "recovery_action": "retry",
        "recoverable": True,
        "last_worker": "retrieval_service",
        "last_agent_step": "rewrite_query",
    }