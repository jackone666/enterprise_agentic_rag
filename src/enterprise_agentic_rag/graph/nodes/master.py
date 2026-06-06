"""Master agent node — owns routing decisions.

Reads the current state (last_agent_step, tool_errors, retry_count, …)
and asks ``MasterAgent.decide()`` for the next node. The decision path
(LLM vs rule-based) is recorded as ``routing_path`` in the state so
operators can see how often the LLM path actually runs in production.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.config.settings import get_settings
from enterprise_agentic_rag.graph.dependencies import master_agent, recovery
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)


async def master_agent_node(state: AgentState) -> dict[str, Any]:
    """Run the master agent and persist its routing decision."""
    settings = get_settings()
    step_count = int(state.get("graph_step_count", 0)) + 1
    if step_count > settings.app.max_graph_steps:
        history = list(state.get("master_decisions", []))
        history.append({
            "from_worker": state.get("last_worker", ""),
            "from_step": state.get("last_agent_step", ""),
            "next_node": "human_fallback",
            "reason": f"graph step budget exceeded ({step_count}>{settings.app.max_graph_steps})",
        })
        fb = recovery.evaluate_failure(dict(state), fallback_type="step_budget_exceeded")
        return {
            **fb,
            "graph_step_count": step_count,
            "master_next": "human_fallback",
            "master_reason": "graph step budget exceeded",
            "master_decisions": history,
            "routing_path": "rule",
        }

    decision = await master_agent.decide(dict(state))
    history = list(state.get("master_decisions", []))
    history.append({
        "from_worker": state.get("last_worker", ""),
        "from_step": state.get("last_agent_step", ""),
        **decision.to_dict(),
    })
    return {
        "graph_step_count": step_count,
        "master_next": decision.next_node,
        "master_reason": decision.reason,
        "master_decisions": history,
        "routing_path": decision.routing_path,
    }
