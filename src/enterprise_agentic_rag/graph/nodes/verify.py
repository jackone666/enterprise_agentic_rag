"""Verifier node — checks that the draft answer is grounded in evidence.

Calls the verifier agent, records a verification event with the
tracer, and on failure triggers a fallback (regenerate or escalate).
"""

from __future__ import annotations

import time
from typing import Any

from enterprise_agentic_rag.agents.verifier_agent import verify_answer_async
from enterprise_agentic_rag.graph.dependencies import recovery, tracer
from enterprise_agentic_rag.graph.state import AgentState


async def verify_answer_node(state: AgentState) -> dict[str, Any]:
    draft = state.get("draft_answer", "")
    citations = state.get("citations", [])
    docs = state.get("retrieved_docs", [])

    t0 = time.time()
    verified, reason = await verify_answer_async(draft, citations, docs)
    latency_ms = (time.time() - t0) * 1000

    tracer.record_verification_event(
        dict(state),
        verified=verified,
        reason=reason,
        latency_ms=latency_ms,
    )

    if not verified:
        fb = recovery.evaluate_failure(dict(state), fallback_type="answer_not_grounded")
        return {
            **fb,
            "verified": verified,
            "verification_reason": reason,
            "need_human": True,
            "last_worker": "verifier_agent",
            "last_agent_step": "verify_answer",
        }

    return {
        "verified": verified,
        "verification_reason": reason,
        "need_human": False,
        "last_worker": "verifier_agent",
        "last_agent_step": "verify_answer",
    }
