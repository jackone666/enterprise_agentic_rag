"""Graph builder — wires the 16 nodes into a StateGraph.

This module is the only place that knows about the *shape* of the
graph (which node follows which, which edges are conditional). The
individual node implementations live under ``graph/nodes/`` and are
imported through ``graph/nodes/__init__.py``.

Routing strategy:
  - All worker nodes feed back to ``master_agent`` so the master picks
    the next slave based on the latest state.
  - Terminal nodes (finalize / human_fallback / final_refusal) feed
    into ``save_memory`` and then END.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from enterprise_agentic_rag.graph.dependencies import tracer
from enterprise_agentic_rag.graph.nodes import (
    build_context,
    call_tools_node,
    check_permission,
    deep_intent_recognition_node,
    execute_code_node,
    final_refusal_node,
    finalize_answer_node,
    generate_answer_node,
    generate_code_node,
    human_fallback_node,
    load_memory,
    master_agent_node,
    retrieve_knowledge,
    rewrite_query,
    save_memory,
    verify_answer_node,
)
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def after_permission(state: AgentState) -> Literal["deep_intent_recognition", "final_refusal"]:
    """Permission denied → polite refusal. Otherwise → continue."""
    perms = state.get("permissions", [])
    if "knowledge_search" not in perms:
        return "final_refusal"
    return "deep_intent_recognition"


def after_deep_intent(state: AgentState) -> Literal["retrieve_knowledge", "master_agent"]:
    """Happy path: intent → retrieval directly. Exception: route through master_agent."""
    deep_intent = state.get("deep_intent", {})
    needs_clarification = deep_intent.get("needs_clarification", False)
    confidence = deep_intent.get("confidence", 0.5)

    # Exception: needs clarification with low confidence → human fallback via master
    if needs_clarification and confidence < 0.2:
        return "master_agent"

    # Exception: requires tools (error_diagnosis) → master routes to call_tools
    primary = deep_intent.get("primary_intent", "")
    if primary == "error_diagnosis":
        return "master_agent"

    # Happy path: direct to retrieval
    return "retrieve_knowledge"


def after_retrieval(state: AgentState) -> Literal["build_context", "master_agent"]:
    """Happy path: retrieval → context build directly. Exception: route through master."""
    docs = state.get("retrieved_docs", [])
    fallback_reason = state.get("fallback_reason", "")

    # Happy path: usable docs found
    if docs and any(d.get("score", 0) > 0 for d in docs):
        return "build_context"

    # Exception: low score with retry exhausted → human fallback via master
    if fallback_reason == "low_retrieval_score":
        return "master_agent"

    # Happy path: continue to context build even with empty docs (code request)
    return "build_context"


def after_context(state: AgentState) -> Literal["generate_answer", "generate_code"]:
    """Context ready: direct to answer generation unless code required."""
    from enterprise_agentic_rag.agents.master_agent import MasterAgent

    if MasterAgent._requires_code(state) and not state.get("code_snippet", ""):
        return "generate_code"
    return "generate_answer"


def after_answer(state: AgentState) -> Literal["verify_answer", "finalize_answer"]:
    """Draft answer ready → direct to verification."""
    return "verify_answer"


def after_verification(state: AgentState) -> Literal["finalize_answer", "human_fallback"]:
    """Verification complete → finalize. Verification failed → human_fallback."""
    if state.get("verified", False):
        return "finalize_answer"
    # Verification failed → human fallback (direct, not through master)
    return "human_fallback"


def after_master(
    state: AgentState,
) -> Literal[
    "call_tools",
    "retrieve_knowledge",
    "rewrite_query",
    "build_context",
    "generate_code",
    "execute_code",
    "generate_answer",
    "verify_answer",
    "finalize_answer",
    "human_fallback",
]:
    """Route to the next slave agent selected by MasterAgent."""
    return state.get("master_next", "retrieve_knowledge")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_workflow() -> StateGraph:
    """Construct and compile the LangGraph state machine.

    Returns a compiled graph that can be invoked with ``ainvoke`` /
    ``astream`` from the FastAPI layer.
    """
    t = tracer  # trigger lazy init
    builder = StateGraph(AgentState)  # type: ignore[arg-type]

    # Nodes — each wrapped with tracing instrumentation
    builder.add_node("load_memory", t.traced_node("load_memory", load_memory))
    builder.add_node("check_permission", t.traced_node("check_permission", check_permission))
    builder.add_node("master_agent", t.traced_node("master_agent", master_agent_node))
    builder.add_node("deep_intent_recognition", t.traced_node("deep_intent_recognition", deep_intent_recognition_node))
    builder.add_node("retrieve_knowledge", t.traced_node("retrieve_knowledge", retrieve_knowledge))
    builder.add_node("rewrite_query", t.traced_node("rewrite_query", rewrite_query))
    builder.add_node("call_tools", t.traced_node("call_tools", call_tools_node))
    builder.add_node("build_context", t.traced_node("build_context", build_context))
    builder.add_node("generate_answer", t.traced_node("generate_answer", generate_answer_node))
    builder.add_node("verify_answer", t.traced_node("verify_answer", verify_answer_node))
    builder.add_node("finalize_answer", t.traced_node("finalize_answer", finalize_answer_node))
    builder.add_node("human_fallback", t.traced_node("human_fallback", human_fallback_node))
    builder.add_node("final_refusal", t.traced_node("final_refusal", final_refusal_node))
    builder.add_node("save_memory", t.traced_node("save_memory", save_memory))
    builder.add_node("generate_code", t.traced_node("generate_code", generate_code_node))
    builder.add_node("execute_code", t.traced_node("execute_code", execute_code_node))

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    # Entry
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "check_permission")

    # Permission gate
    builder.add_conditional_edges(
        "check_permission",
        after_permission,
        {
            "deep_intent_recognition": "deep_intent_recognition",
            "final_refusal": "final_refusal",
        },
    )

    # Happy path: intent → retrieval (bypass master)
    builder.add_conditional_edges(
        "deep_intent_recognition",
        after_deep_intent,
        {
            "retrieve_knowledge": "retrieve_knowledge",
            "master_agent": "master_agent",
        },
    )

    # Happy path: retrieval → context build (bypass master)
    builder.add_conditional_edges(
        "retrieve_knowledge",
        after_retrieval,
        {
            "build_context": "build_context",
            "master_agent": "master_agent",
        },
    )

    # Happy path: context → answer generation (bypass master)
    builder.add_conditional_edges(
        "build_context",
        after_context,
        {
            "generate_answer": "generate_answer",
            "generate_code": "generate_code",
        },
    )

    # Happy path: answer → verification (bypass master)
    builder.add_conditional_edges(
        "generate_answer",
        after_answer,
        {
            "verify_answer": "verify_answer",
            "finalize_answer": "finalize_answer",
        },
    )

    # Happy path: verification → finalize (bypass master); exception → human_fallback
    builder.add_conditional_edges(
        "verify_answer",
        after_verification,
        {
            "finalize_answer": "finalize_answer",
            "human_fallback": "human_fallback",
        },
    )

    # Exception paths: call_tools, rewrite_query, generate_code, execute_code,
    # human_fallback → route through master_agent
    for worker in (
        "call_tools",
        "rewrite_query",
        "generate_code",
        "execute_code",
    ):
        builder.add_edge(worker, "master_agent")
    # human_fallback is terminal (no master routing after escalation)
    builder.add_edge("human_fallback", "save_memory")

    # Master → selected slave / terminal path
    builder.add_conditional_edges(
        "master_agent",
        after_master,
        {
            "call_tools": "call_tools",
            "retrieve_knowledge": "retrieve_knowledge",
            "rewrite_query": "rewrite_query",
            "build_context": "build_context",
            "generate_code": "generate_code",
            "execute_code": "execute_code",
            "generate_answer": "generate_answer",
            "verify_answer": "verify_answer",
            "finalize_answer": "finalize_answer",
            "human_fallback": "human_fallback",
        },
    )

    # Terminal nodes → save_memory → END
    for terminal in ("finalize_answer", "human_fallback", "final_refusal"):
        builder.add_edge(terminal, "save_memory")
    builder.add_edge("save_memory", END)

    graph = builder.compile()
    graph.name = "EnterpriseAgenticRAG"
    return graph
