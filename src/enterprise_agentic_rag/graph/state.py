"""Unified state for the enterprise agentic RAG workflow.

Uses TypedDict for LangGraph compatibility.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Global state flowing through the LangGraph workflow.

    v3.2 — trimmed from 72 to ~30 fields (decorative cleanup).
    """

    # ── Input ──
    query: str
    user_id: str
    session_id: str

    # ── Permission ──
    user_role: str
    permissions: list[str]

    # ── Routing ──
    intent: str
    complexity: str

    # ── RAG ──
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]

    # ── Tool system ──
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_errors: list[str]
    tool_audit_logs: list[dict[str, Any]]

    # ── Generation ──
    draft_answer: str

    # ── Verification ──
    verified: bool
    verification_reason: str

    # ── Output ──
    final_answer: str
    citations: list[dict[str, Any]]
    need_human: bool

    # ── Memory tier ──
    chat_history: list[dict[str, Any]]
    session_summary: str
    user_profile: dict[str, Any]
    memory_context: dict[str, Any]
    memory_ckpt_id: str

    # ── Context management ──
    structured_context: dict[str, Any]
    context_window: str
    prompt_context: dict[str, Any]
    token_budget: dict[str, Any]

    # ── Fallback & recovery ──
    fallback_reason: str
    recovery_action: str
    retry_count: dict[str, int]
    retry_history: list[dict[str, Any]]
    human_fallback_payload: dict[str, Any]
    recoverable: bool

    # ── Observability ──
    trace_id: str
    tool_events: list[dict[str, Any]]
    retrieval_events: list[dict[str, Any]]
    verification_events: list[dict[str, Any]]

    # ── Graph RAG fields ──
    query_analysis: dict[str, Any]
    retrieval_plan: dict[str, Any]
    retrieval_mode: str
    retrieval_errors: list[str]

    # ── Code generation & execution ──
    code_snippet: str
    code_language: str
    code_execution_result: dict[str, Any]
    code_verified: bool
    code_retry_count: int

    # ── Error handling ──
    error: str

    # ── Agent routing fields ──
    deep_intent: dict[str, Any]
    deep_intent_confidence: float
    last_worker: str
    last_agent_step: str
    master_next: str
    master_reason: str
    master_decisions: list[dict[str, Any]]
    graph_step_count: int
    routing_path: str

    # ── Messages (LangGraph add_messages reducer) ──
    messages: Annotated[list[Any], add_messages]
