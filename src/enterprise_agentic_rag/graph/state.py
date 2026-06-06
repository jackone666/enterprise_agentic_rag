"""Unified state for the enterprise agentic RAG workflow.

Uses TypedDict for LangGraph compatibility.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Global state flowing through the LangGraph workflow.

    Fields:
        query: The raw user question.
        user_id: Caller identifier.
        session_id: Session identifier for checkpointing.
        user_role: Authorization role (admin, developer, basic).
        permissions: Granted permission list.
        intent: Deep intent primary intent label.
        complexity: Estimated question complexity.
        retrieved_docs: Documents fetched from the knowledge base.
        reranked_docs: Documents after relevance re-ranking.
        draft_answer: Raw answer before verification.
        verified: Whether the answer passed verification.
        verification_reason: Explanation when verification fails.
        final_answer: Polished answer returned to the user.
        citations: Source citations attached to the answer.
        need_human: Whether escalation to a human is required.
        tool_calls: Log of tool invocations (name + params).
        tool_results: Serialised results from executed tools.
        tool_errors: Error messages from failed tool executions.
        pending_tool_confirmations: Sensitive tools awaiting approval.
        tool_audit_logs: Full audit trail of tool executions.

        # Memory tier fields
        chat_history: Recent conversation turns (list of {role, content, intent}).
        session_summary: Compressed summary string from SummaryMemory.
        user_profile: User profile dict from UserMemory.
        memory_context: Aggregated memory metadata dict.
        memory_ckpt_id: Latest checkpoint identifier.

        # Context management fields
        structured_context: Full structured context dict from ContextManager.
        context_window: Combined system/user context string.
        prompt_context: Per-agent assembled prompts dict.
        token_budget: TokenBudget allocation object (or dict).

        # Fallback & recovery fields
        fallback_reason: Standardised failure type label.
        recovery_action: Action the recovery manager recommends.
        retry_count: Per-node retry attempt counters.
        retry_history: Structured log of every retry attempt.
        human_fallback_payload: Full context dump for human escalation.
        recoverable: Whether the current failure can be recovered.

        # Observability fields
        trace_id: Unique identifier for this request trace.
        node_events: Serialised NodeEvent dicts from each graph node.
        tool_events: Serialised ToolEvent dicts from tool executions.
        retrieval_events: Serialised RetrievalEvent dicts.
        verification_events: Serialised VerificationEvent dicts.
        metrics_snapshot: Latest metrics snapshot dict.

        error: Last error message (if any).
        messages: Accumulated conversation messages.
    """

    # Input
    query: str
    user_id: str
    session_id: str

    # Permission
    user_role: str
    permissions: list[str]

    # Routing
    intent: str
    complexity: str

    # RAG
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]

    # Tool system
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_errors: list[str]
    pending_tool_confirmations: list[dict[str, Any]]
    tool_audit_logs: list[dict[str, Any]]
    tool_call_mode: str          # "mock" | "production"
    external_call_trace: list[dict[str, Any]]  # external API call trace
    pending_confirmation_id: str  # for sensitive tool approval flow

    # Generation
    draft_answer: str

    # Verification
    verified: bool
    verification_reason: str

    # Output
    final_answer: str
    citations: list[dict[str, Any]]
    need_human: bool

    # Memory tier
    chat_history: list[dict[str, Any]]
    session_summary: str
    user_profile: dict[str, Any]
    memory_context: dict[str, Any]
    memory_ckpt_id: str

    # Context management
    structured_context: dict[str, Any]
    context_window: str
    prompt_context: dict[str, Any]
    token_budget: dict[str, Any]

    # Fallback & recovery
    fallback_reason: str
    recovery_action: str
    retry_count: dict[str, int]
    retry_history: list[dict[str, Any]]
    human_fallback_payload: dict[str, Any]
    recoverable: bool

    # Observability
    trace_id: str
    node_events: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    retrieval_events: list[dict[str, Any]]
    verification_events: list[dict[str, Any]]
    metrics_snapshot: dict[str, Any]

    # Graph RAG fields
    query_analysis: dict[str, Any]
    retrieval_plan: dict[str, Any]
    retrieval_mode: str
    graph_paths: list[dict[str, Any]]
    graph_candidates: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    degraded_from: str
    degraded_to: str
    retrieval_errors: list[str]

    # Code generation & execution
    code_snippet: str                   # Generated code snippet
    code_language: str                  # Language of the generated code
    code_execution_result: dict[str, Any]  # {stdout, stderr, exit_code, execution_time_ms}
    code_verified: bool                 # Whether code execution succeeded
    code_retry_count: int               # Code fix retry counter
    code_retry_attempted: bool          # Whether code has been retried at least once

    # External search
    external_search_results: list[dict[str, Any]]  # Results from external knowledge sources
    external_search_used: bool          # Whether external search was triggered

    # Error handling
    error: str

    # ── Agent routing fields ──
    deep_intent: dict[str, Any]           # DeepIntentResult as dict
    deep_intent_confidence: float         # Router confidence score
    retrieval_plan_config: dict[str, Any] # RetrievalPlan as dict
    last_worker: str                      # Last worker/service that updated state
    last_agent_step: str                  # Last internal workflow step from that worker/service
    master_next: str                      # Next node selected by MasterAgent
    master_reason: str                    # Human-readable routing reason
    master_decisions: list[dict[str, Any]] # Master routing history
    graph_step_count: int                 # Master routing step counter / request budget
    routing_path: str                     # Which path produced the master decision: "llm" / "rule" / "rule_direct"

    # Messages (for LangGraph add_messages reducer)
    messages: Annotated[list[Any], add_messages]
