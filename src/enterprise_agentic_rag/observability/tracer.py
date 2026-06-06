"""Tracer — orchestrates trace ID generation and per-event recording.

Integrates with EventLogger (JSONL) and MetricsCollector (in-memory stats).

Usage:
    tracer = get_tracer()
    trace_id = tracer.new_trace()

    # Wrap a node
    wrapped = tracer.traced_node("retrieve_knowledge", my_node_fn)

    # Record tool / retrieval / verification events inside nodes
    tracer.record_tool_event(state, tool_name, success, latency_ms)
    tracer.record_retrieval_event(state, query, num_docs, latency_ms)
    tracer.record_verification_event(state, verified, reason, latency_ms)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from enterprise_agentic_rag.observability.event_schema import (
    EventType,
    NodeEvent,
    RetrievalEvent,
    ToolEvent,
    VerificationEvent,
    event_to_dict,
    make_summary,
)
from enterprise_agentic_rag.observability.logger import EventLogger
from enterprise_agentic_rag.observability.metrics import get_metrics_collector


class Tracer:
    """Unified tracer — creates trace IDs and wraps nodes with observability."""

    def __init__(self, logger: EventLogger | None = None) -> None:
        self.logger = logger or EventLogger()
        self.metrics = get_metrics_collector()

    # ------------------------------------------------------------------
    # Trace ID
    # ------------------------------------------------------------------
    @staticmethod
    def new_trace() -> str:
        """Generate a unique trace ID (short UUID)."""
        return uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # Node wrapper — used by LangGraph nodes
    # ------------------------------------------------------------------
    def traced_node(
        self,
        node_name: str,
        node_fn: Any,
    ) -> Any:
        """Wrap an async node function with start/end event recording.

        Returns a new async function that:
        1. Records a ``node_start`` event
        2. Calls the original node function
        3. Records a ``node_end`` event with timing and success/error
        4. Appends both events to ``node_events`` in the returned state

        Node functions must have signature: ``async def fn(state: AgentState) -> dict``
        """

        async def _wrapped(state: Any) -> dict[str, Any]:
            trace_id = state.get("trace_id", "")
            session_id = state.get("session_id", "")
            user_id = state.get("user_id", "")
            query = state.get("query", "")
            t0 = time.time()

            # --- Start event ---
            start_evt = NodeEvent(
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                event_type=EventType.NODE_START,
                node_name=node_name,
                input_summary=make_summary(query, 100),
            )

            # --- Execute ---
            success = True
            error_msg = ""
            try:
                output = await node_fn(state)
            except Exception as exc:
                success = False
                error_msg = str(exc)
                output = {"error": error_msg}

            latency_ms = (time.time() - t0) * 1000

            # --- End event ---
            output_summary = make_summary(
                self._summarise_output(output), 120
            )
            end_evt = NodeEvent(
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                event_type=EventType.NODE_END,
                node_name=node_name,
                input_summary=make_summary(query, 100),
                output_summary=output_summary,
                latency_ms=round(latency_ms, 2),
                success=success,
                error=error_msg,
            )

            # --- Persist ---
            self.logger.write_event(event_to_dict(start_evt))
            self.logger.write_event(event_to_dict(end_evt))

            # --- Append to state ---
            node_events = list(state.get("node_events", []))
            node_events.append(event_to_dict(start_evt))
            node_events.append(event_to_dict(end_evt))

            if not success:
                output["error"] = output.get("error", error_msg)

            output["node_events"] = node_events
            return output

        # Preserve metadata for debugging
        _wrapped.__name__ = f"traced_{node_name}"  # type: ignore[attr-defined]
        return _wrapped

    # ------------------------------------------------------------------
    # Tool event
    # ------------------------------------------------------------------
    def record_tool_event(
        self,
        state: dict[str, Any],
        tool_name: str,
        params: dict[str, Any] | None = None,
        output: str = "",
        latency_ms: float = 0.0,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Record a single tool execution event."""
        evt = ToolEvent(
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            user_id=state.get("user_id", ""),
            event_type=EventType.TOOL_CALL,
            tool_name=tool_name,
            input_params=params or {},
            output_summary=make_summary(output, 120),
            latency_ms=round(latency_ms, 2),
            success=success,
            error=error,
        )
        self.logger.write_event(event_to_dict(evt))
        self.metrics.record_tool_call(success=success)

        # Append to state
        tool_events = list(state.get("tool_events", []))
        tool_events.append(event_to_dict(evt))
        # Note: state dict mutation happens in the node that calls this

    # ------------------------------------------------------------------
    # Retrieval event
    # ------------------------------------------------------------------
    def record_retrieval_event(
        self,
        state: dict[str, Any],
        query: str = "",
        num_docs: int = 0,
        top_score: float = 0.0,
        latency_ms: float = 0.0,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Record a retrieval operation event."""
        evt = RetrievalEvent(
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            user_id=state.get("user_id", ""),
            event_type=EventType.RETRIEVAL,
            query=make_summary(query, 120),
            num_docs_retrieved=num_docs,
            top_score=round(top_score, 4),
            latency_ms=round(latency_ms, 2),
            success=success,
            error=error,
        )
        self.logger.write_event(event_to_dict(evt))
        self.metrics.record_retrieval(num_docs=num_docs)

    # ------------------------------------------------------------------
    # Verification event
    # ------------------------------------------------------------------
    def record_verification_event(
        self,
        state: dict[str, Any],
        verified: bool = False,
        reason: str = "",
        latency_ms: float = 0.0,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Record an answer verification event."""
        evt = VerificationEvent(
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            user_id=state.get("user_id", ""),
            event_type=EventType.VERIFICATION,
            verified=verified,
            verification_reason=make_summary(reason, 200),
            latency_ms=round(latency_ms, 2),
            success=success,
            error=error,
        )
        self.logger.write_event(event_to_dict(evt))
        self.metrics.record_verification(verified=verified)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _summarise_output(output: dict[str, Any]) -> str:
        """Build a compact summary of what a node returned."""
        if not output:
            return "{}"
        keys = list(output.keys())
        # Filter out verbose / large fields
        skip_keys = {
            "node_events", "retrieval_events", "verification_events",
            "tool_events", "retrieved_docs", "reranked_docs",
            "chat_history", "structured_context", "prompt_context",
            "human_fallback_payload", "retry_history",
        }
        summary_keys = [k for k in keys if k not in skip_keys]
        parts = []
        for k in summary_keys[:8]:  # first 8 interesting keys
            v = output[k]
            if isinstance(v, str):
                parts.append(f"{k}={make_summary(v, 40)}")
            elif isinstance(v, bool):
                parts.append(f"{k}={v}")
            elif isinstance(v, (int, float)):
                parts.append(f"{k}={v}")
            elif isinstance(v, list):
                parts.append(f"{k}=[{len(v)} items]")
            elif isinstance(v, dict):
                parts.append(f"{k}={{...}}")
        return ", ".join(parts) if parts else "ok"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
