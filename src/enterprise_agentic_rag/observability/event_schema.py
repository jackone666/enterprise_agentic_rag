"""Standardised event schemas for the observability system.

Every event has: trace_id, session_id, user_id, event_type, timestamp.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------
class EventType:
    """Well-known event type labels."""

    NODE_START = "node_start"
    NODE_END = "node_end"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    VERIFICATION = "verification"
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------
@dataclass
class BaseEvent:
    """Common fields for every observability event."""

    trace_id: str
    session_id: str
    user_id: str
    event_type: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Node event
# ---------------------------------------------------------------------------
@dataclass
class NodeEvent(BaseEvent):
    """Recorded when a LangGraph node starts or ends."""

    node_name: str = ""
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Tool event
# ---------------------------------------------------------------------------
@dataclass
class ToolEvent(BaseEvent):
    """Recorded for each individual tool execution."""

    tool_name: str = ""
    input_params: dict[str, Any] = field(default_factory=dict)
    output_summary: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Retrieval event
# ---------------------------------------------------------------------------
@dataclass
class RetrievalEvent(BaseEvent):
    """Recorded for every retrieval operation."""

    query: str = ""
    num_docs_retrieved: int = 0
    top_score: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Verification event
# ---------------------------------------------------------------------------
@dataclass
class VerificationEvent(BaseEvent):
    """Recorded for every answer verification."""

    verified: bool = False
    verification_reason: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Request-level event
# ---------------------------------------------------------------------------
@dataclass
class RequestEvent(BaseEvent):
    """Bookend events for the full /chat request lifecycle."""

    query: str = ""
    intent: str = ""
    final_answer: str = ""
    total_latency_ms: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_summary(text: str, max_chars: int = 120) -> str:
    """Truncate *text* to a safe summary length (no sensitive data exposure)."""
    if not text:
        return ""
    clean = text.replace("\n", " ").replace("\r", " ")
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def event_to_dict(event: BaseEvent) -> dict[str, Any]:
    """Convert any event dataclass to a plain dict for JSON serialisation."""
    d = asdict(event)
    # Make timestamps readable
    if "timestamp" in d:
        from datetime import datetime, timezone
        ts = d["timestamp"]
        d["timestamp_iso"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return d
