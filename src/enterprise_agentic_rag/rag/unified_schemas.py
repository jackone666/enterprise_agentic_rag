"""Unified input/output schemas for retrieval tools.

Shared between keyword_search, vector_search, and graph_search tools.
This is the single source of truth for the tool I/O contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedToolInput:
    """Unified input schema for all retrieval tools."""

    query: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    intent: str = ""
    scenario: str = ""
    entities: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedToolOutput:
    """Unified output schema for all retrieval tools."""

    tool_name: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
