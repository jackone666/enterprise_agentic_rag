"""Retrieval trace — structured JSON logging for Graph-Augmented Hybrid RAG.

Captures the complete retrieval pipeline trace for every request:
- query analysis & routing
- per-source retrieval results (keyword, vector, graph)
- fusion & reranking
- degradation events
- timing breakdown
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTrace:
    """Complete trace of a single retrieval execution."""

    trace_id: str = ""
    query: str = ""
    query_analysis: dict[str, Any] = field(default_factory=dict)

    # Routing
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    enabled_retrievers: list[str] = field(default_factory=list)

    # Counts
    keyword_hit_count: int = 0
    vector_hit_count: int = 0
    graph_hit_count: int = 0
    external_hit_count: int = 0
    merged_count: int = 0
    reranked_count: int = 0
    graph_paths_count: int = 0

    # Timing (ms)
    keyword_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    graph_latency_ms: float = 0.0
    external_search_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Degradation
    degraded_from: str = ""
    degraded_to: str = ""
    graph_failed: bool = False

    # Routing reason
    route_reason: str = ""

    # Errors
    errors: list[str] = field(default_factory=list)

    # Fusion
    fusion_method: str = "rrf"
    fusion_weights: dict[str, float] = field(default_factory=dict)

    # Expanded query (graph_first mode)
    original_query: str = ""
    expanded_query: str = ""
    expansion_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON logging."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "query_analysis": self.query_analysis,
            "retrieval_plan": self.retrieval_plan,
            "mode": self.mode,
            "enabled_retrievers": self.enabled_retrievers,
            "keyword_hit_count": self.keyword_hit_count,
            "vector_hit_count": self.vector_hit_count,
            "graph_hit_count": self.graph_hit_count,
            "external_hit_count": self.external_hit_count,
            "merged_count": self.merged_count,
            "reranked_count": self.reranked_count,
            "graph_paths_count": self.graph_paths_count,
            "keyword_latency_ms": self.keyword_latency_ms,
            "vector_latency_ms": self.vector_latency_ms,
            "graph_latency_ms": self.graph_latency_ms,
            "external_search_latency_ms": self.external_search_latency_ms,
            "fusion_latency_ms": self.fusion_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "degraded_from": self.degraded_from,
            "degraded_to": self.degraded_to,
            "graph_failed": self.graph_failed,
            "route_reason": self.route_reason,
            "errors": self.errors,
            "fusion_method": self.fusion_method,
            "fusion_weights": self.fusion_weights,
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "expansion_terms": self.expansion_terms,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    def log_summary(self) -> None:
        """Log a human-readable summary of the trace."""
        lines = [
            f"╔══ Retrieval Trace: {self.trace_id} ══╗",
            f"║ Query: {self.query[:80]}",
            f"║ Mode: {self.mode} | Plan: {self.retrieval_plan.get('reason', 'N/A')}",
            f"║ Retrievers: {self.enabled_retrievers}",
        ]

        if self.degraded_from:
            lines.append(f"║ ⚠ Degraded: {self.degraded_from} → {self.degraded_to}")
        if self.graph_failed:
            lines.append(f"║ ⚠ Graph failed: graph weight zeroed, continuing with keyword+vector")

        lines.extend([
            f"║ Hits — KW:{self.keyword_hit_count} VEC:{self.vector_hit_count} GRAPH:{self.graph_hit_count} EXT:{self.external_hit_count}",
            f"║ Counts — Merged:{self.merged_count} Reranked:{self.reranked_count} Paths:{self.graph_paths_count}",
        ])

        if self.expanded_query:
            lines.append(f"║ Expanded: +{len(self.expansion_terms)} terms")

        lines.extend([
            f"║ Timing — KW:{self.keyword_latency_ms:.1f}ms VEC:{self.vector_latency_ms:.1f}ms "
            f"GRAPH:{self.graph_latency_ms:.1f}ms FUSION:{self.fusion_latency_ms:.1f}ms "
            f"TOTAL:{self.total_latency_ms:.1f}ms",
            f"║ Fusion: {self.fusion_method} | Weights: {self.fusion_weights}",
        ])

        if self.errors:
            lines.append(f"║ Errors: {'; '.join(self.errors[:3])}")

        lines.append("╚" + "═" * 48 + "╝")
        logger.info("\n".join(lines))


class RetrievalTracer:
    """Factory for RetrievalTrace objects with timing helpers."""

    def __init__(self) -> None:
        self._timers: dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed ms."""
        if name in self._timers:
            elapsed = (time.time() - self._timers[name]) * 1000
            del self._timers[name]
            return round(elapsed, 2)
        return 0.0

    def create_trace(
        self,
        trace_id: str = "",
        query: str = "",
        query_analysis: dict[str, Any] | None = None,
    ) -> RetrievalTrace:
        """Create a new RetrievalTrace with defaults."""
        return RetrievalTrace(
            trace_id=trace_id,
            query=query,
            query_analysis=query_analysis or {},
        )

    def populate_from_plan(
        self,
        trace: RetrievalTrace,
        plan: Any,  # RetrievalPlan
    ) -> None:
        """Populate trace fields from a RetrievalPlan."""
        trace.mode = plan.mode
        trace.enabled_retrievers = list(plan.enabled_retrievers)
        trace.route_reason = plan.reason
        trace.retrieval_plan = {
            "mode": plan.mode,
            "enabled_retrievers": plan.enabled_retrievers,
            "top_k": plan.top_k,
            "weights": plan.weights,
            "graph_depth": plan.graph_depth,
            "reason": plan.reason,
            "need_query_expansion": getattr(plan, "need_query_expansion", False),
            "degraded_from": getattr(plan, "degraded_from", ""),
            "degraded_to": getattr(plan, "degraded_to", ""),
        }
        trace.fusion_weights = dict(plan.weights)


# ===========================================================================
# Singleton
# ===========================================================================

_tracer: RetrievalTracer | None = None


def get_retrieval_tracer() -> RetrievalTracer:
    global _tracer
    if _tracer is None:
        _tracer = RetrievalTracer()
    return _tracer
