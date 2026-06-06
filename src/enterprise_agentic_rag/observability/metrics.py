"""Prometheus + JSON metrics — dual-format exposure.

- Prometheus: exposed at GET /prometheus_metrics
- JSON: exposed at GET /metrics (existing, unchanged)

All Prometheus metrics are named with `agent_` prefix.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, generate_latest


# ===================================================================
# Prometheus metrics
# ===================================================================
agent_requests_total = Counter(
    "agent_requests_total", "Total /chat requests", ["intent"]
)
agent_request_latency_seconds = Histogram(
    "agent_request_latency_seconds", "Request latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
agent_fallback_total = Counter(
    "agent_fallback_total", "Fallback events"
)
agent_human_fallback_total = Counter(
    "agent_human_fallback_total", "Human fallback events"
)
agent_tool_calls_total = Counter(
    "agent_tool_calls_total", "Tool executions", ["tool_name"]
)
agent_tool_failures_total = Counter(
    "agent_tool_failures_total", "Failed tool executions", ["tool_name"]
)
agent_retrieval_score = Histogram(
    "agent_retrieval_score", "Retrieval relevance score",
    buckets=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
)
agent_verification_pass_total = Counter(
    "agent_verification_pass_total", "Verification results", ["verified"]
)
agent_llm_calls_total = Counter(
    "agent_llm_calls_total", "LLM API calls", ["provider", "model"]
)
agent_llm_failures_total = Counter(
    "agent_llm_failures_total", "Failed LLM calls", ["provider", "model"]
)
agent_uptime_seconds = Gauge("agent_uptime_seconds", "Service uptime")


# ===================================================================
# In-memory metrics (backward-compatible JSON snapshot)
# ===================================================================
class MetricsCollector:
    """Thread-safe cumulative metrics (JSON /metrics endpoint)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.total_success: int = 0
        self.total_latency_ms: float = 0.0
        self.intent_counts: dict[str, int] = {}
        self.retrieval_attempts: int = 0
        self.retrieval_hits: int = 0
        self.verification_attempts: int = 0
        self.verification_passes: int = 0
        self.tool_attempts: int = 0
        self.tool_successes: int = 0
        self.fallback_count: int = 0
        self.human_fallback_count: int = 0
        self._first_request_at: float | None = None

    def record_request(self, intent: str = "", latency_ms: float = 0.0, success: bool = True,
                       need_human: bool = False, has_fallback: bool = False) -> None:
        with self._lock:
            now = time.time()
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if success:
                self.total_success += 1
            if self._first_request_at is None:
                self._first_request_at = now
            self.intent_counts[intent or "unknown"] = self.intent_counts.get(intent or "unknown", 0) + 1
            if has_fallback:
                self.fallback_count += 1
            if need_human:
                self.human_fallback_count += 1

        # Update Prometheus
        agent_requests_total.labels(intent=intent or "unknown").inc()
        agent_request_latency_seconds.observe(latency_ms / 1000)
        if has_fallback:
            agent_fallback_total.inc()
        if need_human:
            agent_human_fallback_total.inc()
        agent_uptime_seconds.set(time.time() - (self._first_request_at or now))

    def record_retrieval(self, num_docs: int = 0, top_score: float = 0.0) -> None:
        with self._lock:
            self.retrieval_attempts += 1
            if num_docs > 0:
                self.retrieval_hits += 1
        if top_score > 0:
            agent_retrieval_score.observe(top_score)

    def record_verification(self, verified: bool = False) -> None:
        with self._lock:
            self.verification_attempts += 1
            if verified:
                self.verification_passes += 1
        agent_verification_pass_total.labels(verified="true" if verified else "false").inc()

    def record_tool_call(self, success: bool = True, tool_name: str = "") -> None:
        with self._lock:
            self.tool_attempts += 1
            if success:
                self.tool_successes += 1
        agent_tool_calls_total.labels(tool_name=tool_name or "unknown").inc()
        if not success:
            agent_tool_failures_total.labels(tool_name=tool_name or "unknown").inc()

    def record_llm_call(self, provider: str = "", model: str = "", success: bool = True) -> None:
        agent_llm_calls_total.labels(provider=provider, model=model).inc()
        if not success:
            agent_llm_failures_total.labels(provider=provider, model=model).inc()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_success": self.total_success,
                "success_rate": round(self.total_success / max(self.total_requests, 1), 4),
                "avg_latency_ms": round(self.total_latency_ms / max(self.total_requests, 1), 2),
                "intent_distribution": dict(self.intent_counts),
                "retrieval_hit_rate": round(self.retrieval_hits / max(self.retrieval_attempts, 1), 4),
                "verification_pass_rate": round(self.verification_passes / max(self.verification_attempts, 1), 4),
                "tool_success_rate": round(self.tool_successes / max(self.tool_attempts, 1), 4),
                "fallback_rate": round(self.fallback_count / max(self.total_requests, 1), 4),
                "human_fallback_rate": round(self.human_fallback_count / max(self.total_requests, 1), 4),
                "uptime_seconds": round(time.time() - (self._first_request_at or time.time()), 1),
            }

    def reset(self) -> None:
        with self._lock:
            self.__init__()  # type: ignore[misc]


def get_metrics_collector() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


_metrics: MetricsCollector | None = None


def get_prometheus_metrics() -> bytes:
    """Return Prometheus text format metrics."""
    return generate_latest()
