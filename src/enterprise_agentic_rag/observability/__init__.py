"""Observability system — structured tracing, logging, and metrics.

Components:
- EventSchema: Standardised event structures
- EventLogger: JSONL file writer (failure-safe)
- MetricsCollector: Cumulative request statistics
- Tracer: Trace ID generation + per-node/per-tool/per-retrieval recording
"""

from enterprise_agentic_rag.observability.metrics import MetricsCollector, get_metrics_collector
from enterprise_agentic_rag.observability.tracer import Tracer, get_tracer

__all__ = ["Tracer", "get_tracer", "MetricsCollector", "get_metrics_collector"]
