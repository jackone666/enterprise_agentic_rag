"""OpenTelemetry integration — distributed tracing for production observability.

Replaces/upgrades the JSONL-based EventLogger with OTLP-exported spans
that can be visualized in Jaeger, Tempo, or any OTLP-compatible backend.

Features:
- Auto-instrumentation for FastAPI, httpx, SQLAlchemy
- Manual span creation for LangGraph nodes
- Graceful fallback when OTel collector is unavailable
- Trace ID propagation across services

Reference:
    TECHNICAL_DEEP_DIVE.md §40.4 — "接入 OpenTelemetry"
    Expected impact: trace 查询时间下降, 跨服务链路追踪
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_OTEL_ENABLED = os.getenv("OTEL_ENABLED", "0").lower() in ("1", "true", "yes", "on")
_OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
_OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "enterprise-agentic-rag")

# Lazy imports to avoid hard dependency
_tracer_provider = None
_tracer = None
_initialized = False


def is_available() -> bool:
    """Check if OpenTelemetry SDK packages are installed."""
    try:
        import opentelemetry  # noqa: F401
        return True
    except ImportError:
        return False


def initialize() -> bool:
    """Initialize OpenTelemetry SDK with OTLP exporter.

    Returns True if initialization succeeded, False otherwise.
    """
    global _tracer_provider, _tracer, _initialized

    if _initialized:
        return _tracer is not None

    _initialized = True

    if not _OTEL_ENABLED:
        logger.info("OpenTelemetry is disabled (OTEL_ENABLED=0)")
        return False

    if not is_available():
        logger.info("OpenTelemetry SDK not installed — tracing disabled")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: F401
        from opentelemetry.instrumentation.httpx import HTTPXInstrumentor  # noqa: F401
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(attributes={SERVICE_NAME: _OTEL_SERVICE_NAME})

        exporter = OTLPSpanExporter(endpoint=_OTEL_ENDPOINT, insecure=True)

        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_tracer_provider)

        _tracer = trace.get_tracer(__name__)

        logger.info(
            "OpenTelemetry initialized: service=%s endpoint=%s",
            _OTEL_SERVICE_NAME, _OTEL_ENDPOINT,
        )
        return True

    except Exception as exc:
        logger.warning("OpenTelemetry initialization failed: %s — tracing disabled", exc)
        return False


def get_tracer():
    """Get the OpenTelemetry tracer instance.

    Returns None if OpenTelemetry is not available.
    """
    global _tracer
    if _tracer is None and _OTEL_ENABLED:
        initialize()
    return _tracer


def instrument_app(app) -> None:
    """Auto-instrument a FastAPI application with OpenTelemetry.

    Call this once after app creation:
        instrument_app(app)

    This automatically creates spans for all HTTP requests.
    """
    if not _OTEL_ENABLED:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI auto-instrumented with OpenTelemetry")
    except Exception as exc:
        logger.warning("FastAPI instrumentation failed: %s", exc)

    try:
        from opentelemetry.instrumentation.httpx import HTTPXInstrumentor
        HTTPXInstrumentor().instrument()
    except Exception:
        pass


@contextmanager
def traced_span(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for manual span creation.

    Usage:
        with traced_span("retrieve_knowledge", {"query": query}):
            results = _do_retrieval(query)

    Falls back gracefully when OTel is unavailable.
    """
    otel_tracer = get_tracer()

    if otel_tracer is None:
        # No-op context manager
        yield None
        return

    with otel_tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value)[:256])
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(2, str(exc)[:256])  # ERROR
            raise


async def traced_async(name: str, coro, attributes: dict[str, Any] | None = None):
    """Trace an async operation with a span.

    Usage:
        result = await traced_async("vector_search", vector_search(query), {"query": query})
    """
    with traced_span(name, attributes) as span:
        try:
            result = await coro
            if span:
                span.set_status(0)  # OK
            return result
        except Exception as exc:
            if span:
                span.record_exception(exc)
            raise


def shutdown() -> None:
    """Gracefully shut down the OTel tracer provider."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry tracer provider shut down")
        _tracer_provider = None
