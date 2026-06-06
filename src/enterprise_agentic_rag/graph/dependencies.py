"""Shared singleton instances for the LangGraph workflow.

These objects are constructed once at import time and shared across all
graph nodes. Centralising them here avoids hidden module-level state in
individual node files and makes multi-tenant configuration a single
edit point (P1 follow-up).

The tracer is exposed as a module-level lazy proxy because it depends
on the metrics collector that itself lazy-initialises its sink.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.agents.master_agent import MasterAgent
from enterprise_agentic_rag.context.context_manager import ContextManager
from enterprise_agentic_rag.memory.memory_manager import MemoryManager
from enterprise_agentic_rag.rag.retriever import Retriever
from enterprise_agentic_rag.recovery.recovery_manager import RecoveryManager

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
retriever = Retriever(chunk_size=500, top_k=5)
memory = MemoryManager()
context_manager = ContextManager(max_tokens=4096)
recovery = RecoveryManager()
master_agent = MasterAgent(recovery)


class _LazyTracer:
    """Lazy proxy for the tracer so importing this module stays cheap."""

    def __init__(self) -> None:
        self._instance: Any = None

    def _ensure(self) -> Any:
        if self._instance is None:
            from enterprise_agentic_rag.observability.tracer import get_tracer

            self._instance = get_tracer()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure(), name)


tracer = _LazyTracer()
