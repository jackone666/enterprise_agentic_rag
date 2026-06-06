"""Workflows — reusable retrieval execution patterns.

All 4 mode-specific workflows (hybrid, graph-first, error-first,
code-generation) are merged into a single :class:`BaseRAGWorkflow` class
that dispatches by ``mode`` parameter.

Available modes (maps to ``RetrievalMode`` in deep_intent/schema):
    hybrid_only  — keyword + vector parallel (default / fallback)
    graph_first  — graph search → expanded query → keyword + vector
    parallel     — multi-source parallel → keyword + vector
"""

from enterprise_agentic_rag.workflows.base_rag_workflow import BaseRAGWorkflow

__all__ = [
    "BaseRAGWorkflow",
]
