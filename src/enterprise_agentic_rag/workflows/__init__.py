"""Workflows — reusable retrieval execution patterns.

Each workflow implements a specific retrieval mode:
- hybrid_rag_workflow.py — keyword + vector hybrid (default fallback)
- graph_first_workflow.py — graph → expand → keyword + vector
- error_first_workflow.py — error KB → faq/ticket/official/vector
- code_generation_workflow.py — sample → api_ref → official_doc → code_review
"""

from enterprise_agentic_rag.workflows.code_generation_workflow import CodeGenerationWorkflow
from enterprise_agentic_rag.workflows.error_first_workflow import ErrorFirstWorkflow
from enterprise_agentic_rag.workflows.graph_first_workflow import GraphFirstWorkflow
from enterprise_agentic_rag.workflows.hybrid_rag_workflow import HybridRAGWorkflow

__all__ = [
    "HybridRAGWorkflow",
    "GraphFirstWorkflow",
    "ErrorFirstWorkflow",
    "CodeGenerationWorkflow",
]
