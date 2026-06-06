"""Enterprise Agentic RAG workflow — public entry point.

This module is intentionally a thin re-export layer. The actual graph
construction lives in :mod:`enterprise_agentic_rag.graph.builder` and
the 16 node implementations live under :mod:`enterprise_agentic_rag.graph.nodes`.
Splitting the file (from a single 1080-line module) is the v3.1
refactor; see ``technical_deep_dive/10-数据结构技术选型与项目指标.md``.

Pipeline (high level):

    START → load_memory → check_permission
      ├─ permission denied → final_refusal → save_memory → END
      └─ permission ok → deep_intent_recognition → master_agent
           ├─ low-confidence clarification → human_fallback → save_memory → END
           └─ actionable intent → [call_tools|retrieve_knowledge|rewrite_query|
                                    build_context|generate_code|execute_code|
                                    generate_answer|verify_answer|
                                    finalize_answer|human_fallback]
                (each worker reports back to master_agent)

MasterAgent owns routing decisions; specialised agents and services
execute work and return state patches.
"""

from __future__ import annotations

# Public surface — kept stable so app/main.py and tests/ can keep doing
# `from enterprise_agentic_rag.graph.workflow import build_workflow`
from enterprise_agentic_rag.graph.builder import after_master, after_permission, build_workflow

__all__ = ["build_workflow", "after_master", "after_permission"]
