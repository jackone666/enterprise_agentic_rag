"""Master-slave agent definitions.

The project ships **5 Agents** (1 master + 4 workers):

* :class:`MasterAgent` — Central router (LLM-first, rules-fallback).
* Knowledge generation — :func:`generate_answer_async` (LLM-first, template fallback).
* Tool orchestration — :func:`call_tools` (registry + tier + circuit breaker).
* Code generation — :func:`generate_code` (AST symbol injection + LLM/template).
* Answer verification — :func:`verify_answer_async` (Claim-level, LLM, rules).

Workers are exported as async functions so call sites can do
``from enterprise_agentic_rag.agents import call_tools`` instead of
reaching into individual modules.
"""

from __future__ import annotations

from enterprise_agentic_rag.agents.knowledge_agent import generate_answer_async
from enterprise_agentic_rag.agents.master_agent import MasterAgent, MasterDecision
from enterprise_agentic_rag.agents.tool_agent import call_tools
from enterprise_agentic_rag.agents.verifier_agent import verify_answer_async
from enterprise_agentic_rag.prompts.code_prompts import generate_code

__all__ = [
    # Master
    "MasterAgent",
    "MasterDecision",
    # Workers (4)
    "generate_answer_async",  # KnowledgeAgent
    "call_tools",             # ToolAgent
    "generate_code",          # CodeAgent
    "verify_answer_async",    # VerifierAgent
]
