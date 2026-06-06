"""Code generation + sandboxed execution nodes.

The code path is opt-in — only entered when the master agent decides
the request is a ``code_generation`` intent. ``execute_code`` runs the
snippet in a security-gated sandbox and increments ``code_retry_count``
to break the routing loop if execution fails more than once.
"""

from __future__ import annotations

import time
from typing import Any

from enterprise_agentic_rag.agents.code_executor import CodeExecutor
from enterprise_agentic_rag.graph.state import AgentState
from enterprise_agentic_rag.prompts.code_prompts import generate_code as code_prompts_generate


async def generate_code_node(state: AgentState) -> dict[str, Any]:
    """Generate a code snippet based on retrieved documents."""
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])

    result = code_prompts_generate(query, docs, language=state.get("code_language", ""))

    return {
        "code_snippet": result.get("code_snippet", ""),
        "code_language": result.get("language", "typescript"),
        "last_worker": "code_generator",
        "last_agent_step": "generate_code",
    }


async def execute_code_node(state: AgentState) -> dict[str, Any]:
    """Execute the generated code in the sandbox via CodeExecutor.

    Increments code_retry_count to enable routing gate (avoids infinite loop).
    """
    code = state.get("code_snippet", "")
    language = state.get("code_language", "typescript")

    if not code or not code.strip():
        return {
            "code_execution_result": {"exit_code": -1, "stderr": "无代码可执行", "stdout": ""},
            "code_verified": False,
            "code_retry_count": (state.get("code_retry_count", 0)) + 1,
            "last_worker": "code_executor",
            "last_agent_step": "execute_code",
        }

    t0 = time.time()
    executor = CodeExecutor()
    exec_result = await executor.run(code=code, language=language)
    latency_ms = (time.time() - t0) * 1000

    code_verified = exec_result.get("exit_code") == 0

    return {
        "code_execution_result": exec_result,
        "code_verified": code_verified,
        "code_retry_count": (state.get("code_retry_count", 0)) + 1,
        "last_worker": "code_executor",
        "last_agent_step": "execute_code",
    }
