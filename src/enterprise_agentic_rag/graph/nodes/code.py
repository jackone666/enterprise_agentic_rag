"""Code generation + sandboxed execution nodes.

The code path is opt-in — only entered when the master agent decides
the request is a ``code_generation`` intent. ``execute_code`` runs the
snippet in a security-gated sandbox and increments ``code_retry_count``
to break the routing loop if execution fails more than once.
"""

from __future__ import annotations

import time
from typing import Any

from enterprise_agentic_rag.agents.code_agent import generate_code as code_agent_generate
from enterprise_agentic_rag.graph.state import AgentState


async def generate_code_node(state: AgentState) -> dict[str, Any]:
    """Generate a code snippet based on retrieved documents."""
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])

    result = code_agent_generate(query, docs, language=state.get("code_language", ""))

    return {
        "code_snippet": result.get("code_snippet", ""),
        "code_language": result.get("language", "typescript"),
        "last_worker": "code_generator",
        "last_agent_step": "generate_code",
    }


async def execute_code_node(state: AgentState) -> dict[str, Any]:
    """Execute the generated code in the sandbox.

    Uses CodeExecutionTool for security-gated execution.
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

    from enterprise_agentic_rag.tools.code_execution_tool import get_code_execution_tool

    t0 = time.time()
    tool = get_code_execution_tool()

    try:
        result = await tool.execute(code=code, language=language)
    except Exception as exc:
        result = type("ToolResult", (), {})()
        result.success = False
        result.error = str(exc)
        result.output = {"exit_code": -1, "stderr": str(exc), "stdout": ""}

    latency_ms = (time.time() - t0) * 1000

    exec_result = result.output if result.success else {
        "stdout": "",
        "stderr": result.error or "代码执行失败",
        "exit_code": -1,
    }

    if not isinstance(exec_result, dict):
        exec_result = {"stdout": str(exec_result), "stderr": "", "exit_code": 0}

    code_verified = result.success and exec_result.get("exit_code") == 0

    return {
        "code_execution_result": exec_result,
        "code_verified": code_verified,
        "code_retry_count": (state.get("code_retry_count", 0)) + 1,
        "last_worker": "code_executor",
        "last_agent_step": "execute_code",
    }
