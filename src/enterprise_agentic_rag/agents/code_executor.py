"""CodeExecutor — agent that executes generated code in a sandbox.

Wraps CodeExecutionTool with retry policy and retryable-error detection.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Error patterns that are retryable (transient infrastructure issues)
_RETRYABLE_PATTERNS = [
    "timeout",
    "connection refused",
    "ECONNREFUSED",
    "ETIMEDOUT",
    "temporary failure",
    "rate limit",
    "429",
    "503",
    "504",
]


class CodeExecutor:
    """Execute code snippets in a security-gated sandbox with retry support."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    async def run(
        self,
        code: str,
        language: str = "typescript",
    ) -> dict[str, Any]:
        """Execute code in sandbox, with automatic retry on transient failures.

        Args:
            code: The code snippet to execute.
            language: Programming language (javascript/typescript/python/bash).

        Returns:
            Dict with keys: stdout, stderr, exit_code, execution_time_ms.
        """
        from enterprise_agentic_rag.tools.code_execution_tool import (
            get_code_execution_tool,
        )

        tool = get_code_execution_tool()
        attempt = 0
        last_error = ""

        while attempt <= self.max_retries:
            attempt += 1
            try:
                result = await tool.execute(code=code, language=language)

                if result.success:
                    return result.output if isinstance(result.output, dict) else {
                        "stdout": str(result.output),
                        "stderr": "",
                        "exit_code": 0,
                    }

                last_error = result.error or "unknown execution error"

                if self._is_retryable(last_error) and attempt <= self.max_retries:
                    logger.warning(
                        "Code execution attempt %d/%d failed with retryable error: %s",
                        attempt, self.max_retries, last_error,
                    )
                    continue

                # Non-retryable or final attempt — return failure
                return {
                    "stdout": "",
                    "stderr": last_error,
                    "exit_code": -1,
                }

            except Exception as exc:
                last_error = str(exc)
                if self._is_retryable(last_error) and attempt <= self.max_retries:
                    logger.warning(
                        "Code execution attempt %d/%d raised retryable exception: %s",
                        attempt, self.max_retries, last_error,
                    )
                    continue
                return {
                    "stdout": "",
                    "stderr": last_error,
                    "exit_code": -1,
                }

        # Exhausted retries
        return {
            "stdout": "",
            "stderr": f"代码执行失败（已重试 {self.max_retries} 次）: {last_error}",
            "exit_code": -1,
        }

    @staticmethod
    def _is_retryable(error: str) -> bool:
        """Return True if the error is transient and worth retrying."""
        if not error:
            return False
        error_lower = error.lower()
        return any(pat.lower() in error_lower for pat in _RETRYABLE_PATTERNS)
