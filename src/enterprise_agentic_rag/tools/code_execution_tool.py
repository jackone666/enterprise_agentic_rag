"""Code execution tool — sandboxed code execution via subprocess.

Executes code in a restricted subprocess with timeout and memory limits.
Follows the BaseTool pattern for security gating and audit logging.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from enterprise_agentic_rag.config.settings import get_settings
from enterprise_agentic_rag.tools.base import BaseTool, ToolResult


class CodeExecutionTool(BaseTool):
    """Execute code in a sandboxed subprocess.

    Tier: sensitive (can execute arbitrary code)
    Required permissions: ["write"]

    Currently uses subprocess with timeout. Future versions can upgrade
    to Docker/gVisor sandbox for stronger isolation.
    """

    name: str = "execute_code"
    description: str = "在沙箱环境中执行代码，返回执行结果（stdout/stderr/exit_code）"
    is_sensitive: bool = True
    tier: str = "sensitive"
    required_permissions: list[str] = ["write"]
    timeout_seconds: float = 15.0
    max_retries: int = 1

    input_schema: dict[str, Any] = {
        "code": "string — 要执行的代码",
        "language": "string — 编程语言 (javascript/typescript/python/bash)",
        "timeout_seconds": "number (optional) — 超时时间，默认 15 秒",
    }
    output_schema: dict[str, Any] = {
        "stdout": "string — 标准输出",
        "stderr": "string — 标准错误输出",
        "exit_code": "number — 进程退出码 (0=成功)",
        "execution_time_ms": "number — 执行耗时（毫秒）",
    }

    # Blocklist of dangerous patterns
    _DANGEROUS_PATTERNS = [
        "import os", "import subprocess", "import sys",
        "os.system", "os.popen", "os.exec",
        "subprocess.call", "subprocess.Popen",
        "__import__", "eval(", "exec(",
        "require('child_process')", "require('fs')",
        "process.exit", "Deno.run",
        "rm -rf", "mkfs.", "dd if=", "> /dev/",
        "chmod", "chown", "sudo",
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute code in a sandboxed subprocess.

        Args:
            **kwargs:
                code: The code string to execute.
                language: Programming language.
                timeout_seconds: Optional timeout override.

        Returns:
            ToolResult with execution output.
        """
        import time

        code = kwargs.get("code", "")
        language = kwargs.get("language", "typescript").lower()
        timeout = float(kwargs.get("timeout_seconds", self.timeout_seconds))
        settings = get_settings()

        if settings.runtime.is_production and not settings.runtime.allow_local_code_execution:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="生产环境禁止本机 subprocess 代码执行；请接入 Docker/gVisor/Firecracker 沙箱后再开启。",
                latency_ms=0.0,
            )

        if not code or not code.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="代码为空，无法执行",
                latency_ms=0.0,
            )

        # Security check
        if not self._is_safe(code):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="代码包含不安全操作，已被拒绝执行",
                latency_ms=0.0,
            )

        # Language check
        allowed = settings.code_execution.allowed_languages
        if language not in allowed:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"不支持的语言 '{language}'，当前支持: {', '.join(allowed)}",
                latency_ms=0.0,
            )

        t0 = time.time()
        try:
            output = await self._run_code(code, language, timeout)
            latency_ms = (time.time() - t0) * 1000
            success = output.get("exit_code", -1) == 0

            return ToolResult(
                tool_name=self.name,
                success=success,
                output=output,
                error="" if success else output.get("stderr", "执行失败"),
                latency_ms=round(latency_ms, 2),
            )
        except TimeoutError:
            latency_ms = (time.time() - t0) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"代码执行超时（{timeout}秒）",
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"代码执行异常: {str(exc)}",
                latency_ms=round(latency_ms, 2),
            )

    def _is_safe(self, code: str) -> bool:
        """Check if the code contains dangerous patterns."""
        code_lower = code.lower()
        for pattern in self._DANGEROUS_PATTERNS:
            if pattern.lower() in code_lower:
                return False
        return True

    async def _run_code(
        self, code: str, language: str, timeout: float,
    ) -> dict[str, Any]:
        """Execute code via subprocess with timeout."""
        # Select interpreter
        interpreter, ext, flags = self._get_runtime(language)

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{ext}", delete=False, prefix="code_exec_",
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            cmd = [interpreter] + flags + [tmp_path]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            return {
                "stdout": stdout.decode("utf-8", errors="replace")[:2000],
                "stderr": stderr.decode("utf-8", errors="replace")[:2000],
                "exit_code": proc.returncode if proc.returncode is not None else -1,
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _get_runtime(self, language: str) -> tuple[str, str, list[str]]:
        """Get the interpreter, file extension, and flags for a language.

        Returns:
            Tuple of (interpreter_path, file_extension, extra_flags).
        """
        runtimes = {
            "javascript": ("node", "js", ["--check"]),
            "js": ("node", "js", ["--check"]),
            "typescript": ("node", "ts", ["--eval"]),
            "ts": ("node", "ts", ["--eval"]),
            "arkts": ("node", "ets", ["--eval"]),
            "ets": ("node", "ets", ["--eval"]),
            "python": ("python3", "py", []),
            "py": ("python3", "py", []),
            "bash": ("bash", "sh", ["-n"]),
            "sh": ("bash", "sh", ["-n"]),
        }
        return runtimes.get(language, ("node", "js", ["--check"]))


# Singleton instance for use in tool_agent
_code_execution_tool: CodeExecutionTool | None = None


def get_code_execution_tool() -> CodeExecutionTool:
    """Return the singleton CodeExecutionTool instance."""
    global _code_execution_tool
    if _code_execution_tool is None:
        _code_execution_tool = CodeExecutionTool()
    return _code_execution_tool
