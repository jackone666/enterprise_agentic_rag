"""Unified tool executor — enhanced with timeout, retry, circuit breaker.

Wraps every tool call with safety gates, resilience patterns, and auditing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enterprise_agentic_rag.tools.base import BaseTool, ToolResult
from enterprise_agentic_rag.tools.policies import evaluate_policy
from enterprise_agentic_rag.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Simple circuit breaker state per tool
_circuit_state: dict[str, dict[str, Any]] = {}


class ToolExecutor:
    """Execute tools with safety gates, retry, timeout, and audit logging."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        user_permissions: list[str] | None = None,
        skip_confirmation: bool = False,
        trace_id: str = "",
    ) -> ToolResult:
        user_permissions = user_permissions or []

        # --- Lookup ---
        try:
            tool = self.registry.get(tool_name)
        except KeyError as exc:
            return ToolResult(tool_name=tool_name, success=False, error=f"工具未注册: {exc}")

        # --- Circuit breaker check ---
        if _is_circuit_open(tool_name):
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"熔断器已打开 — {tool_name} 暂时不可用",
            )

        # --- Policy gate (tier-aware) ---
        policy = evaluate_policy(
            tool_name=tool_name,
            tier=tool.tier,
            required_permissions=tool.required_permissions,
            user_permissions=user_permissions,
            skip_confirmation=skip_confirmation,
        )

        if policy.decision == "denied":
            return ToolResult(tool_name=tool_name, success=False, error=policy.reason)

        if policy.decision == "pending":
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"⚠️ 需要确认: {policy.reason}",
            )

        # --- Execute with timeout + retry ---
        result = await self._execute_with_timeout(tool, params, trace_id)

        # --- Circuit breaker tracking ---
        _record_result(tool_name, result.success)

        # --- Audit log (PostgreSQL) ---
        self._audit_pg(tool_name, params, result, trace_id)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _execute_with_timeout(
        self, tool: BaseTool, params: dict[str, Any], trace_id: str
    ) -> ToolResult:
        timeout = getattr(tool, "timeout_seconds", 30.0) or 30.0
        t0 = time.monotonic()

        for attempt in range(1, max(1, tool.max_retries) + 1):
            try:
                result = await asyncio.wait_for(
                    tool.execute(**params), timeout=timeout
                )
                result.latency_ms = (time.monotonic() - t0) * 1000
                result.tool_name = tool.name
                return result
            except TimeoutError:
                if attempt < tool.max_retries:
                    continue
                return ToolResult(
                    tool_name=tool.name,
                    success=False,
                    error=f"执行超时 ({timeout}s, 已重试 {attempt} 次)",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            except Exception as exc:
                if attempt < tool.max_retries:
                    await asyncio.sleep(0.1 * attempt)
                    continue
                return ToolResult(
                    tool_name=tool.name,
                    success=False,
                    error=f"执行失败 (已重试 {attempt} 次): {exc}",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

        return ToolResult(tool_name=tool.name, success=False, error="未知错误")

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    def _audit_pg(
        self, tool_name: str, params: dict, result: ToolResult, trace_id: str
    ) -> None:
        try:
            import asyncio as aio

            from enterprise_agentic_rag.storage.repositories import Repository

            repo = Repository()
            coro = repo.insert_tool_audit_log(
                trace_id=trace_id,
                tool_name=tool_name,
                input_summary=str(params)[:200],
                output_summary=str(result.output)[:200] if result.success else "",
                success=result.success,
                error=result.error or "",
                latency_ms=result.latency_ms,
            )
            loop = aio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(repo.available())
                loop.run_until_complete(coro)
            else:
                task = aio.ensure_future(coro)
                task.add_done_callback(_log_background_audit_error)
        except Exception:
            logger.warning("Tool audit scheduling failed", exc_info=True)


# ---------------------------------------------------------------------------
# Simple circuit breaker helpers
# ---------------------------------------------------------------------------
def _is_circuit_open(tool_name: str) -> bool:
    state = _circuit_state.get(tool_name, {})
    if state.get("open") and time.time() - state.get("since", 0) < 30:
        return True
    return False


def _record_result(tool_name: str, success: bool) -> None:
    entry = _circuit_state.setdefault(tool_name, {"failures": 0, "open": False, "since": 0})
    if success:
        entry["failures"] = 0
        entry["open"] = False
    else:
        entry["failures"] += 1
        if entry["failures"] >= 5:
            entry["open"] = True
            entry["since"] = time.time()


def _log_background_audit_error(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except Exception:
        logger.warning("Tool audit persistence failed", exc_info=True)
