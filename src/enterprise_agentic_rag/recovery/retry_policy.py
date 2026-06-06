"""Retry policy — defines max retries and backoff per workflow node.

Node retry limits:
- retrieve_knowledge: 1 retry (via query rewrite)
- call_tools:          2 retries
- generate_answer:     1 retry (via regenerate)
- verify_answer:       1 retry (via regenerate, counted under verify)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetryConfig:
    """Configuration for a single node's retry behaviour."""

    max_retries: int
    backoff_seconds: float = 0.1  # mock — instant in tests
    description: str = ""


class RetryPolicy:
    """Central retry configuration for all recoverable nodes."""

    def __init__(self) -> None:
        self._limits: dict[str, RetryConfig] = {
            "retrieve": RetryConfig(
                max_retries=1,
                backoff_seconds=0.1,
                description="检索节点 — 查询改写后重试 1 次",
            ),
            "tool_call": RetryConfig(
                max_retries=2,
                backoff_seconds=0.1,
                description="工具调用节点 — 最多重试 2 次",
            ),
            "generate": RetryConfig(
                max_retries=1,
                backoff_seconds=0.1,
                description="答案生成节点 — 重新生成 1 次",
            ),
            "verify": RetryConfig(
                max_retries=1,
                backoff_seconds=0.1,
                description="答案校验节点 — 校验失败后允许重新生成 1 次",
            ),
            "code_generation": RetryConfig(
                max_retries=2,
                backoff_seconds=0.1,
                description="代码生成节点 — 生成失败或执行失败后修复重试最多 2 次",
            ),
            "code_execution": RetryConfig(
                max_retries=2,
                backoff_seconds=0.1,
                description="代码执行节点 — 执行失败后修复重试最多 2 次",
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_limits(self) -> dict[str, int]:
        """Return a simple ``{node_key: max_retries}`` mapping."""
        return {k: v.max_retries for k, v in self._limits.items()}

    def get_config(self, node_key: str) -> RetryConfig | None:
        """Return the full config for a specific node, or None."""
        return self._limits.get(node_key)

    def can_retry(self, node_key: str, current_count: int) -> bool:
        """Check whether a node can still be retried."""
        cfg = self._limits.get(node_key)
        if cfg is None:
            return False
        return current_count < cfg.max_retries

    def next_backoff(self, node_key: str) -> float:
        """Return the backoff duration for the next retry attempt."""
        cfg = self._limits.get(node_key)
        return cfg.backoff_seconds if cfg else 0.0

    @property
    def all_limits(self) -> dict[str, RetryConfig]:
        return dict(self._limits)

    # ------------------------------------------------------------------
    # Building retry history entries
    # ------------------------------------------------------------------
    @staticmethod
    def build_retry_entry(
        node_key: str,
        attempt: int,
        reason: str = "",
    ) -> dict[str, Any]:
        """Create a structured retry history record."""
        return {
            "node": node_key,
            "attempt": attempt,
            "reason": reason,
        }
