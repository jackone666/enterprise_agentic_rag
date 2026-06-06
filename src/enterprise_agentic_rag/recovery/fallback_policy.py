"""Fallback policy — defines fallback types, reasons, and decision logic.

Each fallback type maps to a stable reason label used across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FallbackType(str, Enum):
    """Standardised fallback type labels."""

    PERMISSION_DENIED = "permission_denied"
    NO_RELEVANT_DOCS = "no_relevant_docs"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    TOOL_FAILURE = "tool_failure"
    ANSWER_NOT_GROUNDED = "answer_not_grounded"
    LLM_FAILURE = "llm_failure"
    UNKNOWN_INTENT = "unknown_intent"
    CODE_EXECUTION_FAILED = "code_execution_failed"
    CODE_GENERATION_FAILED = "code_generation_failed"


class RecoveryAction(str, Enum):
    """Actions the recovery manager can instruct the workflow to take."""

    RETRY = "retry"
    REWRITE_QUERY = "rewrite_query"
    USE_KEYWORD_RETRIEVER = "use_keyword_retriever"
    REGENERATE_ANSWER = "regenerate_answer"
    HUMAN_FALLBACK = "human_fallback"
    FINAL_REFUSAL = "final_refusal"


# ---------------------------------------------------------------------------
# Fallback decision table
# ---------------------------------------------------------------------------
# Maps fallback_type → preferred recovery action (first choice)
FALLBACK_ACTION_MAP: dict[FallbackType, RecoveryAction] = {
    FallbackType.PERMISSION_DENIED: RecoveryAction.FINAL_REFUSAL,
    FallbackType.NO_RELEVANT_DOCS: RecoveryAction.REWRITE_QUERY,
    FallbackType.LOW_RETRIEVAL_SCORE: RecoveryAction.USE_KEYWORD_RETRIEVER,
    FallbackType.TOOL_FAILURE: RecoveryAction.RETRY,
    FallbackType.ANSWER_NOT_GROUNDED: RecoveryAction.REGENERATE_ANSWER,
    FallbackType.LLM_FAILURE: RecoveryAction.RETRY,
    FallbackType.UNKNOWN_INTENT: RecoveryAction.HUMAN_FALLBACK,
    FallbackType.CODE_EXECUTION_FAILED: RecoveryAction.RETRY,
    FallbackType.CODE_GENERATION_FAILED: RecoveryAction.RETRY,
}

# Escalation path — when the primary action fails, what's next?
FALLBACK_ESCALATION: dict[RecoveryAction, RecoveryAction] = {
    RecoveryAction.RETRY: RecoveryAction.HUMAN_FALLBACK,
    RecoveryAction.REWRITE_QUERY: RecoveryAction.HUMAN_FALLBACK,
    RecoveryAction.USE_KEYWORD_RETRIEVER: RecoveryAction.HUMAN_FALLBACK,
    RecoveryAction.REGENERATE_ANSWER: RecoveryAction.HUMAN_FALLBACK,
    RecoveryAction.HUMAN_FALLBACK: RecoveryAction.FINAL_REFUSAL,
    RecoveryAction.FINAL_REFUSAL: RecoveryAction.FINAL_REFUSAL,  # terminal
}


@dataclass
class FallbackDecision:
    """Result of evaluating a fallback scenario."""

    fallback_type: FallbackType
    recovery_action: RecoveryAction
    reason: str
    recoverable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class FallbackPolicy:
    """Determines what recovery action to take for a given failure."""

    @staticmethod
    def evaluate(
        fallback_type: str | FallbackType,
        retry_count: dict[str, int] | None = None,
        reason: str = "",
    ) -> FallbackDecision:
        """Evaluate the best recovery action for a failure.

        Args:
            fallback_type: One of the ``FallbackType`` values.
            retry_count: Current per-node retry counts.
            reason: Human-readable description of the failure.

        Returns:
            A :class:`FallbackDecision` with the recommended action.
        """
        if isinstance(fallback_type, str):
            try:
                fb = FallbackType(fallback_type)
            except ValueError:
                fb = FallbackType.UNKNOWN_INTENT
        else:
            fb = fallback_type

        retry_count = retry_count or {}
        action = FALLBACK_ACTION_MAP.get(fb, RecoveryAction.HUMAN_FALLBACK)

        # Check if we've already exhausted retries for this type
        exhausted = FallbackPolicy._is_exhausted(fb, retry_count)
        if exhausted:
            action = FALLBACK_ESCALATION.get(action, RecoveryAction.HUMAN_FALLBACK)
            recoverable = action != RecoveryAction.FINAL_REFUSAL
        else:
            recoverable = action != RecoveryAction.FINAL_REFUSAL

        return FallbackDecision(
            fallback_type=fb,
            recovery_action=action,
            reason=reason or FallbackPolicy._default_reason(fb),
            recoverable=recoverable,
            metadata={"exhausted": exhausted},
        )

    @staticmethod
    def determine_fallback_type(state: dict[str, Any]) -> FallbackType:
        """Infer the fallback type from the current workflow state.

        Checks multiple signals in priority order and returns the first match.
        """
        # Permission check
        perms = state.get("permissions", [])
        if "knowledge_search" not in perms:
            return FallbackType.PERMISSION_DENIED

        # Unknown intent
        intent = state.get("intent", "")
        if intent == "unknown":
            return FallbackType.UNKNOWN_INTENT

        # No relevant docs
        docs = state.get("retrieved_docs", [])
        if not docs:
            return FallbackType.NO_RELEVANT_DOCS

        # Low retrieval score
        if all(d.get("score", 0) < 0.1 for d in docs):
            return FallbackType.LOW_RETRIEVAL_SCORE

        # Tool failure
        tool_errors = state.get("tool_errors", [])
        if tool_errors:
            return FallbackType.TOOL_FAILURE

        # Answer not grounded (verification failed)
        if state.get("verified") is False:
            return FallbackType.ANSWER_NOT_GROUNDED

        # Default — nothing wrong
        return FallbackType.UNKNOWN_INTENT

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_exhausted(fb: FallbackType, retry_count: dict[str, int]) -> bool:
        """Check if retries for this fallback type are exhausted."""
        from enterprise_agentic_rag.recovery.retry_policy import RetryPolicy

        policy = RetryPolicy()
        limits = policy.get_limits()

        key_map: dict[FallbackType, str] = {
            FallbackType.NO_RELEVANT_DOCS: "retrieve",
            FallbackType.LOW_RETRIEVAL_SCORE: "retrieve",
            FallbackType.TOOL_FAILURE: "tool_call",
            FallbackType.ANSWER_NOT_GROUNDED: "verify",
            FallbackType.LLM_FAILURE: "generate",
            FallbackType.CODE_EXECUTION_FAILED: "code_execution",
            FallbackType.CODE_GENERATION_FAILED: "code_generation",
        }

        node_key = key_map.get(fb)
        if node_key is None:
            # Not a retryable type — use primary action as-is (no escalation)
            return False

        limit = limits.get(node_key, 0)
        current = retry_count.get(node_key, 0)
        return current >= limit

    @staticmethod
    def _default_reason(fb: FallbackType) -> str:
        reasons: dict[FallbackType, str] = {
            FallbackType.PERMISSION_DENIED: "用户权限不足，无法访问知识库",
            FallbackType.NO_RELEVANT_DOCS: "未检索到相关文档",
            FallbackType.LOW_RETRIEVAL_SCORE: "检索结果相关度过低",
            FallbackType.TOOL_FAILURE: "工具执行失败",
            FallbackType.ANSWER_NOT_GROUNDED: "答案校验未通过",
            FallbackType.LLM_FAILURE: "LLM 调用失败",
            FallbackType.UNKNOWN_INTENT: "无法识别用户意图",
            FallbackType.CODE_EXECUTION_FAILED: "代码沙箱执行失败",
            FallbackType.CODE_GENERATION_FAILED: "代码生成失败",
        }
        return reasons.get(fb, "未知错误")
