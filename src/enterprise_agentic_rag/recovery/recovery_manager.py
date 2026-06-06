"""Recovery Manager — unified orchestrator for fallback and retry.

Takes an error/failure signal from the workflow, evaluates the best
recovery action, and returns structured guidance for the LangGraph router.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.recovery.fallback_policy import (
    FallbackDecision,
    FallbackPolicy,
    FallbackType,
    RecoveryAction,
)
from enterprise_agentic_rag.recovery.retry_policy import RetryPolicy


class RecoveryManager:
    """Decides recovery actions based on failure type and retry state."""

    def __init__(
        self,
        fallback_policy: FallbackPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.fallback_policy = fallback_policy or FallbackPolicy()
        self.retry_policy = retry_policy or RetryPolicy()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def evaluate_failure(
        self,
        state: dict[str, Any],
        fallback_type: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate the current failure and return state updates.

        Args:
            state: Current workflow state.
            fallback_type: Explicit type; auto-detected from state if omitted.

        Returns:
            Dict of state fields to merge (fallback_reason, recovery_action,
            recoverable, human_fallback_payload).
        """
        if fallback_type is None:
            fb = self.fallback_policy.determine_fallback_type(state)
        else:
            try:
                fb = FallbackType(fallback_type)
            except ValueError:
                fb = FallbackType.UNKNOWN_INTENT

        retry_count = state.get("retry_count", {})
        decision = self.fallback_policy.evaluate(
            fallback_type=fb,
            retry_count=retry_count,
        )

        payload = self._build_human_payload(state) if not decision.recoverable else {}

        return {
            "fallback_reason": decision.fallback_type.value,
            "recovery_action": decision.recovery_action.value,
            "recoverable": decision.recoverable,
            "human_fallback_payload": payload,
            "error": decision.reason if not decision.recoverable else state.get("error", ""),
        }

    def can_retry(self, node_key: str, retry_count: dict[str, int]) -> bool:
        """Check if *node_key* still has retries available."""
        return self.retry_policy.can_retry(node_key, retry_count.get(node_key, 0))

    def record_retry(
        self,
        state: dict[str, Any],
        node_key: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Increment the retry counter for *node_key* and log the attempt.

        Returns state updates that merge into the workflow state.
        """
        retry_count = dict(state.get("retry_count", {}))
        current = retry_count.get(node_key, 0) + 1
        retry_count[node_key] = current

        retry_history = list(state.get("retry_history", []))
        retry_history.append(
            self.retry_policy.build_retry_entry(
                node_key=node_key,
                attempt=current,
                reason=reason,
            )
        )

        return {
            "retry_count": retry_count,
            "retry_history": retry_history,
        }

    # ------------------------------------------------------------------
    # Decision helpers — used by routing functions
    # ------------------------------------------------------------------
    @staticmethod
    def get_action(state: dict[str, Any]) -> RecoveryAction:
        """Read the current recovery_action from state."""
        raw = state.get("recovery_action", "")
        try:
            return RecoveryAction(raw)
        except ValueError:
            return RecoveryAction.HUMAN_FALLBACK

    @staticmethod
    def get_fallback_type(state: dict[str, Any]) -> FallbackType:
        """Read the current fallback_reason from state."""
        raw = state.get("fallback_reason", "")
        try:
            return FallbackType(raw)
        except ValueError:
            return FallbackType.UNKNOWN_INTENT

    # ------------------------------------------------------------------
    # Human fallback payload builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_human_payload(state: dict[str, Any]) -> dict[str, Any]:
        """Build a complete payload for human escalation.

        Required fields per spec:
        - query, user_id, session_id, intent
        - retrieved_docs, tool_results
        - verification_reason, error
        """
        return {
            "query": state.get("query", ""),
            "user_id": state.get("user_id", ""),
            "session_id": state.get("session_id", ""),
            "intent": state.get("intent", "unknown"),
            "user_role": state.get("user_role", ""),
            "fallback_reason": state.get("fallback_reason", ""),
            "retrieved_docs": state.get("retrieved_docs", []),
            "tool_results": state.get("tool_results", []),
            "tool_errors": state.get("tool_errors", []),
            "verification_reason": state.get("verification_reason", ""),
            "draft_answer": state.get("draft_answer", ""),
            "retry_history": state.get("retry_history", []),
            "error": state.get("error", ""),
            "timestamp": "",  # filled by consumer
        }

    # ------------------------------------------------------------------
    # Query rewrite for retrieval retry
    # ------------------------------------------------------------------
    @staticmethod
    def rewrite_query(original_query: str) -> str:
        """Simple query rewrite: drop stop-words and reorder keywords.

        In production this would use an LLM.  Our mock version extracts
        the longest tokens (likely domain terms) and rearranges them.
        """
        tokens = original_query.split()
        if not tokens:
            return original_query

        stop_words = {
            "的", "了", "是", "在", "和", "也", "就", "都", "而", "及",
            "与", "着", "或", "一个", "没有", "我们", "你们", "他们",
            "这个", "那个", "什么", "怎么", "为什么", "如何",
            "我", "你", "他", "她", "它",
        }

        # Keep the longest tokens first (they tend to be domain terms)
        meaningful = [t for t in tokens if t.lower() not in stop_words]
        if not meaningful:
            meaningful = tokens

        # Sort by length descending, then rejoin
        meaningful.sort(key=len, reverse=True)
        return " ".join(meaningful)
