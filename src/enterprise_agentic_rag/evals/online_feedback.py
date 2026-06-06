"""Online feedback handler — thumbs_up/down capture and auto case mining.

Supports:
- Manual thumbs_up / thumbs_down with optional feedback_text
- Auto-capture of failed cases when:
  * need_human == True
  * verified == False
  * fallback_reason is not empty
  * User gives thumbs_down

Failed cases are written to data/eval/failed_cases.jsonl (Data Flywheel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from enterprise_agentic_rag.evals.dataset import EvalDataset, FailedCase


@dataclass
class FeedbackRecord:
    """A single piece of user feedback."""

    trace_id: str
    session_id: str
    thumbs_up: bool = True
    feedback_text: str = ""
    user_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class FeedbackHandler:
    """Processes user feedback and automatically mines failed cases."""

    def __init__(self, dataset: EvalDataset | None = None) -> None:
        self.dataset = dataset or EvalDataset()

    # ------------------------------------------------------------------
    # Process feedback
    # ------------------------------------------------------------------
    def process_feedback(
        self,
        feedback: FeedbackRecord,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a thumbs_up/down feedback.

        Args:
            feedback: The user feedback record.
            result: Optional prior /chat result dict (for auto-mining context).

        Returns:
            Dict with ``auto_captured`` (bool) and ``reason`` (str).
        """
        auto_captured = False
        reason = ""

        # Explicit thumbs_down → always capture
        if not feedback.thumbs_up:
            self._capture(
                trace_id=feedback.trace_id,
                session_id=feedback.session_id,
                query=result.get("query", "") if result else "",
                intent=result.get("intent", "") if result else "",
                user_id=feedback.user_id,
                final_answer=result.get("final_answer", "") if result else "",
                fallback_reason=feedback.feedback_text or "用户标记为不满意",
                source="feedback",
                metadata={
                    "feedback_text": feedback.feedback_text,
                    "thumbs_up": False,
                },
            )
            # Write to feedback table in PostgreSQL
            self._save_feedback_pg(feedback)
            auto_captured = True
            reason = feedback.feedback_text or "用户 thumbs_down"

        # Always persist user feedback to PG feedback table (even thumbs_up)
        if not auto_captured:
            self._save_feedback_pg(feedback)

        # Auto-capture from result signals
        if result and not auto_captured:
            auto_reason = self._auto_capture_reason(result)
            if auto_reason:
                self._capture(
                    trace_id=feedback.trace_id,
                    session_id=feedback.session_id,
                    query=result.get("query", ""),
                    intent=result.get("intent", ""),
                    user_id=feedback.user_id,
                    final_answer=result.get("final_answer", ""),
                    fallback_reason=auto_reason,
                    source="auto",
                    metadata={
                        "need_human": result.get("need_human", False),
                        "verified": result.get("verified", True),
                        "fallback_reason": result.get("fallback_reason", ""),
                    },
                )
                auto_captured = True
                reason = auto_reason

        return {"auto_captured": auto_captured, "reason": reason}

    # ------------------------------------------------------------------
    # Auto-capture logic
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_capture_reason(result: dict[str, Any]) -> str:
        """Determine if a result should be auto-captured as a failed case.

        Returns the reason string if capture is needed, empty string otherwise.
        """
        reasons: list[str] = []

        if result.get("need_human"):
            reasons.append("need_human=true")
        if not result.get("verified", True):
            reasons.append(f"verified=false: {result.get('verification_reason', '')}")
        if result.get("fallback_reason", ""):
            reasons.append(f"fallback_reason={result.get('fallback_reason')}")

        return "; ".join(reasons) if reasons else ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _capture(
        self,
        trace_id: str,
        session_id: str,
        query: str,
        intent: str,
        user_id: str,
        final_answer: str,
        fallback_reason: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Write a failed case to the dataset."""
        case = FailedCase(
            trace_id=trace_id,
            session_id=session_id,
            query=query,
            intent=intent,
            user_id=user_id,
            final_answer=final_answer,
            fallback_reason=fallback_reason,
            source=source,
            metadata=metadata or {},
        )
        return self.dataset.save_failed_case(case)

    @staticmethod
    def _save_feedback_pg(feedback: FeedbackRecord) -> None:
        """Persist user feedback to the PostgreSQL feedback table.

        Uses ensure_future when called from within an async context
        (e.g. FastAPI route), so the write does not block the response.
        """
        try:
            import asyncio

            from enterprise_agentic_rag.storage.repositories import Repository

            repo = Repository()
            loop = asyncio.get_event_loop()
            coro = repo.insert_feedback(
                trace_id=feedback.trace_id,
                session_id=feedback.session_id,
                user_id=feedback.user_id,
                thumbs_up=feedback.thumbs_up,
                feedback_text=feedback.feedback_text,
            )
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except Exception:
            pass  # feedback persistence is non-critical
