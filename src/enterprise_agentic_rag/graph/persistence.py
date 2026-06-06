"""QA log persistence — fire-and-forget PostgreSQL writes.

Extracted from graph/workflow.py so the graph builder can stay focused
on node wiring. The persistence call is intentionally non-blocking:
a DB outage must never break user-facing responses.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def persist_qa_log(state: dict) -> None:
    """Write the final QA turn to PostgreSQL qa_logs table.

    Called from ``save_memory`` at the end of every request.
    Uses ``ensure_future`` so a DB write failure never blocks the response.
    """
    try:
        from enterprise_agentic_rag.storage.repositories import Repository

        node_events = state.get("node_events", [])
        total_ms = 0.0
        for evt in node_events:
            if evt.get("event_type") == "node_end":
                total_ms += evt.get("latency_ms", 0)

        repo = Repository()
        coro = repo.insert_qa_log(
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            user_id=state.get("user_id", ""),
            query=state.get("query", ""),
            answer=state.get("final_answer", ""),
            intent=state.get("intent", "unknown"),
            citations=state.get("citations", []),
            verified=state.get("verified", True),
            need_human=state.get("need_human", False),
            fallback_reason=state.get("fallback_reason", ""),
            latency_ms=round(total_ms, 2),
        )

        try:
            loop = asyncio.get_running_loop()
            # Inside an async context — schedule without awaiting
            loop.create_task(coro)
        except RuntimeError:
            # No running loop (e.g. test) — run synchronously
            asyncio.run(coro)
    except Exception as exc:
        logger.warning("QA log persistence failed: %s", exc)
