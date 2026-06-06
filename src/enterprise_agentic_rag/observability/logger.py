"""Event logger — PostgreSQL primary, JSONL fallback.

PostgreSQL is written first for structured event types.
JSONL is written only when PostgreSQL is unavailable (fallback-safe).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class EventLogger:
    """Failure-safe logger — PostgreSQL primary, JSONL fallback."""

    def __init__(self, log_path: str | None = None) -> None:
        if log_path is None:
            project_root = Path(__file__).resolve().parents[3]
            log_path = str(project_root / "data" / "logs" / "events.jsonl")
        self._log_path = Path(log_path)
        self._repo = None

    @property
    def repo(self):
        if self._repo is None:
            try:
                from enterprise_agentic_rag.storage.repositories import Repository
                self._repo = Repository()
            except Exception:
                self._repo = False
        return self._repo if self._repo is not False else None

    # ------------------------------------------------------------------
    # Write events
    # ------------------------------------------------------------------
    def write_event(self, event: dict[str, Any]) -> bool:
        """Write event to PostgreSQL; fallback to JSONL when PG unavailable."""
        # Try PostgreSQL first for structured event types
        etype = event.get("event_type", "")
        pg_ok = False
        if etype in ("node_start", "node_end"):
            pg_ok = self._write_node_event_pg(event)
        elif etype == "retrieval":
            pg_ok = self._write_retrieval_event_pg(event)
        elif etype == "verification":
            pg_ok = self._write_verification_event_pg(event)
        elif etype in ("llm_call", "llm_failure"):
            pg_ok = self._write_llm_event_pg(event)
        else:
            pg_ok = self._pg_write_untyped(event)

        # Fallback to JSONL when PG is not available
        if not pg_ok:
            return self._write_jsonl(event)
        return True

    def write_events(self, events: list[dict[str, Any]]) -> int:
        written = 0
        for evt in events:
            if self.write_event(evt):
                written += 1
        return written

    # ------------------------------------------------------------------
    # JSONL
    # ------------------------------------------------------------------
    def _write_jsonl(self, event: dict[str, Any]) -> bool:
        try:
            self._ensure_dir()
            line = json.dumps(event, ensure_ascii=False, default=str)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return True
        except Exception:
            return False

    def read_events(self, tail_n: int | None = None) -> list[dict[str, Any]]:
        if not self._log_path.exists():
            return []
        try:
            with open(self._log_path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception:
            return []
        if tail_n is not None:
            lines = lines[-tail_n:]
        events: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def clear(self) -> None:
        try:
            self._log_path.write_text("", encoding="utf-8")
        except Exception:
            pass

    @property
    def log_path(self) -> str:
        return str(self._log_path)

    def _ensure_dir(self) -> None:
        os.makedirs(self._log_path.parent, exist_ok=True)

    # ------------------------------------------------------------------
    # PostgreSQL event writers
    # ------------------------------------------------------------------
    def _pg_insert(self, table: str, data: dict) -> bool:
        """Insert into PostgreSQL. Returns True on success, False on failure."""
        repo = self.repo
        if repo is None:
            return False

        # Quick connectivity check — skip PG if known unavailable
        if not self._pg_is_available():
            return False

        try:
            from enterprise_agentic_rag.storage.models import (
                Base,
                LLMEventModel,
                NodeEventModel,
                RetrievalEventModel,
                VerificationEventModel,
            )
        except ImportError:
            return False

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Non-blocking fire-and-forget in async context
                asyncio.ensure_future(self._async_insert(table, data))
                # Also write JSONL as safety net since we can't await confirmation
                return False
            loop.run_until_complete(self._async_insert(table, data))
            return True
        except Exception:
            return False

    def _pg_is_available(self) -> bool:
        """Check PG connectivity without creating a new engine."""
        try:
            from enterprise_agentic_rag.storage.database import get_db_manager
            dbm = get_db_manager()
            return dbm.available
        except Exception:
            return False

    def _pg_write_untyped(self, event: dict[str, Any]) -> bool:
        """Write untyped events to PG node_events table as catch-all."""
        etype = event.get("event_type", "unknown")
        data = dict(event)
        data.setdefault("node_name", etype)
        data.setdefault("latency_ms", 0.0)
        data.setdefault("success", True)
        return self._pg_insert("node_events", data)

    async def _async_insert(self, table: str, data: dict) -> None:
        from enterprise_agentic_rag.storage.database import get_db_manager
        dbm = get_db_manager()
        if not await dbm.check_connection():
            raise RuntimeError("PostgreSQL unavailable")

        import json as _json

        from enterprise_agentic_rag.storage.models import (
            LLMEventModel,
            NodeEventModel,
            RetrievalEventModel,
            VerificationEventModel,
        )

        model_map = {
            "node_events": NodeEventModel,
            "retrieval_events": RetrievalEventModel,
            "verification_events": VerificationEventModel,
            "llm_events": LLMEventModel,
        }
        model_cls = model_map.get(table)
        if model_cls is None:
            raise ValueError(f"Unknown event table: {table}")

        instance = model_cls(
            trace_id=data.get("trace_id", ""),
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", ""),
            event_type=data.get("event_type", ""),
            node_name=data.get("node_name", data.get("tool_name", "")),
            input_summary=data.get("input_summary", ""),
            output_summary=data.get("output_summary", ""),
            latency_ms=data.get("latency_ms", 0.0),
            success=data.get("success", True),
            error=data.get("error", ""),
            meta_json=_json.dumps(data, ensure_ascii=False, default=str),
        )
        async with dbm.session() as sess:
            sess.add(instance)
            await sess.commit()

    def _write_node_event_pg(self, e: dict) -> bool:
        return self._pg_insert("node_events", e)

    def _write_retrieval_event_pg(self, e: dict) -> bool:
        return self._pg_insert("retrieval_events", e)

    def _write_verification_event_pg(self, e: dict) -> bool:
        return self._pg_insert("verification_events", e)

    def _write_llm_event_pg(self, e: dict) -> bool:
        return self._pg_insert("llm_events", e)
