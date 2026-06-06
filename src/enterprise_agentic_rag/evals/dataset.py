"""Eval dataset — PostgreSQL primary, JSONL fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    query: str
    expected_intent: str = ""
    expected_sources: list[str] = field(default_factory=list)
    expected_answer_keywords: list[str] = field(default_factory=list)
    user_role: str = "basic"
    difficulty: str = "medium"
    prompt_version: str = "v1"
    actual_intent: str = ""
    actual_answer: str = ""
    actual_sources: list[str] = field(default_factory=list)
    verified: bool = False
    passed: bool | None = None
    failure_reason: str = ""


@dataclass
class FailedCase:
    trace_id: str = ""
    session_id: str = ""
    query: str = ""
    intent: str = ""
    user_id: str = ""
    final_answer: str = ""
    fallback_reason: str = ""
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "eval"


def _default_dataset_path() -> str:
    return str(_data_dir() / "regression_cases.jsonl")


def _failed_cases_path() -> str:
    return str(_data_dir() / "failed_cases.jsonl")


class EvalDataset:
    """PG-first eval dataset with JSONL fallback."""

    def __init__(self, dataset_path: str | None = None, failed_path: str | None = None) -> None:
        self.dataset_path = dataset_path or _default_dataset_path()
        self.failed_path = failed_path or _failed_cases_path()

    # ------------------------------------------------------------------
    # Load — PostgreSQL first, JSONL fallback
    # ------------------------------------------------------------------
    def load_cases(self) -> list[EvalCase]:
        pg_cases = self._load_pg()
        if pg_cases:
            return pg_cases
        return self._load_jsonl()

    def _load_pg(self) -> list[EvalCase]:
        try:
            import asyncio
            from enterprise_agentic_rag.storage.database import get_db_manager
            from sqlalchemy import text
            dbm = get_db_manager()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            if not loop.run_until_complete(dbm.check_connection()):
                return []

            async def _fetch():
                async with dbm.session() as sess:
                    r = await sess.execute(text("SELECT * FROM eval_cases ORDER BY id"))
                    return r.fetchall()
            rows = loop.run_until_complete(_fetch())
            cases = []
            for row in rows:
                m = row._mapping
                cases.append(EvalCase(
                    query=m.get("query", ""),
                    expected_intent=m.get("expected_intent", ""),
                    expected_sources=json.loads(m.get("expected_sources", "[]")),
                    expected_answer_keywords=json.loads(m.get("expected_answer_keywords", "[]")),
                    difficulty=m.get("difficulty", "medium"),
                    prompt_version=m.get("prompt_version", "v1"),
                ))
            return cases
        except Exception:
            return []

    def _load_jsonl(self) -> list[EvalCase]:
        cases: list[EvalCase] = []
        if not os.path.exists(self.dataset_path):
            return cases
        with open(self.dataset_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cases.append(EvalCase(
                        query=obj.get("query", ""),
                        expected_intent=obj.get("expected_intent", ""),
                        expected_sources=obj.get("expected_sources", []),
                        expected_answer_keywords=obj.get("expected_answer_keywords", []),
                        user_role=obj.get("user_role", "basic"),
                        difficulty=obj.get("difficulty", "medium"),
                        prompt_version=obj.get("prompt_version", "v1"),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
        return cases

    def count(self) -> int:
        return len(self.load_cases())

    # ------------------------------------------------------------------
    # Save failed case — PostgreSQL + JSONL (dual-write)
    # ------------------------------------------------------------------
    def save_failed_case(self, case: FailedCase) -> bool:
        # PG first, JSONL fallback
        pg_ok = self._save_pg(case)
        if pg_ok:
            return True
        return self._save_jsonl(case)

    def _save_jsonl(self, case: FailedCase) -> bool:
        try:
            os.makedirs(os.path.dirname(self.failed_path), exist_ok=True)
            d = {
                "trace_id": case.trace_id, "session_id": case.session_id,
                "query": case.query, "intent": case.intent, "user_id": case.user_id,
                "final_answer": case.final_answer[:500], "fallback_reason": case.fallback_reason,
                "source": case.source, "timestamp": case.timestamp, "metadata": case.metadata,
            }
            with open(self.failed_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
            return True
        except Exception:
            return False

    def _save_pg(self, case: FailedCase) -> bool:
        try:
            import asyncio
            from enterprise_agentic_rag.storage.repositories import Repository
            repo = Repository()
            loop = asyncio.get_event_loop()
            coro = repo.insert_failed_case(
                trace_id=case.trace_id, session_id=case.session_id,
                query=case.query, reason=case.fallback_reason,
                source=case.source, payload=case.metadata,
            )
            if loop.is_running():
                asyncio.ensure_future(coro)
                return True  # best-effort: assume success in async context
            else:
                loop.run_until_complete(coro)
            return True
        except Exception:
            return False

    def load_failed_cases(self) -> list[dict[str, Any]]:
        # Try PG first
        try:
            import asyncio
            from enterprise_agentic_rag.storage.database import get_db_manager
            from sqlalchemy import text
            dbm = get_db_manager()
            loop = asyncio.get_event_loop()
            if not loop.is_running() and loop.run_until_complete(dbm.check_connection()):
                async def _f():
                    async with dbm.session() as sess:
                        r = await sess.execute(text("SELECT * FROM failed_cases ORDER BY created_at DESC LIMIT 500"))
                        return [dict(row._mapping) for row in r.fetchall()]
                return loop.run_until_complete(_f())
        except Exception:
            pass

        # JSONL fallback
        cases: list[dict[str, Any]] = []
        if not os.path.exists(self.failed_path):
            return cases
        with open(self.failed_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return cases

    def clear_failed_cases(self) -> None:
        try:
            Path(self.failed_path).write_text("", encoding="utf-8")
        except Exception:
            pass
