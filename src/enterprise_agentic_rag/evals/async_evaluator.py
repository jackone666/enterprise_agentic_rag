"""Async evaluator — background evaluation with bad case persistence."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalJob:
    query: str
    answer: str
    retrieved_docs: list[dict]
    trace_id: str = ""
    session_id: str = ""
    user_id: str = ""


class AsyncEvaluator:
    """Runs evaluation in background thread, persists bad cases to PostgreSQL."""

    def __init__(self) -> None:
        self._queue: list[EvalJob] = []
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, job: EvalJob) -> None:
        with self._lock:
            self._queue.append(job)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def get_result(self, trace_id: str) -> dict | None:
        with self._lock:
            return self._results.get(trace_id)

    def _run(self) -> None:
        while True:
            job = None
            with self._lock:
                if self._queue:
                    job = self._queue.pop(0)
            if job is None:
                break

            try:
                result = self._evaluate_sync(job)
                with self._lock:
                    self._results[job.trace_id] = result

                # Persist bad cases
                if not result.get("passing", True):
                    self._save_bad_case(job, result)
            except Exception:
                pass

    def _evaluate_sync(self, job: EvalJob) -> dict:
        from enterprise_agentic_rag.evals.eval_judge import EvalJudge
        judge = EvalJudge()
        r = judge.evaluate(job.query, job.answer, job.retrieved_docs)
        return {
            "overall": r.overall,
            "precision": r.precision,
            "recall": r.recall,
            "faithfulness": r.faithfulness,
            "relevance": r.relevance,
            "passing": r.passing,
            "details": r.details,
        }

    def _save_bad_case(self, job: EvalJob, result: dict) -> None:
        try:
            from enterprise_agentic_rag.storage.database import get_db_manager
            dbm = get_db_manager()
            if not asyncio.get_event_loop().is_running():
                async def _save():
                    if not await dbm.check_connection():
                        return
                    async with dbm.session() as sess:
                        from sqlalchemy import text
                        payload = json.dumps({
                            "query": job.query,
                            "answer": job.answer[:500],
                            "eval": result,
                        }, ensure_ascii=False)
                        await sess.execute(
                            text("INSERT INTO failed_cases (trace_id, session_id, query, reason, source, payload) VALUES (:tid, :sid, :q, :r, 'eval', :p)"),
                            {"tid": job.trace_id, "sid": job.session_id, "q": job.query, "r": f"eval score={result['overall']}", "p": payload},
                        )
                        await sess.commit()
                asyncio.run(_save())
        except Exception:
            pass  # bad case persistence is non-critical


# Singleton
_evaluator: AsyncEvaluator | None = None

def get_async_evaluator() -> AsyncEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = AsyncEvaluator()
    return _evaluator
