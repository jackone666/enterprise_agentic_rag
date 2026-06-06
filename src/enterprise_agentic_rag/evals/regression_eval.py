"""Regression evaluation — run full workflow against test cases.

Usage:
    python -m enterprise_agentic_rag.evals.regression_eval

Outputs:
    - Pass rate per metric (intent, retrieval, answer, overall)
    - Failed case details
    - Failed cases written to data/eval/failed_cases.jsonl
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

from enterprise_agentic_rag.evals.answer_eval import AnswerEvaluator, AnswerMetrics
from enterprise_agentic_rag.evals.dataset import EvalCase, EvalDataset, FailedCase
from enterprise_agentic_rag.evals.rag_eval import RAGEvaluator, RAGMetrics
from enterprise_agentic_rag.graph.workflow import build_workflow


@dataclass
class RegressionResult:
    """Full results from a regression evaluation run."""

    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    prompt_version: str = "v1"

    # Sub-metrics
    intent_accuracy: float = 0.0
    keyword_match_rate: float = 0.0

    # RAG metrics
    rag_metrics: RAGMetrics | None = None

    # Answer metrics
    answer_metrics: AnswerMetrics | None = None

    # Failed cases detail
    failed_cases: list[dict[str, Any]] = field(default_factory=list)


class RegressionEvaluator:
    """Runs the full LangGraph pipeline against eval cases and reports results."""

    def __init__(self, dataset: EvalDataset | None = None) -> None:
        self.dataset = dataset or EvalDataset()
        self.rag_evaluator = RAGEvaluator(k=5)
        self.answer_evaluator = AnswerEvaluator()
        self._workflow = build_workflow()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    async def run(self) -> RegressionResult:
        """Execute the full regression suite."""
        cases = self.dataset.load_cases()
        if not cases:
            print("No eval cases found.")
            return RegressionResult()

        # Run each case through the workflow
        results: list[tuple[EvalCase, dict[str, Any]]] = []
        for case in cases:
            state = {
                "query": case.query,
                "user_id": _role_to_user_id(case.user_role),
                "session_id": f"eval_{hash(case.query) % 10000}",
            }
            output = await self._workflow.ainvoke(state)
            results.append((case, output))

        # ---- Evaluate ----
        queries = [c.query for c in cases]
        intents = [r.get("intent", "") for c, r in results]
        answers = [r.get("final_answer", "") for c, r in results]
        docs_lists = [r.get("retrieved_docs", []) for c, r in results]
        expected_sources_lists = [c.expected_sources for c in cases]
        expected_keywords_lists = [c.expected_answer_keywords for c in cases]

        # Intent accuracy
        intent_correct = 0
        for case, (_, result) in zip(cases, results):
            if result.get("intent", "") == case.expected_intent:
                intent_correct += 1
        intent_acc = round(intent_correct / len(cases), 4) if cases else 0.0

        # Keyword match rate
        kw_total = 0
        kw_found = 0
        for case, (_, result) in zip(cases, results):
            answer_lower = result.get("final_answer", "").lower()
            for kw in case.expected_answer_keywords:
                kw_total += 1
                if kw.lower() in answer_lower:
                    kw_found += 1
        kw_rate = round(kw_found / kw_total, 4) if kw_total else 0.0

        # RAG metrics
        rag_metrics = self.rag_evaluator.evaluate(
            queries, docs_lists, expected_sources_lists,
        )

        # Answer metrics
        answer_metrics = self.answer_evaluator.evaluate(
            queries, answers, docs_lists, expected_keywords_lists,
        )

        # ---- Identify failures ----
        failed_cases: list[dict[str, Any]] = []
        passed_count = 0
        for i, (case, (_, result)) in enumerate(zip(cases, results)):
            intent_ok = result.get("intent", "") == case.expected_intent
            keywords_ok = _check_keywords(
                result.get("final_answer", ""), case.expected_answer_keywords
            )
            verified_ok = result.get("verified", False)

            if intent_ok and keywords_ok and verified_ok:
                passed_count += 1
            else:
                reasons = []
                if not intent_ok:
                    reasons.append(
                        f"intent mismatch: got {result.get('intent')}, expected {case.expected_intent}"
                    )
                if not keywords_ok:
                    reasons.append("missing expected keywords")
                if not verified_ok:
                    reasons.append(
                        f"verification failed: {result.get('verification_reason', '')}"
                    )
                failed_cases.append({
                    "query": case.query,
                    "expected_intent": case.expected_intent,
                    "actual_intent": result.get("intent", ""),
                    "expected_keywords": case.expected_answer_keywords,
                    "actual_answer": result.get("final_answer", "")[:300],
                    "failure_reasons": reasons,
                    "difficulty": case.difficulty,
                })

        # ---- Save failed cases to file ----
        for fc in failed_cases:
            self.dataset.save_failed_case(FailedCase(
                query=fc["query"],
                intent=fc["actual_intent"],
                final_answer=fc["actual_answer"],
                fallback_reason="; ".join(fc["failure_reasons"]),
                source="regression",
                metadata={"difficulty": fc["difficulty"]},
            ))

        pass_rate = round(passed_count / len(cases), 4) if cases else 0.0

        return RegressionResult(
            total_cases=len(cases),
            passed=passed_count,
            failed=len(failed_cases),
            pass_rate=pass_rate,
            intent_accuracy=intent_acc,
            keyword_match_rate=kw_rate,
            rag_metrics=rag_metrics,
            answer_metrics=answer_metrics,
            failed_cases=failed_cases,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _role_to_user_id(role: str) -> str:
    mapping = {"admin": "u001", "developer": "u002", "basic": "u003"}
    return mapping.get(role, "u003")


def _check_keywords(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run regression eval from the command line."""
    print("=" * 60)
    print("Enterprise Agentic RAG — Regression Evaluation")
    print("=" * 60)

    evaluator = RegressionEvaluator()

    async def _run() -> None:
        result = await evaluator.run()
        print(f"\n总用例数: {result.total_cases}")
        print(f"通过: {result.passed}")
        print(f"失败: {result.failed}")
        print(f"通过率: {result.pass_rate:.1%}")
        print(f"意图准确率: {result.intent_accuracy:.1%}")
        print(f"关键词命中率: {result.keyword_match_rate:.1%}")

        if result.rag_metrics:
            rm = result.rag_metrics
            print(f"\n--- RAG 指标 ---")
            print(f"hit@{evaluator.rag_evaluator.k}: {rm.hit_at_k:.1%}")
            print(f"recall@{evaluator.rag_evaluator.k}: {rm.recall_at_k:.1%}")
            print(f"MRR: {rm.mrr:.4f}")
            print(f"平均检索分: {rm.avg_retrieval_score:.4f}")

        if result.answer_metrics:
            am = result.answer_metrics
            print(f"\n--- 答案指标 ---")
            print(f"引用率: {am.citation_present_rate:.1%}")
            print(f"依据性: {am.groundedness_rate:.1%}")
            print(f"相关性: {am.answer_relevance_rate:.1%}")
            print(f"拒绝正确率: {am.refusal_correctness_rate:.1%}")
            print(f"综合分: {am.overall_score:.4f}")

        if result.failed_cases:
            print(f"\n--- 失败用例 ({len(result.failed_cases)}) ---")
            for fc in result.failed_cases:
                print(f"  ❌ {fc['query'][:60]}...")
                print(f"     原因: {'; '.join(fc['failure_reasons'])}")
        else:
            print("\n🎉 全部通过！")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
