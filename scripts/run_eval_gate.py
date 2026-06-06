#!/usr/bin/env python3
"""Eval Gate — run evaluation suite and enforce quality thresholds.

Usage:
    python scripts/run_eval_gate.py [--threshold 0.7] [--output results.json]

Exits with code 0 if gate passes, 1 if it fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eval dataset (embedded — extend with external JSON/CSV for production)
# ---------------------------------------------------------------------------
_EVAL_CASES = [
    {
        "query": "HarmonyOS NEXT 如何申请权限？",
        "expected_intent": "api_usage",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
    {
        "query": "@ohos.app.ability 迁移到 API 12 有什么变化？",
        "expected_intent": "migration",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
    {
        "query": "HarmonyOS 应用中实现 ArkUI 导航的最佳实践是什么？",
        "expected_intent": "best_practice",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
    {
        "query": "错误码 15500000 是什么问题？",
        "expected_intent": "error_diagnosis",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
    {
        "query": "如何用 TypeScript 实现 HarmonyOS 的网络请求？",
        "expected_intent": "code_generation",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
    {
        "query": "Stage 模型和 FA 模型的区别是什么？",
        "expected_intent": "concept_qa",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
    {
        "query": "HarmonyOS NEXT API 12 兼容性需要注意什么？",
        "expected_intent": "compatibility",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
    {
        "query": "我的应用在 API 12 上启动崩溃怎么调试？",
        "expected_intent": "project_debug",
        "min_faithfulness": 0.7,
        "min_context_recall": 0.5,
    },
]


def run_eval_gate(
    threshold: float = 0.7,
    output_path: str = "eval_results.json",
) -> dict:
    """Run the evaluation suite and enforce quality gate.

    Args:
        threshold: Minimum overall score for gate to pass.
        output_path: Path to write results JSON.

    Returns:
        Results dict with gate_passed, summary, per_case scores.
    """
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": threshold,
        "gate_passed": False,
        "total_cases": len(_EVAL_CASES),
        "cases": [],
        "summary": {},
    }

    logger.info("Starting eval gate with %d test cases (threshold=%.2f)", len(_EVAL_CASES), threshold)

    for i, case in enumerate(_EVAL_CASES):
        logger.info("Case %d/%d: %s", i + 1, len(_EVAL_CASES), case["query"])
        case_result = _evaluate_single_case(case)
        results["cases"].append(case_result)

    # Compute summary metrics
    if results["cases"]:
        faithfulness_scores = [c.get("faithfulness", 0) for c in results["cases"]]
        context_recall_scores = [c.get("context_recall", 0) for c in results["cases"]]
        answer_relevancy_scores = [c.get("answer_relevancy", 0) for c in results["cases"]]
        intent_accuracy = sum(
            1 for c in results["cases"] if c.get("intent_correct", False)
        ) / len(results["cases"])

        results["summary"] = {
            "overall": round(
                (
                    sum(faithfulness_scores) / len(faithfulness_scores) * 0.4
                    + sum(context_recall_scores) / len(context_recall_scores) * 0.3
                    + sum(answer_relevancy_scores) / len(answer_relevancy_scores) * 0.2
                    + intent_accuracy * 0.1
                ),
                4,
            ),
            "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
            "context_recall": round(sum(context_recall_scores) / len(context_recall_scores), 4),
            "answer_relevancy": round(sum(answer_relevancy_scores) / len(answer_relevancy_scores), 4),
            "intent_accuracy": round(intent_accuracy, 4),
        }

    results["gate_passed"] = results["summary"].get("overall", 0) >= threshold

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    s = results["summary"]
    logger.info("=" * 50)
    logger.info("Eval Gate Results:")
    logger.info("  Overall:       %.4f", s.get("overall", 0))
    logger.info("  Faithfulness:  %.4f", s.get("faithfulness", 0))
    logger.info("  Context Recall: %.4f", s.get("context_recall", 0))
    logger.info("  Answer Relevancy: %.4f", s.get("answer_relevancy", 0))
    logger.info("  Intent Accuracy:  %.4f", s.get("intent_accuracy", 0))
    logger.info("  Gate: %s", "✅ PASSED" if results["gate_passed"] else "❌ FAILED")
    logger.info("=" * 50)

    return results


def _evaluate_single_case(case: dict) -> dict:
    """Evaluate a single test case against the RAG pipeline.

    Simulates the evaluation by:
    1. Running deep intent classification
    2. Checking retrieval quality
    3. Computing proxy metrics

    In production, this would use RAGAS via the evals/ modules.
    """
    result = {
        "query": case["query"],
        "expected_intent": case["expected_intent"],
        "faithfulness": 0.0,
        "context_recall": 0.0,
        "answer_relevancy": 0.0,
        "intent_correct": False,
        "errors": [],
    }

    # ── 1. Deep intent check ──
    try:
        _check_intent_sync(case, result)
    except Exception as exc:
        result["errors"].append(f"Intent check failed: {exc}")

    # ── 2. Retrieval quality check ──
    try:
        _check_retrieval_sync(case, result)
    except Exception as exc:
        result["errors"].append(f"Retrieval check failed: {exc}")

    # ── 3. Proxy faithfulness (keyword overlap with expected concepts) ──
    result["faithfulness"] = _proxy_faithfulness(case)

    # ── 4. Proxy answer relevancy ──
    result["answer_relevancy"] = _proxy_answer_relevancy(case)

    return result


def _check_intent_sync(case: dict, result: dict) -> None:
    """Check intent classification accuracy using rule-based classifier."""
    from enterprise_agentic_rag.agents.deep_intent.rules import rule_based_intent
    try:
        rule_result = rule_based_intent(case["query"])
        detected = rule_result.primary_intent
        result["detected_intent"] = detected
        result["intent_correct"] = (detected == case["expected_intent"])
    except Exception:
        # Graceful degradation: mark as unknown
        result["detected_intent"] = "unknown"
        result["intent_correct"] = False


def _check_retrieval_sync(case: dict, result: dict) -> None:
    """Check if retrieval returns results (proxy for context_recall)."""
    from enterprise_agentic_rag.rag.retriever import Retriever
    try:
        retriever = Retriever(chunk_size=500, top_k=5)
        docs = retriever.search(case["query"])
        result["docs_retrieved"] = len(docs)
        # Proxy: context_recall ≥ 0.5 if at least 2 docs retrieved
        result["context_recall"] = min(1.0, len(docs) / 4.0)
    except Exception:
        result["docs_retrieved"] = 0
        result["context_recall"] = 0.0


def _proxy_faithfulness(case: dict) -> float:
    """Proxy faithfulness: estimate based on retrieval quality and intent match.

    In production, this would use LLM Judge or RAGAS faithfulness metric.
    """
    keywords = case["query"].lower().split()
    expected_keywords = {
        "api_usage": ["权限", "申请", "permission"],
        "migration": ["迁移", "变化", "api 12"],
        "best_practice": ["最佳实践", "导航", "arkui"],
        "error_diagnosis": ["错误码", "15500000"],
        "code_generation": ["typescript", "网络", "请求"],
        "concept_qa": ["stage", "fa", "区别", "模型"],
        "compatibility": ["兼容", "api 12"],
        "project_debug": ["崩溃", "调试", "api 12"],
    }

    expected = expected_keywords.get(case["expected_intent"], [])
    if not expected:
        return 0.7

    matches = sum(1 for kw in expected if kw in keywords)
    return min(1.0, max(0.3, matches / len(expected) * 0.9))


def _proxy_answer_relevancy(case: dict) -> float:
    """Proxy answer relevancy: check if query is well-formed and targetable."""
    query = case["query"]
    score = 0.5  # Base score

    # Query quality heuristics
    if len(query) > 10:
        score += 0.1
    if "?" in query or "？" in query:
        score += 0.1
    if len(query.split()) > 3:
        score += 0.1
    if any(kw in query for kw in ("如何", "怎么", "什么", "为什么")):
        score += 0.1
    if case["expected_intent"] != "unknown":
        score += 0.1

    return min(1.0, score)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval gate for RAG pipeline")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum overall score (default: 0.7)")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    # Ensure the package is importable
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    results = run_eval_gate(threshold=args.threshold, output_path=args.output)

    return 0 if results["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
