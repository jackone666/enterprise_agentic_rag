"""Answer evaluation metrics — generation quality assessment.

Metrics:
- citation_present — does the answer contain source citations?
- groundedness — does answer reference retrieved docs?
- answer_relevance — does answer address the query?
- refusal_correctness — when there are no docs, is the answer a proper refusal?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnswerMetrics:
    """Result of an answer evaluation run."""

    citation_present_rate: float = 0.0
    groundedness_rate: float = 0.0
    answer_relevance_rate: float = 0.0
    refusal_correctness_rate: float = 0.0
    overall_score: float = 0.0          # mean of the four rates
    total_queries: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)


class AnswerEvaluator:
    """Compute generation quality metrics."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        queries: list[str],
        answers: list[str],
        retrieved_docs_list: list[list[dict[str, Any]]],
        expected_keywords_list: list[list[str]],
    ) -> AnswerMetrics:
        """Run answer evaluation across all queries.

        Args:
            queries: Original user queries.
            answers: Generated final answers.
            retrieved_docs_list: Retrieved documents per query (may be empty).
            expected_keywords_list: Expected keywords per query.

        Returns:
            Aggregated :class:`AnswerMetrics`.
        """
        total_citation = 0
        total_grounded = 0
        total_relevance = 0
        total_refusal_correct = 0
        per_query: list[dict[str, Any]] = []

        for i, (query, answer, docs, keywords) in enumerate(
            zip(queries, answers, retrieved_docs_list, expected_keywords_list)
        ):
            cit = self._citation_present(answer)
            grd = self._groundedness(answer, docs)
            rel = self._answer_relevance(answer, keywords)
            ref = self._refusal_correctness(answer, docs)

            total_citation += cit
            total_grounded += grd
            total_relevance += rel
            total_refusal_correct += ref

            per_query.append({
                "query": query[:100],
                "citation_present": cit,
                "groundedness": grd,
                "answer_relevance": rel,
                "refusal_correctness": ref,
                "answer_snippet": answer[:200],
            })

        n = len(queries) or 1
        rates = {
            "citation_present_rate": round(total_citation / n, 4),
            "groundedness_rate": round(total_grounded / n, 4),
            "answer_relevance_rate": round(total_relevance / n, 4),
            "refusal_correctness_rate": round(total_refusal_correct / n, 4),
        }
        overall = round(sum(rates.values()) / len(rates), 4)

        return AnswerMetrics(
            citation_present_rate=rates["citation_present_rate"],
            groundedness_rate=rates["groundedness_rate"],
            answer_relevance_rate=rates["answer_relevance_rate"],
            refusal_correctness_rate=rates["refusal_correctness_rate"],
            overall_score=overall,
            total_queries=len(queries),
            per_query=per_query,
        )

    # ------------------------------------------------------------------
    # Individual metric checks
    # ------------------------------------------------------------------
    @staticmethod
    def _citation_present(answer: str) -> float:
        """1.0 if the answer contains citation markers like [1], 参考来源, etc."""
        markers = ["[1]", "[2]", "参考来源", "引用", "citation"]
        return 1.0 if any(m in answer for m in markers) else 0.0

    @staticmethod
    def _groundedness(
        answer: str,
        docs: list[dict[str, Any]],
    ) -> float:
        """Heuristic: if docs exist and answer is non-empty → grounded.

        In production this would use NLI or LLM-as-judge.  Our mock version
        checks that the answer contains at least some tokens from docs.
        """
        if not docs:
            return 0.0  # no docs → can't be grounded
        if not answer.strip():
            return 0.0

        # Check token overlap between answer and all docs
        answer_tokens = set(answer.lower().split())
        doc_tokens: set[str] = set()
        for d in docs:
            doc_tokens.update(d.get("content", "").lower().split())

        if not doc_tokens:
            return 0.0

        overlap = len(answer_tokens & doc_tokens)
        # If at least 3 tokens overlap, consider it grounded
        return 1.0 if overlap >= 3 else (overlap / max(3, 1))

    @staticmethod
    def _answer_relevance(answer: str, expected_keywords: list[str]) -> float:
        """Proportion of expected keywords found in the answer."""
        if not expected_keywords:
            return 1.0
        answer_lower = answer.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return found / len(expected_keywords)

    @staticmethod
    def _refusal_correctness(
        answer: str,
        docs: list[dict[str, Any]],
    ) -> float:
        """If no docs were retrieved, the answer should be a proper refusal.

        A proper refusal contains phrases like "抱歉", "没有找到", "知识库".
        When docs exist, refusal correctness is trivially 1.0 (no refusal needed).
        """
        if docs:
            return 1.0  # Not a refusal scenario — trivially correct

        refusal_phrases = ["抱歉", "没有找到", "知识库", "信息不足", "无法"]
        return 1.0 if any(p in answer for p in refusal_phrases) else 0.0
