"""RAG evaluation metrics — retrieval quality assessment.

Metrics:
- hit@k — was at least 1 relevant doc in top-k?
- recall@k — proportion of expected sources found in top-k
- MRR (Mean Reciprocal Rank) — 1 / rank_of_first_relevant
- avg_retrieval_score — mean score of retrieved docs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGMetrics:
    """Result of a RAG evaluation run."""

    hit_at_k: float = 0.0       # hit@k averaged across queries
    recall_at_k: float = 0.0     # recall@k averaged across queries
    mrr: float = 0.0             # MRR averaged across queries
    avg_retrieval_score: float = 0.0  # mean retrieval score
    total_queries: int = 0
    queries_with_hits: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)


class RAGEvaluator:
    """Compute retrieval quality metrics."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        queries: list[str],
        retrieved_docs_list: list[list[dict[str, Any]]],
        expected_sources_list: list[list[str]],
    ) -> RAGMetrics:
        """Run retrieval evaluation across all queries.

        Args:
            queries: List of query strings.
            retrieved_docs_list: For each query, the top-k retrieved docs.
            expected_sources_list: For each query, the expected source filenames.

        Returns:
            Aggregated :class:`RAGMetrics`.
        """
        hit_count = 0
        total_recall = 0.0
        total_mrr = 0.0
        total_score = 0.0
        per_query: list[dict[str, Any]] = []

        for i, (query, docs, expected) in enumerate(
            zip(queries, retrieved_docs_list, expected_sources_list)
        ):
            sources = [d.get("source", "") for d in docs[: self.k]]
            scores = [d.get("score", 0.0) for d in docs[: self.k]]

            # hit@k
            hit = any(exp in sources for exp in expected) if expected else len(sources) > 0
            if hit:
                hit_count += 1

            # recall@k
            if expected:
                found = sum(1 for exp in expected if exp in sources)
                recall = found / len(expected)
            else:
                recall = 1.0 if sources else 0.0
            total_recall += recall

            # MRR
            mrr = 0.0
            for rank, src in enumerate(sources, start=1):
                if expected and src in expected:
                    mrr = 1.0 / rank
                    break
            total_mrr += mrr

            # avg score
            avg_score = sum(scores) / len(scores) if scores else 0.0
            total_score += avg_score

            per_query.append({
                "query": query[:100],
                "hit": hit,
                "recall": round(recall, 4),
                "mrr": round(mrr, 4),
                "avg_score": round(avg_score, 4),
                "retrieved_sources": sources,
                "expected_sources": expected,
            })

        n = len(queries) or 1
        return RAGMetrics(
            hit_at_k=round(hit_count / n, 4),
            recall_at_k=round(total_recall / n, 4),
            mrr=round(total_mrr / n, 4),
            avg_retrieval_score=round(total_score / n, 4),
            total_queries=len(queries),
            queries_with_hits=hit_count,
            per_query=per_query,
        )

    # ------------------------------------------------------------------
    # Single-query helpers
    # ------------------------------------------------------------------
    @staticmethod
    def hit_at_k_single(
        retrieved_sources: list[str],
        expected_sources: list[str],
        k: int = 5,
    ) -> bool:
        """Check if any expected source is in top-k retrieved sources."""
        return any(exp in retrieved_sources[:k] for exp in expected_sources)

    @staticmethod
    def recall_at_k_single(
        retrieved_sources: list[str],
        expected_sources: list[str],
    ) -> float:
        """Proportion of expected sources found in retrieved list."""
        if not expected_sources:
            return 1.0 if retrieved_sources else 0.0
        found = sum(1 for exp in expected_sources if exp in retrieved_sources)
        return found / len(expected_sources)

    @staticmethod
    def mrr_single(
        retrieved_sources: list[str],
        expected_sources: list[str],
    ) -> float:
        """Reciprocal rank of the first relevant document."""
        for rank, src in enumerate(retrieved_sources, start=1):
            if src in expected_sources:
                return 1.0 / rank
        return 0.0
