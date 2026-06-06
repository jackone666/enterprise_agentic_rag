"""Merger — multi-source result fusion with intent-aware weights.

Combines results from keyword, vector, graph, and other sources
using weighted RRF (Reciprocal Rank Fusion) with intent-aware weights
from DynamicWeights.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.rag.retrieval_router import DynamicWeights

logger = logging.getLogger(__name__)


class Merger:
    """Merge and fuse results from multiple retrieval sources.

    Uses weighted RRF (Reciprocal Rank Fusion) with intent-aware weights.
    Also applies source quality boost for official docs, API references, etc.
    """

    def merge(
        self,
        source_results: dict[str, list[dict[str, Any]]],
        weights: DynamicWeights | None = None,
        k: int = 60,
        top_n: int = 15,
    ) -> list[dict[str, Any]]:
        """Merge results from multiple sources with weighted RRF."""
        if not source_results:
            return []

        weights = weights or DynamicWeights()

        # Map source names to RRF weights
        source_weight_map = {
            "keyword_search": weights.keyword_weight,
            "vector_search": weights.vector_weight,
            "graph_search": weights.graph_weight,
            "official_doc": weights.official_doc_weight,
            "api_reference": weights.api_reference_weight,
            "sample_code": weights.sample_code_weight,
            "error_diagnosis": weights.error_knowledge_weight,
            "faq": weights.faq_weight,
            "ticket": weights.ticket_weight,
            "migration_guide": weights.migration_guide_weight,
            "version_meta": weights.version_meta_weight,
        }

        # Build RRF scores
        scores: dict[str, float] = {}
        docs: dict[str, dict[str, Any]] = {}
        matched_sources: dict[str, list[str]] = {}

        for source_name, results in source_results.items():
            if not results:
                continue

            source_weight = source_weight_map.get(source_name, 0.15)

            for rank, doc in enumerate(results, start=1):
                cid = str(doc.get("id", doc.get("chunk_id", f"{source_name}_{rank}")))
                rrf_score = source_weight / (k + rank)
                scores[cid] = scores.get(cid, 0) + rrf_score

                if cid not in docs:
                    docs[cid] = dict(doc)
                matched_sources.setdefault(cid, []).append(source_name)

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build final merged results
        merged: list[dict[str, Any]] = []
        seen_ids: set[tuple[str, str]] = set()

        for cid, score in ranked[: top_n * 2]:
            doc = docs[cid]
            doc_id = str(doc.get("id", doc.get("chunk_id", cid)))
            source = str(doc.get("source", ""))

            dedup_key = (doc_id, source)
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)

            # Apply source quality boost
            final_score = self._apply_source_quality_boost(score, doc, weights)

            doc["fused_score"] = round(final_score, 4)
            doc["original_score"] = doc.get("score", 0)
            doc["score"] = round(final_score, 4)
            doc["matched_sources"] = matched_sources.get(cid, [])

            merged.append(doc)

        # Sort by final score
        merged.sort(key=lambda d: d.get("score", 0), reverse=True)

        logger.info(
            "Merger: %d sources → %d merged results",
            len(source_results),
            len(merged[:top_n]),
        )

        return merged[:top_n]

    @staticmethod
    def _apply_source_quality_boost(
        score: float, doc: dict[str, Any], weights: DynamicWeights,
    ) -> float:
        """Apply source quality boost based on document type/source."""
        doc_type = doc.get("doc_type", "")
        source = doc.get("source", "")

        boost = 1.0

        # Official documentation boost
        if doc_type in ("official_doc", "api_reference"):
            boost += weights.official_doc_weight * 0.2

        # API reference boost
        if doc_type == "api_reference" or "api_reference" in doc.get("matched_sources", []):
            boost += weights.api_reference_weight * 0.15

        # Sample code boost
        if doc_type == "sample_code" or "code" in source.lower():
            boost += weights.sample_code_weight * 0.15

        # Error knowledge base boost
        if doc_type in ("error_knowledge", "faq", "ticket"):
            boost += weights.error_knowledge_weight * 0.1

        # Migration guide boost
        if "migration" in source.lower() or doc_type == "migration_guide":
            boost += weights.migration_guide_weight * 0.2

        # Version metadata boost
        if doc_type == "version_meta" or "version" in source.lower():
            boost += weights.version_meta_weight * 0.15

        return score * boost


def merge_results(
    source_results: dict[str, list[dict[str, Any]]],
    weights: DynamicWeights | None = None,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """Convenience function for merging results."""
    merger = Merger()
    return merger.merge(source_results, weights, top_n=top_n)
