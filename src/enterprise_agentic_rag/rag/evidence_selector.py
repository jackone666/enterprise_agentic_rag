"""Evidence selector — selects most relevant evidence chunks for final answer.

After merge and rerank, EvidenceSelector picks the most informative
chunks based on:
1. Relevance score
2. Factual density
3. Source authority
4. Content completeness
"""

from __future__ import annotations

from typing import Any


class EvidenceSelector:
    """Select the best evidence chunks for answer generation.

    Ensures selected evidence is:
    - Relevant (high score)
    - Diverse (covers multiple aspects)
    - Authoritative (prefers official/official docs)
    - Complete (enough context to answer)
    """

    def select(
        self,
        documents: list[dict[str, Any]],
        primary_intent: str = "",
        max_chunks: int = 5,
        min_score: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Select evidence chunks from reranked documents.

        Args:
            documents: Reranked document list.
            primary_intent: Detected primary intent.
            max_chunks: Maximum evidence chunks to select.
            min_score: Minimum relevance threshold.

        Returns:
            Selected evidence chunks with metadata.
        """
        if not documents:
            return []

        # Stage 1: Filter by score
        candidates = [d for d in documents if d.get("score", 0) >= min_score]
        if not candidates:
            candidates = documents[:max_chunks]

        # Stage 2: Score each candidate on multiple dimensions
        scored = []
        for doc in candidates:
            evidence_score = self._score_evidence(doc, primary_intent)
            doc["evidence_score"] = evidence_score
            scored.append(doc)

        # Stage 3: Select top evidence chunks with diversity
        selected = self._select_diverse(scored, max_chunks, primary_intent)

        # Annotate selected with evidence metadata
        for i, doc in enumerate(selected):
            doc["evidence_index"] = i + 1
            doc["selected_as_evidence"] = True

        return selected

    def _score_evidence(self, doc: dict[str, Any], intent: str) -> float:
        """Score a document chunk as evidence.

        Components:
        - relevance: base score (0.0 - 0.5)
        - factual_density: content has useful info (0.0 - 0.2)
        - source_authority: official/official docs preferred (0.0 - 0.2)
        - completeness: enough context to be useful (0.0 - 0.1)
        """
        score = 0.0

        # Relevance (0.0 - 0.5)
        base_score = float(doc.get("score", doc.get("rerank_score", 0)))
        score += min(0.5, base_score * 0.5)

        # Factual density (0.0 - 0.2)
        content = doc.get("content", "")
        if len(content) > 100:
            # Check for code examples, API references, error codes
            has_code = "```" in content
            has_api = "@ohos" in content or "API" in content
            has_error = "error" in content.lower() or "错误" in content
            if has_code or has_api or has_error:
                score += 0.15
            # Substantial content
            if len(content) > 500:
                score += 0.05

        # Source authority (0.0 - 0.2)
        source = str(doc.get("source", ""))
        doc_type = str(doc.get("doc_type", ""))
        if "official" in source.lower() or doc_type == "official_doc":
            score += 0.2
        elif "api" in source.lower() or doc_type == "api_reference":
            score += 0.15
        elif "error" in source.lower() or doc_type == "error_knowledge":
            score += 0.1
        elif doc_type == "sample_code":
            score += 0.12

        # Completeness (0.0 - 0.1)
        if len(content) > 200:
            score += 0.05
        if len(content) > 500:
            score += 0.05

        return round(score, 4)

    def _select_diverse(
        self,
        scored: list[dict[str, Any]],
        max_chunks: int,
        intent: str,
    ) -> list[dict[str, Any]]:
        """Select diverse evidence chunks.

        Ensures:
        - Mix of sources (not all from the same document)
        - Priority for official docs for concept_qa/compatibility
        - Priority for code examples for code_generation/api_usage
        - Priority for error knowledge for error_diagnosis
        """
        # Sort by evidence score
        scored.sort(key=lambda d: d.get("evidence_score", 0), reverse=True)

        selected: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        seen_titles: set[str] = set()

        # Intent-based priority buckets
        high_priority: list[dict[str, Any]] = []
        medium_priority: list[dict[str, Any]] = []
        low_priority: list[dict[str, Any]] = []

        for doc in scored:
            source = str(doc.get("source", ""))
            doc_type = str(doc.get("doc_type", ""))

            if intent == "code_generation" and ("code" in source.lower() or "sample" in doc_type):
                high_priority.append(doc)
            elif intent == "error_diagnosis" and ("error" in source.lower() or "faq" in doc_type):
                high_priority.append(doc)
            elif intent == "migration" and ("migration" in source.lower() or "migration" in doc_type):
                high_priority.append(doc)
            elif doc_type in ("official_doc", "api_reference"):
                high_priority.append(doc)
            elif doc_type == "sample_code":
                medium_priority.append(doc)
            else:
                low_priority.append(doc)

        # Select from high priority first, then medium, then low
        for priority_list in [high_priority, medium_priority, low_priority]:
            for doc in priority_list:
                if len(selected) >= max_chunks:
                    break
                source = str(doc.get("source", ""))
                title = str(doc.get("title", ""))
                # Allow same source if different title (different chunks)
                if source in seen_sources and title in seen_titles:
                    continue
                seen_sources.add(source)
                seen_titles.add(title)
                selected.append(doc)

        return selected[:max_chunks]


def select_evidence(
    documents: list[dict[str, Any]],
    primary_intent: str = "",
    max_chunks: int = 5,
) -> list[dict[str, Any]]:
    """Convenience function for evidence selection."""
    selector = EvidenceSelector()
    return selector.select(documents, primary_intent, max_chunks)
