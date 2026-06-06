"""Query Expander — expand queries using graph retrieval results.

Used in graph_first mode:

1. GraphRetriever finds entities, paths, and evidence
2. QueryExpander extracts expansion terms from graph results
3. Expanded query is fed to keyword + vector retrievers

IMPORTANT: original_query is preserved. expanded_query is used only
for keyword + vector retrieval in the second stage.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.rag.graph.graph_schema import Candidate

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand a query using graph retrieval results.

    Extracts entity names, relation types, lifecycle methods,
    and evidence keywords from graph paths.
    """

    def expand(
        self,
        original_query: str,
        graph_candidates: list[Candidate],
    ) -> dict[str, Any]:
        """Expand a query using graph retrieval results.

        Args:
            original_query: The original user query (preserved).
            graph_candidates: GraphRetriever results with graph_paths.

        Returns:
            Dict with:
            - original_query: preserved original
            - expanded_query: expanded query string for keyword/vector retrieval
            - expansion_terms: list of extracted terms
            - graph_paths: list of serialized GraphPath dicts
        """
        if not graph_candidates:
            logger.debug("QueryExpander: no graph candidates to expand from")
            return {
                "original_query": original_query,
                "expanded_query": original_query,
                "expansion_terms": [],
                "graph_paths": [],
            }

        # Collect expansion terms from graph paths
        terms: set[str] = set()

        for cand in graph_candidates:
            for gp in cand.graph_paths:
                # Add all entity names from the path
                for name in gp.path_entities:
                    terms.add(name)

                # Add relation types as context terms
                for rel in gp.path_relations:
                    # Translate relation types to Chinese keywords
                    rel_terms = _relation_to_keywords(rel)
                    terms.update(rel_terms)

        # Also extract from candidate metadata
        for cand in graph_candidates:
            if cand.content:
                # Add key terms from candidate content
                content_terms = _extract_key_terms(cand.content, max_terms=5)
                terms.update(content_terms)

        # Remove the original query words to avoid redundancy
        query_words = set(original_query.lower().split())
        filtered_terms = [t for t in terms
                          if t.lower() not in query_words and len(t) >= 2]

        # Sort by term length (longer terms are usually more specific)
        filtered_terms.sort(key=len, reverse=True)

        # Limit expansion terms to avoid query bloat
        expansion_terms = filtered_terms[:15]

        # Build expanded query: original + expansion terms
        if expansion_terms:
            expanded_query = original_query + " " + " ".join(expansion_terms)
        else:
            expanded_query = original_query

        # Serialize graph paths for trace
        serialized_paths = []
        for cand in graph_candidates:
            for gp in cand.graph_paths:
                serialized_paths.append({
                    "path_entities": gp.path_entities,
                    "path_relations": gp.path_relations,
                    "evidence_chunk_id": gp.evidence_chunk_id,
                    "path_score": gp.path_score,
                    "path_length": gp.path_length,
                })

        logger.info("QueryExpander: expanded %d terms from %d graph candidates",
                     len(expansion_terms), len(graph_candidates))

        return {
            "original_query": original_query,
            "expanded_query": expanded_query,
            "expansion_terms": expansion_terms,
            "graph_paths": serialized_paths,
        }


# ===========================================================================
# Helpers
# ===========================================================================


def _relation_to_keywords(relation_type: str) -> list[str]:
    """Convert a Neo4j relation type to search query keywords."""
    mapping = {
        "RELATED_TO": ["相关", "关联"],
        "DEPENDS_ON": ["依赖", "需要"],
        "CALLS": ["调用", "执行"],
        "BELONGS_TO": ["属于", "模块"],
        "CAUSES": ["导致", "原因", "引发"],
        "FIXES": ["修复", "解决"],
        "PART_OF": ["包含", "组成"],
        "HAS_LIFECYCLE": ["生命周期"],
        "AFFECTS": ["影响", "波及"],
    }
    return mapping.get(relation_type, [])


def _extract_key_terms(content: str, max_terms: int = 5) -> list[str]:
    """Extract key technical terms from chunk content.

    Focuses on capitalized identifiers, API names, and class names.
    """
    import re

    terms: list[str] = []

    # CamelCase identifiers
    camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content)
    terms.extend(camel)

    # ALL_CAPS identifiers
    caps = re.findall(r'\b[A-Z_]{4,}\b', content)
    terms.extend(caps)

    # @ohos.xxx API references
    apis = re.findall(r'@ohos\.\w+(?:\.\w+)*', content)
    terms.extend(apis)

    # Deduplicate
    seen = set()
    unique = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)

    return unique[:max_terms]
