"""Graph retriever tool — unified wrapper for Neo4j knowledge graph search.

Returns both documents AND entity relationships.
Reuses existing GraphRetriever from rag/graph/.
"""

from __future__ import annotations

import time
from typing import Any

from enterprise_agentic_rag.rag.unified_schemas import UnifiedToolOutput


class GraphRetrieverTool:
    """Graph search tool wrapping Neo4j knowledge graph retrieval.

    Unlike keyword/vector, graph search also returns:
    - Entity relationships
    - Expanded query
    - Graph paths for explainability

    Input: query, filters, top_k, intent, scenario, entities
    Output: UnifiedToolOutput + entity_relations + expanded_query in metadata.
    """

    TOOL_NAME = "graph_search"

    def __init__(self) -> None:
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from enterprise_agentic_rag.rag.graph.graph_retriever import GraphRetriever

                gr = GraphRetriever()
                self._available = gr.available
            except Exception:
                self._available = False
        return self._available

    async def execute(self, **kwargs: Any) -> UnifiedToolOutput:
        """Execute graph-based knowledge retrieval."""
        t0 = time.time()
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 10)
        filters = kwargs.get("filters", {})
        intent = kwargs.get("intent", "")
        scenario = kwargs.get("scenario", "")
        entities = kwargs.get("entities", {})

        results: list[dict[str, Any]] = []
        entity_relations: list[dict[str, Any]] = []
        expanded_query: str | None = None
        error: str | None = None

        if not self.available:
            error = "Graph database unavailable"
            return UnifiedToolOutput(
                tool_name=self.TOOL_NAME,
                results=[],
                confidence=0.0,
                metadata={
                    "intent": intent,
                    "scenario": scenario,
                    "top_k": top_k,
                    "backend": "unavailable",
                    "entity_relations": [],
                    "expanded_query": None,
                },
                error=error,
                latency_ms=round((time.time() - t0) * 1000, 2),
            )

        try:
            from enterprise_agentic_rag.rag.graph.graph_retriever import GraphRetriever
            from enterprise_agentic_rag.rag.graph.query_expander import QueryExpander

            gr = GraphRetriever()

            # Build query_analysis for graph retriever
            entity_list = []
            for cat in ("apis", "components", "errors", "api_levels", "versions", "files"):
                vals = entities.get(cat, [])
                if isinstance(vals, list):
                    entity_list.extend(vals)

            query_analysis = {
                "entities": entity_list,
                "keywords": query.split(),
                "intent": intent,
                "original_query": query,
            }

            depth = 2 if intent in ("migration", "compatibility") else 1

            candidates = gr.retrieve(
                query_analysis=query_analysis,
                query=query,
                top_k=top_k,
                graph_depth=depth,
                filters=filters if filters else None,
            )

            # Convert Candidates to dicts
            for c in candidates:
                d = c.to_dict() if hasattr(c, "to_dict") else c
                results.append({
                    "id": d.get("chunk_id", ""),
                    "title": d.get("doc_id", d.get("source", "graph")),
                    "content": d.get("content", ""),
                    "source": d.get("source_path", d.get("source", "graph")),
                    "doc_type": "graph_result",
                    "score": d.get("graph_score", d.get("fused_score", 0)),
                    "metadata": d.get("metadata", {}),
                    "graph_paths": d.get("graph_paths", []),
                })

            # Extract entity relations from paths
            entity_relations = self._extract_relations(results)

            # Generate expanded query from graph results
            if results:
                try:
                    expander = QueryExpander()
                    expansion = expander.expand(query, candidates)
                    expanded_query = expansion.get("expanded_query")
                except Exception:
                    expanded_query = None

        except Exception as exc:
            error = f"Graph search failed: {exc}"

        latency_ms = (time.time() - t0) * 1000
        confidence = self._calc_confidence(results)

        return UnifiedToolOutput(
            tool_name=self.TOOL_NAME,
            results=results,
            confidence=confidence,
            metadata={
                "intent": intent,
                "scenario": scenario,
                "top_k": top_k,
                "backend": "neo4j" if self.available else "unavailable",
                "entity_relations": entity_relations,
                "expanded_query": expanded_query,
                "entity_count": len(entity_relations),
            },
            error=error,
            latency_ms=round(latency_ms, 2),
        )

    def _extract_relations(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract entity relations from graph paths."""
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for doc in results:
            paths = doc.get("graph_paths", [])
            for path in paths:
                entities = path.get("path_entities", [])
                rels = path.get("path_relations", [])
                for i in range(len(entities) - 1):
                    if i < len(rels):
                        key = (entities[i], rels[i], entities[i + 1])
                        if key not in seen:
                            seen.add(key)
                            relations.append({
                                "source": entities[i],
                                "relation": rels[i],
                                "target": entities[i + 1],
                                "path_score": path.get("path_score", 0.0),
                            })

        return relations

    @staticmethod
    def _calc_confidence(results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        count_factor = min(1.0, len(results) / 10)
        return round(avg_score * count_factor, 4)


async def graph_search(
    query: str,
    top_k: int = 10,
    intent: str = "",
    scenario: str = "",
    entities: dict[str, Any] | None = None,
    filters: dict[str, Any] | None = None,
) -> UnifiedToolOutput:
    """Convenience function for graph search."""
    tool = GraphRetrieverTool()
    return await tool.execute(
        query=query,
        top_k=top_k,
        intent=intent,
        scenario=scenario,
        entities=entities or {},
        filters=filters or {},
    )
