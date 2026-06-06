"""Graph Retriever — retrieve candidates from Neo4j knowledge graph.

Real Neo4j-based retrieval. No mock results.

Capabilities:
- Entity lookup by name / normalized_name
- 1-hop and 2-hop neighbor expansion
- Path extraction with relation details
- Chunk retrieval via evidence_chunk_id
- Graph scoring based on entity match + relation weight + evidence + path length
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.rag.graph.graph_indexer import get_neo4j_connection
from enterprise_agentic_rag.rag.graph.graph_schema import Candidate, GraphPath

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieve candidates from Neo4j knowledge graph.

    Connects to Neo4j via the shared connection singleton.
    Returns empty results when Neo4j is unavailable — no mock data.
    """

    def __init__(self) -> None:
        self._conn = get_neo4j_connection()
        self._data_ready: bool | None = None

    @property
    def available(self) -> bool:
        """Whether Neo4j is reachable and has indexed graph data."""
        return self._conn.available and self.data_ready

    @property
    def data_ready(self) -> bool:
        """Whether the graph contains indexed entity nodes."""
        if self._data_ready is not None:
            return self._data_ready

        driver = self._conn.driver
        if driver is None:
            self._data_ready = False
            return False

        try:
            with driver.session() as session:
                labels = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()
                if not labels or "Entity" not in labels["labels"]:
                    self._data_ready = False
                    return False
                record = session.run("MATCH (e:Entity) RETURN count(e) AS count LIMIT 1").single()
                self._data_ready = bool(record and record["count"] > 0)
        except Exception:
            self._data_ready = False
        return self._data_ready

    # ------------------------------------------------------------------
    # Main retrieve interface
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_analysis: dict[str, Any] | None = None,
        query: str = "",
        top_k: int = 10,
        graph_depth: int = 2,
        filters: dict | None = None,
    ) -> list[Candidate]:
        """Retrieve graph candidates based on query analysis entities.

        Args:
            query_analysis: Dict with ``entities``, ``keywords``, ``intent``.
            query: Raw query string (fallback if query_analysis has no entities).
            top_k: Max candidates to return.
            graph_depth: Traversal depth (1 or 2 hops).
            filters: Optional entity type filters.

        Returns:
            List of Candidate objects with graph_paths populated.
            Returns empty list if Neo4j unavailable or no entities found.
        """
        if not self.available:
            logger.debug("GraphRetriever: Neo4j unavailable — returning empty")
            return []

        query_analysis = query_analysis or {}

        # 1. Collect entity search terms
        entity_terms = self._collect_entity_terms(query_analysis, query)
        if not entity_terms:
            logger.debug("GraphRetriever: no entity terms to search")
            return []

        # 2. Find matching entities in Neo4j
        matched_entities = self._find_entities(entity_terms, filters)
        if not matched_entities:
            logger.debug("GraphRetriever: no entities matched in Neo4j for terms=%s", entity_terms)
            return []

        # 3. Expand neighbors (1 or 2 hops)
        paths = self._expand_neighbors(matched_entities, depth=graph_depth)

        # 4. Collect evidence chunks from paths
        candidates = self._build_candidates(matched_entities, paths, top_k)

        logger.info("GraphRetriever: found %d candidates from %d entities (%d paths)",
                     len(candidates), len(matched_entities), len(paths))
        return candidates

    # ------------------------------------------------------------------
    # Entity term collection
    # ------------------------------------------------------------------
    def _collect_entity_terms(
        self,
        query_analysis: dict[str, Any],
        query: str,
    ) -> list[str]:
        """Collect entity search terms from query analysis and raw query.

        Priority:
        1. query_analysis.entities (list of entity strings)
        2. query_analysis.keywords (fallback)
        3. Raw query tokenized into potential entity names
        """
        terms: list[str] = []

        # From entities in query analysis
        entities = query_analysis.get("entities", [])
        if isinstance(entities, list):
            terms.extend([str(e).strip() for e in entities if str(e).strip()])

        # From keywords as fallback
        if not terms:
            keywords = query_analysis.get("keywords", [])
            if isinstance(keywords, list):
                terms.extend([str(k).strip() for k in keywords if str(k).strip()])

        # From raw query — extract potential entity names (CamelCase, ALL_CAPS, etc.)
        if not terms and query:
            import re
            # Extract capitalized identifiers, error codes, API-like patterns
            patterns = [
                r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",       # CamelCase
                r"\b[A-Z_]{3,}\b",                          # CONSTANTS
                r"\b\d{4,10}\b",                            # Error codes
                r"\b@[\w.]+",                               # Decorators
                r"\b\w+Ability\b",                          # Ability classes
                r"\bohos\.\w+(?:\.\w+)*\b",                 # API modules
            ]
            for pat in patterns:
                matches = re.findall(pat, query)
                terms.extend(matches)

        # Deduplicate, preserve order
        seen = set()
        unique = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        return unique[:20]  # Limit to avoid overly broad search

    # ------------------------------------------------------------------
    # Entity lookup in Neo4j
    # ------------------------------------------------------------------
    def _find_entities(
        self,
        terms: list[str],
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Find matching entity nodes in Neo4j.

        Matches against name (exact) and normalized_name (fuzzy/prefix).
        """
        driver = self._conn.driver
        if driver is None:
            return []

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        try:
            with driver.session() as session:
                for term in terms:
                    term_lower = term.lower().strip()
                    if len(term_lower) < 2:
                        continue

                    # Multi-strategy lookup
                    queries = [
                        # Exact match
                        ("MATCH (e:Entity) WHERE e.normalized_name = $term RETURN e", {"term": term_lower}),
                        # Prefix match
                        ("MATCH (e:Entity) WHERE e.normalized_name STARTS WITH $term RETURN e LIMIT 5", {"term": term_lower}),
                        # Contains match (for longer terms)
                        ("MATCH (e:Entity) WHERE e.normalized_name CONTAINS $term RETURN e LIMIT 5", {"term": term_lower}),
                    ]

                    for cypher, params in queries:
                        try:
                            records = session.run(cypher, **params)
                            for record in records:
                                node = record["e"]
                                node_data = dict(node.items())
                                eid = node_data.get("entity_id", "")
                                if eid not in seen_ids:
                                    seen_ids.add(eid)
                                    # Calculate match score
                                    name = node_data.get("name", "")
                                    match_score = self._entity_match_score(term, name)
                                    node_data["_match_score"] = match_score
                                    results.append(node_data)
                        except Exception:
                            continue

            # Sort by match score, best first
            results.sort(key=lambda x: x.get("_match_score", 0), reverse=True)

        except Exception as exc:
            logger.warning("Entity lookup failed: %s", exc)

        return results[:30]

    @staticmethod
    def _entity_match_score(query_term: str, entity_name: str) -> float:
        """Calculate how well a query term matches an entity name."""
        qt = query_term.lower().strip()
        en = entity_name.lower().strip()
        if qt == en:
            return 1.0
        if en.startswith(qt):
            return 0.8
        if qt in en:
            return 0.6
        return 0.3

    # ------------------------------------------------------------------
    # Neighbor expansion
    # ------------------------------------------------------------------
    def _expand_neighbors(
        self,
        entities: list[dict[str, Any]],
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Expand from matched entities to find connected entities and paths.

        Returns list of path dicts with:
        - path_entities: list of entity names
        - path_relations: list of relation types
        - evidence_chunk_id
        - relation_weight
        - path_length
        """
        driver = self._conn.driver
        if driver is None:
            return []

        depth = min(depth, 2)  # Safety limit
        paths: list[dict[str, Any]] = []

        try:
            with driver.session() as session:
                for entity in entities[:15]:  # Limit to top matches
                    eid = entity.get("entity_id", "")
                    if not eid:
                        continue

                    # 1-hop: direct neighbors
                    cypher_1hop = """
                        MATCH (e:Entity {entity_id: $eid})-[r]-(neighbor:Entity)
                        RETURN e.name AS source, type(r) AS relation, r.weight AS weight,
                               r.evidence_chunk_id AS evidence, neighbor.name AS target,
                               neighbor.entity_id AS target_id, neighbor.type AS target_type,
                               1 AS path_length
                        LIMIT 20
                    """
                    try:
                        records = session.run(cypher_1hop, eid=eid)
                        for rec in records:
                            paths.append({
                                "path_entities": [rec["source"], rec["target"]],
                                "path_relations": [rec["relation"]],
                                "evidence_chunk_id": rec["evidence"] or "",
                                "relation_weight": rec["weight"] or 1.0,
                                "path_length": 1,
                                "target_entity_id": rec["target_id"],
                                "target_type": rec["target_type"],
                            })
                    except Exception:
                        continue

                    # 2-hop: expand further
                    if depth >= 2:
                        cypher_2hop = """
                            MATCH (e:Entity {entity_id: $eid})-[r1]-(mid:Entity)-[r2]-(far:Entity)
                            WHERE far.entity_id <> $eid
                            RETURN e.name AS source, type(r1) AS rel1, r1.weight AS w1,
                                   r1.evidence_chunk_id AS ev1, mid.name AS middle,
                                   type(r2) AS rel2, r2.weight AS w2,
                                   r2.evidence_chunk_id AS ev2, far.name AS target,
                                   far.entity_id AS target_id, far.type AS target_type,
                                   2 AS path_length
                            LIMIT 15
                        """
                        try:
                            records = session.run(cypher_2hop, eid=eid)
                            for rec in records:
                                evidence = rec["ev1"] or rec["ev2"] or ""
                                weight = (rec["w1"] or 1.0) * (rec["w2"] or 1.0)
                                paths.append({
                                    "path_entities": [rec["source"], rec["middle"], rec["target"]],
                                    "path_relations": [rec["rel1"], rec["rel2"]],
                                    "evidence_chunk_id": evidence,
                                    "relation_weight": round(weight, 2),
                                    "path_length": 2,
                                    "target_entity_id": rec["target_id"],
                                    "target_type": rec["target_type"],
                                })
                        except Exception:
                            continue

        except Exception as exc:
            logger.warning("Neighbor expansion failed: %s", exc)

        # Deduplicate by path
        seen = set()
        unique = []
        for p in paths:
            key = "|".join(p["path_entities"] + p["path_relations"])
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique

    # ------------------------------------------------------------------
    # Chunk retrieval from evidence
    # ------------------------------------------------------------------
    def _get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Retrieve chunk content from Neo4j by chunk_id."""
        driver = self._conn.driver
        if driver is None or not chunk_ids:
            return {}

        chunks: dict[str, dict[str, Any]] = {}
        try:
            with driver.session() as session:
                # Batch lookup in chunks of 50
                for i in range(0, len(chunk_ids), 50):
                    batch = chunk_ids[i:i + 50]
                    cypher = """
                        MATCH (c:Chunk)
                        WHERE c.chunk_id IN $ids
                        RETURN c.chunk_id AS chunk_id, c.content AS content,
                               c.doc_id AS doc_id, c.chunk_index AS chunk_index
                    """
                    records = session.run(cypher, ids=batch)
                    for rec in records:
                        chunks[rec["chunk_id"]] = {
                            "chunk_id": rec["chunk_id"],
                            "content": rec["content"] or "",
                            "doc_id": rec["doc_id"] or "",
                            "chunk_index": rec["chunk_index"] or 0,
                        }
        except Exception as exc:
            logger.warning("Chunk lookup failed: %s", exc)

        return chunks

    # ------------------------------------------------------------------
    # Build candidates
    # ------------------------------------------------------------------
    def _build_candidates(
        self,
        entities: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        top_k: int,
    ) -> list[Candidate]:
        """Build Candidate objects from entities and paths.

        Groups paths by evidence_chunk_id to avoid duplicate chunks.
        Calculates graph_score using the scoring formula.
        """
        # Collect all evidence chunk IDs
        evidence_ids = [p["evidence_chunk_id"] for p in paths if p["evidence_chunk_id"]]
        evidence_ids = list(set(evidence_ids))

        # Retrieve chunk content from Neo4j
        chunks_map = self._get_chunks_by_ids(evidence_ids)

        # If no evidence chunks, create candidates from entity metadata
        if not chunks_map and entities:
            return self._build_entity_only_candidates(entities, paths, top_k)

        # Group paths by chunk_id
        paths_by_chunk: dict[str, list[dict]] = {}
        for p in paths:
            cid = p["evidence_chunk_id"]
            if cid and cid in chunks_map:
                paths_by_chunk.setdefault(cid, []).append(p)

        # Build candidates
        candidates: list[Candidate] = []
        for chunk_id, chunk_paths in paths_by_chunk.items():
            ch = chunks_map.get(chunk_id, {})
            content = ch.get("content", "")
            doc_id = ch.get("doc_id", "")

            # Calculate graph score
            entity_match = self._calc_entity_match_score(entities, chunk_paths)
            relation_weight = self._calc_relation_weight(chunk_paths)
            evidence_score = 1.0 if content else 0.0
            path_length_score = self._calc_path_length_penalty(chunk_paths)

            graph_score = (
                entity_match * 0.4
                + relation_weight * 0.3
                + evidence_score * 0.2
                + path_length_score * 0.1
            )

            # Build GraphPath objects
            graph_paths = []
            for p in chunk_paths[:5]:  # Limit paths per candidate
                graph_paths.append(GraphPath(
                    path_entities=p.get("path_entities", []),
                    path_relations=p.get("path_relations", []),
                    evidence_chunk_id=p.get("evidence_chunk_id", chunk_id),
                    relation_weight=p.get("relation_weight", 1.0),
                    path_score=graph_score,
                    path_length=p.get("path_length", 1),
                ))

            candidates.append(Candidate(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                source_path=doc_id,
                graph_score=round(graph_score, 4),
                raw_scores={"graph": graph_score},
                matched_sources=["graph"],
                graph_paths=graph_paths,
                metadata={
                    "entity_count": len(entities),
                    "path_count": len(chunk_paths),
                },
            ))

        # Sort by graph_score descending
        candidates.sort(key=lambda c: c.graph_score, reverse=True)
        return candidates[:top_k]

    def _build_entity_only_candidates(
        self,
        entities: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        top_k: int,
    ) -> list[Candidate]:
        """Build minimal candidates when no evidence chunks are available.

        Uses entity metadata as candidate content.
        """
        candidates: list[Candidate] = []
        for entity in entities[:top_k]:
            name = entity.get("name", "")
            etype = entity.get("type", "")
            eid = entity.get("entity_id", "")
            match_score = entity.get("_match_score", 0.5)

            # Build graph paths from this entity's connections
            entity_paths = [p for p in paths
                            if p.get("path_entities") and p["path_entities"][0] == name]

            graph_paths = []
            for p in entity_paths[:5]:
                graph_paths.append(GraphPath(
                    path_entities=p.get("path_entities", []),
                    path_relations=p.get("path_relations", []),
                    evidence_chunk_id=p.get("evidence_chunk_id", ""),
                    relation_weight=p.get("relation_weight", 1.0),
                    path_score=match_score,
                    path_length=p.get("path_length", 1),
                ))

            candidates.append(Candidate(
                chunk_id=f"entity:{eid}",
                doc_id="graph",
                content=f"[{etype}] {name}: 知识图谱实体",
                source_path="graph",
                graph_score=match_score,
                raw_scores={"graph": match_score},
                matched_sources=["graph"],
                graph_paths=graph_paths,
            ))

        candidates.sort(key=lambda c: c.graph_score, reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_entity_match_score(
        entities: list[dict[str, Any]],
        paths: list[dict[str, Any]],
    ) -> float:
        """Calculate how well paths cover the matched entities."""
        if not entities:
            return 0.0

        entity_names = {e.get("name", "").lower() for e in entities}
        path_entities: set[str] = set()
        for p in paths:
            for name in p.get("path_entities", []):
                path_entities.add(name.lower())

        overlap = entity_names & path_entities
        return len(overlap) / max(len(entity_names), 1)

    @staticmethod
    def _calc_relation_weight(paths: list[dict[str, Any]]) -> float:
        """Calculate average relation weight across paths."""
        weights = [p.get("relation_weight", 1.0) for p in paths]
        if not weights:
            return 0.0
        return sum(weights) / len(weights)

    @staticmethod
    def _calc_path_length_penalty(paths: list[dict[str, Any]]) -> float:
        """Shorter paths get higher scores. 1-hop = 1.0, 2-hop = 0.5."""
        lengths = [p.get("path_length", 2) for p in paths]
        if not lengths:
            return 0.0
        avg_len = sum(lengths) / len(lengths)
        return 1.0 / avg_len  # 1-hop → 1.0, 2-hop → 0.5
