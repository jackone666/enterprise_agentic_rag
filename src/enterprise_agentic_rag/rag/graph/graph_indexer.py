"""Graph indexer — write entities, relations, documents, chunks into Neo4j.

Handles:
1. Neo4j connection management
2. Constraint & index creation
3. Bulk entity/relation writing
4. Graph rebuild from existing chunks/documents

Gracefully degrades when Neo4j is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from enterprise_agentic_rag.config.settings import get_settings
from enterprise_agentic_rag.rag.graph.entity_extractor import Entity, extract_entities_from_chunks
from enterprise_agentic_rag.rag.graph.relation_extractor import Relation, extract_relations_from_chunks

logger = logging.getLogger(__name__)


@dataclass
class GraphIndexReport:
    """Report from a graph indexing run."""

    total_docs: int = 0
    total_chunks: int = 0
    entity_count: int = 0
    relation_count: int = 0
    nodes_created: int = 0
    relations_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.total_chunks > 0 and self.entity_count > 0


class Neo4jConnection:
    """Lazy Neo4j connection manager.

    Only connects when needed. Gracefully handles unavailability.
    """

    def __init__(self) -> None:
        self._driver: Any = None
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Check if Neo4j is reachable."""
        if self._available is None:
            self._available = self._check_health()
        return self._available

    @property
    def driver(self):
        """Lazily initialised Neo4j driver."""
        if self._driver is None and self.available:
            try:
                from neo4j import GraphDatabase
                settings = get_settings()
                self._driver = GraphDatabase.driver(
                    settings.neo4j.uri,
                    auth=(settings.neo4j.user, settings.neo4j.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=10,
                    connection_acquisition_timeout=30,
                )
                # Verify connection
                self._driver.verify_connectivity()
                logger.info("Neo4j connected at %s", settings.neo4j.uri)
            except ImportError:
                logger.warning("neo4j package not installed — graph RAG unavailable")
                self._available = False
                self._driver = None
            except Exception as exc:
                logger.warning("Neo4j connection failed: %s", exc)
                self._available = False
                self._driver = None
        return self._driver

    def _check_health(self) -> bool:
        """Quick connectivity check without full driver init."""
        try:
            import socket
            settings = get_settings()
            # Parse host:port from bolt URI
            uri = settings.neo4j.uri
            host = uri.replace("bolt://", "").replace("neo4j://", "").split(":")[0]
            port = 7687
            if ":" in uri.split("://")[-1]:
                port = int(uri.split("://")[-1].split(":")[-1])
            sock = socket.create_connection((host, port), timeout=2.0)
            sock.close()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            self._available = None


# ===========================================================================
# Global connection singleton
# ===========================================================================

_neo4j_conn: Neo4jConnection | None = None


def get_neo4j_connection() -> Neo4jConnection:
    """Get or create the Neo4j connection singleton."""
    global _neo4j_conn
    if _neo4j_conn is None:
        _neo4j_conn = Neo4jConnection()
    return _neo4j_conn


# ===========================================================================
# Graph indexer
# ===========================================================================


class GraphIndexer:
    """Writes entities, documents, chunks, and relations into Neo4j."""

    def __init__(self, connection: Neo4jConnection | None = None) -> None:
        self._conn = connection or get_neo4j_connection()

    @property
    def available(self) -> bool:
        return self._conn.available

    # ------------------------------------------------------------------
    # Constraint & index creation
    # ------------------------------------------------------------------
    def create_constraints(self) -> bool:
        """Create Neo4j uniqueness constraints and indexes.

        Must be called before any data ingestion.
        Safe to call multiple times — constraints are idempotent.
        """
        driver = self._conn.driver
        if driver is None:
            logger.warning("Neo4j unavailable — skipping constraint creation")
            return False

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
        ]

        try:
            with driver.session() as session:
                for stmt in constraints:
                    session.run(stmt)
                for stmt in indexes:
                    session.run(stmt)
            logger.info("Neo4j constraints and indexes created successfully")
            return True
        except Exception as exc:
            logger.error("Failed to create Neo4j constraints: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Clear graph data
    # ------------------------------------------------------------------
    def clear_graph(self) -> bool:
        """Delete all nodes and relationships in the graph.

        Useful for full rebuild. Destructive operation.
        """
        driver = self._conn.driver
        if driver is None:
            return False

        try:
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared — all nodes and relationships deleted")
            return True
        except Exception as exc:
            logger.error("Failed to clear graph: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Index documents
    # ------------------------------------------------------------------
    def index_documents(self, docs: list[dict]) -> int:
        """Write Document nodes to Neo4j.

        Args:
            docs: List of document dicts with ``filename``, ``source``.

        Returns:
            Number of documents created.
        """
        driver = self._conn.driver
        if driver is None:
            return 0

        created = 0
        try:
            with driver.session() as session:
                for doc in docs:
                    doc_id = doc.get("filename", doc.get("source", ""))
                    title = doc.get("filename", doc.get("source", ""))
                    source_path = doc.get("source", "")
                    file_type = source_path.rsplit(".", 1)[-1] if "." in source_path else "md"

                    session.run("""
                        MERGE (d:Document {doc_id: $doc_id})
                        SET d.title = $title,
                            d.source_path = $source_path,
                            d.file_type = $file_type
                    """, doc_id=doc_id, title=title, source_path=source_path, file_type=file_type)
                    created += 1
            logger.info("Indexed %d documents to Neo4j", created)
        except Exception as exc:
            logger.error("Failed to index documents: %s", exc)
        return created

    # ------------------------------------------------------------------
    # Index chunks
    # ------------------------------------------------------------------
    def index_chunks(self, chunks: list[dict]) -> int:
        """Write Chunk nodes to Neo4j and create HAS_CHUNK relations.

        Args:
            chunks: List of chunk dicts with ``chunk_id``, ``doc_id``,
                    ``content``, ``chunk_index``.

        Returns:
            Number of chunks created.
        """
        driver = self._conn.driver
        if driver is None:
            return 0

        created = 0
        try:
            with driver.session() as session:
                for ch in chunks:
                    chunk_id = ch.get("chunk_id", "")
                    doc_id = ch.get("doc_id", ch.get("source", ""))
                    content = ch.get("content", "")[:5000]  # Truncate long content
                    chunk_index = ch.get("chunk_index", 0)

                    session.run("""
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.doc_id = $doc_id,
                            c.content = $content,
                            c.chunk_index = $chunk_index
                        WITH c
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """, chunk_id=chunk_id, doc_id=doc_id, content=content, chunk_index=chunk_index)
                    created += 1
            logger.info("Indexed %d chunks to Neo4j", created)
        except Exception as exc:
            logger.error("Failed to index chunks: %s", exc)
        return created

    # ------------------------------------------------------------------
    # Index entities
    # ------------------------------------------------------------------
    def index_entities(self, entities: list[Entity]) -> int:
        """Write Entity nodes and MENTIONS relations to Neo4j.

        Deduplicates by (normalized_name, type).
        Creates MENTIONS relations from Chunk to Entity.

        Args:
            entities: List of Entity objects.

        Returns:
            Number of unique entities created.
        """
        driver = self._conn.driver
        if driver is None:
            return 0

        # Deduplicate: keep first occurrence for metadata
        seen: dict[tuple[str, str], Entity] = {}
        for e in entities:
            key = (e.normalized_name, e.type)
            if key not in seen:
                seen[key] = e

        created = 0
        try:
            with driver.session() as session:
                for (norm_name, etype), ent in seen.items():
                    entity_id = f"{etype}:{norm_name}"

                    session.run("""
                        MERGE (e:Entity {entity_id: $entity_id})
                        SET e.name = $name,
                            e.type = $type,
                            e.normalized_name = $normalized_name
                    """, entity_id=entity_id, name=ent.name, type=ent.type, normalized_name=norm_name)
                    created += 1

                # Create MENTIONS relations
                for ent in entities:
                    entity_id = f"{ent.type}:{ent.normalized_name}"
                    if ent.chunk_id:
                        try:
                            session.run("""
                                MATCH (c:Chunk {chunk_id: $chunk_id})
                                MATCH (e:Entity {entity_id: $entity_id})
                                MERGE (c)-[:MENTIONS]->(e)
                            """, chunk_id=ent.chunk_id, entity_id=entity_id)
                        except Exception:
                            pass  # Chunk may not exist yet

            logger.info("Indexed %d entities to Neo4j (%d MENTIONS relations)",
                         created, len(entities))
        except Exception as exc:
            logger.error("Failed to index entities: %s", exc)
        return created

    # ------------------------------------------------------------------
    # Index relations
    # ------------------------------------------------------------------
    def index_relations(self, relations: list[Relation]) -> int:
        """Write relationship edges between entities in Neo4j.

        Merges edges to avoid duplicates.

        Args:
            relations: List of Relation objects.

        Returns:
            Number of relationships created.
        """
        driver = self._conn.driver
        if driver is None:
            return 0

        # Deduplicate
        seen_edges: set[tuple[str, str, str]] = set()
        created = 0

        try:
            with driver.session() as session:
                for rel in relations:
                    src_id = f"{rel.source_entity.type}:{rel.source_entity.normalized_name}"
                    tgt_id = f"{rel.target_entity.type}:{rel.target_entity.normalized_name}"
                    edge_key = (src_id, tgt_id, rel.relation_type)

                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    # Use dynamic relationship type via APOC or string formatting
                    # Since we use a fixed set of relation types, direct Cypher is safe
                    cypher = f"""
                        MATCH (src:Entity {{entity_id: $src_id}})
                        MATCH (tgt:Entity {{entity_id: $tgt_id}})
                        MERGE (src)-[r:{rel.relation_type}]->(tgt)
                        SET r.weight = $weight,
                            r.evidence_chunk_id = $evidence_chunk_id
                    """

                    try:
                        session.run(
                            cypher,
                            src_id=src_id,
                            tgt_id=tgt_id,
                            weight=rel.weight,
                            evidence_chunk_id=rel.evidence_chunk_id or "",
                        )
                        created += 1
                    except Exception as exc:
                        logger.debug("Failed to create relation %s: %s", edge_key, exc)

            logger.info("Indexed %d relations to Neo4j", created)
        except Exception as exc:
            logger.error("Failed to index relations: %s", exc)

        return created

    # ------------------------------------------------------------------
    # Full graph build pipeline
    # ------------------------------------------------------------------
    def build_graph(
        self,
        docs: list[dict] | None = None,
        chunks: list[dict] | None = None,
    ) -> GraphIndexReport:
        """Full graph building pipeline from documents and chunks.

        Steps:
        1. Index Document nodes
        2. Index Chunk nodes + HAS_CHUNK relations
        3. Extract entities from chunks
        4. Index Entity nodes + MENTIONS relations
        5. Extract relations from chunks
        6. Index relationship edges

        Args:
            docs: List of document dicts.
            chunks: List of chunk dicts.

        Returns:
            GraphIndexReport with counts and errors.
        """
        import time
        t0 = time.time()
        report = GraphIndexReport()

        if not self.available:
            report.errors.append("Neo4j unavailable — graph not built")
            report.duration_ms = (time.time() - t0) * 1000
            return report

        docs = docs or []
        chunks = chunks or []

        report.total_docs = len(docs)
        report.total_chunks = len(chunks)

        # 1. Index documents
        n_docs = self.index_documents(docs)
        report.nodes_created += n_docs

        # 2. Index chunks
        n_chunks = self.index_chunks(chunks)
        report.nodes_created += n_chunks

        # 3. Extract entities from chunks
        entities = extract_entities_from_chunks(chunks)
        report.entity_count = len(entities)

        # 4. Index entities
        n_entities = self.index_entities(entities)
        report.nodes_created += n_entities

        # 5. Extract relations
        entities_by_chunk: dict[str, list[Entity]] = {}
        for e in entities:
            entities_by_chunk.setdefault(e.chunk_id, []).append(e)
        relations = extract_relations_from_chunks(chunks, entities_by_chunk=entities_by_chunk)
        report.relation_count = len(relations)

        # 6. Index relations
        n_relations = self.index_relations(relations)
        report.relations_created = n_relations

        report.duration_ms = round((time.time() - t0) * 1000, 2)
        return report
