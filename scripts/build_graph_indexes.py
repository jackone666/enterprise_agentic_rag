#!/usr/bin/env python3
"""Build Neo4j graph indexes and constraints for Graph-Augmented RAG.

Creates:
- Uniqueness constraints on Document.doc_id, Chunk.chunk_id, Entity.entity_id
- Indexes on Entity.normalized_name, Entity.type, Chunk.doc_id

Usage:
    python scripts/build_graph_indexes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enterprise_agentic_rag.config.settings import get_settings
from enterprise_agentic_rag.rag.graph.graph_indexer import GraphIndexer, get_neo4j_connection


def main() -> None:
    print("=" * 60)
    print("  Build Neo4j Graph Indexes & Constraints")
    print("=" * 60)

    settings = get_settings()

    # Check config
    print(f"\n[config] Graph RAG enabled: {settings.graph_rag.enabled}")
    print(f"[config] Neo4j URI: {settings.neo4j.uri}")
    print(f"[config] Neo4j User: {settings.neo4j.user}")

    if not settings.graph_rag.enabled:
        print("\n⚠  Graph RAG is disabled (ENABLE_GRAPH_RAG=false).")
        print("   Set ENABLE_GRAPH_RAG=true in .env to enable.")
        return

    # Check Neo4j connectivity
    print("\n[step 1/3] Checking Neo4j connection...")
    conn = get_neo4j_connection()
    if not conn.available:
        print("✗  Neo4j is NOT reachable. Please ensure:")
        print("   1. docker compose up -d neo4j")
        print("   2. Neo4j is healthy: docker compose ps neo4j")
        print("   3. NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are correct in .env")
        sys.exit(1)
    print("✓  Neo4j connection OK")

    # Create constraints and indexes
    print("\n[step 2/3] Creating constraints and indexes...")
    t0 = time.time()
    indexer = GraphIndexer(connection=conn)
    ok = indexer.create_constraints()
    elapsed = (time.time() - t0) * 1000

    if ok:
        print(f"✓  Constraints and indexes created successfully ({elapsed:.0f}ms)")
    else:
        print("✗  Failed to create constraints/indexes")
        sys.exit(1)

    # Verify
    print("\n[step 3/3] Verifying indexes...")
    driver = conn.driver
    if driver:
        try:
            with driver.session() as session:
                result = session.run("SHOW INDEXES")
                indexes = list(result)
                print(f"✓  {len(indexes)} indexes/constraints exist:")
                for idx in indexes:
                    name = idx.get("name", "?")
                    idx_type = idx.get("type", "?")
                    print(f"     - {name} ({idx_type})")
        except Exception as exc:
            print(f"⚠  Could not verify indexes: {exc}")

    print("\n" + "=" * 60)
    print("  ✓ Graph indexes ready. Run ingest_graph.py to build the graph.")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
