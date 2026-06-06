#!/usr/bin/env python3
"""Build knowledge graph from existing documents and chunks.

Reads from:
- data/docs/ — existing markdown documents
- Elasticsearch / in-memory chunks (from ingestion)

Extracts:
- Entities (API, CLASS, FUNCTION, ERROR_CODE, etc.)
- Relations (RELATED_TO, DEPENDS_ON, CALLS, etc.)

Writes to Neo4j.

Usage:
    # Full rebuild (clears existing graph first)
    python scripts/ingest_graph.py --full

    # Incremental (add to existing graph)
    python scripts/ingest_graph.py

    # Dry run (extract only, don't write to Neo4j)
    python scripts/ingest_graph.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from existing documents and chunks"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Clear existing graph before building (full rebuild)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Extract entities/relations but don't write to Neo4j"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Build Knowledge Graph from Documents & Chunks")
    print("=" * 60)

    # 1. Check Neo4j connectivity
    from enterprise_agentic_rag.config.settings import get_settings
    from enterprise_agentic_rag.rag.graph.graph_indexer import (
        GraphIndexer,
        get_neo4j_connection,
    )

    settings = get_settings()
    if not settings.graph_rag.enabled:
        print("\n⚠  Graph RAG is disabled (ENABLE_GRAPH_RAG=false). Exiting.")
        return

    print(f"\n[config] Neo4j URI: {settings.neo4j.uri}")

    conn = get_neo4j_connection()

    if args.dry_run:
        print("[mode] DRY RUN — entities and relations will be extracted but NOT written to Neo4j")
    elif not conn.available:
        print("\n✗  Neo4j is NOT reachable. Start it first: docker compose up -d neo4j")
        sys.exit(1)
    else:
        print("✓  Neo4j connection OK")

    # 2. Load documents and chunks
    print("\n[step 1/5] Loading documents and chunks...")
    from enterprise_agentic_rag.rag.document_loader import load_markdown_files
    from enterprise_agentic_rag.rag.splitter import split_documents

    raw_docs = load_markdown_files()
    print(f"  Loaded {len(raw_docs)} documents")

    if not raw_docs:
        print("  ✗ No documents found in data/docs/")
        print("    Run ingest_docs.py first to import documents.")
        sys.exit(1)

    chunks = split_documents(raw_docs, chunk_size=500)
    for i, ch in enumerate(chunks):
        ch["chunk_id"] = f"{ch.get('source', 'doc')}_{ch.get('chunk_index', i)}"
        ch["doc_id"] = ch.get("source", "unknown")
    print(f"  Split into {len(chunks)} chunks")

    # 3. Extract entities
    print("\n[step 2/5] Extracting entities...")
    t0 = time.time()
    from enterprise_agentic_rag.rag.graph.entity_extractor import extract_entities_from_chunks

    entities = extract_entities_from_chunks(chunks)
    entity_ms = (time.time() - t0) * 1000

    # Entity type breakdown
    type_counts: dict[str, int] = {}
    for e in entities:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1
    print(f"  Extracted {len(entities)} entities in {entity_ms:.0f}ms")
    for etype, count in sorted(type_counts.items()):
        print(f"    {etype}: {count}")

    # 4. Extract relations
    print("\n[step 3/5] Extracting relations...")
    t0 = time.time()

    entities_by_chunk: dict[str, list] = {}
    for e in entities:
        entities_by_chunk.setdefault(e.chunk_id, []).append(e)

    from enterprise_agentic_rag.rag.graph.relation_extractor import extract_relations_from_chunks

    relations = extract_relations_from_chunks(chunks, entities_by_chunk=entities_by_chunk)
    rel_ms = (time.time() - t0) * 1000

    # Relation type breakdown
    rel_counts: dict[str, int] = {}
    for r in relations:
        rel_counts[r.relation_type] = rel_counts.get(r.relation_type, 0) + 1
    print(f"  Extracted {len(relations)} relations in {rel_ms:.0f}ms")
    for rtype, count in sorted(rel_counts.items()):
        print(f"    {rtype}: {count}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("  DRY RUN COMPLETE — no data written to Neo4j")
        print(f"  Would index: {len(raw_docs)} docs, {len(chunks)} chunks")
        print(f"  Would index: {len(entities)} entities, {len(relations)} relations")
        print("=" * 60)
        return

    # 5. Write to Neo4j
    print("\n[step 4/5] Writing to Neo4j...")
    indexer = GraphIndexer(connection=conn)

    # Create constraints if needed
    indexer.create_constraints()

    # Clear existing graph for full rebuild
    if args.full:
        print("  Clearing existing graph (--full)...")
        indexer.clear_graph()

    # Build graph
    t0 = time.time()
    report = indexer.build_graph(docs=raw_docs, chunks=chunks)
    graph_ms = (time.time() - t0) * 1000

    print(f"  Indexed in {graph_ms:.0f}ms")
    print(f"    Documents: {report.total_docs}")
    print(f"    Chunks: {report.total_chunks}")
    print(f"    Entities: {report.entity_count}")
    print(f"    Relations: {report.relation_count}")
    print(f"    Nodes created: {report.nodes_created}")
    print(f"    Relationships created: {report.relations_created}")

    if report.errors:
        print(f"  Errors: {report.errors}")

    # 6. Verify
    print("\n[step 5/5] Verifying...")
    driver = conn.driver
    if driver:
        try:
            with driver.session() as session:
                # Count nodes by label
                for label in ["Document", "Chunk", "Entity"]:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                    cnt = result.single()["cnt"]
                    print(f"  {label} nodes: {cnt}")

                # Count relationships
                result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
                cnt = result.single()["cnt"]
                print(f"  Relationships: {cnt}")

                # Sample entities
                result = session.run(
                    "MATCH (e:Entity) RETURN e.name AS name, e.type AS type LIMIT 5"
                )
                print("  Sample entities:")
                for record in result:
                    print(f"    [{record['type']}] {record['name']}")
        except Exception as exc:
            print(f"  ⚠  Verification query failed: {exc}")

    print("\n" + "=" * 60)
    print("  ✓ Knowledge graph built successfully")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
