#!/usr/bin/env python3
"""Run Graph-Augmented Hybrid RAG and display full trace.

Demonstrates the complete Graph RAG pipeline with retrieval traces.

Usage:
    python scripts/run_graph_rag.py "Ability 和页面生命周期有什么关系？"
    python scripts/run_graph_rag.py "9568321 是什么错误？"
    python scripts/run_graph_rag.py "为什么白屏？" --mode parallel
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def print_separator(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{'─' * 10} {title} {'─' * (width - len(title) - 12)}")
    else:
        print("─" * width)


def print_trace_summary(trace: dict) -> None:
    """Pretty-print the retrieval trace summary."""
    print_separator("RETRIEVAL TRACE")

    print(f"  trace_id:        {trace.get('trace_id', 'N/A')}")
    print(f"  mode:            {trace.get('mode', 'N/A')}")
    print(f"  retrievers:      {trace.get('enabled_retrievers', [])}")

    plan = trace.get("retrieval_plan", {})
    if plan:
        print(f"  plan reason:     {plan.get('reason', 'N/A')}")

    degraded_from = trace.get("degraded_from", "")
    degraded_to = trace.get("degraded_to", "")
    if degraded_from or degraded_to:
        print(f"  ⚠  DEGRADED:       {degraded_from} → {degraded_to}")

    print(f"\n  --- Hits ---")
    print(f"  keyword_hits:    {trace.get('keyword_hit_count', 0)}")
    print(f"  vector_hits:     {trace.get('vector_hit_count', 0)}")
    print(f"  graph_hits:      {trace.get('graph_hit_count', 0)}")
    print(f"  merged_count:    {trace.get('merged_count', 0)}")
    print(f"  reranked_count:  {trace.get('reranked_count', 0)}")
    print(f"  graph_paths:     {trace.get('graph_paths_count', 0)}")

    print(f"\n  --- Timing (ms) ---")
    print(f"  keyword:         {trace.get('keyword_latency_ms', 0):.1f}")
    print(f"  vector:          {trace.get('vector_latency_ms', 0):.1f}")
    print(f"  graph:           {trace.get('graph_latency_ms', 0):.1f}")
    print(f"  fusion:          {trace.get('fusion_latency_ms', 0):.1f}")
    print(f"  total:           {trace.get('total_latency_ms', 0):.1f}")

    print(f"\n  --- Fusion ---")
    print(f"  method:          {trace.get('fusion_method', 'N/A')}")
    print(f"  weights:         {trace.get('fusion_weights', {})}")

    expanded = trace.get("expanded_query", "")
    if expanded:
        print(f"\n  --- Query Expansion ---")
        print(f"  original:        {trace.get('original_query', '')[:80]}")
        print(f"  expanded:        {expanded[:80]}")
        print(f"  terms:           {trace.get('expansion_terms', [])}")

    errors = trace.get("errors", [])
    if errors:
        print(f"\n  --- Errors ---")
        for e in errors:
            print(f"  ✗ {e}")


def print_results(docs: list[dict]) -> None:
    """Print retrieved document summaries."""
    print_separator("RETRIEVED DOCUMENTS")

    if not docs:
        print("  (no documents retrieved)")
        return

    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] score={doc.get('score', 0):.4f} | sources={doc.get('matched_sources', [])}")
        content = doc.get("content", "")[:200]
        print(f"      {content}...")
        print(f"      source: {doc.get('source', doc.get('chunk_id', '?'))}")

        paths = doc.get("graph_paths", [])
        if paths:
            print(f"      graph_paths ({len(paths)}):")
            for p in paths[:3]:
                entities = " → ".join(p.get("path_entities", []))
                rels = " → ".join(p.get("path_relations", []))
                print(f"        {entities}  [{rels}]")


async def main_async(query: str) -> None:
    from enterprise_agentic_rag.rag.graph_rag_orchestrator import GraphRAGOrchestrator

    orchestrator = GraphRAGOrchestrator()
    result = await orchestrator.retrieve(query=query, top_k=5)

    # Print trace
    trace = result.get("retrieval_trace", {})
    print_trace_summary(trace)

    # Print results
    docs = result.get("retrieved_docs", [])
    print_results(docs)

    # Print full trace as JSON (compact)
    print_separator("FULL TRACE JSON")
    print(json.dumps(trace, ensure_ascii=False, indent=2, default=str))

    # Query analysis
    qa = result.get("query_analysis", {})
    if qa:
        print_separator("QUERY ANALYSIS")
        print(json.dumps(qa, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Graph-Augmented Hybrid RAG pipeline"
    )
    parser.add_argument(
        "query", nargs="?", default="Ability 和页面生命周期有什么关系？",
        help="Query to run (default: relationship query)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Graph-Augmented Hybrid RAG — Retrieval Pipeline")
    print("=" * 60)
    print(f"\n  Query: {args.query}")

    t0 = time.time()
    asyncio.run(main_async(args.query))
    total_s = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  Total wall time: {total_s:.2f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
