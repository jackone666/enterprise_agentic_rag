"""Hybrid retrieval fusion — weighted RRF + score fusion.

Combines Milvus vector search with ES keyword search via
Weighted Reciprocal Rank Fusion.

Includes code block boost for developer-facing scenarios.
"""

from __future__ import annotations

import math
from typing import Any


# ===========================================================================
# Code block detection for retrieval boost
# ===========================================================================

_CODE_FENCE_RE = None  # lazy import


def _get_code_fence_pattern():
    global _CODE_FENCE_RE
    if _CODE_FENCE_RE is None:
        import re
        _CODE_FENCE_RE = re.compile(r"```\w*\s*\n.*?```", re.DOTALL)
    return _CODE_FENCE_RE


def _detect_code_density(chunk: dict[str, Any]) -> float:
    """Calculate the fraction of content that is code blocks.

    Args:
        chunk: Document chunk dict with ``content`` field.

    Returns:
        Float in [0.0, 1.0] — proportion of content in code fences.
    """
    content = chunk.get("content", "")
    if not content:
        return 0.0

    # Quick check: does content contain code markers?
    if "```" not in content:
        return 0.0

    try:
        pattern = _get_code_fence_pattern()
        code_chars = sum(m.end() - m.start() for m in pattern.finditer(content))
        return min(1.0, code_chars / max(len(content), 1))
    except Exception:
        return 0.0


def _apply_code_boost(
    chunk: dict[str, Any],
    score: float,
    boost_factor: float = 0.5,
) -> float:
    """Apply a code density boost to a chunk's score.

    Formula: score *= (1.0 + boost_factor * code_density)

    Only applies when the chunk actually contains code blocks.
    When boost_factor is 0, returns score unchanged.

    Args:
        chunk: Document chunk dict with ``content`` field.
        score: Current fused score.
        boost_factor: Maximum boost multiplier (default 0.5 → +50%).

    Returns:
        Boosted score.
    """
    if boost_factor <= 0:
        return score

    density = _detect_code_density(chunk)
    if density > 0:
        return score * (1.0 + boost_factor * density)
    return score


def weighted_rrf_fusion(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    k: int = 60,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Weighted Reciprocal Rank Fusion.

    Combines vector and keyword results with configurable weights.
    Higher weight = more influence from that source.

    Formula: score(chunk) = Σ weight_i / (k + rank_i)

    Args:
        vector_results: Ranked results from Milvus.
        keyword_results: Ranked results from ES / memory keyword search.
        vector_weight: Weight for the vector source (default 0.6).
        keyword_weight: Weight for the keyword source (default 0.4).
        k: Smoothing constant — prevents rank=1 from dominating (default 60).
        top_n: Number of fused results to return.

    Returns:
        Fused and deduplicated (by source) result list.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results, start=1):
        cid = doc.get("chunk_id", doc.get("source", f"v{rank}"))
        scores[cid] = scores.get(cid, 0) + vector_weight / (k + rank)
        docs[cid] = doc

    for rank, doc in enumerate(keyword_results, start=1):
        cid = doc.get("chunk_id", doc.get("source", f"k{rank}"))
        scores[cid] = scores.get(cid, 0) + keyword_weight / (k + rank)
        docs[cid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for cid, score in ranked:
        doc = docs[cid]
        source = doc.get("source", "")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        doc["score"] = round(score, 4)
        fused.append(doc)
        if len(fused) >= top_n:
            break

    return fused


def fusion_retrieve(
    query: str,
    top_k: int = 5,
    vector_weight: float = 0.6,
) -> list[dict[str, Any]]:
    """Full fusion retrieval pipeline.

    1. ES keyword search (or memory Jaccard fallback)
    2. Milvus vector search
    3. Weighted RRF fusion

    Returns:
        Fused result list, or keyword-only results if Milvus is down.
    """
    from enterprise_agentic_rag.rag.embedding_provider import get_embedding_provider
    from enterprise_agentic_rag.rag.es_keyword_store import ESKeywordStore
    from enterprise_agentic_rag.rag.milvus_store import MilvusStore
    from enterprise_agentic_rag.rag.retriever import KeywordRetriever

    # --- Keyword search ---
    es_kw = ESKeywordStore()
    if es_kw.available:
        keyword_results = es_kw.search(query, top_k=top_k)
        if not keyword_results and not es_kw.stats().get("index_exists", False):
            kr = KeywordRetriever(top_k=top_k)
            keyword_results = kr.search(query)
    else:
        # Fallback to in-memory Jaccard
        kr = KeywordRetriever(top_k=top_k)
        keyword_results = kr.search(query)

    # --- Vector search ---
    ms = MilvusStore()
    if ms.available:
        try:
            ep = get_embedding_provider()
            vec = ep.embed_query(query)
            vector_results = ms.search(vec, top_k=top_k)
            return weighted_rrf_fusion(
                vector_results, keyword_results,
                vector_weight=vector_weight,
                keyword_weight=1.0 - vector_weight,
                top_n=top_k,
            )
        except Exception:
            pass

    # Milvus unavailable or empty → return keyword-only results
    return keyword_results


# ===========================================================================
# Three-way fusion — keyword + vector + graph
# ===========================================================================


def three_way_rrf_fusion(
    keyword_candidates: list[dict[str, Any]],
    vector_candidates: list[dict[str, Any]],
    graph_candidates: list[dict[str, Any]] | None = None,
    keyword_weight: float = 0.3,
    vector_weight: float = 0.5,
    graph_weight: float = 0.2,
    k: int = 60,
    top_n: int = 5,
    code_boost_factor: float = 0.0,
) -> list[dict[str, Any]]:
    """Three-way Weighted Reciprocal Rank Fusion.

    Combines keyword, vector, and graph results with configurable weights.
    Preserves matched_sources and graph_paths from all three sources.

    If graph_weight is 0 or graph_candidates is empty/None, falls back to
    two-way keyword + vector fusion.

    Formula: score(chunk) = Σ weight_i / (k + rank_i)

    Args:
        keyword_candidates: Ranked results from ES/memory keyword search.
        vector_candidates: Ranked results from Milvus vector search.
        graph_candidates: Ranked results from GraphRetriever (optional).
        keyword_weight: Weight for keyword source (default 0.3).
        vector_weight: Weight for vector source (default 0.5).
        graph_weight: Weight for graph source (default 0.2).
        k: Smoothing constant (default 60).
        top_n: Number of fused results to return.
        code_boost_factor: Boost factor for code-containing chunks (0=disabled, 0.5=+50%).

    Returns:
        Fused and deduplicated result list.
        Each result includes ``matched_sources`` and ``graph_paths`` if present.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    matched_sources: dict[str, list[str]] = {}
    graph_paths_map: dict[str, list[dict]] = {}

    def _chunk_key(doc: dict[str, Any], rank: int, prefix: str) -> str:
        cid = doc.get("chunk_id", doc.get("source", f"{prefix}_{rank}"))
        return str(cid)

    # --- Keyword ---
    if keyword_weight > 0:
        for rank, doc in enumerate(keyword_candidates, start=1):
            cid = _chunk_key(doc, rank, "kw")
            scores[cid] = scores.get(cid, 0) + keyword_weight / (k + rank)
            if cid not in docs:
                docs[cid] = dict(doc)
            matched_sources.setdefault(cid, []).append("keyword")

    # --- Vector ---
    if vector_weight > 0:
        for rank, doc in enumerate(vector_candidates, start=1):
            cid = _chunk_key(doc, rank, "vec")
            scores[cid] = scores.get(cid, 0) + vector_weight / (k + rank)
            if cid not in docs:
                docs[cid] = dict(doc)
            matched_sources.setdefault(cid, []).append("vector")

    # --- Graph ---
    graph_candidates = graph_candidates or []
    if graph_weight > 0 and graph_candidates:
        for rank, doc in enumerate(graph_candidates, start=1):
            cid = _chunk_key(doc, rank, "graph")
            scores[cid] = scores.get(cid, 0) + graph_weight / (k + rank)
            if cid not in docs:
                docs[cid] = dict(doc)
            matched_sources.setdefault(cid, []).append("graph")

            # Preserve graph_paths
            gp = doc.get("graph_paths", [])
            if gp:
                existing = graph_paths_map.get(cid, [])
                # Merge, deduplicate by path entities
                seen_paths = {"|".join(p.get("path_entities", [])) for p in existing}
                for p in gp:
                    key = "|".join(p.get("path_entities", []))
                    if key not in seen_paths:
                        seen_paths.add(key)
                        existing.append(p)
                graph_paths_map[cid] = existing

    # --- Sort by fused score ---
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # --- Build fused results ---
    fused: list[dict[str, Any]] = []
    seen_by_doc_id: dict[str, set[str]] = {}  # doc_id → set of chunk_ids already included

    for cid, score in ranked:
        doc = docs[cid]
        doc_id = doc.get("doc_id", doc.get("source", ""))
        chunk_id = doc.get("chunk_id", cid)

        # Deduplicate by (doc_id, chunk_id)
        if doc_id not in seen_by_doc_id:
            seen_by_doc_id[doc_id] = set()
        if chunk_id in seen_by_doc_id[doc_id]:
            continue
        seen_by_doc_id[doc_id].add(chunk_id)

        # Apply code block boost before storing final score
        boosted_score = _apply_code_boost(doc, score, boost_factor=code_boost_factor)

        doc["fused_score"] = round(boosted_score, 4)
        doc["score"] = round(boosted_score, 4)
        doc["pre_boost_score"] = round(score, 4)  # Preserve original score for trace
        doc["code_density"] = round(_detect_code_density(doc), 3)
        doc["matched_sources"] = matched_sources.get(cid, [])
        doc["raw_scores"] = {
            src: 1.0 / (k + 1)  # approximate per-source score from RRF
            for src in matched_sources.get(cid, [])
        }
        if cid in graph_paths_map:
            doc["graph_paths"] = graph_paths_map[cid]

        fused.append(doc)
        if len(fused) >= top_n:
            break

    return fused


def multi_way_rrf_fusion(
    candidate_sources: dict[str, list[dict[str, Any]]],
    k: int = 60,
    top_n: int = 5,
    code_boost_factor: float = 0.0,
) -> list[dict[str, Any]]:
    """Generic multi-way Weighted Reciprocal Rank Fusion.

    Supports any number of named retrieval sources. Each source contributes
    a list of ranked chunks with an associated weight.

    Args:
        candidate_sources: Dict mapping source name to (weight, candidates) tuple.
            Example: {"keyword": 0.3, "vector": 0.5, "graph": 0.2, "external": 0.15}
            Weight values are normalized internally, so they don't need to sum to 1.
        k: Smoothing constant (default 60).
        top_n: Number of fused results to return.
        code_boost_factor: Boost factor for code-containing chunks.

    Returns:
        Fused and deduplicated result list.
    """
    if not isinstance(candidate_sources, dict):
        raise TypeError("candidate_sources must be a dict mapping name → list of chunks")

    # Filter out sources with empty candidates
    active_sources = {
        name: weight
        for name, weight in candidate_sources.items()
        if isinstance(weight, (int, float)) and weight > 0
    }
    if not active_sources:
        return []

    # Normalize weights
    total_weight = sum(active_sources.values())
    if total_weight > 0:
        active_sources = {k: v / total_weight for k, v in active_sources.items()}

    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    matched_sources: dict[str, list[str]] = {}

    for source_name, weight in active_sources.items():
        candidates = []
        # Allow passing (weight, list) pairs
        raw = candidate_sources.get(source_name)
        if isinstance(raw, tuple):
            _, candidates = raw
        elif isinstance(raw, list):
            candidates = raw
        else:
            continue

        for rank, doc in enumerate(candidates, start=1):
            cid = str(doc.get("chunk_id", doc.get("source", f"{source_name}_{rank}")))
            scores[cid] = scores.get(cid, 0) + weight / (k + rank)
            if cid not in docs:
                docs[cid] = dict(doc)
            matched_sources.setdefault(cid, []).append(source_name)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for cid, score in ranked:
        doc = docs[cid]
        doc_id = str(doc.get("doc_id", doc.get("source", "")))
        chunk_id = str(doc.get("chunk_id", cid))

        dedup_key = (doc_id, chunk_id)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        boosted = _apply_code_boost(doc, score, boost_factor=code_boost_factor)
        doc["fused_score"] = round(boosted, 4)
        doc["score"] = round(boosted, 4)
        doc["pre_boost_score"] = round(score, 4)
        doc["matched_sources"] = matched_sources.get(cid, [])
        doc["code_density"] = round(_detect_code_density(doc), 3)

        fused.append(doc)
        if len(fused) >= top_n:
            break

    return fused


def normalize_weights_for_fallback(
    weights: dict[str, float],
    available_retrievers: list[str],
) -> dict[str, float]:
    """Normalize weights when some retrievers are unavailable.

    If graph is unavailable, its weight is zeroed out and the remaining
    weights (keyword, vector) are normalized to sum to 1.0.

    Args:
        weights: Original weight dict (e.g. {"keyword": 0.3, "vector": 0.5, "graph": 0.2}).
        available_retrievers: List of available retriever names.

    Returns:
        Normalized weight dict with unavailable sources zeroed out.
    """
    normalized = dict(weights)
    available_set = set(available_retrievers)

    # Zero out unavailable retrievers
    for key in list(normalized.keys()):
        if key not in available_set:
            normalized[key] = 0.0

    # Normalize remaining to sum to 1.0
    total = sum(normalized.values())
    if total > 0:
        for key in normalized:
            normalized[key] = normalized[key] / total

    return normalized
