"""Graph RAG data schemas — RetrievalPlan, GraphPath, Candidate, RetrievalResult.

All new data structures for Graph-Augmented Hybrid RAG.
Existing Hybrid RAG structures remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# Entity & Relation type enums
# ===========================================================================

ENTITY_TYPES = frozenset({
    "API",
    "CLASS",
    "FUNCTION",
    "ERROR_CODE",
    "COMPONENT",
    "MODULE",
    "CONFIG",
    "LIFECYCLE",
    "CONCEPT",
    "IMPORT",
    "METHOD_CALL",
    "PROPERTY",
    "TYPE",
    "INTERFACE",
    "CODE_BLOCK",
})

RELATION_TYPES = frozenset({
    "RELATED_TO",
    "DEPENDS_ON",
    "CALLS",
    "BELONGS_TO",
    "CAUSES",
    "FIXES",
    "PART_OF",
    "HAS_LIFECYCLE",
    "AFFECTS",
    "IMPORTS",
    "EXTENDS",
    "IMPLEMENTS",
})


# ===========================================================================
# Retrieval Plan
# ===========================================================================

@dataclass
class RetrievalPlan:
    """Dynamic retrieval plan produced by the RetrievalRouter.

    Determines which retrievers to use, in what order, and with what weights.

    Every plan records a ``reason`` explaining the routing decision for
    full observability.
    """

    mode: str = "parallel"
    # parallel | graph_first | keyword_first | vector_first | hybrid_only

    enabled_retrievers: list[str] = field(default_factory=lambda: ["keyword", "vector"])
    # Possible values: "keyword", "vector", "graph"

    top_k: dict[str, int] = field(default_factory=dict)
    # e.g. {"keyword": 5, "vector": 5, "graph": 10}

    weights: dict[str, float] = field(default_factory=dict)
    # e.g. {"keyword": 0.3, "vector": 0.5, "graph": 0.2}

    graph_depth: int = 1
    # Neo4j traversal depth (1 or 2 hops)

    need_query_expansion: bool = False
    # Whether graph_first should produce an expanded_query

    need_second_stage_retrieval: bool = False
    # Whether graph_first should trigger keyword+vector after graph

    fallback_to_hybrid: bool = True
    # Whether to auto-fallback to hybrid_only on graph failure

    reason: str = ""
    # Human-readable reason for this plan (the routing decision)

    # ------------------------------------------------------------------
    # Degradation tracking — set by orchestrator when fallback occurs
    # ------------------------------------------------------------------
    degraded_from: str = ""
    # Original mode before degradation, e.g. "graph_first"

    degraded_to: str = ""
    # Target mode after degradation, e.g. "hybrid_only"

    graph_failed: bool = False
    # Whether the graph retriever failed during this retrieval

    # ------------------------------------------------------------------
    # External search
    # ------------------------------------------------------------------
    enable_external: bool = False
    # Whether to trigger external knowledge source search

    external_sources: list[str] = field(default_factory=list)
    # e.g. ["github", "stackoverflow", "web"]


# ===========================================================================
# Graph Path
# ===========================================================================

@dataclass
class GraphPath:
    """A path found through the knowledge graph connecting entities.

    Used in context building to explain how entities relate.
    """

    path_entities: list[str] = field(default_factory=list)
    # Ordered list of entity names along the path

    path_relations: list[str] = field(default_factory=list)
    # Ordered list of relation types along the path

    evidence_chunk_id: str = ""
    # The chunk_id that provides evidence for this path

    relation_weight: float = 1.0
    # Aggregate weight of relations in the path

    path_score: float = 0.0
    # Computed relevance score for this path

    path_length: int = 0
    # Number of hops


# ===========================================================================
# Candidate (extended)
# ===========================================================================

@dataclass
class Candidate:
    """A single retrieval candidate — extended for graph support.

    Compatible with existing dict-based chunks used throughout the project.
    """

    chunk_id: str = ""
    doc_id: str = ""
    content: str = ""
    source_path: str = ""

    # Scores
    keyword_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0

    # Source tracking
    matched_sources: list[str] = field(default_factory=list)
    # e.g. ["keyword", "vector", "graph"]

    raw_scores: dict[str, float] = field(default_factory=dict)
    # e.g. {"keyword": 0.85, "vector": 0.72, "graph": 0.45}

    # Graph enrichment
    graph_paths: list[GraphPath] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (compatible with existing chunk format)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source_path,
            "source_path": self.source_path,
            "score": self.fused_score or max(self.raw_scores.values()) if self.raw_scores else 0.0,
            "keyword_score": self.keyword_score,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "fused_score": self.fused_score,
            "rerank_score": self.rerank_score,
            "matched_sources": self.matched_sources,
            "raw_scores": self.raw_scores,
            "graph_paths": [{
                "path_entities": gp.path_entities,
                "path_relations": gp.path_relations,
                "evidence_chunk_id": gp.evidence_chunk_id,
                "relation_weight": gp.relation_weight,
                "path_score": gp.path_score,
                "path_length": gp.path_length,
            } for gp in self.graph_paths],
            "metadata": self.metadata,
        }


# ===========================================================================
# Retrieval Result (full trace)
# ===========================================================================

@dataclass
class RetrievalResult:
    """Complete retrieval result with trace information.

    Captures everything that happened during retrieval for observability.
    """

    query: str = ""
    retrieval_plan: RetrievalPlan | None = None

    # Per-source candidates
    keyword_candidates: list[Candidate] = field(default_factory=list)
    vector_candidates: list[Candidate] = field(default_factory=list)
    graph_candidates: list[Candidate] = field(default_factory=list)

    # Merged & fused
    merged_candidates: list[Candidate] = field(default_factory=list)
    fused_candidates: list[Candidate] = field(default_factory=list)
    reranked_candidates: list[Candidate] = field(default_factory=list)

    # Final
    final_context: list[dict[str, Any]] = field(default_factory=list)

    # Observability
    trace: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    degraded_from: str = ""   # e.g. "graph_first"
    degraded_to: str = ""     # e.g. "hybrid_only"

    # Timing (ms)
    keyword_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    graph_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Counts
    keyword_hit_count: int = 0
    vector_hit_count: int = 0
    graph_hit_count: int = 0
    merged_count: int = 0
    reranked_count: int = 0
    graph_paths_count: int = 0

    # Fusion config
    fusion_method: str = "rrf"
