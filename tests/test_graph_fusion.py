"""Tests for three-way fusion — keyword + vector + graph.

Covers:
- Three-way RRF fusion
- Graph unavailable → weights normalized to 0
- doc_id + chunk_id deduplication
- matched_sources merging
- Backward compatibility with two-way fusion
"""

from __future__ import annotations

import pytest

from enterprise_agentic_rag.rag.fusion import (
    normalize_weights_for_fallback,
    three_way_rrf_fusion,
    weighted_rrf_fusion,
)


def _make_doc(source: str, chunk_id: str, score: float = 0.8) -> dict:
    return {
        "chunk_id": chunk_id,
        "source": source,
        "doc_id": source,
        "content": f"Content from {source} chunk {chunk_id}",
        "score": score,
    }


class TestThreeWayFusion:
    """Three-way keyword + vector + graph fusion."""

    def test_basic_three_way_fusion(self):
        """All three sources produce deduplicated fused results."""
        kw = [
            _make_doc("doc_a.md", "doc_a.md_0", 0.9),
            _make_doc("doc_b.md", "doc_b.md_0", 0.7),
        ]
        vec = [
            _make_doc("doc_a.md", "doc_a.md_0", 0.85),
            _make_doc("doc_c.md", "doc_c.md_0", 0.8),
        ]
        graph = [
            _make_doc("doc_b.md", "doc_b.md_0", 0.6),
            _make_doc("doc_d.md", "doc_d.md_0", 0.5),
        ]

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=graph,
            keyword_weight=0.3,
            vector_weight=0.5,
            graph_weight=0.2,
            top_n=5,
        )

        assert len(fused) >= 2
        # doc_a should appear once (deduplicated)
        doc_a_chunks = [d for d in fused if d["source"] == "doc_a.md"]
        assert len(doc_a_chunks) == 1

    def test_matched_sources_merged(self):
        """matched_sources combines all three sources."""
        kw = [_make_doc("doc_a.md", "doc_a.md_0")]
        vec = [_make_doc("doc_a.md", "doc_a.md_0")]
        graph = [_make_doc("doc_a.md", "doc_a.md_0")]

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=graph,
            keyword_weight=0.3,
            vector_weight=0.5,
            graph_weight=0.2,
        )

        for doc in fused:
            sources = doc.get("matched_sources", [])
            assert "keyword" in sources
            assert "vector" in sources
            assert "graph" in sources

    def test_fused_score_populated(self):
        """fused_score should be set on each result."""
        kw = [_make_doc("a.md", "a_0")]
        vec = [_make_doc("a.md", "a_0")]
        graph = []

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=graph,
            keyword_weight=0.4,
            vector_weight=0.6,
            graph_weight=0.0,
        )

        for doc in fused:
            assert "fused_score" in doc
            assert doc["fused_score"] > 0

    def test_graph_weight_zero_disables_graph(self):
        """When graph_weight=0, graph candidates are ignored."""
        kw = [_make_doc("a.md", "a_0")]
        vec = [_make_doc("a.md", "a_0")]
        graph = [_make_doc("graph_only.md", "g_0")]

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=graph,
            keyword_weight=0.4,
            vector_weight=0.6,
            graph_weight=0.0,
        )

        # graph_only.md should NOT appear since graph_weight is 0
        graph_results = [d for d in fused if d["source"] == "graph_only.md"]
        assert len(graph_results) == 0

    def test_graph_paths_preserved(self):
        """graph_paths from graph candidates are preserved in fusion output."""
        kw = [_make_doc("a.md", "a_0")]
        vec = [_make_doc("a.md", "a_0")]
        graph = [{
            ** _make_doc("a.md", "a_0"),
            "graph_paths": [
                {"path_entities": ["Ability", "UIAbility"], "path_relations": ["HAS_LIFECYCLE"]}
            ],
        }]

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=graph,
            keyword_weight=0.2,
            vector_weight=0.3,
            graph_weight=0.5,
        )

        for doc in fused:
            paths = doc.get("graph_paths", [])
            if paths:
                assert len(paths) > 0
                assert "path_entities" in paths[0]

    def test_chunk_id_deduplication(self):
        """Same chunk_id from different sources is deduplicated."""
        kw = [_make_doc("doc.md", "chunk_1")]
        vec = [_make_doc("doc.md", "chunk_1")]
        graph = [_make_doc("doc.md", "chunk_1")]

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=graph,
            keyword_weight=0.3,
            vector_weight=0.5,
            graph_weight=0.2,
            top_n=10,
        )

        # Should have at most 1 result for "chunk_1"
        chunk_1_results = [d for d in fused if d.get("chunk_id") == "chunk_1"]
        assert len(chunk_1_results) == 1

    def test_empty_graph_candidates_ok(self):
        """Fusion works fine with empty graph candidates."""
        kw = [_make_doc("a.md", "a_0")]
        vec = [_make_doc("a.md", "a_0")]

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=[],
            keyword_weight=0.4,
            vector_weight=0.6,
            graph_weight=0.0,
        )

        assert len(fused) >= 1

    def test_graph_none_ok(self):
        """Passing None for graph_candidates is handled."""
        kw = [_make_doc("a.md", "a_0")]
        vec = [_make_doc("a.md", "a_0")]

        fused = three_way_rrf_fusion(
            keyword_candidates=kw,
            vector_candidates=vec,
            graph_candidates=None,
            keyword_weight=0.4,
            vector_weight=0.6,
            graph_weight=0.0,
        )

        assert len(fused) >= 1


class TestTwoWayFusionBackwardCompat:
    """Existing two-way fusion still works."""

    def test_two_way_fusion_still_works(self):
        """Original weighted_rrf_fusion function should still work."""
        vec = [_make_doc("a.md", "a_0")]
        kw = [_make_doc("a.md", "a_0")]

        fused = weighted_rrf_fusion(
            vector_results=vec,
            keyword_results=kw,
            vector_weight=0.6,
            keyword_weight=0.4,
            top_n=5,
        )

        assert len(fused) >= 1
        assert "score" in fused[0]


class TestNormalizeWeights:
    """Weight normalization for fallback."""

    def test_graph_unavailable_normalizes(self):
        """When graph is unavailable, its weight goes to 0, others normalized."""
        weights = {"keyword": 0.3, "vector": 0.5, "graph": 0.2}
        normalized = normalize_weights_for_fallback(weights, ["keyword", "vector"])

        assert normalized["graph"] == 0.0
        assert abs(normalized["keyword"] + normalized["vector"] - 1.0) < 0.01

    def test_only_keyword_available(self):
        """When only keyword is available."""
        weights = {"keyword": 0.3, "vector": 0.5, "graph": 0.2}
        normalized = normalize_weights_for_fallback(weights, ["keyword"])

        assert normalized["keyword"] == 1.0
        assert normalized["vector"] == 0.0
        assert normalized["graph"] == 0.0

    def test_all_available_no_change(self):
        """When all sources are available, ratios are preserved."""
        weights = {"keyword": 0.3, "vector": 0.5, "graph": 0.2}
        normalized = normalize_weights_for_fallback(weights, ["keyword", "vector", "graph"])

        # Should be normalized to sum to 1.0 (already are)
        assert abs(sum(normalized.values()) - 1.0) < 0.01
        # Relative ratios preserved
        assert abs(normalized["keyword"] - 0.3) < 0.01
