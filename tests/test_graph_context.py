"""Tests for context builder with graph_paths support.

Covers:
- graph_paths in document metadata
- Context window includes graph path summaries
- Context without graph_paths works as before
- Graph path enrichment
"""

from __future__ import annotations

import pytest

from enterprise_agentic_rag.context.context_manager import (
    ContextManager,
    _build_graph_path_summaries,
)


@pytest.fixture
def context_mgr():
    return ContextManager(max_tokens=4096)


class TestGraphPathContext:
    """Graph paths in context building."""

    def test_graph_paths_in_retrieved_docs(self, context_mgr):
        """Docs with graph_paths should have them preserved in context."""
        docs = [
            {
                "chunk_id": "doc.md_0",
                "doc_id": "doc.md",
                "source": "doc.md",
                "content": "UIAbility 是应用的基本单元",
                "score": 0.85,
                "matched_sources": ["vector", "graph"],
                "graph_paths": [
                    {
                        "path_entities": ["UIAbility", "onCreate", "onWindowStageCreate"],
                        "path_relations": ["HAS_LIFECYCLE", "CALLS"],
                        "evidence_chunk_id": "doc.md_0",
                        "relation_weight": 1.0,
                        "path_score": 0.9,
                        "path_length": 2,
                    }
                ],
                "metadata": {},
            },
        ]

        enriched = context_mgr.enrich_docs_with_graph_paths(docs)
        assert enriched[0].get("metadata", {}).get("graph_enriched") is True
        assert enriched[0].get("metadata", {}).get("graph_path_count") == 1

    def test_docs_without_graph_paths_unchanged(self, context_mgr):
        """Docs without graph_paths are not modified."""
        docs = [
            {
                "chunk_id": "doc.md_0",
                "source": "doc.md",
                "content": "普通文档内容",
                "score": 0.7,
                "matched_sources": ["keyword"],
            },
        ]

        enriched = context_mgr.enrich_docs_with_graph_paths(docs)
        # Should not fail, and doc should be preserved
        assert enriched[0]["content"] == "普通文档内容"
        # No graph metadata
        assert not enriched[0].get("metadata", {}).get("graph_enriched", False)

    def test_build_context_with_graph_paths(self, context_mgr):
        """build_context should handle docs with graph_paths."""
        docs = [
            {
                "chunk_id": "doc.md_0",
                "doc_id": "doc.md",
                "source": "doc.md",
                "content": "UIAbility 生命周期包括 onCreate 和 onWindowStageCreate",
                "score": 0.9,
                "matched_sources": ["graph", "vector"],
                "graph_paths": [
                    {
                        "path_entities": ["UIAbility", "Ability"],
                        "path_relations": ["RELATED_TO"],
                        "evidence_chunk_id": "doc.md_0",
                        "relation_weight": 1.0,
                        "path_score": 0.8,
                        "path_length": 1,
                    }
                ],
            },
        ]

        structured = context_mgr.build_context(
            query="UIAbility 的生命周期是什么？",
            retrieved_docs=docs,
        )

        assert "truncated_docs" in structured
        assert "context_window" in structured

    def test_build_context_without_graph_paths_still_works(self, context_mgr):
        """build_context without graph_paths should work as before."""
        docs = [
            {
                "chunk_id": "doc.md_0",
                "source": "doc.md",
                "content": "普通文档内容",
                "score": 0.7,
            },
        ]

        structured = context_mgr.build_context(
            query="测试问题",
            retrieved_docs=docs,
        )

        assert "truncated_docs" in structured
        assert len(structured["truncated_docs"]) >= 1

    def test_empty_docs_with_graph_paths(self, context_mgr):
        """Empty docs list should not crash."""
        enriched = context_mgr.enrich_docs_with_graph_paths([])
        assert enriched == []


class TestGraphPathSummaries:
    """Build graph path summaries for context window."""

    def test_empty_docs_no_summary(self):
        """Empty docs produce empty summary."""
        result = _build_graph_path_summaries([])
        assert result == ""

    def test_docs_without_graph_paths_no_summary(self):
        """Docs without graph_paths produce empty summary."""
        docs = [{"chunk_id": "a_0", "content": "text"}]
        result = _build_graph_path_summaries(docs)
        assert result == ""

    def test_docs_with_graph_paths_produce_summary(self):
        """Docs with graph_paths should produce a summary."""
        docs = [
            {
                "chunk_id": "a_0",
                "graph_paths": [
                    {
                        "path_entities": ["Ability", "UIAbility"],
                        "path_relations": ["RELATED_TO"],
                    },
                    {
                        "path_entities": ["UIAbility", "onCreate"],
                        "path_relations": ["HAS_LIFECYCLE"],
                    },
                ],
            },
        ]

        result = _build_graph_path_summaries(docs)
        assert "[知识图谱关系路径]" in result
        assert "Ability" in result
        assert "UIAbility" in result

    def test_summary_deduplicates_paths(self):
        """Duplicate paths should be deduplicated."""
        path = {
            "path_entities": ["A", "B"],
            "path_relations": ["RELATED_TO"],
        }
        docs = [
            {"chunk_id": "a_0", "graph_paths": [path]},
            {"chunk_id": "b_0", "graph_paths": [path]},  # Same path
        ]

        result = _build_graph_path_summaries(docs)
        # Should appear only once
        assert result.count("A → B") <= 1

    def test_summary_limits_to_10_paths(self):
        """Should not exceed 10 paths."""
        docs = [
            {
                "chunk_id": f"c_{i}",
                "graph_paths": [
                    {"path_entities": [f"E{i}", f"E{i+1}"], "path_relations": ["RELATED_TO"]}
                ],
            }
            for i in range(15)
        ]

        result = _build_graph_path_summaries(docs)
        # Count numbered lines (each path starts with a number)
        lines = [l for l in result.split("\n") if l.strip().startswith("  ") and ". " in l]
        assert len(lines) <= 10
