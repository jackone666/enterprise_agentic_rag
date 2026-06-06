"""Tests for GraphRetriever.

Covers:
- Entity lookup with matching terms
- Retrieval returns empty when Neo4j unavailable
- GraphRetriever.available check
- Candidate structure validation
"""

from __future__ import annotations

import pytest

from enterprise_agentic_rag.rag.graph.graph_retriever import GraphRetriever


@pytest.fixture
def retriever():
    return GraphRetriever()


class TestGraphRetrieverBasic:
    """Basic functionality tests (no Neo4j required)."""

    def test_available_property(self, retriever):
        """available should return a boolean."""
        avail = retriever.available
        assert isinstance(avail, bool)

    def test_retrieve_empty_when_unavailable(self, retriever):
        """When Neo4j is not available, retrieve returns empty list."""
        if retriever.available:
            pytest.skip("Neo4j is available — skipping unavailable test")

        result = retriever.retrieve(
            query_analysis={"entities": ["Ability"]},
            top_k=5,
        )
        assert result == []

    def test_retrieve_accepts_query_analysis(self, retriever):
        """retrieve should accept query_analysis dict."""
        if not retriever.available:
            result = retriever.retrieve(
                query_analysis={
                    "entities": ["Ability", "UIAbility"],
                    "keywords": ["生命周期", "页面"],
                    "intent": "technical_question",
                },
                top_k=5,
                graph_depth=1,
            )
            # Returns empty when not available (not crashing is the test)
            assert isinstance(result, list)

    def test_retrieve_accepts_raw_query_fallback(self, retriever):
        """When no entities in query_analysis, falls back to raw query."""
        if not retriever.available:
            result = retriever.retrieve(
                query="EntryAbility onCreate 参数",
                query_analysis={},
                top_k=5,
            )
            assert isinstance(result, list)

    def test_entity_term_collection_from_query_analysis(self, retriever):
        """_collect_entity_terms extracts entities from query_analysis."""
        terms = retriever._collect_entity_terms(
            query_analysis={"entities": ["Ability", "onCreate"]},
            query="",
        )
        assert "Ability" in terms or "ability" in [t.lower() for t in terms]
        assert "onCreate" in terms or "oncreate" in [t.lower() for t in terms]

    def test_entity_term_collection_from_keywords_fallback(self, retriever):
        """When entities is empty, uses keywords."""
        terms = retriever._collect_entity_terms(
            query_analysis={"keywords": ["API", "认证"]},
            query="",
        )
        assert len(terms) >= 2

    def test_entity_term_collection_from_raw_query(self, retriever):
        """When both entities and keywords are empty, extracts from raw query."""
        terms = retriever._collect_entity_terms(
            query_analysis={},
            query="EntryAbility onCreate 9568321 @ohos.router",
        )
        # Should extract CamelCase, numbers, and decorator patterns
        assert len(terms) > 0

    def test_entity_match_score_exact(self, retriever):
        """Exact match should score 1.0."""
        score = retriever._entity_match_score("Ability", "Ability")
        assert score == 1.0

    def test_entity_match_score_prefix(self, retriever):
        """Prefix match should score 0.8."""
        score = retriever._entity_match_score("Abil", "Ability")
        assert score == 0.8

    def test_entity_match_score_contains(self, retriever):
        """Contains match should score 0.6."""
        score = retriever._entity_match_score("bilit", "Ability")
        assert score == 0.6

    def test_graph_score_calculation(self, retriever):
        """Scoring helpers return valid values."""
        entities = [{"name": "Ability"}]
        paths = [{"path_entities": ["Ability", "UIAbility"], "relation_weight": 1.0, "path_length": 1}]

        entity_score = retriever._calc_entity_match_score(entities, paths)
        assert 0.0 <= entity_score <= 1.0

        rel_weight = retriever._calc_relation_weight(paths)
        assert rel_weight > 0

        path_penalty = retriever._calc_path_length_penalty(paths)
        assert path_penalty > 0


class TestGraphRetrieverWithNeo4j:
    """Tests that require Neo4j to be running with data."""

    @pytest.mark.skip(reason="Requires Neo4j with indexed data")
    def test_entity_lookup(self, retriever):
        """Find entities in Neo4j by name."""
        if not retriever.available:
            pytest.skip("Neo4j not available")

        entities = retriever._find_entities(["Ability"])
        assert len(entities) >= 0

    @pytest.mark.skip(reason="Requires Neo4j with indexed data")
    def test_neighbor_expansion(self, retriever):
        """Expand from entity to neighbors."""
        if not retriever.available:
            pytest.skip("Neo4j not available")

        entity = {
            "entity_id": "CLASS:ability",
            "name": "Ability",
            "type": "CLASS",
            "normalized_name": "ability",
        }
        paths = retriever._expand_neighbors([entity], depth=1)
        assert isinstance(paths, list)

    @pytest.mark.skip(reason="Requires Neo4j with indexed data")
    def test_two_hop_expansion(self, retriever):
        """Two-hop neighbor expansion."""
        if not retriever.available:
            pytest.skip("Neo4j not available")

        entity = {
            "entity_id": "CLASS:ability",
            "name": "Ability",
            "type": "CLASS",
            "normalized_name": "ability",
        }
        paths_1hop = retriever._expand_neighbors([entity], depth=1)
        paths_2hop = retriever._expand_neighbors([entity], depth=2)
        # 2-hop should find at least as many paths as 1-hop
        assert len(paths_2hop) >= len(paths_1hop)
