"""Tests for retrieval planning primitives used by the graph workflow.

Covers:
- DynamicWeights intent-aware weight generation
"""

from enterprise_agentic_rag.rag.retrieval_router import DynamicWeights


# ===========================================================================
# DynamicWeights tests
# ===========================================================================


class TestDynamicWeights:
    """Test intent-aware dynamic weights."""

    def test_api_usage_weights(self):
        w = DynamicWeights.for_intent("api_usage")
        assert w.keyword_weight > 0.3  # Higher keyword weight for API usage
        assert w.api_reference_weight > 0.3
        assert w.sample_code_weight > 0.2

    def test_error_diagnosis_weights(self):
        w = DynamicWeights.for_intent("error_diagnosis")
        assert w.keyword_weight > 0.4  # Highest keyword weight for errors
        assert w.error_knowledge_weight > 0.3
        assert w.faq_weight > 0.1

    def test_migration_weights(self):
        w = DynamicWeights.for_intent("migration")
        assert w.graph_weight > 0.3  # Graph-heavy for migration
        assert w.migration_guide_weight > 0.3

    def test_compatibility_weights(self):
        w = DynamicWeights.for_intent("compatibility")
        assert w.version_meta_weight > 0.3

    def test_code_generation_weights(self):
        w = DynamicWeights.for_intent("code_generation")
        assert w.sample_code_weight > 0.3  # Code-heavy

    def test_concept_qa_weights(self):
        w = DynamicWeights.for_intent("concept_qa")
        assert w.vector_weight > 0.3  # Semantic-heavy
        assert w.official_doc_weight > 0.3

    def test_weights_serializable(self):
        w = DynamicWeights.for_intent("api_usage")
        d = w.to_dict()
        assert isinstance(d, dict)
        assert "keyword_weight" in d
