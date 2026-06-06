"""Tests for LLM deep intent classifier.

Covers:
- _build_llm_prompt: prompt construction with rule hints and entities
- _extract_json: JSON extraction from various LLM response formats
- _fallback_from_rules: rule-based fallback when LLM fails
- llm_deep_intent_classifier: integration with mock provider
"""

import pytest

from enterprise_agentic_rag.agents.deep_intent.llm_classifier import (
    _build_llm_prompt,
    _extract_json,
    _fallback_from_rules,
    llm_deep_intent_classifier,
)
from enterprise_agentic_rag.agents.deep_intent.rules import RuleIntentResult


# ===========================================================================
# TestPromptBuilding
# ===========================================================================


class TestPromptBuilding:
    """Test _build_llm_prompt constructs correct prompts."""

    def test_build_prompt_with_rule_result(self):
        rule = RuleIntentResult(
            candidate_intents=["api_usage", "code_generation"],
            scenario_hints=["SDK调用"],
            suggested_tools=["keyword_search", "api_reference_search"],
            suggested_mode="parallel",
        )
        prompt = _build_llm_prompt("如何使用X API", rule, None)
        assert "api_usage" in prompt
        assert "code_generation" in prompt
        assert "SDK调用" in prompt
        assert "如何使用X API" in prompt

    def test_build_prompt_with_entities(self):
        entities = {
            "apis": ["abilityManager"],
            "components": ["UIAbility"],
            "errors": [],
            "versions": [],
        }
        prompt = _build_llm_prompt("test query", None, entities)
        assert "abilityManager" in prompt
        assert "UIAbility" in prompt

    def test_build_prompt_minimal(self):
        prompt = _build_llm_prompt("简单问题", None, None)
        assert "用户问题" in prompt
        assert "简单问题" in prompt
        assert "请输出 JSON" in prompt


# ===========================================================================
# TestJsonExtraction
# ===========================================================================


class TestJsonExtraction:
    """Test _extract_json handles various response formats."""

    def test_extract_plain_json(self):
        result = _extract_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_extract_json_with_fence(self):
        result = _extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_json_with_fence_no_lang(self):
        result = _extract_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_json_with_surrounding_text(self):
        result = _extract_json('some text prefix {"key": "value"} and suffix')
        assert result == {"key": "value"}

    def test_extract_json_with_brace_matching(self):
        result = _extract_json('{"outer": {"inner": [1, 2, 3]}, "other": true}')
        assert result == {"outer": {"inner": [1, 2, 3]}, "other": True}

    def test_extract_invalid_json(self):
        assert _extract_json("not json at all") is None

    def test_extract_empty_string(self):
        assert _extract_json("") is None
        assert _extract_json("   ") is None

    def test_extract_no_braces(self):
        assert _extract_json("plain text without braces") is None


# ===========================================================================
# TestFallbackFromRules
# ===========================================================================


class TestFallbackFromRules:
    """Test _fallback_from_rules produces valid fallback dicts."""

    def test_fallback_with_rule_result(self):
        rule = RuleIntentResult(
            candidate_intents=["api_usage", "concept_qa"],
            scenario_hints=["API调用"],
            suggested_tools=["api_reference_search"],
            suggested_mode="parallel",
        )
        result = _fallback_from_rules("test query", rule, None)
        assert result["primary_intent"] == "api_usage"
        assert result["secondary_intents"] == ["concept_qa"]
        assert result["confidence"] == 0.3
        assert result["retrieval_plan"]["mode"] == "parallel"

    def test_fallback_without_rule_result(self):
        result = _fallback_from_rules("test query", None, None)
        assert result["primary_intent"] == "concept_qa"
        assert result["confidence"] == 0.3
        assert result["needs_clarification"] is False

    def test_fallback_confidence_is_low(self):
        result = _fallback_from_rules("test", None, None)
        assert result["confidence"] == 0.3

    def test_fallback_includes_query(self):
        result = _fallback_from_rules("复杂的API问题", None, None)
        assert "复杂的API问题" in result["user_goal"]


# ===========================================================================
# TestLLMClassifierIntegration
# ===========================================================================


@pytest.mark.asyncio
class TestLLMClassifierIntegration:
    """Test llm_deep_intent_classifier with mock provider."""

    async def test_classifier_with_mock_provider(self):
        """Mock provider returns '{}' — classifier should fall back to rules."""
        result = await llm_deep_intent_classifier(
            query="如何使用UIAbility",
            rule_result=None,
            entities=None,
            max_retries=0,
        )
        assert isinstance(result, dict)
        # With mock provider, it falls back to rule-based result
        assert "primary_intent" in result

    async def test_classifier_parse_failure_fallback(self):
        """Verify fallback works even with no rule hints."""
        result = await llm_deep_intent_classifier(
            query="test",
            rule_result=None,
            entities=None,
            max_retries=0,
        )
        assert "primary_intent" in result
        assert result["primary_intent"] in (
            "concept_qa", "api_usage", "code_generation", "error_diagnosis",
            "migration", "compatibility", "project_debug", "best_practice",
            "architecture", "learning_guidance",
        )

    async def test_classifier_with_rule_hints(self):
        """Classifier should propagate rule hints into fallback."""
        rule = RuleIntentResult(
            candidate_intents=["error_diagnosis"],
            scenario_hints=["崩溃排查"],
            suggested_tools=["error_diagnosis_search"],
            suggested_mode="error_first",
        )
        result = await llm_deep_intent_classifier(
            query="应用启动报错401",
            rule_result=rule,
            entities=None,
            max_retries=0,
        )
        assert result["primary_intent"] == "error_diagnosis"
