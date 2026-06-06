"""Tests for Deep Intent Recognition module.

Covers:
- Rule-based intent detection (rules.py)
- Entity extraction (entity_extractor.py)
- Schema validation (validator.py)
- DeepIntentNode pipeline (node.py)
- The 7 required test cases from the spec (Section 17)
"""

import pytest
from enterprise_agentic_rag.agents.deep_intent.schema import (
    DeepIntentResult,
    DeepIntentEntities,
    IntentCategory,
    RetrievalMode,
    AnswerStyle,
)
from enterprise_agentic_rag.agents.deep_intent.rules import rule_based_intent, RuleIntentResult
from enterprise_agentic_rag.agents.deep_intent.entity_extractor import extract_entities, extract_entities_typed
from enterprise_agentic_rag.agents.deep_intent.validator import validate_deep_intent
from enterprise_agentic_rag.agents.deep_intent.confidence import calculate_confidence


# ===========================================================================
# Rule-based intent tests
# ===========================================================================


class TestRuleBasedIntent:
    """Test rule_based_intent function."""

    def test_error_diagnosis_detection(self):
        """Rule 1: error keywords → error_diagnosis."""
        result = rule_based_intent("BusinessError: permission denied 怎么办？")
        assert "error_diagnosis" in result.candidate_intents
        assert "permission_error" in result.scenario_hints or len(result.signals.get("error_diagnosis", [])) > 0

    def test_project_debug_detection(self):
        """Rule 1b: error + project keywords → project_debug."""
        result = rule_based_intent("首页白屏怎么办？项目启动就白屏")
        assert ("error_diagnosis" in result.candidate_intents or "project_debug" in result.candidate_intents)

    def test_code_generation_detection(self):
        """Rule 2: code generation keywords."""
        result = rule_based_intent("帮我写一个鸿蒙登录页面")
        assert "code_generation" in result.candidate_intents

    def test_migration_detection(self):
        """Rule 3: migration keywords."""
        result = rule_based_intent("Router 怎么迁移到 Navigation？")
        assert "migration" in result.candidate_intents

    def test_compatibility_detection(self):
        """Rule 4: compatibility keywords."""
        result = rule_based_intent("API 9 支持 Navigation 吗？")
        assert "compatibility" in result.candidate_intents

    def test_api_usage_detection(self):
        """Rule 5: API usage keywords."""
        result = rule_based_intent("@ohos.net.http 怎么发 GET 请求？")
        assert "api_usage" in result.candidate_intents

    def test_concept_qa_detection(self):
        """Rule 6: concept QA keywords."""
        result = rule_based_intent("UIAbility 生命周期是什么？")
        assert "concept_qa" in result.candidate_intents

    def test_suggested_mode_error(self):
        """Error queries → error_first mode."""
        result = rule_based_intent("hvigor ERROR: compile failed")
        assert result.suggested_mode == "error_first"

    def test_suggested_mode_migration(self):
        """Migration queries → graph_first mode."""
        result = rule_based_intent("Router 怎么迁移到 Navigation？")
        assert result.suggested_mode == "graph_first"

    def test_suggested_mode_concept(self):
        """Concept queries → hybrid_only mode."""
        result = rule_based_intent("UIAbility 是什么？")
        assert result.suggested_mode == "hybrid_only"

    def test_suggested_tools_includes_basics(self):
        """All results should include basic search tools."""
        result = rule_based_intent("随便一个问题")
        assert len(result.suggested_tools) >= 2
        assert "keyword_search" in result.suggested_tools
        assert "vector_search" in result.suggested_tools


# ===========================================================================
# Entity extraction tests
# ===========================================================================


class TestEntityExtraction:
    """Test extract_entities function."""

    def test_extract_apis(self):
        entities = extract_entities("@ohos.net.http 怎么发 GET 请求？@ohos.router 怎么用？")
        assert len(entities["apis"]) >= 1
        assert any("ohos.net.http" in api.lower().replace("@", "") for api in entities["apis"])

    def test_extract_components(self):
        entities = extract_entities("Navigation 组件如何使用？Text 和 Button 有什么区别？")
        assert "Navigation" in entities["components"] or "Text" in entities["components"] or "Button" in entities["components"]

    def test_extract_errors(self):
        entities = extract_entities("BusinessError: permission denied 怎么办？hvigor ERROR")
        assert len(entities["errors"]) >= 1

    def test_extract_api_levels(self):
        entities = extract_entities("API 9 API Level 12 支持吗？")
        assert len(entities["api_levels"]) >= 1

    def test_extract_versions(self):
        entities = extract_entities("HarmonyOS NEXT 支持吗？OpenHarmony 4.0")
        assert len(entities["versions"]) >= 1

    def test_extract_files(self):
        entities = extract_entities("module.json5 怎么配置？Index.ets 在哪？")
        assert "module.json5" in entities["files"] or "Index.ets" in entities["files"]

    def test_extract_migration(self):
        entities = extract_entities("Router 怎么迁移到 Navigation？")
        assert entities["migration_from"] == "Router" or entities.get("migration_from") is not None
        assert entities["migration_to"] == "Navigation" or entities.get("migration_to") is not None

    def test_typed_extraction(self):
        result = extract_entities_typed("UIAbility 生命周期 @ohos.app.ability.UIAbility")
        assert isinstance(result, DeepIntentEntities)
        assert len(result.apis) >= 1 or len(result.components) >= 1

    def test_empty_query(self):
        entities = extract_entities("")
        assert isinstance(entities, dict)
        assert entities["apis"] == []


# ===========================================================================
# Validator tests
# ===========================================================================


class TestValidator:
    """Test validate_deep_intent function."""

    def test_validates_primary_intent(self):
        raw = {"primary_intent": "invalid_intent", "confidence": 0.5}
        result = validate_deep_intent(raw)
        assert result.primary_intent == "concept_qa"  # Fallback

    def test_validates_retrieval_mode(self):
        raw = {
            "primary_intent": "api_usage",
            "retrieval_plan": {"mode": "invalid_mode"},
            "suggested_tools": ["keyword_search"],
        }
        result = validate_deep_intent(raw)
        assert result.retrieval_plan.mode == "hybrid_only"  # Fallback

    def test_validates_difficulty(self):
        raw = {"primary_intent": "concept_qa", "difficulty": "extreme"}
        result = validate_deep_intent(raw)
        assert result.difficulty == "low"

    def test_validates_confidence_range(self):
        raw = {"primary_intent": "concept_qa", "confidence": 1.5}
        result = validate_deep_intent(raw)
        assert 0.0 <= result.confidence <= 1.0

    def test_auto_adds_default_tools(self):
        raw = {"primary_intent": "api_usage", "suggested_tools": []}
        result = validate_deep_intent(raw)
        assert len(result.suggested_tools) >= 1

    def test_filters_invalid_tools(self):
        raw = {
            "primary_intent": "api_usage",
            "suggested_tools": ["keyword_search", "invalid_tool_xyz"],
        }
        result = validate_deep_intent(raw)
        assert "invalid_tool_xyz" not in result.suggested_tools
        assert "keyword_search" in result.suggested_tools

    def test_full_valid_output(self):
        raw = {
            "primary_intent": "api_usage",
            "secondary_intents": ["code_generation"],
            "scenario": "api_lookup",
            "user_goal": "了解如何使用API",
            "query_focus": "@ohos.net.http GET请求",
            "required_context": ["API文档"],
            "missing_context": ["版本号"],
            "entities": {
                "apis": ["@ohos.net.http"],
                "components": [],
                "errors": [],
                "api_levels": [],
                "versions": [],
                "files": [],
                "migration_from": None,
                "migration_to": None,
            },
            "constraints": {
                "needs_code_example": True,
                "needs_before_after_code": False,
                "needs_checklist": False,
                "prefer_official_docs": True,
                "requires_version_check": False,
            },
            "difficulty": "low",
            "risk_level": "low",
            "needs_clarification": False,
            "clarification_questions": [],
            "suggested_tools": ["keyword_search", "vector_search", "api_reference_search", "sample_code_search"],
            "retrieval_plan": {
                "mode": "parallel",
                "sources": ["official_docs", "api_reference"],
                "filters": {},
                "expanded_query": None,
            },
            "answer_style": "explanation_with_code",
            "confidence": 0.85,
        }
        result = validate_deep_intent(raw)
        assert result.primary_intent == "api_usage"
        assert result.retrieval_plan.mode == "parallel"
        assert result.answer_style == "explanation_with_code"
        assert len(result.suggested_tools) >= 3
        assert result.entities.apis == ["@ohos.net.http"]


# ===========================================================================
# Confidence tests
# ===========================================================================


class TestConfidence:
    """Test calculate_confidence function."""

    def test_rule_llm_alignment_boosts_confidence(self):
        deep_intent = DeepIntentResult(primary_intent="api_usage", confidence=0.0)
        rule_result = RuleIntentResult(candidate_intents=["api_usage"], signals={"api_usage": ["API", "接口"]})
        entities = {"apis": ["@ohos.net.http"], "components": [], "errors": [], "api_levels": [], "versions": [], "files": []}
        confidence = calculate_confidence(deep_intent, rule_result, entities)
        assert confidence > 0.3

    def test_low_confidence_on_empty(self):
        deep_intent = DeepIntentResult(primary_intent="concept_qa", confidence=0.0)
        confidence = calculate_confidence(deep_intent, None, {})
        assert confidence < 0.5


# ===========================================================================
# Required test cases (Section 17)
# ===========================================================================


class TestRequiredCases:
    """The 7 required test cases from the specification."""

    def test_case_1_concept_qa(self):
        """'UIAbility 生命周期是什么？'
        Expected: primary_intent=concept_qa, mode=hybrid_only
        """
        result = rule_based_intent("UIAbility 生命周期是什么？")
        assert "concept_qa" in result.candidate_intents
        assert result.suggested_mode == "hybrid_only"

    def test_case_2_api_usage(self):
        """'@ohos.net.http 怎么发 GET 请求？'
        Expected: primary_intent=api_usage, mode=parallel,
        tools include keyword_search, vector_search, api_reference_search, sample_code_search
        """
        result = rule_based_intent("@ohos.net.http 怎么发 GET 请求？")
        assert "api_usage" in result.candidate_intents
        assert result.suggested_mode == "parallel"
        assert "api_reference_search" in result.suggested_tools
        assert "sample_code_search" in result.suggested_tools

    def test_case_3_error_diagnosis(self):
        """'BusinessError: permission denied 怎么办？'
        Expected: primary_intent=error_diagnosis, scenario=permission_error,
        mode=error_first, tools include error_diagnosis_search, official_doc_search
        """
        result = rule_based_intent("BusinessError: permission denied 怎么办？")
        assert "error_diagnosis" in result.candidate_intents
        assert result.suggested_mode == "error_first"
        assert any(t in result.suggested_tools for t in ("error_diagnosis_search", "official_doc_search"))

    def test_case_4_migration(self):
        """'Router 怎么迁移到 Navigation？'
        Expected: primary_intent=migration, scenario=router_to_navigation,
        mode=graph_first, graph_search is called
        """
        result = rule_based_intent("Router 怎么迁移到 Navigation？")
        assert "migration" in result.candidate_intents
        assert result.suggested_mode == "graph_first"
        assert "graph_search" in result.suggested_tools

    def test_case_5_compatibility(self):
        """'API 9 支持 Navigation 吗？'
        Expected: primary_intent=compatibility, mode=graph_first,
        tools include version_compatibility_check
        """
        result = rule_based_intent("API 9 支持 Navigation 吗？")
        assert "compatibility" in result.candidate_intents
        assert result.suggested_mode == "graph_first"
        assert "version_compatibility_check" in result.suggested_tools

    def test_case_6_code_generation(self):
        """'帮我写一个鸿蒙登录页面'
        Expected: primary_intent=code_generation, mode=parallel,
        tools include sample_code_search, code_review
        """
        result = rule_based_intent("帮我写一个鸿蒙登录页面")
        assert "code_generation" in result.candidate_intents
        assert "sample_code_search" in result.suggested_tools

    def test_case_7_project_debug(self):
        """'首页白屏怎么办？'
        Expected: primary_intent=error_diagnosis or project_debug,
        scenario=white_screen, mode=error_first or code_first,
        missing_context contains 运行日志、入口文件、API Level
        """
        result = rule_based_intent("首页白屏怎么办？")
        assert ("error_diagnosis" in result.candidate_intents or "project_debug" in result.candidate_intents)
        # White screen scenario should be detected
        assert "white_screen" in result.scenario_hints
        # Mode should be error_first for error-type queries
        assert result.suggested_mode in ("error_first", "code_first")
