"""DeepIntentNode — orchestrates the full deep intent recognition pipeline.

Pipeline:
    1. rule_based_intent(query) → RuleIntentResult
    2. extract_entities(query) → entities dict
    3. llm_deep_intent_classifier(query, rule_result, entities) → raw dict
    4. validate_deep_intent(raw_dict) → DeepIntentResult
    5. calculate_confidence(result, rule_result, entities) → confidence

This node can be used standalone or as a LangGraph node.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from enterprise_agentic_rag.agents.deep_intent.confidence import calculate_confidence
from enterprise_agentic_rag.agents.deep_intent.entity_extractor import extract_entities
from enterprise_agentic_rag.agents.deep_intent.llm_classifier import llm_deep_intent_classifier
from enterprise_agentic_rag.agents.deep_intent.rules import RuleIntentResult, rule_based_intent
from enterprise_agentic_rag.agents.deep_intent.schema import DeepIntentResult
from enterprise_agentic_rag.agents.deep_intent.validator import validate_deep_intent

logger = logging.getLogger(__name__)


class DeepIntentNode:
    """Orchestrate deep intent recognition with fallback.

    Usage::

        node = DeepIntentNode()
        result = await node.recognize("UIAbility 生命周期是什么？")
        # result.primary_intent → "concept_qa"
        # result.retrieval_plan.mode → "hybrid_only"
    """

    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = use_llm

    async def recognize(self, query: str) -> DeepIntentResult:
        """Run full deep intent recognition pipeline.

        Args:
            query: Raw user query string.

        Returns:
            Validated DeepIntentResult with confidence score.
        """
        t0 = time.time()

        # Stage 1: Rule-based detection (always runs, fast)
        rule_result = rule_based_intent(query)

        # Stage 2: Entity extraction (always runs, fast)
        entities = extract_entities(query)

        # Stage 3: LLM deep intent classification (optional, fallback on failure)
        raw_result: dict[str, Any]
        if self._use_llm:
            try:
                raw_result = await llm_deep_intent_classifier(
                    query=query,
                    rule_result=rule_result,
                    entities=entities,
                )
            except Exception as exc:
                logger.warning("LLM deep intent failed, using rule fallback: %s", exc)
                raw_result = _build_rule_dict(query, rule_result, entities)
        else:
            raw_result = _build_rule_dict(query, rule_result, entities)

        # Stage 4: Validate and correct
        deep_intent = validate_deep_intent(raw_result)

        # Stage 5: Calculate confidence
        confidence = calculate_confidence(
            deep_intent=deep_intent,
            rule_result=rule_result,
            entities_raw=entities,
        )
        deep_intent.confidence = confidence

        latency_ms = (time.time() - t0) * 1000
        logger.info(
            "DeepIntent: primary=%s mode=%s confidence=%.2f tools=%s latency=%.0fms",
            deep_intent.primary_intent,
            deep_intent.retrieval_plan.mode,
            deep_intent.confidence,
            deep_intent.suggested_tools,
            latency_ms,
        )

        return deep_intent


def _build_rule_dict(
    query: str,
    rule_result: RuleIntentResult,
    entities: dict[str, Any],
) -> dict[str, Any]:
    """Build a DeepIntentResult dict from rule-based analysis only.

    Used as fallback when LLM is unavailable or fails.
    """
    primary = rule_result.candidate_intents[0] if rule_result.candidate_intents else "concept_qa"
    secondary = rule_result.candidate_intents[1:] if len(rule_result.candidate_intents) > 1 else []

    # Determine answer_style from primary intent
    style_map = {
        "error_diagnosis": "diagnosis_steps",
        "project_debug": "diagnosis_steps",
        "code_generation": "explanation_with_code",
        "migration": "migration_plan",
        "api_usage": "explanation_with_code",
        "architecture": "architecture_proposal",
        "learning_guidance": "learning_path",
        "concept_qa": "direct_answer",
        "best_practice": "direct_answer",
        "compatibility": "direct_answer",
    }

    return {
        "primary_intent": primary,
        "secondary_intents": secondary,
        "scenario": rule_result.scenario_hints[0] if rule_result.scenario_hints else "",
        "user_goal": f"用户查询: {query[:100]}",
        "query_focus": "",
        "required_context": [],
        "missing_context": [],
        "entities": entities,
        "constraints": {
            "needs_code_example": primary == "code_generation",
            "needs_before_after_code": primary == "migration",
            "needs_checklist": primary in ("error_diagnosis", "project_debug"),
            "prefer_official_docs": True,
            "requires_version_check": primary == "compatibility",
        },
        "difficulty": "medium",
        "risk_level": "low",
        "needs_clarification": False,
        "clarification_questions": [],
        "suggested_tools": rule_result.suggested_tools or ["keyword_search", "vector_search"],
        "retrieval_plan": {
            "mode": rule_result.suggested_mode or "hybrid_only",
            "sources": ["official_docs", "internal_kb"],
            "filters": {},
            "expanded_query": None,
        },
        "answer_style": style_map.get(primary, "direct_answer"),
        "confidence": 0.3,
    }


# ===========================================================================
# Convenience function
# ===========================================================================


async def recognize_deep_intent(query: str, use_llm: bool = True) -> DeepIntentResult:
    """Convenience function for deep intent recognition.

    Args:
        query: Raw user query.
        use_llm: Whether to use LLM classification.

    Returns:
        Validated DeepIntentResult.
    """
    node = DeepIntentNode(use_llm=use_llm)
    return await node.recognize(query)
