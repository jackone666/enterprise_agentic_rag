"""Deep Intent Recognition — multi-layer intent analysis for HarmonyOS developer Q&A.

Pipeline:
    1. rule_based_intent(query) → candidate intents + signals
    2. extract_entities(query) → structured entity dict
    3. llm_deep_intent_classifier(query, rule_result, entities) → DeepIntentResult
    4. validate_deep_intent(result) → validated DeepIntentResult
    5. calculate_confidence(result) → confidence score

The DeepIntentNode orchestrates this pipeline with automatic fallback.
"""

from enterprise_agentic_rag.agents.deep_intent.schema import (
    DeepIntentResult,
    DeepIntentEntities,
    DeepIntentConstraints,
    RetrievalPlanConfig,
    IntentCategory,
    RetrievalMode,
    AnswerStyle,
    Difficulty,
    RiskLevel,
)
from enterprise_agentic_rag.agents.deep_intent.rules import rule_based_intent, RuleIntentResult
from enterprise_agentic_rag.agents.deep_intent.entity_extractor import extract_entities
from enterprise_agentic_rag.agents.deep_intent.llm_classifier import llm_deep_intent_classifier
from enterprise_agentic_rag.agents.deep_intent.validator import validate_deep_intent
from enterprise_agentic_rag.agents.deep_intent.confidence import calculate_confidence
from enterprise_agentic_rag.agents.deep_intent.node import DeepIntentNode, recognize_deep_intent

__all__ = [
    "DeepIntentResult",
    "DeepIntentEntities",
    "DeepIntentConstraints",
    "RetrievalPlanConfig",
    "IntentCategory",
    "RetrievalMode",
    "AnswerStyle",
    "Difficulty",
    "RiskLevel",
    "RuleIntentResult",
    "rule_based_intent",
    "extract_entities",
    "llm_deep_intent_classifier",
    "validate_deep_intent",
    "calculate_confidence",
    "DeepIntentNode",
    "recognize_deep_intent",
]
