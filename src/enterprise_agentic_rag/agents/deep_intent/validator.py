"""DeepIntentResult validator — ensures LLM output conforms to schema.

Performs structural and semantic validation:
1. Field type checks
2. Enum value constraints
3. Logical consistency rules
4. Automatic correction where possible
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.agents.deep_intent.schema import (
    ALLOWED_DIFFICULTIES,
    ALLOWED_INTENTS,
    ALLOWED_MODES,
    ALLOWED_RISK_LEVELS,
    ALLOWED_STYLES,
    ALLOWED_TOOLS,
    DeepIntentResult,
)

logger = logging.getLogger(__name__)


def validate_deep_intent(data: dict[str, Any]) -> DeepIntentResult:
    """Validate and correct a raw deep intent dict into a DeepIntentResult.

    Args:
        data: Raw dict from LLM or rule-based pipeline.

    Returns:
        Validated DeepIntentResult with corrections applied.
    """
    corrections: list[str] = []

    # ── primary_intent ──
    primary_intent = str(data.get("primary_intent", "concept_qa")).lower().strip()
    if primary_intent not in ALLOWED_INTENTS:
        corrections.append(f"primary_intent '{primary_intent}' → 'concept_qa'")
        primary_intent = "concept_qa"

    # ── secondary_intents ──
    secondary_intents = data.get("secondary_intents", [])
    if not isinstance(secondary_intents, list):
        secondary_intents = []
    secondary_intents = [
        s.lower().strip() for s in secondary_intents
        if isinstance(s, str) and s.lower().strip() in ALLOWED_INTENTS
    ]
    # Remove primary from secondary
    secondary_intents = [s for s in secondary_intents if s != primary_intent]

    # ── scenario ──
    scenario = str(data.get("scenario", "")).strip()

    # ── user_goal, query_focus ──
    user_goal = str(data.get("user_goal", "")).strip()
    query_focus = str(data.get("query_focus", "")).strip()

    # ── required_context, missing_context ──
    required_context = _ensure_str_list(data.get("required_context", []))
    missing_context = _ensure_str_list(data.get("missing_context", []))

    # ── entities ──
    entities_raw = data.get("entities", {})
    if not isinstance(entities_raw, dict):
        entities_raw = {}
    entities = {
        "apis": _ensure_str_list(entities_raw.get("apis", [])),
        "components": _ensure_str_list(entities_raw.get("components", [])),
        "errors": _ensure_str_list(entities_raw.get("errors", [])),
        "api_levels": _ensure_str_list(entities_raw.get("api_levels", [])),
        "versions": _ensure_str_list(entities_raw.get("versions", [])),
        "files": _ensure_str_list(entities_raw.get("files", [])),
        "migration_from": _ensure_str_or_none(entities_raw.get("migration_from")),
        "migration_to": _ensure_str_or_none(entities_raw.get("migration_to")),
    }

    # ── constraints ──
    constraints_raw = data.get("constraints", {})
    if not isinstance(constraints_raw, dict):
        constraints_raw = {}
    constraints = {
        "needs_code_example": bool(constraints_raw.get("needs_code_example", False)),
        "needs_before_after_code": bool(constraints_raw.get("needs_before_after_code", False)),
        "needs_checklist": bool(constraints_raw.get("needs_checklist", False)),
        "prefer_official_docs": bool(constraints_raw.get("prefer_official_docs", True)),
        "requires_version_check": bool(constraints_raw.get("requires_version_check", False)),
    }

    # ── difficulty ──
    difficulty = str(data.get("difficulty", "low")).lower().strip()
    if difficulty not in ALLOWED_DIFFICULTIES:
        corrections.append(f"difficulty '{difficulty}' → 'low'")
        difficulty = "low"

    # ── risk_level ──
    risk_level = str(data.get("risk_level", "low")).lower().strip()
    if risk_level not in ALLOWED_RISK_LEVELS:
        corrections.append(f"risk_level '{risk_level}' → 'low'")
        risk_level = "low"

    # ── needs_clarification ──
    needs_clarification = bool(data.get("needs_clarification", False))

    # ── clarification_questions ──
    clarification_questions = _ensure_str_list(data.get("clarification_questions", []))

    # ── suggested_tools ──
    suggested_tools = _ensure_str_list(data.get("suggested_tools", []))
    suggested_tools = [t for t in suggested_tools if t in ALLOWED_TOOLS]

    # ── retrieval_plan ──
    retrieval_plan_raw = data.get("retrieval_plan", {})
    if not isinstance(retrieval_plan_raw, dict):
        retrieval_plan_raw = {}
    mode = str(retrieval_plan_raw.get("mode", "hybrid_only")).lower().strip()
    if mode not in ALLOWED_MODES:
        corrections.append(f"retrieval_plan.mode '{mode}' → 'hybrid_only'")
        mode = "hybrid_only"
    retrieval_plan = {
        "mode": mode,
        "sources": _ensure_str_list(retrieval_plan_raw.get("sources", [])),
        "filters": retrieval_plan_raw.get("filters", {}) if isinstance(retrieval_plan_raw.get("filters"), dict) else {},
        "expanded_query": _ensure_str_or_none(retrieval_plan_raw.get("expanded_query")),
    }

    # ── answer_style ──
    answer_style = str(data.get("answer_style", "direct_answer")).lower().strip()
    if answer_style not in ALLOWED_STYLES:
        corrections.append(f"answer_style '{answer_style}' → 'direct_answer'")
        answer_style = "direct_answer"

    # ── confidence ──
    confidence = data.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    # ── Logical consistency corrections ──
    if needs_clarification and not clarification_questions:
        clarification_questions = ["请提供更多信息以便准确定位问题。"]

    if not suggested_tools:
        suggested_tools = ["keyword_search", "vector_search"]

    if not retrieval_plan["sources"]:
        retrieval_plan["sources"] = ["official_docs", "internal_kb"]

    if corrections:
        logger.info("Deep intent validation corrections: %s", corrections)

    from enterprise_agentic_rag.agents.deep_intent.schema import (
        DeepIntentConstraints,
        DeepIntentEntities,
        RetrievalPlanConfig,
    )

    return DeepIntentResult(
        primary_intent=primary_intent,
        secondary_intents=secondary_intents,
        scenario=scenario,
        user_goal=user_goal,
        query_focus=query_focus,
        required_context=required_context,
        missing_context=missing_context,
        entities=DeepIntentEntities.from_dict(entities),
        constraints=DeepIntentConstraints.from_dict(constraints),
        difficulty=difficulty,
        risk_level=risk_level,
        needs_clarification=needs_clarification,
        clarification_questions=clarification_questions,
        suggested_tools=suggested_tools,
        retrieval_plan=RetrievalPlanConfig.from_dict(retrieval_plan),
        answer_style=answer_style,
        confidence=confidence,
    )


# ===========================================================================
# Helpers
# ===========================================================================


def _ensure_str_list(val: Any) -> list[str]:
    """Ensure a value is a list of strings."""
    if not isinstance(val, list):
        return []
    return [str(v).strip() for v in val if v is not None and str(v).strip()]


def _ensure_str_or_none(val: Any) -> str | None:
    """Ensure a value is a string or None."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None
