"""Confidence scoring for deep intent recognition.

Calculates a confidence score [0.0, 1.0] based on:
1. Rule-LLM alignment
2. Entity extraction quality
3. Query clarity
4. Scenario detection certainty
5. Primary intent signal strength
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.agents.deep_intent.rules import RuleIntentResult
from enterprise_agentic_rag.agents.deep_intent.schema import DeepIntentResult


def calculate_confidence(
    deep_intent: DeepIntentResult,
    rule_result: RuleIntentResult | None = None,
    entities_raw: dict[str, Any] | None = None,
    llm_raw_response: str = "",
) -> float:
    """Calculate overall confidence score for deep intent result.

    Components:
        - intent_signal_strength: How many rule signals support the primary intent (0.0-0.3)
        - entity_coverage: How well entities are extracted (0.0-0.2)
        - scenario_certainty: Whether scenario is detected (0.0-0.15)
        - llm_rule_alignment: Whether LLM agrees with rules (0.0-0.15)
        - query_clarity: How clear the query is (0.0-0.1)
        - tool_coverage: Whether suggested tools make sense (0.0-0.1)

    Args:
        deep_intent: The validated DeepIntentResult.
        rule_result: Optional RuleIntentResult for alignment check.
        entities_raw: Optional raw entity dict for coverage check.
        llm_raw_response: Raw LLM response for quality check.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    score = 0.0

    # 1. Intent signal strength (0.0 - 0.3)
    if rule_result and rule_result.signals:
        primary = deep_intent.primary_intent
        signal_count = len(rule_result.signals.get(primary, []))
        total_signals = sum(len(v) for v in rule_result.signals.values())
        if total_signals > 0:
            ratio = signal_count / total_signals
            # More signals → higher confidence
            if signal_count >= 5:
                score += 0.3
            elif signal_count >= 3:
                score += 0.25
            elif signal_count >= 2:
                score += 0.2
            elif signal_count >= 1:
                score += 0.15
        else:
            score += 0.05  # No signals → very low confidence

        # Multiple candidate intents → lower confidence
        if len(rule_result.candidate_intents) > 3:
            score -= 0.05
    else:
        score += 0.1  # Baseline without rules

    # 2. Entity coverage (0.0 - 0.2)
    if entities_raw:
        coverage = _entity_coverage_score(entities_raw)
        score += coverage * 0.2

    # 3. Scenario certainty (0.0 - 0.15)
    if deep_intent.scenario:
        score += 0.15
    elif rule_result and rule_result.scenario_hints:
        score += 0.1

    # 4. LLM-Rule alignment (0.0 - 0.15)
    if rule_result and rule_result.candidate_intents:
        if deep_intent.primary_intent in rule_result.candidate_intents:
            score += 0.15
        elif any(i in rule_result.candidate_intents for i in deep_intent.secondary_intents):
            score += 0.08

    # 5. Query clarity (0.0 - 0.1)
    if not deep_intent.needs_clarification:
        score += 0.1
    else:
        # Fewer clarification questions → more certain
        n_questions = len(deep_intent.clarification_questions)
        score += max(0.0, 0.1 - n_questions * 0.03)

    # 6. Tool coverage (0.0 - 0.1)
    if len(deep_intent.suggested_tools) >= 3:
        score += 0.1
    elif len(deep_intent.suggested_tools) >= 1:
        score += 0.05

    # 7. LLM response quality bonus (0.0 - 0.05)
    if llm_raw_response:
        # Valid JSON with all required fields → bonus
        if len(llm_raw_response) > 100:
            score += 0.05
        elif len(llm_raw_response) > 50:
            score += 0.03

    # Clamp
    return max(0.0, min(1.0, round(score, 4)))


def _entity_coverage_score(entities_raw: dict[str, Any]) -> float:
    """Calculate how well entities are extracted from the query.

    Returns a score in [0.0, 1.0].
    """
    if not isinstance(entities_raw, dict):
        return 0.0

    categories = ["apis", "components", "errors", "api_levels", "versions", "files"]
    total = 0
    for cat in categories:
        vals = entities_raw.get(cat, [])
        if isinstance(vals, list) and vals:
            total += 1

    # Migration detection bonus
    if entities_raw.get("migration_from") and entities_raw.get("migration_to"):
        total += 1

    return min(1.0, total / max(len(categories), 1))
