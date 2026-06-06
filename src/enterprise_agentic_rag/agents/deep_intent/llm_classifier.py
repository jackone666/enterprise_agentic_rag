"""LLM-based deep intent classifier for HarmonyOS developer queries.

Invokes the LLM to produce a structured DeepIntentResult.
Includes automatic retry on JSON parse failure and fallback to rule-based result.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from enterprise_agentic_rag.agents.deep_intent.rules import RuleIntentResult

logger = logging.getLogger(__name__)

# ===========================================================================
# LLM Prompt
# ===========================================================================

DEEP_INTENT_SYSTEM_PROMPT = """你是鸿蒙开发智能问答系统的深意图识别器。
你的任务**不是回答用户问题**，而是识别用户真实开发目标，并输出结构化 JSON。

你需要判断并输出以下 JSON 结构：
{
  "primary_intent": "concept_qa | api_usage | code_generation | error_diagnosis | migration",
  "secondary_intents": [],
  "scenario": "",
  "user_goal": "",
  "query_focus": "",
  "required_context": [],
  "missing_context": [],
  "entities": {
    "apis": [],
    "components": [],
    "errors": [],
    "api_levels": [],
    "versions": [],
    "files": [],
    "migration_from": null,
    "migration_to": null
  },
  "constraints": {
    "needs_code_example": false,
    "needs_before_after_code": false,
    "needs_checklist": false,
    "prefer_official_docs": true,
    "requires_version_check": false
  },
  "difficulty": "low | medium | high",
  "risk_level": "low | medium | high",
  "needs_clarification": false,
  "clarification_questions": [],
  "suggested_tools": [],
  "retrieval_plan": {
    "mode": "hybrid_only | parallel | graph_first",
    "sources": [],
    "filters": {},
    "expanded_query": null
  },
  "answer_style": "direct_answer | explanation_with_code | diagnosis_steps | migration_plan | architecture_proposal | learning_path",
  "confidence": 0.0
}

判断原则：
1. 报错类问题优先 error_diagnosis。
2. 用户贴代码或描述项目现象（白屏、黑屏、启动失败、首页打不开等），优先 error_diagnosis。
3. 用户要求写代码、生成示例、封装，必须包含 code_generation。
4. 用户提到迁移、升级、替换、废弃 API，优先 migration。
5. 不要轻易追问。能先给通用答案时 needs_clarification=false，但 missing_context 要列出缺失信息。
6. 如果没有关键上下文无法定位问题（例如只知道白屏但不知道任何版本/环境），needs_clarification=true。
7. suggested_tools 必须从允许工具列表中选择。
8. retrieval_plan.mode 必须从允许模式中选择。
9. 如果检测到错误码或报错信息，scenario 应准确反映错误类型。
10. 如果有迁移关系，entities 中 migration_from 和 migration_to 必须填充。

允许工具：keyword_search, vector_search, graph_search, hybrid_rag_search, official_doc_search, api_reference_search, sample_code_search, error_diagnosis_search, ticket_search, version_compatibility_check, code_review

允许 retrieval_plan.mode：hybrid_only, parallel, graph_first

只输出 JSON，不要输出任何其他文本。"""


# ===========================================================================
# LLM Classifier
# ===========================================================================


async def llm_deep_intent_classifier(
    query: str,
    rule_result: RuleIntentResult | None = None,
    entities: dict[str, Any] | None = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Invoke LLM to classify deep intent.

    Args:
        query: Raw user query.
        rule_result: Rule-based intent analysis result (optional).
        entities: Pre-extracted entities (optional).
        max_retries: Max JSON parse retries (default 1).

    Returns:
        Raw dict output from LLM (unvalidated — pass to validate_deep_intent).
        If all retries fail, returns a fallback dict built from rule_result + entities.
    """
    prompt = _build_llm_prompt(query, rule_result, entities)

    for attempt in range(max_retries + 1):
        try:
            raw = await _call_llm(prompt)

            # Extract JSON from response
            parsed = _extract_json(raw)
            if parsed:
                return parsed

            logger.warning(
                "LLM deep intent JSON parse failed (attempt %d/%d). Raw: %s...",
                attempt + 1, max_retries + 1, raw[:200],
            )

        except Exception as exc:
            logger.warning(
                "LLM deep intent call failed (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, exc,
            )

    # All retries exhausted → fallback to rule-based
    logger.warning("LLM deep intent all attempts failed — falling back to rule-based result")
    return _fallback_from_rules(query, rule_result, entities)


# ===========================================================================
# Prompt builder
# ===========================================================================


def _build_llm_prompt(
    query: str,
    rule_result: RuleIntentResult | None,
    entities: dict[str, Any] | None,
) -> str:
    """Build the full LLM prompt with rule hints and extracted entities."""
    parts = [DEEP_INTENT_SYSTEM_PROMPT]

    # Add rule result as hints
    if rule_result:
        parts.append("\n## 规则识别提示")
        parts.append(f"候选意图: {', '.join(rule_result.candidate_intents)}")
        if rule_result.scenario_hints:
            parts.append(f"场景提示: {', '.join(rule_result.scenario_hints)}")
        if rule_result.suggested_tools:
            parts.append(f"建议工具: {', '.join(rule_result.suggested_tools)}")
        if rule_result.suggested_mode:
            parts.append(f"建议检索模式: {rule_result.suggested_mode}")

    # Add extracted entities
    if entities:
        parts.append("\n## 已提取实体")
        for key, val in entities.items():
            if val:
                if isinstance(val, list) and val:
                    parts.append(f"- {key}: {', '.join(str(v) for v in val)}")
                elif isinstance(val, str) and val:
                    parts.append(f"- {key}: {val}")

    # Add user query
    parts.append(f"\n## 用户问题\n{query}")
    parts.append("\n请输出 JSON：")

    return "\n".join(parts)


# ===========================================================================
# LLM invocation
# ===========================================================================


async def _call_llm(prompt: str) -> str:
    """Call LLM provider and return raw response text."""
    try:
        from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
        provider = get_llm_provider()

        if provider.provider_name == "mock":
            return "{}"

        resp = await provider.generate(prompt, temperature=0.0, max_tokens=2048)
        if resp.success and resp.content:
            return resp.content.strip()
        return "{}"
    except Exception as exc:
        logger.warning("LLM provider call failed: %s", exc)
        raise


# ===========================================================================
# JSON extraction
# ===========================================================================


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM response text.

    Handles:
    - Plain JSON: {"key": "value"}
    - JSON in code fences: ```json {...} ```
    - JSON with surrounding text
    """
    if not text or not text.strip():
        return None

    # Try code fence extraction
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find JSON object boundaries
    # First try: the whole text is JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: extract text between { and } using brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                break

    return None


# ===========================================================================
# Fallback from rules
# ===========================================================================


def _fallback_from_rules(
    query: str,
    rule_result: RuleIntentResult | None,
    entities: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a fallback DeepIntentResult dict from rule results and entities.

    Used when LLM classification fails entirely.
    """
    rule = rule_result or RuleIntentResult()
    ents = entities or {}

    primary = rule.candidate_intents[0] if rule.candidate_intents else "concept_qa"
    secondary = rule.candidate_intents[1:] if len(rule.candidate_intents) > 1 else []

    return {
        "primary_intent": primary,
        "secondary_intents": secondary,
        "scenario": rule.scenario_hints[0] if rule.scenario_hints else "",
        "user_goal": f"用户查询: {query[:100]}",
        "query_focus": "",
        "required_context": [],
        "missing_context": [],
        "entities": ents,
        "constraints": {
            "needs_code_example": primary == "code_generation",
            "needs_before_after_code": primary == "migration",
            "needs_checklist": primary == "error_diagnosis",
            "prefer_official_docs": True,
            "requires_version_check": False,
        },
        "difficulty": "medium",
        "risk_level": "low",
        "needs_clarification": False,
        "clarification_questions": [],
        "suggested_tools": rule.suggested_tools or ["keyword_search", "vector_search"],
        "retrieval_plan": {
            "mode": rule.suggested_mode or "hybrid_only",
            "sources": ["official_docs", "internal_kb"],
            "filters": {},
            "expanded_query": None,
        },
        "answer_style": "direct_answer",
        "confidence": 0.3,  # Low confidence marker
    }
