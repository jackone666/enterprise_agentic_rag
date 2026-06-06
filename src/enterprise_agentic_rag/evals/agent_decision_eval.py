"""Agent Decision Evaluation Set — assess MasterAgent routing accuracy.

Evaluates whether the MasterAgent correctly routes queries to the right
downstream agents/workflows. Each test case specifies:
- The user query
- The expected next_node after deep_intent recognition
- The expected retrieval_mode
- Acceptance criteria

Reference:
    TECHNICAL_DEEP_DIVE.md §38.4 — "MasterAgent 决策评估集"
    Expected impact: master_route_accuracy >= 0.92
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DecisionTestCase:
    """A single agent routing test case."""

    query: str
    expected_intent: str
    expected_next_node: str  # First action after intent recognition
    expected_retrieval_mode: str = "hybrid_only"
    needs_tools: bool = False
    needs_code: bool = False
    difficulty: str = "moderate"
    tags: list[str] = field(default_factory=list)


@dataclass
class DecisionEvalResult:
    """Overall agent decision evaluation result."""

    total_cases: int = 0
    intent_correct: int = 0
    routing_correct: int = 0
    mode_correct: int = 0
    cases: list[dict[str, Any]] = field(default_factory=list)

    @property
    def intent_accuracy(self) -> float:
        return self.intent_correct / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def routing_accuracy(self) -> float:
        return self.routing_correct / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def mode_accuracy(self) -> float:
        return self.mode_correct / self.total_cases if self.total_cases > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "routing_accuracy": round(self.routing_accuracy, 4),
            "mode_accuracy": round(self.mode_accuracy, 4),
            "intent_correct": self.intent_correct,
            "routing_correct": self.routing_correct,
            "mode_correct": self.mode_correct,
            "cases": self.cases,
        }


# ---------------------------------------------------------------------------
# Evaluation dataset — 8 representative test cases (v3.2: 22 → 8)
# ---------------------------------------------------------------------------

AGENT_DECISION_EVAL_SET: list[DecisionTestCase] = [
    # ── concept_qa ──
    DecisionTestCase(
        query="HarmonyOS 应用的生命周期是什么？",
        expected_intent="concept_qa",
        expected_next_node="retrieve_knowledge",
        expected_retrieval_mode="hybrid_only",
        tags=["concept", "lifecycle"],
    ),

    # ── api_usage ──
    DecisionTestCase(
        query="@ohos.app.ability 怎么创建 Ability？",
        expected_intent="api_usage",
        expected_next_node="retrieve_knowledge",
        expected_retrieval_mode="parallel",
        tags=["api", "ability"],
    ),

    # ── code_generation ──
    DecisionTestCase(
        query="用 TypeScript 写一个 HarmonyOS 的网络请求示例",
        expected_intent="code_generation",
        expected_next_node="retrieve_knowledge",
        expected_retrieval_mode="parallel",
        needs_code=True,
        tags=["code", "typescript"],
    ),

    # ── error_diagnosis ──
    DecisionTestCase(
        query="错误码 15500000 是什么原因？",
        expected_intent="error_diagnosis",
        expected_next_node="call_tools",
        expected_retrieval_mode="hybrid_only",
        needs_tools=True,
        tags=["error", "error_code"],
    ),

    # ── migration ──
    DecisionTestCase(
        query="从 FA 模型迁移到 Stage 模型有什么变化？",
        expected_intent="migration",
        expected_next_node="retrieve_knowledge",
        expected_retrieval_mode="graph_first",
        tags=["migration", "fa", "stage"],
    ),

    # ── Edge case (ambiguous query) ──
    DecisionTestCase(
        query="帮我查一下",
        expected_intent="concept_qa",
        expected_next_node="retrieve_knowledge",
        expected_retrieval_mode="hybrid_only",
        difficulty="hard",
        tags=["edge_case", "ambiguous"],
    ),
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_agent_decisions(use_llm: bool = False) -> DecisionEvalResult:
    """Evaluate MasterAgent routing decisions against the eval set.

    Args:
        use_llm: Whether to use LLM-based routing (default: rule-based only).

    Returns:
        DecisionEvalResult with accuracy metrics and per-case details.
    """
    from enterprise_agentic_rag.agents.master_agent import MasterAgent

    master = MasterAgent()
    result = DecisionEvalResult()
    result.total_cases = len(AGENT_DECISION_EVAL_SET)

    for case in AGENT_DECISION_EVAL_SET:
        case_result = _eval_single_case(case, master)
        result.cases.append(case_result)

        if case_result["intent_correct"]:
            result.intent_correct += 1
        if case_result["routing_correct"]:
            result.routing_correct += 1
        if case_result["mode_correct"]:
            result.mode_correct += 1

    logger.info(
        "Agent decision eval: intent=%.2f routing=%.2f mode=%.2f (%d cases)",
        result.intent_accuracy, result.routing_accuracy,
        result.mode_accuracy, result.total_cases,
    )

    return result


def _eval_single_case(case: DecisionTestCase, master: MasterAgent) -> dict[str, Any]:
    """Evaluate a single test case."""
    from enterprise_agentic_rag.agents.deep_intent.rules import rule_based_intent

    # Simulate deep intent
    rule_result = rule_based_intent(case.query)
    detected_intent = rule_result.primary_intent

    retrieval_plan = rule_result.retrieval_plan
    detected_mode = retrieval_plan.get("mode", "hybrid_only") if retrieval_plan else "hybrid_only"

    # Simulate state after deep intent
    sim_state = {
        "query": case.query,
        "deep_intent": rule_result.to_dict() if hasattr(rule_result, 'to_dict') else {},
        "deep_intent_confidence": getattr(rule_result, 'confidence', 0.5),
        "last_agent_step": "recognize_intent",
        "retrieved_docs": [],
        "retry_count": {},
        "tool_errors": [],
        "code_snippet": "",
        "code_verified": False,
        "code_retry_count": 0,
        "verified": False,
        "fallback_reason": "",
    }

    # Master decides
    decision = master._rule_decide(sim_state)

    # Check correctness
    intent_correct = detected_intent == case.expected_intent
    routing_correct = decision.next_node == case.expected_next_node
    mode_correct = detected_mode == case.expected_retrieval_mode

    return {
        "query": case.query[:80],
        "expected_intent": case.expected_intent,
        "detected_intent": detected_intent,
        "intent_correct": intent_correct,
        "expected_next": case.expected_next_node,
        "actual_next": decision.next_node,
        "routing_correct": routing_correct,
        "expected_mode": case.expected_retrieval_mode,
        "detected_mode": detected_mode,
        "mode_correct": mode_correct,
        "difficulty": case.difficulty,
        "tags": case.tags,
    }
