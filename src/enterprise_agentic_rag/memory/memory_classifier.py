"""Rule-based memory routing for conversation facts.

The classifier is deliberately cheap and explainable. It decides which memory
layer should receive a message before optional LLM extraction is introduced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryTarget(StrEnum):
    """Writable memory layers."""

    SHORT_TERM = "short_term"
    SUMMARY = "summary"
    USER_PROFILE = "user_profile"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CHECKPOINT = "checkpoint"
    SKIP = "skip"


@dataclass(frozen=True)
class MemoryDecision:
    """A routed memory candidate with audit-friendly reasons."""

    target: MemoryTarget
    content: str
    score: int
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


SENSITIVE_PATTERNS = [
    r"\b\d{6}\b",
    r"(?i)api[_-]?key\s*[:=]\s*\S+",
    r"(?i)password\s*[:=]\s*\S+",
    r"(?i)token\s*[:=]\s*\S+",
    r"\b1[3-9]\d{9}\b",
]

PROFILE_PATTERNS = {
    "preferred_language": [
        r"以后.*(中文|英文).*回答",
        r"以后.*用(中文|英文)",
        r"回答.*用(中文|英文)",
    ],
    "answer_style": [
        r"以后.*(简洁|详细|分步骤|直接).*回答",
        r"回答.*(简洁|详细|分步骤|直接)",
    ],
    "primary_stack": [
        r"我主要(写|做|负责).*(Python|Java|Go|前端|后端|RAG|Agent)",
        r"我的技术栈.*(Python|Java|Go|React|FastAPI|LangGraph)",
    ],
}

LONG_TERM_PATTERNS = [
    r"我在做.*(项目|系统|平台)",
    r"我负责.*(项目|模块|系统|平台)",
    r"我们项目.*(使用|采用|基于)",
    r"长期.*(目标|计划|需求)",
    r"之后.*都要.*",
]

ONE_OFF_PATTERNS = [
    r"这次先",
    r"临时",
    r"今天",
    r"明天",
    r"刚才",
    r"等会",
]

CORRECTION_PATTERNS = [
    r"不是.*是.*",
    r"我刚才说错了",
    r"纠正一下",
]

PROJECT_KEYWORDS = {
    "rag",
    "agent",
    "langgraph",
    "milvus",
    "elasticsearch",
    "neo4j",
    "redis",
    "postgresql",
    "评测",
    "部署",
}


def contains_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def detect_profile_field(text: str) -> str | None:
    for field_name, patterns in PROFILE_PATTERNS.items():
        if contains_any(patterns, text):
            return field_name
    return None


def score_memory_event(
    text: str,
    *,
    repeated_fact: bool = False,
    model_inferred: bool = False,
) -> tuple[int, list[str]]:
    """Score a candidate memory event.

    Positive values mean the fact is stable and useful across turns/sessions.
    Negative values mean it is short-lived, sensitive, or low confidence.
    """

    score = 0
    reasons: list[str] = []
    lowered = text.lower()

    profile_field = detect_profile_field(text)
    if profile_field:
        score += 3
        reasons.append(f"explicit_user_profile:{profile_field}")

    if contains_any(LONG_TERM_PATTERNS, text):
        score += 2
        reasons.append("stable_project_or_goal_fact")

    if any(keyword in lowered for keyword in PROJECT_KEYWORDS):
        score += 2
        reasons.append("related_to_long_running_project")

    if repeated_fact:
        score += 1
        reasons.append("repeated_across_turns_or_sessions")

    if contains_any(CORRECTION_PATTERNS, text):
        score += 1
        reasons.append("user_correction")

    if contains_any(ONE_OFF_PATTERNS, text):
        score -= 3
        reasons.append("one_off_context")

    if contains_any(SENSITIVE_PATTERNS, text):
        score -= 5
        reasons.append("sensitive_or_secret")

    if model_inferred:
        score -= 2
        reasons.append("model_inferred_not_user_stated")

    return score, reasons


class MemoryClassifier:
    """Route conversation text into memory layers."""

    def __init__(
        self,
        summary_turn_threshold: int = 20,
        summary_token_threshold: int = 6000,
        semantic_threshold: int = 3,
        episodic_threshold: int = 4,
    ) -> None:
        self.summary_turn_threshold = summary_turn_threshold
        self.summary_token_threshold = summary_token_threshold
        self.semantic_threshold = semantic_threshold
        self.episodic_threshold = episodic_threshold

    def decide(
        self,
        text: str,
        *,
        session_token_count: int = 0,
        session_turn_count: int = 0,
        is_workflow_state: bool = False,
        repeated_fact: bool = False,
        model_inferred: bool = False,
    ) -> list[MemoryDecision]:
        text = (text or "").strip()
        if not text:
            return []

        decisions = [
            MemoryDecision(
                target=MemoryTarget.SHORT_TERM,
                content=text,
                score=0,
                reasons=["default_recent_message"],
            )
        ]

        if is_workflow_state:
            decisions.append(
                MemoryDecision(
                    target=MemoryTarget.CHECKPOINT,
                    content=text,
                    score=0,
                    reasons=["workflow_execution_state"],
                )
            )
            return decisions

        if (
            session_turn_count >= self.summary_turn_threshold
            or session_token_count >= self.summary_token_threshold
        ):
            decisions.append(
                MemoryDecision(
                    target=MemoryTarget.SUMMARY,
                    content=text,
                    score=0,
                    reasons=["session_history_exceeds_budget"],
                )
            )

        score, reasons = score_memory_event(
            text,
            repeated_fact=repeated_fact,
            model_inferred=model_inferred,
        )

        if "sensitive_or_secret" in reasons:
            decisions.append(
                MemoryDecision(
                    target=MemoryTarget.SKIP,
                    content=text,
                    score=score,
                    reasons=reasons + ["skip_persistent_memory"],
                )
            )
            return decisions

        profile_field = detect_profile_field(text)
        if profile_field and score >= self.semantic_threshold:
            decisions.append(
                MemoryDecision(
                    target=MemoryTarget.USER_PROFILE,
                    content=text,
                    score=score,
                    reasons=reasons,
                    metadata={"profile_field": profile_field},
                )
            )
            decisions.append(
                MemoryDecision(
                    target=MemoryTarget.SEMANTIC,
                    content=text,
                    score=score,
                    reasons=reasons + ["promote_profile_to_semantic_memory"],
                    metadata={"profile_field": profile_field},
                )
            )

        if score >= self.episodic_threshold and not contains_any(ONE_OFF_PATTERNS, text):
            decisions.append(
                MemoryDecision(
                    target=MemoryTarget.EPISODIC,
                    content=text,
                    score=score,
                    reasons=reasons,
                )
            )

        return decisions


def decide_memory_target(text: str, **kwargs: Any) -> list[MemoryDecision]:
    """Convenience wrapper for examples and tests."""

    return MemoryClassifier().decide(text, **kwargs)
