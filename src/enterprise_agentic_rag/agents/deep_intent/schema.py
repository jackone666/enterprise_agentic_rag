"""DeepIntentResult schema — structured intent recognition output.

All types are dataclasses for easy serialisation and LangGraph compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ===========================================================================
# Enums
# ===========================================================================


class IntentCategory(str, Enum):
    """Primary intent categories for HarmonyOS development queries."""
    CONCEPT_QA = "concept_qa"
    API_USAGE = "api_usage"
    CODE_GENERATION = "code_generation"
    ERROR_DIAGNOSIS = "error_diagnosis"
    MIGRATION = "migration"
    COMPATIBILITY = "compatibility"
    PROJECT_DEBUG = "project_debug"
    BEST_PRACTICE = "best_practice"
    ARCHITECTURE = "architecture"
    LEARNING_GUIDANCE = "learning_guidance"


class RetrievalMode(str, Enum):
    """Allowed retrieval plan modes."""
    HYBRID_ONLY = "hybrid_only"
    PARALLEL = "parallel"
    GRAPH_FIRST = "graph_first"
    ERROR_FIRST = "error_first"
    CODE_FIRST = "code_first"


class AnswerStyle(str, Enum):
    """Answer presentation styles."""
    DIRECT_ANSWER = "direct_answer"
    EXPLANATION_WITH_CODE = "explanation_with_code"
    DIAGNOSIS_STEPS = "diagnosis_steps"
    MIGRATION_PLAN = "migration_plan"
    ARCHITECTURE_PROPOSAL = "architecture_proposal"
    LEARNING_PATH = "learning_path"


class Difficulty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ===========================================================================
# Allowed values for validation
# ===========================================================================

ALLOWED_INTENTS = frozenset(e.value for e in IntentCategory)
ALLOWED_MODES = frozenset(e.value for e in RetrievalMode)
ALLOWED_STYLES = frozenset(e.value for e in AnswerStyle)
ALLOWED_DIFFICULTIES = frozenset(e.value for e in Difficulty)
ALLOWED_RISK_LEVELS = frozenset(e.value for e in RiskLevel)

ALLOWED_TOOLS = frozenset({
    "keyword_search",
    "vector_search",
    "graph_search",
    "hybrid_rag_search",
    "official_doc_search",
    "api_reference_search",
    "sample_code_search",
    "error_diagnosis_search",
    "ticket_search",
    "version_compatibility_check",
    "code_review",
})

ALLOWED_SOURCES = frozenset({
    "official_docs",
    "api_reference",
    "sample_code",
    "error_knowledge",
    "faq",
    "ticket",
    "migration_guides",
    "version_metadata",
    "community",
    "internal_kb",
})


# ===========================================================================
# Sub-models
# ===========================================================================


@dataclass
class DeepIntentEntities:
    """Entities extracted from the user query."""
    apis: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    api_levels: list[str] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    migration_from: str | None = None
    migration_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "apis": self.apis,
            "components": self.components,
            "errors": self.errors,
            "api_levels": self.api_levels,
            "versions": self.versions,
            "files": self.files,
            "migration_from": self.migration_from,
            "migration_to": self.migration_to,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeepIntentEntities:
        return cls(
            apis=d.get("apis", []),
            components=d.get("components", []),
            errors=d.get("errors", []),
            api_levels=d.get("api_levels", []),
            versions=d.get("versions", []),
            files=d.get("files", []),
            migration_from=d.get("migration_from"),
            migration_to=d.get("migration_to"),
        )


@dataclass
class DeepIntentConstraints:
    """Constraints inferred from the query context."""
    needs_code_example: bool = False
    needs_before_after_code: bool = False
    needs_checklist: bool = False
    prefer_official_docs: bool = True
    requires_version_check: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_code_example": self.needs_code_example,
            "needs_before_after_code": self.needs_before_after_code,
            "needs_checklist": self.needs_checklist,
            "prefer_official_docs": self.prefer_official_docs,
            "requires_version_check": self.requires_version_check,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeepIntentConstraints:
        return cls(
            needs_code_example=d.get("needs_code_example", False),
            needs_before_after_code=d.get("needs_before_after_code", False),
            needs_checklist=d.get("needs_checklist", False),
            prefer_official_docs=d.get("prefer_official_docs", True),
            requires_version_check=d.get("requires_version_check", False),
        )


@dataclass
class RetrievalPlanConfig:
    """Retrieval plan configuration embedded in DeepIntentResult."""
    mode: str = "hybrid_only"  # one of RetrievalMode
    sources: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    expanded_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sources": self.sources,
            "filters": self.filters,
            "expanded_query": self.expanded_query,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetrievalPlanConfig:
        return cls(
            mode=d.get("mode", "hybrid_only"),
            sources=d.get("sources", []),
            filters=d.get("filters", {}),
            expanded_query=d.get("expanded_query"),
        )


# ===========================================================================
# Main DeepIntentResult
# ===========================================================================


@dataclass
class DeepIntentResult:
    """Complete deep intent recognition result.

    All fields correspond to the schema defined in the technical specification.
    """

    primary_intent: str = "concept_qa"
    secondary_intents: list[str] = field(default_factory=list)
    scenario: str = ""
    user_goal: str = ""
    query_focus: str = ""
    required_context: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    entities: DeepIntentEntities = field(default_factory=DeepIntentEntities)
    constraints: DeepIntentConstraints = field(default_factory=DeepIntentConstraints)
    difficulty: str = "low"
    risk_level: str = "low"
    needs_clarification: bool = False
    clarification_questions: list[str] = field(default_factory=list)
    suggested_tools: list[str] = field(default_factory=list)
    retrieval_plan: RetrievalPlanConfig = field(default_factory=RetrievalPlanConfig)
    answer_style: str = "direct_answer"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "secondary_intents": self.secondary_intents,
            "scenario": self.scenario,
            "user_goal": self.user_goal,
            "query_focus": self.query_focus,
            "required_context": self.required_context,
            "missing_context": self.missing_context,
            "entities": self.entities.to_dict(),
            "constraints": self.constraints.to_dict(),
            "difficulty": self.difficulty,
            "risk_level": self.risk_level,
            "needs_clarification": self.needs_clarification,
            "clarification_questions": self.clarification_questions,
            "suggested_tools": self.suggested_tools,
            "retrieval_plan": self.retrieval_plan.to_dict(),
            "answer_style": self.answer_style,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeepIntentResult:
        return cls(
            primary_intent=d.get("primary_intent", "concept_qa"),
            secondary_intents=d.get("secondary_intents", []),
            scenario=d.get("scenario", ""),
            user_goal=d.get("user_goal", ""),
            query_focus=d.get("query_focus", ""),
            required_context=d.get("required_context", []),
            missing_context=d.get("missing_context", []),
            entities=DeepIntentEntities.from_dict(d.get("entities", {})),
            constraints=DeepIntentConstraints.from_dict(d.get("constraints", {})),
            difficulty=d.get("difficulty", "low"),
            risk_level=d.get("risk_level", "low"),
            needs_clarification=d.get("needs_clarification", False),
            clarification_questions=d.get("clarification_questions", []),
            suggested_tools=d.get("suggested_tools", []),
            retrieval_plan=RetrievalPlanConfig.from_dict(d.get("retrieval_plan", {})),
            answer_style=d.get("answer_style", "direct_answer"),
            confidence=d.get("confidence", 0.0),
        )
