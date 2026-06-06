"""Retrieval Router — dynamic routing for Graph-Augmented Hybrid RAG.

Analyzes query characteristics and outputs a RetrievalPlan that determines:
- mode: parallel | graph_first | hybrid_only
- which retrievers to enable
- top_k and weights per retriever
- whether to do query expansion or second-stage retrieval

Routing rules (in priority order):

1.  ENABLE_GRAPH_RAG=false → hybrid_only
2.  Graph DB unavailable / empty / init failure → hybrid_only
3.  Relational query → graph_first
    (contains 关系/依赖/调用链/影响/导致/上下游/链路/生命周期/路径/关联,
     or intent=relational,
     or multiple entities + asking about connections)
4.  Exact match query → parallel, keyword-heavy
    (error_code / API name / class name / function name / config,
     or intent=exact)
5.  Semantic query → parallel, vector-heavy
    (为什么/怎么解决/可能原因/如何优化/区别/排查/怎么办,
     or intent=semantic)
6.  Default → parallel, balanced (keyword:0.3 vector:0.5 graph:0.2)

7.  parallel mode:
    - keyword/vector/graph via asyncio.gather concurrently
    - graph failure non-fatal → graph_candidates=[], graph weight → 0
    - remaining weights re-normalized
    - trace records graph_failed=true

8.  graph_first mode:
    - graph executes first
    - on success: expands query → keyword+vector with expanded_query
    - merges 3-way results
    - on failure: degraded_from=graph_first, degraded_to=hybrid_only

9.  hybrid_only mode:
    - keyword+vector hybrid search only
    - no graph retriever called
    - no graph_paths generated
"""

from __future__ import annotations

import logging
import re
from typing import Any

from dataclasses import dataclass, field

from enterprise_agentic_rag.config.settings import get_settings
from enterprise_agentic_rag.rag.graph.graph_schema import RetrievalPlan

logger = logging.getLogger(__name__)

# ===========================================================================
# Dynamic Weights (migrated from retrieval/retrieval_router.py)
# ===========================================================================


@dataclass
class DynamicWeights:
    """Intent-aware dynamic fusion weights."""

    keyword_weight: float = 0.30
    vector_weight: float = 0.35
    graph_weight: float = 0.25
    source_quality_weight: float = 0.10

    # Source type weights (for multi-source fusion)
    official_doc_weight: float = 0.25
    api_reference_weight: float = 0.20
    sample_code_weight: float = 0.20
    error_knowledge_weight: float = 0.15
    faq_weight: float = 0.10
    ticket_weight: float = 0.10
    migration_guide_weight: float = 0.15
    version_meta_weight: float = 0.10

    def to_dict(self) -> dict[str, float]:
        return {
            "keyword_weight": self.keyword_weight,
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "source_quality_weight": self.source_quality_weight,
            "official_doc_weight": self.official_doc_weight,
            "api_reference_weight": self.api_reference_weight,
            "sample_code_weight": self.sample_code_weight,
            "error_knowledge_weight": self.error_knowledge_weight,
            "faq_weight": self.faq_weight,
            "ticket_weight": self.ticket_weight,
            "migration_guide_weight": self.migration_guide_weight,
            "version_meta_weight": self.version_meta_weight,
        }

    @classmethod
    def for_intent(cls, intent: str, scenario: str = "") -> DynamicWeights:
        """Create DynamicWeights adjusted for a specific intent.

        Adjustments per Section 8:

        api_usage:
            keyword_weight ↑, api_reference_weight ↑↑, sample_code_weight ↑

        error_diagnosis:
            keyword_weight ↑↑, error_knowledge_weight ↑↑, faq_weight ↑, ticket_weight ↑

        migration:
            graph_weight ↑↑, migration_guide_weight ↑↑

        compatibility:
            graph_weight ↑, version_meta_weight ↑↑

        code_generation:
            sample_code_weight ↑↑, official_doc_weight ↑, api_reference_weight ↑

        concept_qa:
            vector_weight ↑↑, official_doc_weight ↑↑
        """
        w = cls()

        if intent == "api_usage":
            w.keyword_weight = 0.35
            w.vector_weight = 0.25
            w.graph_weight = 0.15
            w.source_quality_weight = 0.15
            w.api_reference_weight = 0.35
            w.sample_code_weight = 0.30
            w.official_doc_weight = 0.20

        elif intent == "error_diagnosis":
            w.keyword_weight = 0.45
            w.vector_weight = 0.25
            w.graph_weight = 0.20
            w.source_quality_weight = 0.10
            w.error_knowledge_weight = 0.40
            w.faq_weight = 0.25
            w.ticket_weight = 0.20
            w.official_doc_weight = 0.15

        elif intent == "project_debug":
            w.keyword_weight = 0.40
            w.vector_weight = 0.25
            w.graph_weight = 0.25
            w.error_knowledge_weight = 0.30
            w.official_doc_weight = 0.25
            w.faq_weight = 0.20
            w.ticket_weight = 0.15

        elif intent == "migration":
            w.keyword_weight = 0.20
            w.vector_weight = 0.25
            w.graph_weight = 0.45
            w.source_quality_weight = 0.10
            w.migration_guide_weight = 0.45
            w.official_doc_weight = 0.25
            w.api_reference_weight = 0.15

        elif intent == "compatibility":
            w.keyword_weight = 0.25
            w.vector_weight = 0.25
            w.graph_weight = 0.35
            w.source_quality_weight = 0.15
            w.version_meta_weight = 0.40
            w.official_doc_weight = 0.30
            w.graph_weight = 0.25  # override from above

        elif intent == "code_generation":
            w.keyword_weight = 0.25
            w.vector_weight = 0.25
            w.graph_weight = 0.15
            w.source_quality_weight = 0.15
            w.sample_code_weight = 0.40
            w.official_doc_weight = 0.25
            w.api_reference_weight = 0.20

        elif intent == "concept_qa":
            w.keyword_weight = 0.20
            w.vector_weight = 0.45
            w.graph_weight = 0.20
            w.source_quality_weight = 0.15
            w.official_doc_weight = 0.45
            w.api_reference_weight = 0.20

        elif intent == "best_practice":
            w.keyword_weight = 0.25
            w.vector_weight = 0.35
            w.graph_weight = 0.20
            w.source_quality_weight = 0.20
            w.official_doc_weight = 0.40

        elif intent == "architecture":
            w.keyword_weight = 0.20
            w.vector_weight = 0.30
            w.graph_weight = 0.35
            w.source_quality_weight = 0.15
            w.official_doc_weight = 0.35
            w.migration_guide_weight = 0.15

        # learning_guidance: use defaults

        return w

# ---------------------------------------------------------------------------
# Pattern sets for query classification
# ---------------------------------------------------------------------------

# Rule 3: Relational / reasoning keywords
_RELATIONAL_KEYWORDS = re.compile(
    r"关系|依赖|调用链|影响.*哪些|导致|上下游|链路|生命周期|路径|关联|"
    r"有关|涉及|联系|相连|连接|触发|回调|监听|订阅|通知|"
    r"依赖关系|调用关系|什么关系|之间.*关系",
)

# Rule 4: Exact match — error codes / API / class / function / config
_EXACT_ERROR_CODE = re.compile(
    r"\b\d{4,10}\b|\bERR[A-Z_]+\b|\bAUTH_\w+\b|\b[A-Z]{2,6}_\d{3,8}\b"
)
_EXACT_API_NAME = re.compile(
    r"@ohos\.\w+|@\w+\.\w+|\b\w+API\b|\bimport\s+\{"
)
_EXACT_CLASS_NAME = re.compile(
    r"\b[A-Z][a-z]+Ability\b|\bclass\s+\w+|\bextends\s+\w+|\bComponent\b"
)
_EXACT_FUNCTION_NAME = re.compile(
    r"\bon\w+Create\b|\bon\w+Destroy\b|\b\w+\(\)"
)
_EXACT_CONFIG_NAME = re.compile(
    r"\b[A-Z_]{3,30}\b|module\.json|app\.json|\.env\b"
)

# Rule 5: Semantic — explanatory / reasoning questions
_SEMANTIC_KEYWORDS = re.compile(
    r"为什么|怎么解决|可能原因|如何优化|区别|排查|怎么办|"
    r"如何|怎么|怎样|什么是|含义|原因|可能|建议|推荐|最佳实践|"
    r"有哪些|介绍|说明|解释"
)

# Optional routing hints from query analysis.
_INTENT_RELATIONAL = frozenset({"relational", "relationship", "graph_query"})
_INTENT_EXACT = frozenset({"exact", "precise", "keyword_lookup"})
_INTENT_SEMANTIC = frozenset({"semantic", "explanation", "technical_question"})
_DEEP_INTENT_KEYWORD_HEAVY = frozenset({
    "api_usage",
    "code_generation",
    "error_diagnosis",
    "project_debug",
    "compatibility",
})
_DEEP_INTENT_VECTOR_HEAVY = frozenset({
    "concept_qa",
    "best_practice",
    "learning_guidance",
})
_DEEP_INTENT_GRAPH_HEAVY = frozenset({
    "migration",
    "architecture",
})

# ---------------------------------------------------------------------------
# Code-related query patterns — used for code-boost activation
_CODE_QUERY_KEYWORDS = re.compile(
    r"代码|示例|example|怎么写|怎么调用|demo|snippet|接入|使用示例|"
    r"用法|implement|how\s+to\s+use|调用方法|怎么用|使用方法|"
    r"code\s+sample|example\s+code|完整示例|参考代码|代码片段"
)

# Multi-entity indicator — non-CJK separators between named things
# ---------------------------------------------------------------------------
_MULTI_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][a-zA-Z]+\b.*(?:和|与|、|,|and).*\b[A-Z][a-zA-Z]+\b"
)


class RetrievalRouter:
    """Analyze query and decide retrieval strategy.

    Pure rule-based routing — no LLM calls, no online API.
    Every routing decision writes a human-readable ``reason`` into the plan.

    Usage::

        router = RetrievalRouter()
        plan = router.route(query, query_analysis)
        # plan.mode → "parallel" | "graph_first" | "hybrid_only"
        # plan.reason → human-readable decision rationale
    """

    def __init__(self) -> None:
        self._refresh_settings()
        # Override for tests: set to True/False to bypass real Neo4j check.
        # When None (default), performs the actual health check.
        self._graph_available_override: bool | None = None

    # ------------------------------------------------------------------
    # Settings cache
    # ------------------------------------------------------------------
    def _refresh_settings(self) -> None:
        """Re-read settings (useful after env changes at runtime / in tests)."""
        s = get_settings()
        self._graph_enabled = s.graph_rag.enabled
        self._dynamic_router = s.router.dynamic_router_enabled
        self._default_mode = s.router.default_mode
        self._graph_depth = s.graph_rag.graph_depth
        self._graph_top_k = s.graph_rag.graph_top_k

    @property
    def graph_available(self) -> bool:
        """Check whether Neo4j is reachable *and* graph RAG is enabled.

        Encapsulates Rule 1 + Rule 2 (partial — index emptiness checked
        separately by the orchestrator).

        Tests can set ``router._graph_available_override = True`` to
        bypass the real connectivity check.
        """
        if self._graph_available_override is not None:
            return self._graph_available_override
        if not self._graph_enabled:
            return False
        try:
            from enterprise_agentic_rag.rag.graph.graph_retriever import GraphRetriever
            gr = GraphRetriever()
            return gr.available
        except Exception:
            logger.warning("RetrievalRouter: GraphRetriever init failed — "
                           "graph unavailable")
            return False

    # ==================================================================
    # Main route method — implements Rules 1–6
    # ==================================================================
    def route(
        self,
        query: str,
        query_analysis: dict[str, Any] | None = None,
    ) -> RetrievalPlan:
        """Analyze query and produce a routing plan.

        Args:
            query: Raw user query string.
            query_analysis: Optional pre-computed analysis dict with keys:
                - intent: classified intent string
                - entities: list of detected entity strings
                - keywords: list of keyword tokens
                - query_type: "semantic" | "keyword" | "mixed"

        Returns:
            RetrievalPlan with mode, enabled_retrievers, weights, top_k,
            graph_depth, need_query_expansion, fallback_to_hybrid, reason.
        """
        self._refresh_settings()
        qa = query_analysis or {}

        # ── Rule 1: Graph RAG disabled globally ─────────────────────
        if not self._graph_enabled:
            reason = "ENABLE_GRAPH_RAG=false，Graph RAG 全局关闭 → hybrid_only"
            logger.info("RetrievalRouter: %s", reason)
            return self._make_hybrid_only(reason)

        # ── Rule 2: Graph DB unavailable / init failure ─────────────
        if not self.graph_available:
            reason = "Graph 数据库不可用或 GraphRetriever 初始化失败 → hybrid_only"
            logger.info("RetrievalRouter: %s", reason)
            return self._make_hybrid_only(reason)

        # ── Feature feature extraction ──────────────────────────────
        features = self._analyze_query(query, qa)

        # ── Deep intent overrides: the retrieval service tunes weights from intent facts.
        deep_intent = features.get("deep_intent", "")
        if deep_intent in _DEEP_INTENT_GRAPH_HEAVY:
            plan = self._make_graph_first(
                features,
                qa,
                reason=f"deep_intent={deep_intent} → graph-first / graph-heavy",
            )
            logger.info("RetrievalRouter: mode=%s reason=%s", plan.mode, plan.reason)
            return plan

        if deep_intent in _DEEP_INTENT_KEYWORD_HEAVY:
            plan = self._make_deep_intent_parallel(
                deep_intent,
                keyword=0.6,
                vector=0.25,
                graph=0.15,
                keyword_top_k=10,
                vector_top_k=5,
                graph_top_k=max(5, self._graph_top_k // 2),
                reason="deep intent 精确/API/代码/排障类问题 → keyword-heavy",
            )
            logger.info("RetrievalRouter: mode=%s reason=%s", plan.mode, plan.reason)
            return plan

        if deep_intent in _DEEP_INTENT_VECTOR_HEAVY:
            plan = self._make_deep_intent_parallel(
                deep_intent,
                keyword=0.2,
                vector=0.65,
                graph=0.15,
                keyword_top_k=5,
                vector_top_k=10,
                graph_top_k=max(5, self._graph_top_k // 2),
                reason="deep intent 概念/最佳实践/学习类问题 → vector-heavy",
            )
            logger.info("RetrievalRouter: mode=%s reason=%s", plan.mode, plan.reason)
            return plan

        # ── Rule 3: Relational / reasoning query → graph_first ──────
        if self._is_relational(query, features, qa):
            plan = self._make_graph_first(features, qa,
                                          reason="关系推理类问题 → graph_first")
            logger.info("RetrievalRouter: mode=%s reason=%s", plan.mode, plan.reason)
            return plan

        # ── Rule 4: Exact match → parallel, keyword-heavy ───────────
        if self._is_exact(query, features, qa):
            plan = self._make_exact_parallel(features, qa)
            logger.info("RetrievalRouter: mode=%s reason=%s", plan.mode, plan.reason)
            return plan

        # ── Rule 5: Semantic → parallel, vector-heavy ───────────────
        if self._is_semantic(query, features, qa):
            plan = self._make_semantic_parallel(features, qa)
            logger.info("RetrievalRouter: mode=%s reason=%s", plan.mode, plan.reason)
            return plan

        # ── Rule 6: Default → parallel, balanced ────────────────────
        plan = self._make_default_parallel(features, qa)
        logger.info("RetrievalRouter: mode=%s reason=%s", plan.mode, plan.reason)
        return plan

    # ==================================================================
    # Query classification helpers (Rules 3–5 detection)
    # ==================================================================

    def _analyze_query(
        self, query: str, qa: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract boolean flags and counts from the query."""
        entities = qa.get("entities", [])
        if isinstance(entities, dict):
            flattened: list[str] = []
            for value in entities.values():
                if isinstance(value, list):
                    flattened.extend(str(v) for v in value if v)
                elif value:
                    flattened.append(str(value))
            entities = flattened
        keywords_list = qa.get("keywords", [])
        intent = qa.get("intent", qa.get("primary_intent", "general_question"))

        return {
            "query_length": len(query),
            "has_entities": len(entities) > 0,
            "entity_count": len(entities),
            "has_multi_entities": bool(_MULTI_ENTITY_PATTERN.search(query)),
            "has_keywords": len(keywords_list) > 0,

            # Exact match signals
            "has_error_code": bool(_EXACT_ERROR_CODE.search(query)),
            "has_api_name": bool(_EXACT_API_NAME.search(query)),
            "has_class_name": bool(_EXACT_CLASS_NAME.search(query)),
            "has_function_name": bool(_EXACT_FUNCTION_NAME.search(query)),
            "has_config_name": bool(_EXACT_CONFIG_NAME.search(query)),

            # Relational signals
            "has_relation_keywords": bool(_RELATIONAL_KEYWORDS.search(query)),
            "intent_is_relational": intent in _INTENT_RELATIONAL,

            # Semantic signals
            "has_semantic_keywords": bool(_SEMANTIC_KEYWORDS.search(query)),
            "intent_is_semantic": intent in _INTENT_SEMANTIC,

            # Exact intent
            "intent_is_exact": intent in _INTENT_EXACT,

            # Carried intent from classifier
            "intent": intent,
            "deep_intent": qa.get("primary_intent", intent),
            "entities": entities,
            "keywords": keywords_list,
        }

    # ------------------------------------------------------------------
    # Rule 3: Relational detection
    # ------------------------------------------------------------------
    def _is_relational(
        self, query: str, features: dict[str, Any], qa: dict[str, Any],
    ) -> bool:
        """Return True if this query is a relational/reasoning question."""
        # 3a: Relational keywords
        if features["has_relation_keywords"]:
            return True

        # 3b: Intent is explicitly relational
        if features["intent_is_relational"]:
            return True

        # 3c: Multiple entities detected AND query asks about connections
        if features.get("has_multi_entities") and (
            "关系" in query
            or "联系" in query
            or "关联" in query
            or "有关" in query
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Rule 4: Exact match detection
    # ------------------------------------------------------------------
    def _is_exact(
        self, query: str, features: dict[str, Any], qa: dict[str, Any],
    ) -> bool:
        """Return True if this is a precise / exact-match query."""
        # Error code → exact
        if features["has_error_code"]:
            return True

        # API name without semantic keywords → exact
        if features["has_api_name"] and not features["has_semantic_keywords"]:
            return True

        # Class name → exact
        if features["has_class_name"]:
            return True

        # Function name → exact
        if features["has_function_name"]:
            return True

        # Config name → exact
        if features["has_config_name"]:
            return True

        # Intent is explicitly exact
        if features["intent_is_exact"]:
            return True

        return False

    # ------------------------------------------------------------------
    # Rule 5: Semantic detection
    # ------------------------------------------------------------------
    def _is_semantic(
        self, query: str, features: dict[str, Any], qa: dict[str, Any],
    ) -> bool:
        """Return True if this is a semantic / explanatory question."""
        # Semantic keywords AND not an exact error-code lookup
        if features["has_semantic_keywords"] and not features["has_error_code"]:
            return True

        # Intent is explicitly semantic
        if features["intent_is_semantic"]:
            return True

        # Query is longer (>20 chars), has no exact markers, and has question words
        if (
            features["query_length"] > 20
            and not features["has_error_code"]
            and not features["has_api_name"]
            and ("什么是" in query or "如何" in query or "为什么" in query)
        ):
            return True

        return False

    # ==================================================================
    # Code-related query detection
    # ==================================================================
    @staticmethod
    def is_code_related_query(query: str) -> bool:
        """Detect whether the query is asking for code examples.

        Returns True for queries like "怎么调用X", "X的使用示例", "X的demo" etc.
        """
        return bool(_CODE_QUERY_KEYWORDS.search(query))

    @staticmethod
    def should_enable_external_search(
        internal_results: list[dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> bool:
        """Decide whether to trigger external search based on internal result quality.

        Triggers when:
        - Fewer than top_k/2 results returned, OR
        - All results have score < min_score

        Args:
            internal_results: Results from internal retrieval.
            top_k: Expected number of results.
            min_score: Minimum score threshold.

        Returns:
            True if external search should be triggered.
        """
        if not internal_results:
            return True
        if len(internal_results) < top_k / 2:
            return True
        if all(r.get("score", 0) < min_score for r in internal_results):
            return True
        return False

    # ==================================================================
    # Plan factories — each returns a fully-populated RetrievalPlan
    # ==================================================================

    # ------------------------------------------------------------------
    # Rule 9 + Rule 1 + Rule 2: hybrid_only
    # ------------------------------------------------------------------
    def _make_hybrid_only(self, reason: str) -> RetrievalPlan:
        """hybrid_only — keyword + vector hybrid search, zero graph involvement."""
        return RetrievalPlan(
            mode="hybrid_only",
            enabled_retrievers=["keyword", "vector"],
            top_k={"keyword": 5, "vector": 5},
            weights={"keyword": 0.4, "vector": 0.6, "graph": 0.0},
            graph_depth=0,
            need_query_expansion=False,
            need_second_stage_retrieval=False,
            fallback_to_hybrid=True,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Rule 3: graph_first
    # ------------------------------------------------------------------
    def _make_graph_first(
        self,
        features: dict[str, Any],
        qa: dict[str, Any],
        reason: str,
    ) -> RetrievalPlan:
        """graph_first — graph retrieval first, then keyword+vector with expanded query."""
        entity_count = features.get("entity_count", 0)
        depth = 2 if entity_count >= 3 else self._graph_depth

        # Build detailed reason
        detail_parts = []
        if features.get("has_relation_keywords"):
            detail_parts.append("命中关系类关键词")
        if features.get("intent_is_relational"):
            detail_parts.append("intent=relational")
        if features.get("has_multi_entities"):
            detail_parts.append(f"检测到{entity_count}个实体且询问实体间联系")
        if detail_parts:
            reason = f"{reason}（{'，'.join(detail_parts)}）"

        return RetrievalPlan(
            mode="graph_first",
            enabled_retrievers=["graph", "keyword", "vector"],
            top_k={"graph": self._graph_top_k, "keyword": 5, "vector": 5},
            weights={"keyword": 0.2, "vector": 0.3, "graph": 0.5},
            graph_depth=depth,
            need_query_expansion=True,
            need_second_stage_retrieval=True,
            fallback_to_hybrid=True,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Rule 4: parallel, keyword-heavy (exact match)
    # ------------------------------------------------------------------
    def _make_exact_parallel(
        self,
        features: dict[str, Any],
        qa: dict[str, Any],
    ) -> RetrievalPlan:
        """parallel mode with keyword-heavy weights for exact match queries."""
        reason_parts = []
        if features.get("has_error_code"):
            reason_parts.append("检测到错误码")
        if features.get("has_api_name"):
            reason_parts.append("检测到 API 名称")
        if features.get("has_class_name"):
            reason_parts.append("检测到类名")
        if features.get("has_function_name"):
            reason_parts.append("检测到函数名")
        if features.get("has_config_name"):
            reason_parts.append("检测到配置项")
        if features.get("intent_is_exact"):
            reason_parts.append("intent=exact")

        reason = (
            f"精确匹配类问题 → parallel（关键词权重最高）"
            f"（{'，'.join(reason_parts)}），"
            f"keyword_top_k 增大，vector 作为兜底，graph 权重较低"
        )

        return RetrievalPlan(
            mode="parallel",
            enabled_retrievers=["keyword", "vector", "graph"],
            top_k={"keyword": 10, "vector": 5, "graph": 0},
            weights={"keyword": 0.7, "vector": 0.3, "graph": 0.0},
            graph_depth=1,
            need_query_expansion=False,
            need_second_stage_retrieval=False,
            fallback_to_hybrid=True,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Rule 5: parallel, vector-heavy (semantic)
    # ------------------------------------------------------------------
    def _make_semantic_parallel(
        self,
        features: dict[str, Any],
        qa: dict[str, Any],
    ) -> RetrievalPlan:
        """parallel mode with vector-heavy weights for semantic queries."""
        reason_parts = []
        if features.get("has_semantic_keywords"):
            reason_parts.append("命中语义关键词（为什么/怎么解决/如何优化等）")
        if features.get("intent_is_semantic"):
            reason_parts.append("intent=semantic")

        reason = (
            f"语义理解类问题 → parallel（向量权重最高）"
            f"（{'，'.join(reason_parts)}），"
            f"keyword 和 graph 辅助"
        )

        return RetrievalPlan(
            mode="parallel",
            enabled_retrievers=["keyword", "vector", "graph"],
            top_k={"keyword": 5, "vector": 10, "graph": self._graph_top_k},
            weights={"keyword": 0.2, "vector": 0.6, "graph": 0.2},
            graph_depth=1,
            need_query_expansion=False,
            need_second_stage_retrieval=False,
            fallback_to_hybrid=True,
            reason=reason,
        )

    def _make_deep_intent_parallel(
        self,
        deep_intent: str,
        *,
        keyword: float,
        vector: float,
        graph: float,
        keyword_top_k: int,
        vector_top_k: int,
        graph_top_k: int,
        reason: str,
    ) -> RetrievalPlan:
        """parallel mode with weights chosen from deep intent."""
        return RetrievalPlan(
            mode="parallel",
            enabled_retrievers=["keyword", "vector", "graph"],
            top_k={"keyword": keyword_top_k, "vector": vector_top_k, "graph": graph_top_k},
            weights={"keyword": keyword, "vector": vector, "graph": graph},
            graph_depth=1,
            need_query_expansion=False,
            need_second_stage_retrieval=False,
            fallback_to_hybrid=True,
            reason=f"{reason}（{deep_intent}: keyword={keyword} vector={vector} graph={graph}）",
        )

    # ------------------------------------------------------------------
    # Rule 6: parallel, balanced (default)
    # ------------------------------------------------------------------
    def _make_default_parallel(
        self,
        features: dict[str, Any],
        qa: dict[str, Any],
    ) -> RetrievalPlan:
        """parallel mode with balanced weights — the default path."""
        return RetrievalPlan(
            mode="parallel",
            enabled_retrievers=["keyword", "vector", "graph"],
            top_k={"keyword": 5, "vector": 5, "graph": self._graph_top_k},
            weights={"keyword": 0.3, "vector": 0.5, "graph": 0.2},
            graph_depth=1,
            need_query_expansion=False,
            need_second_stage_retrieval=False,
            fallback_to_hybrid=True,
            reason="默认并行检索：keyword + vector + graph 三路 asyncio.gather 并发，"
                   "keyword:0.3 vector:0.5 graph:0.2",
        )
