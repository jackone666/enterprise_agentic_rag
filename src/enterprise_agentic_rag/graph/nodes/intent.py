"""Deep intent recognition node.

Runs the multi-layer deep intent analyser (rule-based + LLM classifier)
and writes a ``deep_intent`` dict, plus the routing-plan and entity
extraction that downstream retrieval nodes will read.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.graph.dependencies import tracer
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)


async def deep_intent_recognition_node(state: AgentState) -> dict[str, Any]:
    """Deep intent recognition — rule-based + LLM classifier."""
    from enterprise_agentic_rag.agents.deep_intent.node import recognize_deep_intent

    query = state.get("query", "")

    try:
        deep_intent = await recognize_deep_intent(query, use_llm=True)
        result = deep_intent.to_dict()

        primary_intent = result.get("primary_intent", "concept_qa")
        complexity = "high" if result.get("difficulty") == "high" else "moderate"

        tracer.record_retrieval_event(
            dict(state),
            query=query,
            num_docs=0,
            top_score=result.get("confidence", 0),
            latency_ms=0,
            success=True,
        )

        return {
            "deep_intent": result,
            "intent": primary_intent,
            "complexity": complexity,
            "query_analysis": {
                "intent": primary_intent,
                "primary_intent": primary_intent,
                "secondary_intents": result.get("secondary_intents", []),
                "entities": result.get("entities", {}),
                "keywords": result.get("query_focus", query).split(),
                "scenario": result.get("scenario", ""),
                "retrieval_plan": result.get("retrieval_plan", {}),
                "original_query": query,
            },
            "deep_intent_confidence": result.get("confidence", 0.0),
            "retrieval_plan_config": result.get("retrieval_plan", {}),
            "last_worker": "intent_analyzer",
            "last_agent_step": "recognize_intent",
        }

    except Exception as exc:
        logger.warning("Deep intent recognition failed, using default retrieval intent: %s", exc)

        fallback = {
            "intent": "concept_qa",
            "complexity": "simple",
        }
        fallback["deep_intent"] = {
            "primary_intent": "concept_qa",
            "retrieval_plan": {"mode": "hybrid_only"},
            "confidence": 0.3,
            "suggested_tools": ["keyword_search", "vector_search"],
        }
        fallback["query_analysis"] = {
            "intent": "concept_qa",
            "primary_intent": "concept_qa",
            "entities": {},
            "keywords": query.split(),
            "retrieval_plan": {"mode": "hybrid_only"},
            "original_query": query,
        }
        fallback["deep_intent_confidence"] = 0.3
        fallback["retrieval_plan_config"] = {"mode": "hybrid_only"}
        fallback["last_worker"] = "intent_analyzer"
        fallback["last_agent_step"] = "recognize_intent"
        return fallback
