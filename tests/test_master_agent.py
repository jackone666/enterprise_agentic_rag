"""Tests for MasterAgent routing decisions.

Covers:
- Rule chain: all routing branch points (tested via LLM-fallback path)
- LLM-assisted routing: extraction, validation, prompt building
- Helper methods: _requires_code, _requires_tools, tool_intent
- Integration with workflow state

Note: In this test environment, LLM_PROVIDER=mock (via conftest.py),
so decide() always falls through to the rule-based chain.
"""

import pytest

from enterprise_agentic_rag.agents.master_agent import MasterAgent, MasterDecision

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def master() -> MasterAgent:
    return MasterAgent()


def _state(**overrides) -> dict:
    """Build a minimal state dict with defaults, overridden by kwargs."""
    base = {
        "query": "test query",
        "user_id": "u001",
        "session_id": "s001",
        "deep_intent": {
            "primary_intent": "concept_qa",
            "confidence": 0.8,
            "needs_clarification": False,
        },
        "retrieved_docs": [{"score": 0.6, "content": "test doc"}],
        "tool_errors": [],
        "tool_results": [],
        "retry_count": {},
        "fallback_reason": "",
        "verified": False,
        "code_snippet": "",
        "code_verified": False,
        "code_retry_count": 0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# TestMasterAgentRuleDecisions — all routing branches
# ===========================================================================


class TestMasterAgentRuleDecisions:
    """Test that decide() produces correct next_node for each last_agent_step."""

    async def test_after_deep_intent_routes_to_retrieval(self, master):
        state = _state(last_agent_step="recognize_intent")
        decision = await master.decide(state)
        assert decision.next_node == "retrieve_knowledge"

    async def test_after_deep_intent_low_confidence_routes_to_human(self, master):
        state = _state(
            last_agent_step="recognize_intent",
            deep_intent={"primary_intent": "concept_qa", "confidence": 0.1, "needs_clarification": True},
        )
        decision = await master.decide(state)
        assert decision.next_node == "human_fallback"

    async def test_after_deep_intent_error_diagnosis_routes_to_tools(self, master):
        state = _state(
            last_agent_step="recognize_intent",
            deep_intent={"primary_intent": "error_diagnosis", "confidence": 0.8, "needs_clarification": False},
        )
        decision = await master.decide(state)
        assert decision.next_node == "call_tools"

    async def test_after_tool_agent_success_routes_to_retrieval(self, master):
        state = _state(last_agent_step="call_tools", tool_errors=[])
        decision = await master.decide(state)
        assert decision.next_node == "retrieve_knowledge"

    async def test_after_tool_agent_failure_with_retry(self, master):
        state = _state(last_agent_step="call_tools", tool_errors=["connection refused"], retry_count={"tool_call": 0})
        decision = await master.decide(state)
        assert decision.next_node == "call_tools"  # retry

    async def test_after_tool_agent_failure_exhausted(self, master):
        state = _state(last_agent_step="call_tools", tool_errors=["connection refused"], retry_count={"tool_call": 3})
        decision = await master.decide(state)
        assert decision.next_node == "retrieve_knowledge"

    async def test_after_retrieval_service_has_docs(self, master):
        state = _state(last_agent_step="retrieve", retrieved_docs=[{"score": 0.6, "content": "doc"}])
        decision = await master.decide(state)
        assert decision.next_node == "build_context"

    async def test_after_retrieval_service_no_docs_retry(self, master):
        state = _state(last_agent_step="retrieve", retrieved_docs=[], retry_count={"retrieve": 0})
        decision = await master.decide(state)
        assert decision.next_node == "rewrite_query"

    async def test_after_retrieval_service_exhausted(self, master):
        state = _state(last_agent_step="retrieve", retrieved_docs=[], retry_count={"retrieve": 3})
        decision = await master.decide(state)
        assert decision.next_node == "human_fallback"

    async def test_after_rewrite_query_routes_to_retrieval(self, master):
        state = _state(last_agent_step="rewrite_query")
        decision = await master.decide(state)
        assert decision.next_node == "retrieve_knowledge"

    async def test_after_context_with_code_intent(self, master):
        state = _state(
            last_agent_step="build_context",
            deep_intent={"primary_intent": "code_generation", "confidence": 0.8, "needs_clarification": False},
            code_snippet="",
        )
        decision = await master.decide(state)
        assert decision.next_node == "generate_code"

    async def test_after_context_without_code_intent(self, master):
        state = _state(
            last_agent_step="build_context",
            deep_intent={"primary_intent": "concept_qa", "confidence": 0.8, "needs_clarification": False},
        )
        decision = await master.decide(state)
        assert decision.next_node == "generate_answer"

    async def test_after_generate_code_routes_to_execute(self, master):
        state = _state(last_agent_step="generate_code")
        decision = await master.decide(state)
        assert decision.next_node == "execute_code"

    async def test_after_code_execution_success(self, master):
        state = _state(last_agent_step="execute_code", code_verified=True)
        decision = await master.decide(state)
        assert decision.next_node == "generate_answer"

    async def test_after_code_execution_failure_retry(self, master):
        state = _state(last_agent_step="execute_code", code_verified=False, code_retry_count=0)
        decision = await master.decide(state)
        assert decision.next_node == "generate_code"

    async def test_after_code_execution_failure_exhausted(self, master):
        state = _state(last_agent_step="execute_code", code_verified=False, code_retry_count=1)
        decision = await master.decide(state)
        assert decision.next_node == "finalize_answer"

    async def test_after_generate_answer_routes_to_verify(self, master):
        state = _state(last_agent_step="generate_answer")
        decision = await master.decide(state)
        assert decision.next_node == "verify_answer"

    async def test_after_verifier_success(self, master):
        state = _state(last_agent_step="verify_answer", verified=True)
        decision = await master.decide(state)
        assert decision.next_node == "finalize_answer"

    async def test_after_verifier_failure_with_retry(self, master):
        state = _state(last_agent_step="verify_answer", verified=False, retry_count={"verify": 0})
        decision = await master.decide(state)
        assert decision.next_node == "build_context"

    async def test_after_verifier_failure_exhausted(self, master):
        state = _state(last_agent_step="verify_answer", verified=False, retry_count={"verify": 3})
        decision = await master.decide(state)
        assert decision.next_node == "human_fallback"

    async def test_default_fallback_to_retrieval(self, master):
        state = _state(last_agent_step="unknown_step")
        decision = await master.decide(state)
        assert decision.next_node == "retrieve_knowledge"


# ===========================================================================
# TestMasterAgentLLMFunctions — LLM routing support methods
# ===========================================================================


class TestMasterAgentLLMFunctions:
    """Test the LLM routing support methods (prompt building, JSON extraction, validation)."""

    def test_build_routing_prompt_includes_context(self, master):
        state = _state(last_agent_step="retrieve", query="什么是ArkUI?")
        prompt = master._build_routing_prompt(state)
        assert "last_agent_step" in prompt
        assert "retrieve" in prompt
        assert "什么是ArkUI" in prompt
        assert "next_node" in prompt.lower()

    def test_extract_routing_json_plain(self):
        result = MasterAgent._extract_routing_json('{"next_node": "build_context", "reason": "ok"}')
        assert result == {"next_node": "build_context", "reason": "ok"}

    def test_extract_routing_json_with_fence(self):
        result = MasterAgent._extract_routing_json('```json\n{"next_node": "retrieve_knowledge", "reason": "test"}\n```')
        assert result == {"next_node": "retrieve_knowledge", "reason": "test"}

    def test_extract_routing_json_with_surrounding_text(self):
        result = MasterAgent._extract_routing_json('some text {"next_node": "call_tools", "reason": "need tools"} more text')
        assert result == {"next_node": "call_tools", "reason": "need tools"}

    def test_extract_routing_json_invalid(self):
        result = MasterAgent._extract_routing_json("not json at all")
        assert result is None

    def test_extract_routing_json_empty(self):
        assert MasterAgent._extract_routing_json("") is None
        assert MasterAgent._extract_routing_json("   ") is None

    def test_validate_decision_valid(self):
        result = MasterAgent._validate_decision({"next_node": "build_context", "reason": "ok"})
        assert result is not None
        assert result.next_node == "build_context"

    def test_validate_decision_invalid_node(self):
        result = MasterAgent._validate_decision({"next_node": "invalid_node", "reason": "???"})
        assert result is None

    def test_validate_decision_missing_node(self):
        result = MasterAgent._validate_decision({"reason": "no node"})
        assert result is None

    def test_all_valid_nodes_rejected_on_invalid(self):
        # Ensure every member of _VALID_NODES passes validation
        valid_nodes = {
            "call_tools", "retrieve_knowledge", "rewrite_query", "build_context",
            "generate_code", "execute_code", "generate_answer", "verify_answer",
            "finalize_answer", "human_fallback",
        }
        for node in valid_nodes:
            r = MasterAgent._validate_decision({"next_node": node, "reason": "test"})
            assert r is not None, f"Valid node {node} was rejected"


# ===========================================================================
# TestMasterAgentHelperMethods
# ===========================================================================


class TestMasterAgentHelperMethods:
    """Test MasterAgent static/classmethod helper utilities."""

    def test_requires_code_true(self):
        state = _state(deep_intent={"primary_intent": "code_generation"})
        assert MasterAgent._requires_code(state) is True

    def test_requires_code_false(self):
        state = _state(deep_intent={"primary_intent": "concept_qa"})
        assert MasterAgent._requires_code(state) is False

    def test_requires_tools_error_diagnosis(self):
        state = _state(deep_intent={"primary_intent": "error_diagnosis"})
        assert MasterAgent._requires_tools(state) is True

    def test_requires_tools_project_debug(self):
        state = _state(deep_intent={"primary_intent": "project_debug"})
        assert MasterAgent._requires_tools(state) is True

    def test_requires_tools_ticket_keyword(self):
        state = _state(deep_intent={"primary_intent": "concept_qa"}, query="我的工单状态")
        assert MasterAgent._requires_tools(state) is True

    def test_requires_tools_false_concept_qa(self):
        state = _state(deep_intent={"primary_intent": "concept_qa"}, query="什么是ArkUI")
        assert MasterAgent._requires_tools(state) is False

    def test_tool_intent_error_diagnosis(self):
        state = _state(deep_intent={"primary_intent": "error_diagnosis"})
        assert MasterAgent.tool_intent(state) == "troubleshooting"

    def test_tool_intent_ticket_query(self):
        state = _state(deep_intent={"primary_intent": "concept_qa"}, query="ticket TKT-123")
        assert MasterAgent.tool_intent(state) == "ticket_query"

    def test_tool_intent_default_to_primary(self):
        state = _state(deep_intent={"primary_intent": "api_usage"})
        assert MasterAgent.tool_intent(state) == "api_usage"

    def test_primary_intent_extraction(self):
        state = _state(deep_intent={"primary_intent": "migration"})
        assert MasterAgent._primary_intent(state) == "migration"


# ===========================================================================
# TestMasterDecision
# ===========================================================================


class TestMasterDecision:
    """Test MasterDecision dataclass."""

    def test_decision_to_dict(self):
        d = MasterDecision(
            next_node="retrieve_knowledge",
            reason="test reason",
            routing_path="llm",
        )
        result = d.to_dict()
        assert result == {
            "next_node": "retrieve_knowledge",
            "reason": "test reason",
            "routing_path": "llm",
        }

    def test_decision_is_frozen(self):
        d = MasterDecision(next_node="retrieve_knowledge", reason="test")
        with pytest.raises(Exception):
            d.next_node = "other"  # type: ignore[misc]
