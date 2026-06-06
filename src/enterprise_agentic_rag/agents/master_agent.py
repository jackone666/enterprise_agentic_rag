"""Master agent for the master-slave Agentic RAG architecture.

The MasterAgent owns routing decisions between specialized agents and services.
Workers execute domain work and write results into shared state; the master
reads those results and decides the next node.

Routing strategy: LLM-first with rule-based fallback. When a real LLM provider
is configured, the master tries an LLM-based decision first. On failure or
with the mock provider, it falls back to the deterministic rule chain.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from enterprise_agentic_rag.recovery.recovery_manager import RecoveryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid routing targets
# ---------------------------------------------------------------------------
_VALID_NODES: frozenset[str] = frozenset({
    "call_tools",
    "retrieve_knowledge",
    "rewrite_query",
    "build_context",
    "generate_code",
    "execute_code",
    "generate_answer",
    "verify_answer",
    "finalize_answer",
    "human_fallback",
})

# ---------------------------------------------------------------------------
# LLM routing prompt
# ---------------------------------------------------------------------------
ROUTING_SYSTEM_PROMPT = """你是企业级 RAG 多智能体系统的 Master Agent（路由调度器）。
你的任务是根据当前工作流状态，选择下一个应执行的节点。

候选节点：
- call_tools         — 调用外部工具（工单查询、系统状态、用户信息等）
- retrieve_knowledge — 从知识库检索相关文档
- rewrite_query      — 改写查询后重新检索
- build_context      — 构建 LLM 上下文窗口
- generate_code      — 生成代码示例
- execute_code       — 在沙箱中执行生成的代码
- generate_answer    — 基于证据生成最终回答
- verify_answer      — 验证回答的事实准确性
- finalize_answer    — 最终化回答并返回
- human_fallback     — 转人工处理

选择原则：
1. 如果意图识别后发现需要工具支持（错误诊断、项目调试、工单查询），优先 call_tools
2. 如果意图识别不清且置信度极低（<0.2）且需要澄清，选 human_fallback
3. 检索前先确认是否需要 call_tools；检索后发现分数过低可 rewrite_query 或 human_fallback
4. 代码生成意图且无代码片段时选 generate_code；代码已生成选 execute_code
5. 代码执行成功后选 generate_answer；失败且未重试选 generate_code；失败且已重试选 finalize_answer
6. 答案生成后必须 verify_answer；验证通过选 finalize_answer；未通过且可重试选 build_context
7. 所有重试耗尽后选 human_fallback

只输出 JSON，不要输出任何其他文本。"""


@dataclass(frozen=True)
class MasterDecision:
    """A routing decision produced by the master agent.

    Attributes:
        next_node: The slave node the master selected.
        reason: Human-readable explanation of the decision.
        routing_path: Which decision branch actually produced this — one of
            "llm" (the LLM router was used), "rule" (deterministic rules
            after LLM failure / mock provider), "rule_direct" (rules were
            used without an LLM attempt, e.g. mock provider). Operators
            can use this counter to know whether the LLM router is
            actually running in production.
    """

    next_node: str
    reason: str
    routing_path: str = "rule"  # default — set explicitly by decide()

    def to_dict(self) -> dict[str, str]:
        return {
            "next_node": self.next_node,
            "reason": self.reason,
            "routing_path": self.routing_path,
        }


class MasterAgent:
    """Central controller that delegates work to agents and services.

    The workflow is still the runtime state machine, but business routing
    lives here so each worker can stay focused on its own task.

    Uses LLM-first routing with deterministic rule fallback.
    """

    def __init__(self, recovery: RecoveryManager | None = None) -> None:
        self._recovery = recovery or RecoveryManager()

    # ==================================================================
    # Public API
    # ==================================================================

    async def decide(self, state: dict[str, Any]) -> MasterDecision:
        """Decide the next graph node — LLM-first with rule fallback.

        The decision is tagged with a ``routing_path`` so callers can tell
        which path actually produced the answer:
        - ``"llm"``       — the LLM router returned a decision
        - ``"rule"``      — LLM was attempted but failed, rules won
        - ``"rule_direct"``— LLM was not attempted (mock provider / no LLM configured)

        Args:
            state: Current workflow state dict (AgentState-compatible).

        Returns:
            MasterDecision with the next node name, reason, and routing_path.
        """
        # 1. Try LLM-based routing
        llm_result = await self._llm_decide(state)
        if llm_result is not None:
            return llm_result

        # 2. Fall back to rule-based chain
        rule_decision = self._rule_decide(state)
        return _with_routing_path(rule_decision, _resolve_rule_path(state))

    # ==================================================================
    # Rule-based routing (fallback)
    # ==================================================================

    def _rule_decide(self, state: dict[str, Any]) -> MasterDecision:
        """Deterministic rule-based routing — the current production path."""
        last_step = state.get("last_agent_step", "")

        if last_step == "recognize_intent":
            return self._after_deep_intent(state)
        if last_step == "call_tools":
            return self._after_tool_agent(state)
        if last_step == "retrieve":
            return self._after_retrieval_service(state)
        if last_step == "rewrite_query":
            return MasterDecision("retrieve_knowledge", "query rewritten; retry retrieval")
        if last_step == "build_context":
            return self._after_context_agent(state)
        if last_step == "generate_code":
            return MasterDecision("execute_code", "code generated; execute in sandbox")
        if last_step == "execute_code":
            return self._after_code_execution_agent(state)
        if last_step == "generate_answer":
            return MasterDecision("verify_answer", "draft answer ready; verify grounding")
        if last_step == "verify_answer":
            return self._after_verifier_agent(state)

        return MasterDecision("retrieve_knowledge", "default to retrieval")

    # ==================================================================
    # LLM-based routing
    # ==================================================================

    def _build_routing_prompt(self, state: dict[str, Any]) -> str:
        """Build the LLM prompt with current workflow context."""
        parts = [ROUTING_SYSTEM_PROMPT]

        # Current step
        parts.append("\n## 当前工作流状态")
        parts.append(f"last_agent_step: {state.get('last_agent_step', 'start')}")

        # Deep intent
        deep = state.get("deep_intent", {})
        if deep:
            parts.append(f"primary_intent: {deep.get('primary_intent', 'unknown')}")
            parts.append(f"confidence: {deep.get('confidence', 0):.2f}")
            parts.append(f"needs_clarification: {deep.get('needs_clarification', False)}")

        # Retrieval status
        docs = state.get("retrieved_docs", [])
        parts.append(f"retrieved_docs count: {len(docs)}")
        if docs:
            scores = [d.get("score", 0) for d in docs[:5]]
            parts.append(f"top scores: {scores}")

        # Tool status
        tool_errors = state.get("tool_errors", [])
        parts.append(f"tool_errors: {tool_errors if tool_errors else 'none'}")

        # Recovery status
        fb = state.get("fallback_reason", "")
        if fb:
            parts.append(f"fallback_reason: {fb}")
        retry = state.get("retry_count", {})
        if retry:
            parts.append(f"retry_count: {retry}")

        # Verification
        parts.append(f"verified: {state.get('verified', False)}")

        # Code status
        code = state.get("code_snippet", "")
        parts.append(f"code_snippet present: {bool(code)}")
        parts.append(f"code_verified: {state.get('code_verified', False)}")
        parts.append(f"code_retry_count: {state.get('code_retry_count', 0)}")

        # User query (truncated)
        query = state.get("query", "")
        parts.append(f"\n## 用户问题\n{query[:200]}")

        parts.append(f"\n请输出 JSON（next_node 必须是以下之一: {', '.join(sorted(_VALID_NODES))}）：")

        return "\n".join(parts)

    async def _llm_decide(self, state: dict[str, Any]) -> MasterDecision | None:
        """Try LLM-based routing decision. Returns None on any failure."""
        try:
            from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
            provider = get_llm_provider()

            # Skip LLM call for mock provider — fall straight to rules
            if provider.provider_name == "mock":
                return None

            prompt = self._build_routing_prompt(state)
            resp = await provider.generate(prompt, temperature=0.0, max_tokens=512)

            if not resp.success or not resp.content:
                logger.debug("LLM routing call returned empty/invalid response")
                return None

            # Extract JSON from response
            parsed = self._extract_routing_json(resp.content.strip())
            if parsed is None:
                return None

            decision = self._validate_decision(parsed)
            if decision is None:
                return None
            # LLM path actually produced a valid decision — mark it.
            return _with_routing_path(decision, "llm")

        except Exception as exc:
            logger.debug("LLM routing decision failed, falling back to rules: %s", exc)
            return None

    @staticmethod
    def _extract_routing_json(text: str) -> dict[str, Any] | None:
        """Extract a routing JSON object from LLM response text."""
        if not text:
            return None

        # Try direct parse
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Try code fence extraction
        import re
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        # Try brace matching
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
                        return json.loads(text[start:i + 1])  # type: ignore[no-any-return]
                    except json.JSONDecodeError:
                        pass
                    break

        return None

    @staticmethod
    def _validate_decision(parsed: dict[str, Any]) -> MasterDecision | None:
        """Validate and convert a parsed LLM dict into a MasterDecision."""
        next_node = str(parsed.get("next_node", "")).strip()
        reason = str(parsed.get("reason", "LLM routing decision")).strip()

        if next_node not in _VALID_NODES:
            logger.debug("LLM returned invalid next_node: %r", next_node)
            return None

        return MasterDecision(next_node=next_node, reason=reason)

    def _after_deep_intent(self, state: dict[str, Any]) -> MasterDecision:
        deep_intent = state.get("deep_intent", {})
        needs_clarification = deep_intent.get("needs_clarification", False)
        confidence = deep_intent.get("confidence", 0.5)

        if needs_clarification and confidence < 0.2:
            return MasterDecision("human_fallback", "deep intent needs clarification")
        if self._requires_tools(state):
            return MasterDecision("call_tools", "master selected tool agent from intent/query signals")
        return MasterDecision("retrieve_knowledge", "intent routes to knowledge retrieval")

    def _after_tool_agent(self, state: dict[str, Any]) -> MasterDecision:
        tool_errors = state.get("tool_errors", [])
        if not tool_errors:
            return MasterDecision("retrieve_knowledge", "tools completed; retrieve supporting knowledge")

        retry_count = state.get("retry_count", {})
        if self._recovery.can_retry("tool_call", retry_count):
            return MasterDecision("call_tools", "tool failed and retry budget remains")
        return MasterDecision("retrieve_knowledge", "tool retry exhausted; continue with knowledge retrieval")

    def _after_retrieval_service(self, state: dict[str, Any]) -> MasterDecision:
        docs = state.get("retrieved_docs", [])
        fallback_reason = state.get("fallback_reason", "")
        code_required = self._requires_code(state)

        if fallback_reason == "low_retrieval_score":
            retry_count = state.get("retry_count", {})
            if not self._recovery.can_retry("retrieve", retry_count):
                if code_required:
                    return MasterDecision("build_context", "code request can continue with template context")
                return MasterDecision("human_fallback", "retrieval quality is too low")

        if docs and any(d.get("score", 0) > 0 for d in docs):
            return MasterDecision("build_context", "retrieval found usable evidence")

        retry_count = state.get("retry_count", {})
        if self._recovery.can_retry("retrieve", retry_count):
            return MasterDecision("rewrite_query", "no usable docs; rewrite and retry retrieval")

        if code_required:
            return MasterDecision("build_context", "code request can continue without retrieved docs")

        return MasterDecision("human_fallback", "no usable docs and retrieval retry exhausted")

    @staticmethod
    def _after_context_agent(state: dict[str, Any]) -> MasterDecision:
        if MasterAgent._requires_code(state) and not state.get("code_snippet", ""):
            return MasterDecision("generate_code", "code generation requested")
        return MasterDecision("generate_answer", "context ready for answer generation")

    @staticmethod
    def _after_code_execution_agent(state: dict[str, Any]) -> MasterDecision:
        if state.get("code_verified", False):
            return MasterDecision("generate_answer", "code verified; generate final explanation")
        if state.get("code_retry_count", 0) == 0:
            return MasterDecision("generate_code", "code execution failed; retry code generation")
        return MasterDecision("finalize_answer", "code execution failed after retry; finalize with disclaimer")

    def _after_verifier_agent(self, state: dict[str, Any]) -> MasterDecision:
        if state.get("verified", False):
            return MasterDecision("finalize_answer", "answer verified")

        retry_count = state.get("retry_count", {})
        if self._recovery.can_retry("verify", retry_count):
            return MasterDecision("build_context", "answer not grounded; regenerate with same context")
        return MasterDecision("human_fallback", "answer verification failed and retry exhausted")

    @staticmethod
    def _primary_intent(state: dict[str, Any]) -> str:
        return str(state.get("deep_intent", {}).get("primary_intent", "concept_qa"))

    @staticmethod
    def _query_text(state: dict[str, Any]) -> str:
        return str(state.get("query", "")).lower()

    @classmethod
    def _requires_code(cls, state: dict[str, Any]) -> bool:
        return cls._primary_intent(state) == "code_generation"

    @classmethod
    def _requires_tools(cls, state: dict[str, Any]) -> bool:
        primary = cls._primary_intent(state)
        if primary == "error_diagnosis":
            return True

        query = cls._query_text(state)
        tool_keywords = (
            "工单", "ticket", "tkt-", "系统状态", "服务状态", "健康检查",
            "运行状态", "错误码", "error code", "用户信息", "用户档案",
            "我的信息", "个人信息",
        )
        return any(keyword in query for keyword in tool_keywords)

    @classmethod
    def tool_intent(cls, state: dict[str, Any]) -> str:
        """Map deep intent/query facts to the legacy tool-agent intent labels."""
        primary = cls._primary_intent(state)
        query = cls._query_text(state)
        if primary == "error_diagnosis":
            return "troubleshooting"
        if "工单" in query or "ticket" in query or "tkt-" in query:
            return "ticket_query"
        return primary


# ===========================================================================
# Module-level helpers for routing_path tagging
# ===========================================================================


def _resolve_rule_path(state: dict[str, Any]) -> str:
    """Determine whether rules ran as a fallback or as the primary path.

    Returns ``"rule_direct"`` when the LLM was never even attempted
    (e.g. mock provider); ``"rule"`` when the LLM was attempted but
    failed. This distinction matters for observability — operators
    want to know if the LLM router is actually exercising code paths.
    """
    try:
        from enterprise_agentic_rag.llm.provider_factory import get_llm_provider

        provider = get_llm_provider()
        if provider.provider_name == "mock":
            return "rule_direct"
    except Exception:
        # If we can't even build a provider, treat the rule path as a fallback
        return "rule"
    return "rule"


def _with_routing_path(decision: MasterDecision, path: str) -> MasterDecision:
    """Return a new MasterDecision tagged with the given routing_path.

    ``MasterDecision`` is frozen, so we reconstruct it. Kept as a
    helper because we set the field in two places (decide() and any
    future direct callers).
    """
    return MasterDecision(
        next_node=decision.next_node,
        reason=decision.reason,
        routing_path=path,
    )
