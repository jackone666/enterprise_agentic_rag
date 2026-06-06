"""Mock LLM provider — retains all current behavior, no API dependency."""

from __future__ import annotations

import json
from typing import Any

from enterprise_agentic_rag.llm.base import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Returns deterministic template-based responses — no real LLM."""

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        return self._timed_generate(prompt, temperature=temperature, max_tokens=max_tokens)

    async def structured_generate(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return self._timed_generate(prompt, temperature=temperature)

    # ------------------------------------------------------------------
    # Sync logic (called via _timed_generate)
    # ------------------------------------------------------------------
    def _sync_generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Heuristic response based on prompt content."""
        content = self._mock_response(prompt)
        return LLMResponse(
            content=content,
            success=True,
            provider="mock",
            model="mock",
            usage={"prompt_tokens": len(prompt) // 2, "completion_tokens": len(content) // 2},
        )

    def _mock_response(self, prompt: str) -> str:
        """Return intent-like responses based on prompt keywords."""
        pl = prompt.lower()

        # Intent classification
        if "意图" in prompt or "classify" in pl or "分类" in prompt:
            if "错误" in prompt or "401" in prompt or "排查" in prompt:
                return "troubleshooting"
            if "api" in pl or "接口" in prompt or "sdk" in pl:
                return "technical_question"
            if "工单" in prompt or "ticket" in pl:
                return "ticket_query"
            if "政策" in prompt or "合规" in prompt or "权限" in prompt:
                return "policy_question"
            return "general_question"

        # Answer generation
        if "生成" in prompt or "回答" in prompt or "答案" in prompt:
            # Extract the query part
            return "[Mock LLM] 根据知识库内容生成的回答。在实际部署中，此处将由真实 LLM 基于检索文档生成答案。"

        # Routing decision (MasterAgent LLM-assisted routing)
        if "last_agent_step" in prompt and "next_node" in prompt.lower():
            return json.dumps({
                "next_node": "retrieve_knowledge",
                "reason": "mock routing: default to knowledge retrieval",
            })

        # Verification
        if "校验" in prompt or "verify" in pl or "幻觉" in prompt:
            return json.dumps({"verified": True, "reason": "mock verification passed"})

        # Summary — dynamically detect topics from the prompt
        if "摘要" in prompt or "summary" in pl:
            _summary_keywords = [
                "API", "认证", "错误排查", "配置", "部署", "密码",
                "权限", "工单", "SDK", "性能", "安全", "数据库",
            ]
            topics = [kw for kw in _summary_keywords if kw in prompt]
            topic_str = "、".join(topics) if topics else "API、认证、错误排查"
            user_msgs = [line for line in prompt.split("\n") if line.startswith("[user]")]
            first_q = user_msgs[0][7:57] + "..." if user_msgs else "用户提问"
            return (
                f"摘要：本次对话共涉及 {topic_str} 等主题。"
                f"用户主要关注 {first_q}。"
                f"助手提供了相关解答和建议。\n"
                f"主题：{topic_str}"
            )

        # Fallback
        return "[Mock LLM] 响应 — 升级至真实 LLM 可获得更高质量答案。"
