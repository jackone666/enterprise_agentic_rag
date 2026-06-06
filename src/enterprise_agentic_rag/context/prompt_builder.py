"""Prompt builder — assembles structured prompts for each agent role.

Produces template strings that combine system instructions, user context,
retrieved knowledge, tool outputs, and conversation history.
"""

from __future__ import annotations

from typing import Any


class PromptBuilder:
    """Generates role-specific prompts from structured context."""

    # ------------------------------------------------------------------
    # Router agent prompt
    # ------------------------------------------------------------------
    @staticmethod
    def build_router_prompt(
        query: str,
        chat_history: list[dict[str, Any]] | None = None,
        user_context: str = "",
    ) -> str:
        """Build the intent-classification prompt."""
        history_text = PromptBuilder._format_history(chat_history or [])
        parts = [
            "你是一个企业级意图分类器。",
            "请将用户问题分类为以下意图之一：",
            "policy_question, technical_question, troubleshooting, ticket_query, general_question",
        ]
        if user_context:
            parts.append(f"\n用户信息:\n{user_context}")
        if history_text:
            parts.append(f"\n历史对话:\n{history_text}")
        parts.append(f"\n当前问题:\n{query}")
        parts.append("\n请只返回意图标签。")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Knowledge agent prompt
    # ------------------------------------------------------------------
    @staticmethod
    def build_knowledge_prompt(
        query: str,
        retrieved_docs: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        user_context: str = "",
        session_summary: str = "",
    ) -> str:
        """Build the answer-generation prompt."""
        retrieved_docs = retrieved_docs or []
        tool_results = tool_results or []
        chat_history = chat_history or []

        parts = [
            "你是一个企业知识库问答助手。",
            "请根据提供的参考文档回答用户问题。",
            "在回答中使用 [1]、[2] 等标记引用来源。",
            "如果信息不足，请明确说明。",
        ]

        if user_context:
            parts.append(f"\n## 用户信息\n{user_context}")

        if session_summary:
            parts.append(f"\n## 会话摘要\n{session_summary}")

        history_text = PromptBuilder._format_history(chat_history)
        if history_text:
            parts.append(f"\n## 历史对话\n{history_text}")

        # Documents
        if retrieved_docs:
            doc_lines = ["\n## 参考文档\n"]
            for i, doc in enumerate(retrieved_docs):
                source = doc.get("source", "unknown")
                content = doc.get("content", "")
                doc_lines.append(f"### 文档 {i + 1} — {source}")
                doc_lines.append(content)
            parts.append("\n".join(doc_lines))

        # Tool results
        if tool_results:
            tool_lines = ["\n## 工具执行结果\n"]
            for tr in tool_results:
                name = tr.get("tool_name", "unknown")
                output = tr.get("output", "")
                success = tr.get("success", False)
                status = "✅" if success else "❌"
                tool_lines.append(f"### {status} {name}")
                tool_lines.append(str(output))
            parts.append("\n".join(tool_lines))

        parts.append(f"\n## 用户问题\n{query}")
        parts.append("\n请生成回答：")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Verifier agent prompt
    # ------------------------------------------------------------------
    @staticmethod
    def build_verifier_prompt(
        draft_answer: str,
        retrieved_docs: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build the answer-verification prompt.

        Note: in the mock implementation the verifier does not use this
        prompt (it works via rule checks).  This is included so the prompt
        builder covers all three agents.
        """
        retrieved_docs = retrieved_docs or []
        citations = citations or []

        parts = [
            "你是一个企业级答案校验器。",
            "请从以下维度检查草稿答案是否可靠：",
            "1. 答案是否基于提供的参考文档？",
            "2. 是否包含幻觉信息？",
            "3. 引用标记是否正确？",
            "4. 答案是否完整回答了用户问题？",
        ]

        doc_count = len(retrieved_docs)
        cit_count = len(citations)
        parts.append(f"\n参考文档数量: {doc_count}, 引用数量: {cit_count}")
        parts.append(f"\n## 草稿答案\n{draft_answer}")
        parts.append("\n请返回 JSON: {\"verified\": true/false, \"reason\": \"...\"}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Code agent prompt
    # ------------------------------------------------------------------
    @staticmethod
    def build_code_agent_prompt(
        query: str,
        retrieved_docs: list[dict[str, Any]] | None = None,
        language: str = "typescript",
        execution_result: dict[str, Any] | None = None,
    ) -> str:
        """Build the code-generation prompt.

        Includes optional execution result for the fix-retry loop.
        """
        retrieved_docs = retrieved_docs or []

        parts = [
            f"你是一个{language.upper()}代码生成助手。",
            "请根据提供的参考文档生成一个完整、可运行的代码示例。",
            "",
            "要求：",
            "1. 代码必须完整可运行（包含所有必要的 import/require）",
            "2. 对关键步骤添加注释说明",
            "3. 包含错误处理逻辑",
            "4. 代码不超过 80 行",
        ]

        if retrieved_docs:
            doc_lines = ["\n## 参考文档\n"]
            for i, doc in enumerate(retrieved_docs[:5]):
                source = doc.get("source", "unknown")
                content = doc.get("content", "")[:1500]
                doc_lines.append(f"### 文档 {i + 1} — {source}")
                doc_lines.append(content)
            parts.append("\n".join(doc_lines))

        if execution_result:
            parts.append("\n## 上次执行结果（失败）")
            parts.append(f"- 标准输出: {execution_result.get('stdout', '')}")
            parts.append(f"- 错误输出: {execution_result.get('stderr', '')}")
            parts.append(f"- 退出码: {execution_result.get('exit_code', -1)}")
            parts.append("\n请修复上述错误，重新生成代码。")

        parts.append(f"\n## 用户需求\n{query}")
        parts.append(f"\n请生成{language}代码（只返回代码，不要解释）：")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_history(history: list[dict[str, Any]], last_n: int = 6) -> str:
        """Format the most recent N turns of chat history."""
        if not history:
            return ""
        recent = history[-last_n:]
        lines: list[str] = []
        for turn in recent:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            label = {"user": "👤 用户", "assistant": "🤖 助手", "system": "⚙️ 系统"}.get(role, role)
            lines.append(f"{label}: {content[:200]}")
        return "\n".join(lines)
