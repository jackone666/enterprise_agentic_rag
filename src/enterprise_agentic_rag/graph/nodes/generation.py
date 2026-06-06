"""Answer generation node + chain-of-thought helpers.

Generates the final answer text by calling the knowledge agent, then
appends tool-result and code-execution sections, and finally the
citation footer. The optional ``deep_thinking`` flag adds a CoT
reasoning trace which is exposed to the frontend stream.
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_agentic_rag.agents.knowledge_agent import generate_answer_async
from enterprise_agentic_rag.graph.state import AgentState

logger = logging.getLogger(__name__)


async def generate_answer_node(state: AgentState) -> dict[str, Any]:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    tool_results = state.get("tool_results", [])
    structured_ctx = state.get("structured_context", {})
    deep_thinking = state.get("deep_thinking", True)
    thinking_trace = ""

    if deep_thinking and docs:
        thinking_trace = await _generate_thinking_trace(query, docs)

    answer_text, citations = await generate_answer_async(query, docs)

    if tool_results:
        tool_lines = ["\n---\n## 🔧 工具执行结果\n"]
        for tr in tool_results:
            name = tr.get("tool_name", "unknown")
            if tr.get("success"):
                tool_lines.append(f"- ✅ **{name}**: 执行成功")
            else:
                tool_lines.append(f"- ❌ **{name}**: {tr.get('error', '失败')}")
        answer_text += "\n".join(tool_lines)

    code_result = state.get("code_execution_result", {})
    code_snippet = state.get("code_snippet", "")
    if code_snippet:
        code_section = f"\n---\n## 💻 已验证代码示例（{state.get('code_language', 'ts')}）\n```{state.get('code_language', 'typescript')}\n{code_snippet}\n```"
        if code_result and code_result.get("exit_code") == 0:
            code_section += "\n✅ 代码已在沙箱中成功执行"
        elif code_result:
            code_section += f"\n⚠️ 代码执行有问题: {code_result.get('stderr', '')[:100]}"
        else:
            code_section += "\n⚠️ 代码未经过执行验证，仅供参考"
        answer_text += code_section

    citations_section = structured_ctx.get("citations_section", "")
    if citations_section:
        answer_text += citations_section

    return {
        "draft_answer": answer_text,
        "citations": citations,
        "deep_thinking_content": thinking_trace,
        "thinking_trace": thinking_trace,
        "last_worker": "knowledge_agent",
        "last_agent_step": "generate_answer",
    }


async def _generate_thinking_trace(query: str, docs: list[dict[str, Any]]) -> str:
    """Generate chain-of-thought reasoning trace for the deep thinking feature.

    Makes a lightweight LLM call to produce a structured analysis of how
    to approach the question, what evidence to use, and how to structure the answer.
    """
    from enterprise_agentic_rag.llm.provider_factory import get_llm_provider

    provider = get_llm_provider()
    if provider.provider_name == "mock":
        return _mock_thinking_trace(query, docs)

    doc_summaries = []
    for i, d in enumerate(docs[:5]):
        source = d.get("source", "未知")
        content = d.get("content", "")[:300]
        doc_summaries.append(f"[文档{i + 1} · {source}] {content}")

    prompt = (
        "你是一个智能客服的思考模块。请分析以下用户问题，并输出你的推理过程。\n\n"
        f"用户问题: {query}\n\n"
        "可用参考文档摘要:\n"
        f"{chr(10).join(doc_summaries)}\n\n"
        "请用中文简要输出你的分析思路（200-500字），包括：\n"
        "1. 问题类型分析\n"
        "2. 关键信息识别\n"
        "3. 回答结构规划\n"
        "不要生成最终答案，只输出思考过程。"
    )

    try:
        resp = await provider.generate(prompt, temperature=0.3, max_tokens=800)
        if resp.success and resp.content:
            return resp.content.strip()
    except Exception:
        pass

    return _mock_thinking_trace(query, docs)


def _mock_thinking_trace(query: str, docs: list[dict[str, Any]]) -> str:
    """Generate a rule-based thinking trace when LLM is unavailable."""
    lines = ["分析用户问题..."]
    lines.append(f"问题: {query}")

    if docs:
        lines.append(f"已检索到 {len(docs)} 篇相关文档")
        sources = list({d.get("source", "未知") for d in docs[:5]})
        lines.append(f"参考来源: {', '.join(sources)}")

    keywords_map = {
        "开发": "识别为开发入门类问题，需要提供步骤指南",
        "入门": "识别为开发入门类问题，需要提供步骤指南",
        "API": "识别为API使用类问题，需要提供接口说明和代码示例",
        "错误": "识别为错误诊断类问题，需要提供排查步骤",
        "升级": "识别为版本升级类问题，需要提供升级路径和注意事项",
        "迁移": "识别为迁移类问题，需要提供兼容性和迁移指南",
        "权限": "识别为权限管理类问题，需要说明权限申请流程",
        "生命周期": "识别为概念理解类问题，需要解释生命周期机制",
        "发布": "识别为应用分发类问题，需要说明发布流程",
        "配置": "识别为配置类问题，需要提供配置参数说明",
    }

    for keyword, analysis in keywords_map.items():
        if keyword in query:
            lines.append(analysis)
            break
    else:
        lines.append("识别为通用技术问答，需要综合多个知识来源")

    lines.append("规划回答结构: 1)问题理解 2)核心步骤/要点 3)补充说明 4)参考来源")
    return "\n".join(lines)
