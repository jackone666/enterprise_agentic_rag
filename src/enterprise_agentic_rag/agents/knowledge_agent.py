"""Knowledge agent — LLM-first generation with template fallback."""

from __future__ import annotations

from typing import Any


def generate_answer(
    query: str,
    retrieved_docs: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Generate answer — sync compatibility wrapper.

    Production workflow uses ``generate_answer_async``. This sync wrapper keeps
    older tests/scripts working and falls back to templates when called from an
    already-running event loop.
    """
    from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
    provider = get_llm_provider()

    if provider.provider_name != "mock" and retrieved_docs:
        result = _generate_with_llm(provider, query, retrieved_docs)
        if result is not None:
            return result

    return _generate_template(query, retrieved_docs)


async def generate_answer_async(
    query: str,
    retrieved_docs: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Generate answer asynchronously — true LLM-first production path."""
    from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
    provider = get_llm_provider()

    if provider.provider_name != "mock" and retrieved_docs:
        result = await _generate_with_llm_async(provider, query, retrieved_docs)
        if result is not None:
            return result

    return _generate_template(query, retrieved_docs)


def _generate_template(query: str, retrieved_docs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if not retrieved_docs:
        return ("抱歉，我在知识库中没有找到与您问题相关的信息。", [])

    paragraphs: list[str] = []
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, doc in enumerate(retrieved_docs):
        source = str(doc.get("source", "unknown"))
        content = str(doc.get("content", ""))
        score = float(doc.get("score", 0))
        paragraphs.append(content)
        if source not in seen:
            seen.add(source)
            citations.append({"index": len(citations) + 1, "source": source, "relevance_score": round(score, 3)})

    answer = f"根据知识库检索结果，为您找到以下相关信息：\n\n{chr(10).join(paragraphs)}\n\n---\n参考来源：共引用了 {len(citations)} 个文档。\n"
    for c in citations:
        answer += f"[{c['index']}] {c['source']} (相关度: {c['relevance_score']})\n"
    return answer, citations


def _generate_with_llm(provider, query: str, docs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]] | None:
    import asyncio
    parts = [f"[文档{i + 1}] {d.get('source', '')}\n{d.get('content', '')}" for i, d in enumerate(docs)]
    prompt = ("你是一个企业知识库问答助手。请根据以下参考文档回答用户问题。在回答中使用 [1]、[2] 等标记引用来源。\n\n"
              f"参考文档:\n{chr(10).join(parts)}\n\n用户问题: {query}\n\n请生成回答：")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return None
        resp = loop.run_until_complete(provider.generate(prompt, temperature=0.3, max_tokens=2048))
        if resp.success and resp.content:
            citations = [{"index": i + 1, "source": d.get("source", ""), "relevance_score": round(float(d.get("score", 0)), 3)} for i, d in enumerate(docs)]
            return resp.content, citations
    except Exception:
        pass
    return None


async def _generate_with_llm_async(provider, query: str, docs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]] | None:
    parts = [f"[文档{i + 1}] {d.get('source', '')}\n{d.get('content', '')}" for i, d in enumerate(docs)]
    prompt = ("你是一个企业知识库问答助手。请根据以下参考文档回答用户问题。在回答中使用 [1]、[2] 等标记引用来源。\n\n"
              f"参考文档:\n{chr(10).join(parts)}\n\n用户问题: {query}\n\n请生成回答：")
    try:
        resp = await provider.generate(prompt, temperature=0.3, max_tokens=2048)
        if resp.success and resp.content:
            citations = [
                {
                    "index": i + 1,
                    "source": d.get("source", ""),
                    "relevance_score": round(float(d.get("score", 0)), 3),
                }
                for i, d in enumerate(docs)
            ]
            return resp.content, citations
    except Exception:
        pass
    return None
