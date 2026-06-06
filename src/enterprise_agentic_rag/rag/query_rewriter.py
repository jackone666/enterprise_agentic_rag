"""Query rewriter — HyDE, decomposition, expansion for better retrieval recall."""

from __future__ import annotations

import asyncio
from typing import Any


def rewrite_query(query: str) -> dict[str, Any]:
    """Rewrite query with multiple strategies to improve retrieval recall.

    Returns dict with:
        - rewritten: the primary rewritten query
        - variants: list of alternative query formulations
        - strategy: which strategy was used
    """
    variants: list[str] = [query]

    # 1. Rule-based decomposition (split on Chinese conjunctions)
    decomposed = _decompose(query)
    if len(decomposed) > 1:
        variants.extend(decomposed)
        return {"rewritten": " ".join(decomposed), "variants": variants, "strategy": "decomposition"}

    # 2. Query expansion (add synonyms for key Chinese terms)
    expanded = _expand_keywords(query)
    if expanded != query:
        variants.append(expanded)
        return {"rewritten": expanded, "variants": variants, "strategy": "expansion"}

    return {"rewritten": query, "variants": variants, "strategy": "none"}


def hyde_rewrite(query: str, llm_provider: Any = None) -> str:
    """HyDE: generate a hypothetical document from the query, then use that as the search query.

    Requires an LLM provider. Falls back gracefully when unavailable.
    """
    if llm_provider is None or llm_provider.provider_name == "mock":
        return query

    prompt = (
        "你是一个企业知识库助手。请根据以下用户问题，"
        "写一段简短的假设性文档段落（50-100字），"
        "这段落应该包含可能回答该问题所需的关键信息。\n\n"
        f"用户问题: {query}\n\n假设段落:"
    )
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            resp = loop.run_until_complete(
                llm_provider.generate(prompt, temperature=0.3, max_tokens=200)
            )
            if resp.success and resp.content:
                return resp.content.strip()
    except Exception:
        pass
    return query


def classify_query_type(query: str) -> str:
    """Classify query as semantic, keyword, or mixed for fusion weight tuning."""
    semantic_markers = ["是什么", "为什么", "如何", "怎样", "含义", "区别", "关系", "影响"]
    keyword_markers = ["工单", "TKT-", "错误码", "AUTH_", "版本", "配置"]

    sem_count = sum(1 for m in semantic_markers if m in query)
    kw_count = sum(1 for m in keyword_markers if m in query)

    if sem_count > kw_count:
        return "semantic"
    elif kw_count > sem_count:
        return "keyword"
    return "mixed"


def _decompose(query: str) -> list[str]:
    """Split query on Chinese conjunction words."""
    parts = [query]
    for sep in ["？", "?", "；", ";", "。", "和", "与", "以及", "还有", "另外"]:
        new_parts = []
        for p in parts:
            new_parts.extend(p.split(sep))
        parts = [p.strip() for p in new_parts if p.strip()]
    return parts if len(parts) > 1 else [query]


def _expand_keywords(query: str) -> str:
    """Add synonyms for common Chinese enterprise terms."""
    synonyms = {
        "API": "API 接口 应用程序接口",
        "认证": "认证 鉴权 登录",
        "密码": "密码 口令 凭证",
        "错误": "错误 异常 报错 故障",
        "权限": "权限 授权 访问控制",
        "配置": "配置 设置 参数",
        "部署": "部署 上线 发布",
    }
    result = query
    for k, v in synonyms.items():
        if k in query and v.split()[0] in query:
            result = result.replace(k, v, 1)
    return result
