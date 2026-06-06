"""Code agent — generates code snippets from retrieved documentation.

LLM-first generation with template fallback.
Augments generation with AST-level symbol extraction from retrieved docs
to improve API usage accuracy.
"""

from __future__ import annotations

import re
from typing import Any


def generate_code(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    language: str = "",
) -> dict[str, Any]:
    """Generate a code snippet based on the query and retrieved documentation.

    Returns a dict with:
        code_snippet: The generated code
        language: Detected/requested programming language
        citations: Source citations for the code
        success: Whether generation was successful

    Args:
        query: The user's question.
        retrieved_docs: Retrieved documentation chunks.
        language: Preferred programming language (auto-detected if empty).

    Returns:
        Dict with code generation result.
    """
    from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
    provider = get_llm_provider()

    if not language:
        language = _detect_language(query)

    # Extract code symbols from retrieved docs to enhance generation quality
    symbols_info = None
    if retrieved_docs:
        try:
            symbols_info = _extract_symbols_from_docs(retrieved_docs, language)
        except Exception:
            pass  # Symbol extraction is non-critical — silent fallback

    if provider.provider_name != "mock" and (retrieved_docs or symbols_info):
        result = _generate_with_llm(provider, query, retrieved_docs, language, symbols_info)
        if result is not None:
            return result

    return _generate_template(query, retrieved_docs, language)


def _detect_language(query: str) -> str:
    """Detect the target programming language from the query."""
    query_lower = query.lower()
    lang_keywords = {
        "python": ["python", "py", "pytest"],
        "typescript": ["typescript", "ts", "arkts", "ets", "harmonyos", "@ohos"],
        "javascript": ["javascript", "js", "node"],
        "bash": ["bash", "shell", "sh", "命令行"],
    }
    for lang, keywords in lang_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return lang
    return "typescript"  # default for this project's domain


def _generate_template(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    language: str = "typescript",
) -> dict[str, Any]:
    """Template-based code generation — extracts examples or returns template."""
    # Always produce a template, even without docs (for code_generation intent)
    if not retrieved_docs:
        snippet = _make_template_snippet(query, language)
        return {
            "code_snippet": snippet,
            "language": language,
            "citations": [],
            "success": False,
            "error": "未检索到相关文档，返回模板代码（未验证）",
        }

    # Extract code blocks from retrieved docs
    code_blocks: list[tuple[str, str]] = []
    for doc in retrieved_docs:
        content = doc.get("content", "")
        # Extract fenced code blocks
        fence_pattern = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)
        for m in fence_pattern.finditer(content):
            lang = m.group(1).strip().lower() or "text"
            code = m.group(2).strip()
            if code and len(code) > 20:  # filter trivial blocks
                code_blocks.append((lang, code))

    if code_blocks:
        # Return the longest matching code block
        matching = [cb for cb in code_blocks if language in cb[0]]
        best = matching[0] if matching else code_blocks[0]
        return {
            "code_snippet": best[1],
            "language": best[0] or language,
            "citations": [
                {"source": doc.get("source", ""), "relevance_score": round(float(doc.get("score", 0)), 3)}
                for doc in retrieved_docs[:3]
            ],
            "success": True,
        }

    # No code blocks found — generate a template
    snippet = _make_template_snippet(query, language)

    return {
        "code_snippet": snippet,
        "language": language,
        "citations": [],
        "success": False,
        "error": "检索文档中未找到可用代码块，返回模板代码（未验证）",
    }


def _make_template_snippet(query: str, language: str) -> str:
    """Generate a template code snippet for the given language."""
    templates = {
        "typescript": (
            f"// 关于: {query[:60]}\n"
            "// 代码示例（需根据实际文档补充）\n\n"
            "import { ComponentName } from '@ohos.package';\n\n"
            "function exampleUsage(): void {\n"
            "    // TODO: 根据文档补充具体调用逻辑\n"
            "    const result = ComponentName.method(params);\n"
            "    console.log('Result:', result);\n"
            "}\n"
        ),
        "python": (
            f"# 关于: {query[:60]}\n"
            "# 代码示例（需根据实际文档补充）\n\n"
            "def example_usage():\n"
            "    # TODO: 根据文档补充具体调用逻辑\n"
            "    result = api_client.method(params)\n"
            "    print(f'Result: {result}')\n"
        ),
        "javascript": (
            f"// 关于: {query[:60]}\n"
            "// 代码示例（需根据实际文档补充）\n\n"
            "function exampleUsage() {\n"
            "    // TODO: 根据文档补充具体调用逻辑\n"
            "    const result = library.method(params);\n"
            "    console.log('Result:', result);\n"
            "}\n"
        ),
    }

    return templates.get(language, templates["typescript"])


def _generate_with_llm(
    provider,
    query: str,
    docs: list[dict[str, Any]],
    language: str = "typescript",
    symbols_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Generate code using the LLM provider, enhanced with extracted symbols."""
    import asyncio

    parts = [
        f"[文档{i + 1}] {d.get('source', '')}\n{d.get('content', '')[:1000]}"
        for i, d in enumerate(docs[:5])
    ]

    # Build enhanced prompt with symbol analysis when available
    symbol_section = ""
    if symbols_info:
        symbol_section = _format_symbols_for_prompt(symbols_info, language)

    prompt = (
        f"你是一个{language.upper()}代码生成助手。请根据以下参考文档生成一个可运行的{language}代码示例。\n\n"
        "要求:\n"
        "1. 只返回代码，不要额外解释\n"
        "2. 代码应该完整可运行（包含必要的 import）\n"
        "3. 使用 [1]、[2] 等注释标记引用来源\n"
        "4. 代码不超过 80 行\n\n"
        f"{symbol_section}"
        f"参考文档:\n{chr(10).join(parts)}\n\n"
        f"用户需求: {query}\n\n"
        f"请生成{language}代码:"
    )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return None
        resp = loop.run_until_complete(provider.generate(prompt, temperature=0.3, max_tokens=2048))
        if resp.success and resp.content:
            # Strip markdown code fences if present
            code = resp.content.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                code = "\n".join(lines)

            return {
                "code_snippet": code.strip(),
                "language": language,
                "citations": [
                    {"index": i + 1, "source": d.get("source", ""),
                     "relevance_score": round(float(d.get("score", 0)), 3)}
                    for i, d in enumerate(docs[:5])
                ],
                "success": True,
            }
    except Exception:
        pass
    return None


# ===========================================================================
# Symbol extraction helpers
# ===========================================================================


def _extract_symbols_from_docs(
    retrieved_docs: list[dict[str, Any]],
    target_language: str = "typescript",
) -> dict[str, Any]:
    """Extract code symbols from retrieved documentation chunks.

    Uses the code_symbol_extractor to perform AST-level analysis of code
    blocks within the retrieved documents. Results are grouped by symbol
    type for structured prompt augmentation.

    Args:
        retrieved_docs: Retrieved documentation chunks with 'content' keys.
        target_language: Filter for symbols matching this language.

    Returns:
        Dict with keys: 'symbols' (grouped by type), 'imports', 'functions',
        'classes', 'method_calls', 'total_count'.
        Returns empty dict if no symbols are found.
    """
    try:
        from enterprise_agentic_rag.rag.graph.code_symbol_extractor import (
            extract_code_blocks,
            extract_symbols_from_code,
        )
    except ImportError:
        return {}

    all_symbols: dict[str, list[dict[str, Any]]] = {
        "imports": [],
        "functions": [],
        "classes": [],
        "method_calls": [],
        "types": [],
    }

    for doc in retrieved_docs:
        content = doc.get("content", "")
        if not content:
            continue

        blocks = extract_code_blocks(content)
        for idx, block in enumerate(blocks):
            # Skip blocks in mismatched languages
            if target_language not in ("", block.language) and block.language != target_language:
                continue

            symbols = extract_symbols_from_code(block.code_text, block.language, code_block_idx=idx)
            for sym in symbols:
                entry = {
                    "name": sym.name,
                    "normalized_name": sym.normalized_name,
                    "source_code": sym.source_code[:200],  # truncate
                    "confidence": sym.confidence,
                }
                sym_type = sym.type.upper()
                if sym_type in ("IMPORT", "IMPORT_STATEMENT"):
                    all_symbols["imports"].append(entry)
                elif sym_type in ("FUNCTION", "FUNCTION_DECLARATION", "METHOD_DEFINITION"):
                    all_symbols["functions"].append(entry)
                elif sym_type in ("CLASS", "CLASS_DECLARATION"):
                    all_symbols["classes"].append(entry)
                elif sym_type in ("METHOD_CALL", "CALL_EXPRESSION", "MEMBER_EXPRESSION"):
                    all_symbols["method_calls"].append(entry)
                elif sym_type in ("TYPE", "TYPE_ALIAS", "INTERFACE", "INTERFACE_DECLARATION"):
                    all_symbols["types"].append(entry)

    # Deduplicate by name
    for key in all_symbols:
        seen = set()
        unique = []
        for s in all_symbols[key]:
            if s["normalized_name"] not in seen:
                seen.add(s["normalized_name"])
                unique.append(s)
        all_symbols[key] = unique

    total = sum(len(v) for v in all_symbols.values())
    if total == 0:
        return {}

    return {
        "symbols": all_symbols,
        "total_count": total,
    }


def _format_symbols_for_prompt(symbols_info: dict[str, Any], language: str) -> str:
    """Format extracted symbols into an LLM-friendly prompt section.

    Args:
        symbols_info: Output from _extract_symbols_from_docs.
        language: Target programming language.

    Returns:
        Formatted string ready for injection into the LLM prompt.
    """
    sym = symbols_info.get("symbols", {})
    if not sym:
        return ""

    lines = ["## 从参考文档中提取的代码符号\n"]
    lines.append("以下是检索文档中实际使用的API和符号，请优先使用这些API：\n")

    if sym.get("imports"):
        names = [s["name"] for s in sym["imports"][:10]]
        lines.append(f"- **导入模块**: {', '.join(names)}")

    if sym.get("classes"):
        names = [s["name"] for s in sym["classes"][:10]]
        lines.append(f"- **类/组件**: {', '.join(names)}")

    if sym.get("functions"):
        names = [s["name"] for s in sym["functions"][:10]]
        lines.append(f"- **函数/方法定义**: {', '.join(names)}")

    if sym.get("method_calls"):
        names = [s["name"] for s in sym["method_calls"][:15]]
        lines.append(f"- **方法调用示例**: {', '.join(names)}")

    if sym.get("types"):
        names = [s["name"] for s in sym["types"][:10]]
        lines.append(f"- **类型/接口**: {', '.join(names)}")

    lines.append(f"\n共提取了 {symbols_info.get('total_count', 0)} 个相关代码符号。")
    lines.append("请充分利用上述API和符号信息生成准确、可运行的代码。\n")

    return "\n".join(lines)
