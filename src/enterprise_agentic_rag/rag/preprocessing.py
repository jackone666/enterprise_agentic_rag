"""Document preprocessing — text cleaning, structure extraction, metadata enrichment."""

from __future__ import annotations

import re
from typing import Any


def clean_text(text: str) -> str:
    """清洗控制字符、规范化空白、统一中文标点."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # 统一中文标点
    text = text.replace("，，", "，").replace("。。", "。")
    return text.strip()


def extract_sections(text: str) -> list[dict[str, Any]]:
    """提取 Markdown / 中文编号标题层级."""
    sections: list[dict[str, Any]] = []
    heading_re = re.compile(
        r"^(#{1,6})\s+(.+)$|"                     # Markdown #
        r"^([一二三四五六七八九十]+)[、．.]\s*(.+)$|"   # 中文编号
        r"^(\d+)[.)]\s+(.+)$",                     # 数字编号
        re.MULTILINE,
    )
    for m in heading_re.finditer(text):
        if m.group(1):
            level = len(m.group(1))
            title = m.group(2)
        elif m.group(3):
            level = 2
            title = f"{m.group(3)}、{m.group(4)}"
        else:
            level = 2
            title = f"{m.group(5)}. {m.group(6)}"
        sections.append({"level": level, "title": title.strip(), "pos": m.start()})
    return sections


def extract_tables(text: str) -> list[dict[str, Any]]:
    """检测 Markdown pipe table."""
    tables: list[dict[str, Any]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"^[\|\s\-:]+$", lines[i + 1]):
            start = i
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            tables.append({"start": start, "rows": len(table_lines) - 2, "raw": "\n".join(table_lines)})
        else:
            i += 1
    return tables


def extract_image_refs(text: str) -> list[str]:
    return re.findall(r"!\[.*?\]\((.*?)\)", text)


def extract_citations(text: str) -> list[str]:
    refs: list[str] = []
    refs.extend(re.findall(r"\[\d+\]", text))
    refs.extend(re.findall(r"（[^）]*\d{4}[^）]*）", text))
    return refs


def preprocess_document(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    return {
        "cleaned": cleaned,
        "sections": extract_sections(cleaned),
        "tables": extract_tables(cleaned),
        "images": extract_image_refs(cleaned),
        "citations": extract_citations(cleaned),
        "char_count": len(cleaned),
        "title": extract_sections(cleaned)[0]["title"] if extract_sections(cleaned) else "",
    }
