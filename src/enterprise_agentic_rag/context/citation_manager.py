"""Citation manager — tracks sources, chunk IDs, and relevance scores.

Generates structured citation metadata for inclusion in final answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    """A single citation entry."""

    index: int
    source: str          # e.g. "sample_policy.md"
    chunk_id: str        # e.g. "sample_policy.md_2"
    score: float         # 0.0 - 1.0 relevance
    excerpt: str = ""    # short snippet from the chunk
    section: str = ""    # heading / section name


class CitationManager:
    """Collects and formats citation data from retrieved documents."""

    def __init__(self) -> None:
        self._citations: dict[int, Citation] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_citations(
        self,
        retrieved_docs: list[dict[str, Any]],
        prefix: str = "sample",
    ) -> list[Citation]:
        """Build citation list from retrieved documents.

        Each doc is expected to have: ``source``, ``chunk_id``, ``score``, ``content``.
        """
        self._citations.clear()
        for i, doc in enumerate(retrieved_docs):
            source = doc.get("source", f"{prefix}.md")
            chunk_id = doc.get("chunk_id", f"{source}_{i}")
            score = doc.get("score", 0.0)
            content = doc.get("content", "")
            excerpt = content[:120].replace("\n", " ") if content else ""

            # Try to extract a section heading from the content
            section = self._extract_section(content)

            self._citations[i] = Citation(
                index=i + 1,
                source=source,
                chunk_id=chunk_id,
                score=score,
                excerpt=excerpt,
                section=section,
            )

        return list(self._citations.values())

    def format_citation_line(self, citation: Citation) -> str:
        """Format a single citation as a reference line.

        Example: ``[1] sample_policy.md § 访问控制 — 评分 0.85``
        """
        parts = [f"[{citation.index}] {citation.source}"]
        if citation.section:
            parts.append(f"§ {citation.section}")
        parts.append(f"— 评分 {citation.score:.2f}")
        return " ".join(parts)

    def format_references_section(self, citations: list[Citation] | None = None) -> str:
        """Produce a markdown references section.

        Args:
            citations: Optional override list; defaults to internal store.
        """
        items = citations or list(self._citations.values())
        if not items:
            return ""

        lines = ["\n---\n## 📚 参考来源\n"]
        for c in sorted(items, key=lambda x: x.index):
            lines.append(f"- **[{c.index}]** `{c.source}` — {c.excerpt}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_section(content: str) -> str:
        """Try to extract a markdown heading from content."""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                # Remove leading #s and whitespace
                return stripped.lstrip("#").strip()
        return ""
