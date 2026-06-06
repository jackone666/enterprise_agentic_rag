"""Context manager — orchestrates token budget, citations, and prompt assembly.

Takes all raw context inputs and produces a ``structured_context`` dict
that downstream nodes can consume directly.
"""

from __future__ import annotations

from typing import Any

from enterprise_agentic_rag.context.citation_manager import Citation, CitationManager
from enterprise_agentic_rag.context.prompt_builder import PromptBuilder
from enterprise_agentic_rag.context.token_budget import TokenBudget


def _build_graph_path_summaries(docs: list[dict[str, Any]]) -> str:
    """Build a compact text summary of graph paths from retrieved docs.

    Only included when docs contain graph_paths metadata.
    """
    all_paths: list[dict] = []
    for doc in docs:
        paths = doc.get("graph_paths", [])
        if paths:
            all_paths.extend(paths)

    if not all_paths:
        return ""

    # Deduplicate by path entities
    seen = set()
    unique_paths = []
    for p in all_paths:
        key = "|".join(p.get("path_entities", []))
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    lines = ["[知识图谱关系路径]"]
    for i, p in enumerate(unique_paths[:10], 1):
        entities = p.get("path_entities", [])
        relations = p.get("path_relations", [])
        path_str = " → ".join(entities)
        rel_str = " → ".join(relations) if relations else "关联"
        lines.append(f"  {i}. {path_str} ({rel_str})")

    return "\n".join(lines)


class ContextManager:
    """Assembles and manages the context window for each workflow turn.

    Inputs:  query, chat_history, session_summary, retrieved_docs,
             tool_results, user_profile
    Output:  structured_context dict with token budget, truncated items,
             citations, and per-agent prompts.
    """

    def __init__(self, max_tokens: int = 4096) -> None:
        self.token_budget = TokenBudget(max_tokens=max_tokens)
        self.citation_manager = CitationManager()
        self.prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def build_context(
        self,
        query: str = "",
        chat_history: list[dict[str, Any]] | None = None,
        session_summary: str = "",
        retrieved_docs: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Produce the full structured context for the current turn.

        Returns a dict with these keys:
        - ``budget_allocation``: :class:`BudgetAllocation`
        - ``truncated_docs``: docs after budget truncation
        - ``truncated_history``: chat history after budget truncation
        - ``citations``: list of :class:`Citation`
        - ``citations_section``: formatted markdown references
        - ``router_prompt``: prompt for intent classification
        - ``knowledge_prompt``: prompt for answer generation
        - ``verifier_prompt``: prompt for answer verification
        - ``context_window``: combined system/user context string
        """
        chat_history = chat_history or []
        retrieved_docs = retrieved_docs or []
        tool_results = tool_results or []
        user_profile = user_profile or {}

        # --- Token budget allocation ---
        allocation = self.token_budget.allocate(
            query=query,
            retrieved_docs=retrieved_docs,
            tool_results=tool_results,
            session_summary=session_summary,
            chat_history=chat_history,
        )

        # --- Truncate items to budget ---
        truncated_docs = self.token_budget.truncate_retrieved_docs(
            retrieved_docs, allocation.retrieved_docs
        )
        truncated_history = self.token_budget.truncate_chat_history(
            chat_history, allocation.chat_history
        )

        # --- Build citations ---
        citations = self.citation_manager.build_citations(truncated_docs)
        citations_section = self.citation_manager.format_references_section(citations)

        # --- User context ---
        user_context = self._build_user_context(user_profile)

        # --- Per-agent prompts ---
        router_prompt = self.prompt_builder.build_router_prompt(
            query=query,
            chat_history=truncated_history,
            user_context=user_context,
        )

        knowledge_prompt = self.prompt_builder.build_knowledge_prompt(
            query=query,
            retrieved_docs=truncated_docs,
            tool_results=tool_results,
            chat_history=truncated_history,
            user_context=user_context,
            session_summary=session_summary,
        )

        verifier_prompt = self.prompt_builder.build_verifier_prompt(
            draft_answer="",  # filled in later by generate_answer
            retrieved_docs=truncated_docs,
        )

        # --- Combined context window ---
        context_window = self._build_context_window(
            user_context=user_context,
            session_summary=session_summary,
            truncated_history=truncated_history,
            truncated_docs=truncated_docs,
        )

        return {
            "budget_allocation": allocation,
            "truncated_docs": truncated_docs,
            "truncated_history": truncated_history,
            "citations": [self._citation_to_dict(c) for c in citations],
            "citations_section": citations_section,
            "router_prompt": router_prompt,
            "knowledge_prompt": knowledge_prompt,
            "verifier_prompt": verifier_prompt,
            "context_window": context_window,
            "token_budget_max": self.token_budget.max_tokens,
            "token_budget_used": (self.token_budget.max_tokens - allocation.remaining),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_user_context(profile: dict[str, Any]) -> str:
        """Build a compact user-context string."""
        if not profile:
            return ""
        name = profile.get("name", profile.get("user_id", "未知"))
        role = profile.get("role", "未知")
        dept = profile.get("department", "未知")
        perms = ", ".join(profile.get("permissions", []))
        tickets = ", ".join(profile.get("recent_tickets", [])) or "无"
        return (
            f"用户: {name} | 角色: {role} | 部门: {dept}\n"
            f"权限: {perms}\n"
            f"最近工单: {tickets}"
        )

    @staticmethod
    def _build_context_window(
        user_context: str = "",
        session_summary: str = "",
        truncated_history: list[dict[str, Any]] | None = None,
        truncated_docs: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build a combined system/user context string.

        Includes graph_paths information when available in document metadata.
        """
        truncated_history = truncated_history or []
        truncated_docs = truncated_docs or []

        parts: list[str] = []
        if user_context:
            parts.append(f"[用户信息]\n{user_context}")
        if session_summary:
            parts.append(f"[会话摘要]\n{session_summary}")
        if truncated_history:
            history_lines = ["[历史对话]"]
            for turn in truncated_history[-6:]:
                role = turn.get("role", "?")
                content = turn.get("content", "")
                history_lines.append(f"  {role}: {content[:150]}")
            parts.append("\n".join(history_lines))
        if truncated_docs:
            parts.append(f"[参考文档] ({len(truncated_docs)} 篇)")
            # Add graph path summaries if available
            graph_summaries = _build_graph_path_summaries(truncated_docs)
            if graph_summaries:
                parts.append(graph_summaries)
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _citation_to_dict(c: Citation) -> dict[str, Any]:
        return {
            "index": c.index,
            "source": c.source,
            "chunk_id": c.chunk_id,
            "score": c.score,
            "excerpt": c.excerpt,
            "section": c.section,
        }

    # ------------------------------------------------------------------
    # Graph-aware document enrichment
    # ------------------------------------------------------------------
    @staticmethod
    def enrich_docs_with_graph_paths(
        docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Enrich document dicts with graph_paths metadata for context.

        Each doc may have ``graph_paths`` from fusion. This method ensures
        graph paths are properly formatted for downstream consumption.
        Does NOT modify docs that don't have graph_paths.

        Args:
            docs: List of document dicts (possibly with graph_paths).

        Returns:
            Same list with enriched metadata (mutated in place for efficiency).
        """
        for doc in docs:
            graph_paths = doc.get("graph_paths", [])
            if not graph_paths:
                continue

            # Add graph-aware metadata
            matched_sources = doc.get("matched_sources", [])
            if "graph" in matched_sources:
                doc.setdefault("metadata", {})
                doc["metadata"]["graph_enriched"] = True
                doc["metadata"]["graph_path_count"] = len(graph_paths)

        return docs
