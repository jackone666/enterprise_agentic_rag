"""Simple token budget with priority-based truncation.

Approximates token count using character count (roughly 2-4 chars per token
in CJK contexts; we use 2 chars/token as a conservative estimate).

Priority order (highest first):
1. user query
2. top-k retrieved docs
3. tool_results
4. session_summary
5. recent N rounds of chat_history
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BudgetAllocation:
    """Breakdown of how many tokens are allocated to each component."""

    query: int = 0
    retrieved_docs: int = 0
    tool_results: int = 0
    session_summary: int = 0
    chat_history: int = 0
    remaining: int = 0


class TokenBudget:
    """Manages token allocation and truncation for context windows."""

    # Rough mapping: 2 characters ≈ 1 token for CJK-mixed text
    CHARS_PER_TOKEN = 2

    def __init__(self, max_tokens: int = 4096) -> None:
        self.max_tokens = max_tokens
        self.allocation: BudgetAllocation | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate_tokens(self, text: str) -> int:
        """Rough token count from character length."""
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def allocate(
        self,
        query: str = "",
        retrieved_docs: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        session_summary: str = "",
        chat_history: list[dict[str, Any]] | None = None,
    ) -> BudgetAllocation:
        """Distribute token budget across components by priority.

        Lower-priority items are truncated when the budget is exhausted.
        Returns the allocation breakdown.
        """
        retrieved_docs = retrieved_docs or []
        tool_results = tool_results or []
        chat_history = chat_history or []

        budget = self.max_tokens
        alloc = BudgetAllocation()

        # --- Priority 1: user query (always keep full) ---
        q_tokens = self.estimate_tokens(query)
        alloc.query = min(q_tokens, budget)
        budget -= alloc.query

        # --- Priority 2: top-k retrieved docs ---
        alloc.retrieved_docs = self._fit_docs(retrieved_docs, budget)
        budget -= alloc.retrieved_docs

        # --- Priority 3: tool results ---
        alloc.tool_results = self._fit_tool_results(tool_results, budget)
        budget -= alloc.tool_results

        # --- Priority 4: session summary ---
        summary_tokens = self.estimate_tokens(session_summary)
        alloc.session_summary = min(summary_tokens, budget)
        budget -= alloc.session_summary

        # --- Priority 5: recent N rounds of chat_history ---
        alloc.chat_history = self._fit_chat_history(chat_history, budget)
        budget -= alloc.chat_history

        alloc.remaining = max(0, budget)
        self.allocation = alloc
        return alloc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fit_docs(
        self,
        docs: list[dict[str, Any]],
        budget: int,
    ) -> int:
        """Pack as many docs as possible within *budget* tokens."""
        used = 0
        for doc in docs:
            content = doc.get("content", "")
            tokens = self.estimate_tokens(content)
            if used + tokens <= budget:
                used += tokens
            else:
                break
        return used

    def _fit_tool_results(
        self,
        results: list[dict[str, Any]],
        budget: int,
    ) -> int:
        """Pack tool results within *budget*."""
        used = 0
        for r in results:
            text = str(r.get("output", ""))
            tokens = self.estimate_tokens(text)
            if used + tokens <= budget:
                used += tokens
            else:
                break
        return used

    def _fit_chat_history(
        self,
        history: list[dict[str, Any]],
        budget: int,
    ) -> int:
        """Keep most recent turns that fit within *budget*."""
        used = 0
        kept = 0
        # Walk from the end (most recent)
        for turn in reversed(history):
            content = turn.get("content", "")
            tokens = self.estimate_tokens(content)
            if used + tokens <= budget:
                used += tokens
                kept += 1
            else:
                break
        return used

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------
    def truncate_retrieved_docs(
        self,
        docs: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Return the prefix of *docs* that fits in *max_tokens* tokens."""
        result: list[dict[str, Any]] = []
        used = 0
        for doc in docs:
            content = doc.get("content", "")
            tokens = self.estimate_tokens(content)
            if used + tokens > max_tokens:
                # Truncate the last document's content to fit
                available = max_tokens - used
                if available > 20:  # only bother if enough room
                    truncated = {**doc, "content": content[: available * self.CHARS_PER_TOKEN] + "…"}
                    result.append(truncated)
                break
            used += tokens
            result.append(doc)
        return result

    def truncate_chat_history(
        self,
        history: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Return the most recent turns that fit in *max_tokens* tokens."""
        result: list[dict[str, Any]] = []
        used = 0
        for turn in reversed(history):
            content = turn.get("content", "")
            tokens = self.estimate_tokens(content)
            if used + tokens > max_tokens:
                break
            used += tokens
            result.append(turn)
        result.reverse()
        return result
