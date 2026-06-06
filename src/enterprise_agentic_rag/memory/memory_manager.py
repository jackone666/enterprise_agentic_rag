"""Unified memory orchestrator — Redis + PostgreSQL + in-memory fallback.

Coordinates:
- ContextWindow     (prompt-visible messages, summaries, tool/memory snippets)
- WorkingMemory     (LangGraph State + checkpointed temporary task state)
- ShortTermMemory   (current-session chat history + conversation summary)
- LongTermMemory    (semantic preferences/facts/rules + episodic past events)
"""

from __future__ import annotations

import time
from typing import Any

from enterprise_agentic_rag.memory.checkpoint import CheckpointStore
from enterprise_agentic_rag.memory.long_term_memory import LongTermMemory
from enterprise_agentic_rag.memory.memory_classifier import (
    MemoryClassifier,
    MemoryTarget,
)
from enterprise_agentic_rag.memory.short_term_memory import ShortTermMemory
from enterprise_agentic_rag.memory.summary_memory import SummaryMemory
from enterprise_agentic_rag.memory.user_memory import UserMemory


class MemoryManager:
    """Orchestrates memory/state tiers with graceful degradation."""

    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        summary: SummaryMemory | None = None,
        user: UserMemory | None = None,
        checkpoint: CheckpointStore | None = None,
        long_term: LongTermMemory | None = None,
        classifier: MemoryClassifier | None = None,
    ) -> None:
        self.short_term = short_term or ShortTermMemory()
        self.summary = summary or SummaryMemory()
        self.user = user or UserMemory()
        self.checkpoint = checkpoint or CheckpointStore()
        self.long_term = long_term or LongTermMemory()
        self.classifier = classifier or MemoryClassifier()

    # ------------------------------------------------------------------
    # Load — called at workflow START
    # ------------------------------------------------------------------
    def load_memory_context(
        self,
        session_id: str,
        user_id: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        chat_history = self.short_term.get_history(session_id)

        # Summary
        self.summary.update_summary(session_id, chat_history)
        session_summary = self.summary.get_summary(session_id)

        # User profile
        user_profile = self.user.get_profile(user_id)
        user_context = self.user.get_context_string(user_id)

        # Long-term memories. Semantic memories are stable preferences, facts,
        # knowledge, and rules; episodic memories describe what happened before.
        long_term_memories = self.long_term.retrieve(user_id, query=query, top_k=5)
        semantic_memories = [
            m for m in long_term_memories
            if m.get("memory_type", "episodic") == MemoryTarget.SEMANTIC.value
        ]
        episodic_memories = [
            m for m in long_term_memories
            if m.get("memory_type", "episodic") == MemoryTarget.EPISODIC.value
        ]

        # Checkpoint
        saved = self.checkpoint.load_checkpoint(session_id)
        checkpoint_id = saved.get("checkpoint_id", "") if saved else ""

        context_window = {
            "recent_messages": chat_history,
            "session_summary": session_summary,
            "user_profile_pin": user_profile,
            "semantic_memory_pins": semantic_memories[:3],
            "episodic_memory_snippets": episodic_memories,
        }
        working_memory = {
            "state_store": "LangGraph AgentState",
            "checkpoint_id": checkpoint_id,
            "has_checkpoint": checkpoint_id != "",
            "checkpoint_keys": sorted(saved.keys()) if saved else [],
        }

        return {
            "chat_history": chat_history,
            "session_summary": session_summary,
            "user_profile": user_profile,
            "memory_context": {
                "user_context": user_context,
                "session_summary": session_summary,
                "chat_turns": len(chat_history),
                "has_checkpoint": checkpoint_id != "",
                "long_term_memory_count": len(long_term_memories),
                "episodic_memory_count": len(episodic_memories),
                "semantic_memory_count": len(semantic_memories),
                "memory_layers": [
                    "context_window",
                    "working_memory",
                    "short_term_memory",
                    "long_term_memory",
                ],
            },
            "checkpoint_id": checkpoint_id,
            "context_window": context_window,
            "working_memory": working_memory,
            "long_term_memories": long_term_memories,
            "episodic_memories": episodic_memories,
            "semantic_memories": semantic_memories,
        }

    # ------------------------------------------------------------------
    # Save — called at workflow END
    # ------------------------------------------------------------------
    def save_memory_context(self, session_id: str, state: dict[str, Any]) -> str:
        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        intent = state.get("intent", "")

        # Short-term (Redis + PG)
        self.short_term.add_message(session_id, "user", query, intent)
        self.short_term.add_message(session_id, "assistant", final_answer, intent)

        # Summary (PG + Redis cache)
        history = self.short_term.get_history(session_id)
        self.summary.update_summary(session_id, history)

        user_id = state.get("user_id", "")
        session_tokens = state.get("session_token_count", 0)
        decisions = self.classifier.decide(
            query,
            session_token_count=session_tokens,
            session_turn_count=len(history),
        )

        if not any(d.target == MemoryTarget.SKIP for d in decisions):
            for decision in decisions:
                if decision.target in {MemoryTarget.EPISODIC, MemoryTarget.SEMANTIC}:
                    self.long_term.store_entry(
                        user_id,
                        decision.content,
                        session_id=session_id,
                        importance=min(1.0, max(0.0, decision.score / 6.0)),
                        memory_type=decision.target.value,
                        metadata={
                            "reasons": decision.reasons,
                            **decision.metadata,
                        },
                    )

        # Keep the existing rule scorer as a fallback for high-signal turns.
        self.long_term.extract_and_store(history, user_id, session_id)

        # Checkpoint (Redis)
        safe_state = dict(state)
        safe_state["checkpoint_id"] = f"ckpt_{session_id}_{int(time.time())}"
        cid = self.checkpoint.save_checkpoint(session_id, safe_state)

        return cid
