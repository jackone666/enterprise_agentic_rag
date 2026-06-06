"""Context management system.

Components:
- TokenBudget: Priority-based token allocation and truncation
- CitationManager: Source citation tracking and formatting
- PromptBuilder: Role-specific prompt assembly
- ContextManager: Unified orchestrator
"""

from enterprise_agentic_rag.context.context_manager import ContextManager

__all__ = ["ContextManager"]
