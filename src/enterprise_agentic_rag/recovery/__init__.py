"""Recovery and fallback system.

Components:
- FallbackPolicy: Defines fallback types and decision logic
- RetryPolicy: Retry limits and backoff per node
- RecoveryManager: Unified orchestrator — error → action mapping
"""

from enterprise_agentic_rag.recovery.recovery_manager import RecoveryManager

__all__ = ["RecoveryManager"]
