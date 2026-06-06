"""Cache helpers for the LangGraph workflow.

Extracted from graph/workflow.py to keep the main graph file lean.
Handles semantic-cache key scoping so that different user/permission
contexts never share cache entries (defense against ACL leaks).
"""

from __future__ import annotations


def cache_scope(state: dict) -> str:
    """Build a cache namespace from user/permission facts.

    Used as a prefix for the semantic cache key so that:
    - A basic user never receives admin-scoped cached answers.
    - A revoked permission immediately invalidates the relevant slice
      (next request gets a fresh key with the new scope).
    """
    role = state.get("user_role", "")
    permissions = ",".join(sorted(state.get("permissions", [])))
    return f"role={role};permissions={permissions}"
