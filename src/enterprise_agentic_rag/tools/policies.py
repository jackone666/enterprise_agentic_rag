"""Tool execution policies — tier-based gating.

Tiers:
    safe        → execute immediately
    sensitive   → require pending_confirmation (skip with flag)
    destructive → denied unless ENABLE_DESTRUCTIVE_TOOLS=true
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyDecision(str, Enum):
    ALLOWED = "allowed"
    PENDING = "pending"
    DENIED = "denied"


class ToolTier(str, Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    DESTRUCTIVE = "destructive"


@dataclass
class PolicyResult:
    decision: PolicyDecision
    reason: str = ""
    missing_permissions: list[str] = field(default_factory=list)


def evaluate_policy(
    tool_name: str,
    tier: ToolTier | str = ToolTier.SAFE,
    required_permissions: list[str] | None = None,
    user_permissions: list[str] | None = None,
    *,
    skip_confirmation: bool = False,
) -> PolicyResult:
    """Evaluate whether a tool may be executed.

    Args:
        tool_name: Tool identifier.
        tier: ToolTier.SAFE | SENSITIVE | DESTRUCTIVE.
        required_permissions: Permissions the tool needs.
        user_permissions: Permissions the user has.
        skip_confirmation: Bypass sensitive confirmation.
    """
    required_permissions = required_permissions or []
    user_permissions = user_permissions or []

    if isinstance(tier, str):
        try:
            tier = ToolTier(tier)
        except ValueError:
            tier = ToolTier.SAFE

    # 1. Permission check
    missing = [p for p in required_permissions if p not in user_permissions]
    if missing:
        return PolicyResult(
            decision=PolicyDecision.DENIED,
            reason=f"缺少权限: {', '.join(missing)}",
            missing_permissions=missing,
        )

    # 2. Destructive — deny unless explicitly enabled
    if tier == ToolTier.DESTRUCTIVE:
        enabled = os.getenv("ENABLE_DESTRUCTIVE_TOOLS", "false").lower() in ("1", "true", "yes")
        if not enabled:
            return PolicyResult(
                decision=PolicyDecision.DENIED,
                reason=f"破坏性工具 '{tool_name}' 默认禁止。设置 ENABLE_DESTRUCTIVE_TOOLS=true 开启。",
            )

    # 3. Sensitive — require confirmation
    if tier == ToolTier.SENSITIVE and not skip_confirmation:
        return PolicyResult(
            decision=PolicyDecision.PENDING,
            reason=f"敏感工具 '{tool_name}' 需要用户确认后才能执行",
        )

    return PolicyResult(decision=PolicyDecision.ALLOWED, reason="ok")
