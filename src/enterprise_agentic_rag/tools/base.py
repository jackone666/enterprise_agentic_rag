"""Base tool definition.

All tools inherit from :class:`BaseTool` and return :class:`ToolResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standardised result from a tool execution.

    Attributes:
        tool_name: Name of the executed tool.
        success: Whether the tool completed without error.
        output: The tool's return value (when ``success=True``).
        error: Error message (when ``success=False``).
        latency_ms: Wall-clock duration of the execution in milliseconds.
    """

    tool_name: str = ""
    success: bool = True
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0


class BaseTool(ABC):
    """Abstract base class for all tools in the system.

    Subclasses must set the class-level attributes and implement ``execute``.
    """

    # -- Tool metadata (override in subclasses) --
    name: str = ""
    description: str = ""
    is_sensitive: bool = False
    tier: str = "safe"  # safe | sensitive | destructive
    required_permissions: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    max_retries: int = 1

    # -- Schemas (human-readable descriptions for the agent) --
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with named parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            A :class:`ToolResult` indicating success or failure.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Return a human-readable description for the tool agent."""
        return {
            "name": self.name,
            "description": self.description,
            "is_sensitive": self.is_sensitive,
            "required_permissions": self.required_permissions,
            "input_schema": self.input_schema,
        }
