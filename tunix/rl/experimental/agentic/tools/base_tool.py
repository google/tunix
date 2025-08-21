from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolCall:
    """Represents a single tool call with function name and arguments."""
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolOutput:
    """Standardized output of a tool call."""
    name: str
    output: Optional[str | list | dict] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if self.output is None:
            return ""
        if isinstance(self.output, (dict, list)):
            import json
            return json.dumps(self.output)
        return str(self.output)


class BaseTool(ABC):
    """
    Abstract base class for all tools. Each tool should implement either `apply` or `apply_async`.
    """

    def __init__(self, name: str, description: str):
        """
        Args:
            name: Tool name for referencing in tool calls.
            description: Tool usage description.
        """
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def json(self) -> dict[str, Any]:
        """
        Return OpenAI-compatible function metadata for tool registration.

        Should follow format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "What it does...",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": [ ... ]
                }
            }
        }
        """
        pass

    def to_mcp_json(self) -> dict[str, Any]:
        """
        Return MCP (Model Context Protocol) compliant tool registration.

        Format:
        {
          "type": "function",
          "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.inputSchema,
          }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": getattr(self, "inputSchema", {}),
            },
        }

    def apply(self, **kwargs) -> ToolOutput:
        """Synchronous tool call. Can be overridden."""
        raise NotImplementedError("Tool must implement either `apply()` or `apply_async()`")

    async def apply_async(self, **kwargs) -> ToolOutput:
        """Async version of tool call. Can be overridden."""
        return self.apply(**kwargs)

    def __call__(self, *args, use_async=False, **kwargs):
        if use_async:
            return self.apply_async(*args, **kwargs)
        return self.apply(*args, **kwargs)
