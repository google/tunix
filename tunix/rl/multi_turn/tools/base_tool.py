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

    def to_string(self) -> str:
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
    Abstract base class for all tools. Each tool should implement either `forward` or `async_forward`.
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

    def forward(self, **kwargs) -> ToolOutput:
        """Synchronous tool call. Can be overridden."""
        raise NotImplementedError("Tool must implement either `forward()` or `async_forward()`")

    async def async_forward(self, **kwargs) -> ToolOutput:
        """Async version of tool call. Can be overridden."""
        return self.forward(**kwargs)

    def __call__(self, *args, use_async=False, **kwargs):
        if use_async:
            return self.async_forward(*args, **kwargs)
        return self.forward(*args, **kwargs)
