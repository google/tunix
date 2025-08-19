from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

@dataclass
class ToolCall:

    name: str                  
    arguments: dict[str, Any]   



class ToolParser(ABC):
    """
    Abstract base class for all tool parsers.
    A ToolParser defines how to:
    1. Extract structured tool calls from raw model responses.
    2. Generate tool prompting text (tool specs / examples) for model input.
    """

    @abstractmethod
    def parse(self, model_response: str) -> list[ToolCall]:
        """
        Parse model output and return a list of tool calls.

        Args:
            model_response (str): The full LLM output text.

        Returns:
            list[ToolCall]: Parsed tool call(s).
        """
        pass

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str:
        """
        Generate tool usage instruction prompt from schema (e.g. tool definitions).

        Args:
            tools_schema (str): Tool spec in JSON/XML/OpenAPI format.

        Returns:
            str: Prompt to feed into the model.
        """
        pass

    def parse_tool_outputs(self, model_response: str) -> dict[str, Any]:
        """
        Optional: Parse tool outputs (e.g. <tool_response> blocks).
        Override if your model uses them.

        Returns:
            dict[str, Any]: Mapping from tool name or ID to its result.
        """
        return {}
