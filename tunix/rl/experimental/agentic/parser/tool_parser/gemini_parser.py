"""Tool parser for Gemini models."""

from typing import Any, List

from tunix.rl.experimental.agentic.parser.tool_parser import tool_parser_base
from tunix.rl.experimental.agentic.tools import base_tool

BaseTool = base_tool.BaseTool
ToolCall = tool_parser_base.ToolCall
ToolParser = tool_parser_base.ToolParser


class GeminiToolParser(ToolParser):

  def parse(self, model_response: Any) -> list[ToolCall]:
    return []

  def get_tool_prompt(
      self,
      tools: List[BaseTool],
      *,
      schema_style: str = "gemini",
  ) -> str:
    return "Return a functionCall with name and args."
