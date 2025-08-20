# tunix/rl/agentic/parser/tool_parser/gemini_parser.py
from typing import Any, List
from tunix.rl.agentic.parser.tool_parser.tool_parser_base import ToolParser, ToolCall
from tunix.rl.agentic.tools.base_tool import BaseTool

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
