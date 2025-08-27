"""Registry for different tool parsers."""

from tunix.rl.experimental.agentic.parser.tool_parser import gemini_parser
from tunix.rl.experimental.agentic.parser.tool_parser import qwen_parser
from tunix.rl.experimental.agentic.parser.tool_parser import tool_parser_base

ToolParser = tool_parser_base.ToolParser
QwenToolParser = qwen_parser.QwenToolParser
GeminiToolParser = gemini_parser.GeminiToolParser
_PARSER_REGISTRY = {"qwen": QwenToolParser, "gemini": GeminiToolParser}


def get_tool_parser(parser_name: str = "qwen") -> type[ToolParser]:
  if parser_name not in _PARSER_REGISTRY:
    raise ValueError(f"Unknown parser: {parser_name}")
  return _PARSER_REGISTRY[parser_name]
