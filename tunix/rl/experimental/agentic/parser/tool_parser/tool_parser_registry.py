from tunix.rl.experimental.agentic.parser.tool_parser.tool_parser_base import ToolParser
from tunix.rl.experimental.agentic.parser.tool_parser.qwen_parser import QwenToolParser
from tunix.rl.experimental.agentic.parser.tool_parser.gemini_parser import GeminiToolParser

_PARSER_REGISTRY = {
    "qwen": QwenToolParser,
    "gemini": GeminiToolParser
}


def get_tool_parser(parser_name: str = "qwen") -> type[ToolParser]:
    if parser_name not in _PARSER_REGISTRY:
        raise ValueError(f"Unknown parser: {parser_name}")
    return _PARSER_REGISTRY[parser_name]
