from tunix.rl.agentic.parser.tool_parser.tool_parser_base import ToolParser
from tunix.rl.agentic.parser.tool_parser.qwen_parser import QwenToolParser

_PARSER_REGISTRY = {
    "qwen": QwenToolParser,
    # "openai": OpenAIFunctionToolParser,
}


def get_tool_parser(parser_name: str = "qwen") -> type[ToolParser]:
    if parser_name not in _PARSER_REGISTRY:
        raise ValueError(f"Unknown parser: {parser_name}")
    return _PARSER_REGISTRY[parser_name]
