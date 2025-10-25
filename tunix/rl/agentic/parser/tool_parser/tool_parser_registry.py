# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry for different tool parsers."""

from tunix.rl.agentic.parser.tool_parser import gemini_parser
from tunix.rl.agentic.parser.tool_parser import qwen_parser
from tunix.rl.agentic.parser.tool_parser import tool_parser_base

ToolParser = tool_parser_base.ToolParser
QwenToolParser = qwen_parser.QwenToolParser
GeminiToolParser = gemini_parser.GeminiToolParser
_PARSER_REGISTRY = {"qwen": QwenToolParser, "gemini": GeminiToolParser}


def get_tool_parser(parser_name: str = "qwen") -> type[ToolParser]:
  if parser_name not in _PARSER_REGISTRY:
    raise ValueError(f"Unknown parser: {parser_name}")
  return _PARSER_REGISTRY[parser_name]
