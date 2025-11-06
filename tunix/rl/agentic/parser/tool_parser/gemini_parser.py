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

"""Tool parser for Gemini models."""

from typing import Any, List

from tunix.rl.agentic.parser.tool_parser import tool_parser_base
from tunix.rl.agentic.tools import base_tool

BaseTool = base_tool.BaseTool
ToolCall = base_tool.ToolCall
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
