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

"""Tool parser for Qwen models.

This parser handles the specific tool call format used by Qwen models,
which involves XML-like tags around a JSON object.
"""

import json
from typing import Any

from tunix.rl.agentic.parser.tool_parser import tool_parser_base
from tunix.rl.agentic.tools import base_tool

ToolParser = tool_parser_base.ToolParser
ToolCall = base_tool.ToolCall
BaseTool = base_tool.BaseTool


class QwenToolParser(ToolParser):
  """Tool parser for Qwen models.

  This parser handles the specific tool call format used by Qwen models,
  which involves XML-like tags around a JSON object.
  """

  def __init__(self):
    """Initialize the parser with specified type and model."""
    self.tool_call_begin = "<tool_call>"
    self.tool_call_end = "</tool_call>"
    self.tool_output_begin = "<tool_response>"
    self.tool_output_end = "</tool_response>"

  def parse(self, model_response: str) -> list[ToolCall]:
    """Parse tool calls from model output.

    Args:
        model_response (str): Text containing tool calls

    Returns:
        list[ToolCall]: Parsed tool calls
    """
    tool_calls_dicts = self.parse_qwen_tool_calls(model_response)
    tool_calls = [ToolCall(**tool_call) for tool_call in tool_calls_dicts]
    return tool_calls

  def parse_qwen_tool_calls(self, text: str) -> list[dict[str, Any]]:
    """Parse tool calls from text using a simple token format.

    Format:
    <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

    Args:
        text (str): The input text potentially containing tool calls.

    Returns:
        list[dict]: List of parsed tool calls, each containing 'name' and
        'parameters'
    """

    tool_calls: list[dict[str, Any]] = []

    # Return empty list if no tool calls found
    if self.tool_call_begin not in text:
      return tool_calls

    # Process all tool calls in the text
    while self.tool_call_begin in text:
      start = text.find(self.tool_call_begin) + len(self.tool_call_begin)
      end = text.find(self.tool_call_end)
      if end == -1:
        break

      # Extract and parse the JSON content
      json_content = text[start:end].strip()
      try:
        call_data = json.loads(json_content)
        # Convert to common format matching parse_tool_calls output
        tool_calls.append(
            {"name": call_data["name"], "arguments": call_data["arguments"]}
        )
      except json.JSONDecodeError:
        print(f"Error parsing tool call: {json_content}")
        text = text[end + len(self.tool_call_end) :]
        continue

      # Move to next potential tool call
      text = text[end + len(self.tool_call_end) :]

    return tool_calls

  def get_tool_prompt(
      self,
      tools: list[BaseTool],
      schema_style: str = "openai",
  ) -> str:
    tools_schema = self._tools_schema_dump(tools, schema_style=schema_style)
    return f"""
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call><|im_end|>
""".strip()
