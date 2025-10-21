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

"""Base classes for parsing tool calls and generating tool prompts."""

import abc
import dataclasses
import json
from typing import Any, List, Literal

from tunix.rl.agentic.tools import base_tool

BaseTool = base_tool.BaseTool
abstractmethod = abc.abstractmethod
dataclass = dataclasses.dataclass
ABC = abc.ABC


class ToolParser(ABC):
  """Abstract base class for all tool parsers.

  A ToolParser defines how to: 1. Extract structured tool calls from raw model
  responses. 2. Generate tool prompting text (tool specs / examples) for model
  input.
  """

  @abstractmethod
  def parse(self, model_response: str) -> list[base_tool.ToolCall]:
    """Parse model output and return a list of tool calls.

    Args:
        model_response (str): The full LLM output text.

    Returns:
        list[ToolCall]: Parsed tool call(s).
    """
    pass

  @abstractmethod
  def get_tool_prompt(
      self,
      tools: List[BaseTool],
      *,
      schema_style: Literal["openai", "mcp", "gemini"] = "openai",
  ) -> str:
    """Generate tool-usage instruction prompt from a list of tools.

    Args:
        tools: List of tool instances (BaseTool).
        schema_style: "openai" -> use tool.json (OpenAI function-calling style)
          "mcp"    -> use tool.to_mcp_json() (MCP-compatible format) "gemini" ->
          use Gemini-compatible schema

    Returns:
        str: Prompt text to feed into the model (includes tool schemas).
    """
    pass

  def _tools_schema_dump(
      self,
      tools: List[BaseTool],
      schema_style: str,
  ) -> str:
    """Dumps a list of tool schemas to a JSON string.

    Args:
        tools: List of tool instances (BaseTool).
        schema_style: The style of schema to dump. "openai" and "gemini" use the
          tool's `.json` property, while "mcp" uses `to_mcp_json()`.

    Returns:
        A JSON string representation of the tool schemas.
    """
    if schema_style == "mcp":
      schemas = [t.to_mcp_json() for t in tools]
    elif schema_style == "gemini":
      # Gemini also uses JSON schema, same as OpenAI
      schemas = [t.get_json_schema() for t in tools]
    else:
      schemas = [t.get_json_schema() for t in tools]
    return json.dumps(schemas, ensure_ascii=False, indent=2)

  def parse_tool_outputs(self) -> dict[str, Any]:
    """Optional: Parse tool outputs (e.g. <tool_response> blocks).

    Override if your model uses them.

    Returns:
        dict[str, Any]: Mapping from tool name or ID to its result.
    """
    return {}
