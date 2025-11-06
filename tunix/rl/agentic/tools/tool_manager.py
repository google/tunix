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

"""Manages and executes tools for an agentic system."""

from __future__ import annotations

from concurrent import futures
import inspect
from typing import Any, Dict, List, Type
import uuid

from tunix.rl.agentic.tools import base_tool

as_completed = futures.as_completed
ToolOutput = base_tool.ToolOutput
ToolCall = base_tool.ToolCall
BaseTool = base_tool.BaseTool

ThreadPoolExecutor = futures.ThreadPoolExecutor


class ToolManager:
  """Centralized router and executor for managing multiple tools in an agent system.

  The ToolManager provides a unified interface for tool registration, discovery,
  and execution. It supports both single tool invocation and batch execution
  with optional parallelization for improved performance.

  Tools can be registered either through explicit mapping during initialization
  or dynamically added via the registration interface. The manager maintains
  compatibility with both OpenAI function calling and MCP (Model Context
  Protocol)
  standards, etc.
  """

  def __init__(
      self, tool_map: Dict[str, Type[BaseTool]], *, desc_fallback: str = ""
  ):
    """Initialize the tool manager with a collection of tool classes.

    Instantiates all provided tool classes with appropriate names and
    descriptions, creating a ready-to-use tool registry. Tool descriptions
    are derived from class docstrings when available.

    Args:
        tool_map (Dict[str, Type[BaseTool]]): Mapping of tool names to their
          implementation classes. Each class will be instantiated once and
          reused for all calls.
        desc_fallback (str): Default description used when a tool class lacks a
          docstring. Helps maintain consistent documentation.
    """
    self._tool_dict: Dict[str, BaseTool] = {}
    for name, cls in tool_map.items():
      if inspect.isabstract(cls):
        raise TypeError(
            f"Cannot instantiate abstract tool class '{name}': {cls.__name__}. "
            "Please provide concrete implementations of BaseTool."
        )
      self._tool_dict[name] = cls(  # pytype: disable=not-instantiable
          name=name, description=(cls.__doc__ or desc_fallback)
      )

  @property
  def names(self) -> List[str]:
    """Get a list of all registered tool names.

    Returns:
        List[str]: Names of all available tools for discovery and validation
    """
    return list(self._tool_dict.keys())

  def get_tools(self) -> List[BaseTool]:
    """Get a list of all registered tool instances.

    Returns:
        List[BaseTool]: A list of all tool instances.
    """
    return list(self._tool_dict.values())

  def get_json_schema(self) -> List[dict[str, Any]]:
    """Get OpenAI-compatible JSON schemas for all registered tools.

    Generates function metadata suitable for injection into LLM prompts
    or direct use with OpenAI's function calling API. Each tool's schema
    includes name, description, and parameter specifications.

    Returns:
        List[dict]: OpenAI function calling format schemas for all tools
    """
    return [tool.get_json_schema() for tool in self._tool_dict.values()]

  def get_mcp_schema(self) -> List[dict[str, Any]]:
    """Get MCP (Model Context Protocol) compatible tool metadata.

    Provides tool definitions in the standardized MCP format used by
    various AI platforms including Gemini and Claude. Enables cross-platform
    tool integration and discovery.

    Returns:
        List[dict]: MCP-formatted tool metadata for all registered tools
    """
    return [tool.to_mcp_json() for tool in self._tool_dict.values()]

  def register_mcp_tool(self, tool: BaseTool):
    """Register a pre-instantiated MCP-compatible tool.

    Adds a tool instance directly to the registry, useful for tools
    that require complex initialization or external dependencies.
    The tool's name property is used as the registry key.

    Args:
        tool (BaseTool): Fully initialized tool instance ready for execution
    """
    self._tool_dict[tool.name] = tool

  def run(self, tool_name: str, **kwargs: Any) -> ToolOutput:
    """Execute a single tool by name with the provided arguments.

    Looks up the tool in the registry, executes it with the given parameters,
    and handles any exceptions that occur during execution. Returns a
    standardized ToolOutput regardless of success or failure.

    Args:
        tool_name (str): Name of the registered tool to execute
        **kwargs: Tool-specific parameters to pass to the tool's apply method

    Returns:
        ToolOutput: Standardized result containing output, error information,
            or metadata from the tool execution
    """
    tool = self._tool_dict.get(tool_name)
    if tool is None:
      return ToolOutput(
          name=tool_name, error=f"Tool '{tool_name}' not registered."
      )
    # pylint: disable=broad-exception-caught
    try:
      return tool.apply(**kwargs)
    except Exception as e:
      return ToolOutput(name=tool_name, error=f"{type(e).__name__}: {e}")

  async def run_async(self, tool_name: str, **kwargs: Any) -> ToolOutput:
    """Asynchronously execute a single tool by name with the provided arguments.

    Looks up the tool in the registry, executes it with the given parameters,
    and handles any exceptions that occur during execution. Returns a
    standardized ToolOutput regardless of success or failure.

    Args:
        tool_name (str): Name of the registered tool to execute
        **kwargs: Tool-specific parameters to pass to the tool's apply method

    Returns:
        ToolOutput: Standardized result containing output, error information,
            or metadata from the tool execution
    """
    tool = self._tool_dict.get(tool_name)
    if tool is None:
      return ToolOutput(
          name=tool_name, error=f"Tool '{tool_name}' not registered."
      )
    # pylint: disable=broad-exception-caught
    try:
      return await tool.apply_async(**kwargs)
    except Exception as e:
      return ToolOutput(name=tool_name, error=f"{type(e).__name__}: {e}")

  def execute_calls(
      self, calls: List[ToolCall], parallel: bool = True
  ) -> Dict[str, str]:
    """Execute multiple tool calls with optional parallel processing.

    Processes a batch of tool calls either sequentially or in parallel
    using a thread pool. Each call is assigned a unique ID for result
    tracking, and all outputs are converted to string format for
    consistent handling by downstream systems.

    Args:
        calls (List[ToolCall]): List of tool calls to execute, each containing
          tool name and arguments. Calls may optionally have an 'id' attribute.
        parallel (bool): Whether to execute calls concurrently using threads.
          True enables better performance for I/O-bound tools, False ensures
          sequential execution for debugging or resource constraints.

    Returns:
        Dict[str, str]: Mapping from call IDs to string-formatted tool outputs.
            Call IDs are taken from the ToolCall objects or generated if
            missing.
    """
    outputs = {}

    if not parallel:
      # Sequential execution for debugging or resource-constrained environments
      for call in calls:
        cid = getattr(call, "id", None) or str(uuid.uuid4())
        res = self.run(tool_name=call.name, **call.arguments)
        outputs[cid] = str(res)
      return outputs

    # Parallel execution using thread pool for improved performance
    with ThreadPoolExecutor() as executor:
      future_to_id = {}
      for call in calls:
        cid = getattr(call, "id", None) or str(uuid.uuid4())
        future = executor.submit(
            self.run, tool_name=call.name, **call.arguments
        )
        future_to_id[future] = cid

      for future in as_completed(future_to_id):
        cid = future_to_id[future]
        # pylint: disable=broad-exception-caught
        try:
          res = future.result()
          outputs[cid] = str(res)
        except Exception as e:
          # Handle exceptions that occur during parallel execution
          outputs[cid] = f"Error: {type(e).__name__}: {e}"

    return outputs
