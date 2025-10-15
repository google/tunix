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

"""Base classes and utilities for defining and using tools in agentic systems.

This module provides the foundational structures:
-   `ToolCall`: Represents a request to invoke a tool.
-   `ToolOutput`: Standardized container for tool execution results.
-   `BaseTool`: Abstract base class for all tools, defining the interface
    for registration and execution.
"""

import abc
import dataclasses
import json
from typing import Any, Optional

abstractmethod = abc.abstractmethod
dataclass = dataclasses.dataclass
ABC = abc.ABC


@dataclass
class ToolCall:
  """Represents a single tool invocation request.

  Contains the function name and arguments needed to execute a specific
  tool. This standardized format enables consistent tool calling across
  different agents and environments.

  Attributes:
      name (str): The name of the tool function to invoke
      arguments (dict[str, Any]): Key-value pairs of parameters to pass to the
        tool
  """

  name: str
  arguments: dict[str, Any]


@dataclass
class ToolOutput:
  """Standardized container for tool execution results.

  Provides a unified interface for tool outputs that can handle successful
  results, errors, and metadata. The flexible output field accommodates
  various data types while maintaining consistent error handling.

  Attributes:
      name (str): Name of the tool that produced this output
      output (Optional[str | list[Any] | dict[str, Any]]): The tool's result
        data. Can be string, structured data, or None for no output.
      error (Optional[str]): Error message if tool execution failed. None
        indicates successful execution.
      metadata (Optional[dict[str, Any]]): Additional information about the
        execution such as timing, confidence scores, or debug data.
  """

  name: str
  output: Optional[str | list[Any] | dict[str, Any]] = None
  error: Optional[str] = None
  metadata: Optional[dict[str, Any]] = None

  def __repr__(self) -> str:
    """Generate human-readable string representation of the tool output.

    Prioritizes error messages, then formats structured data as JSON,
    with fallback to string conversion for other types.

    Returns:
        str: Formatted representation suitable for display or logging
    """
    if self.error:
      return f"Error: {self.error}"
    if self.output is None:
      return ""
    if isinstance(self.output, (dict, list)):
      return json.dumps(self.output)
    return str(self.output)


class BaseTool(ABC):
  """Abstract base class defining the interface for all agent tools.

  Tools are reusable components that extend agent capabilities by providing
  access to external systems, computations, or data sources. This base class
  establishes the contract for tool registration, execution, and metadata
  that enables dynamic tool discovery and integration.

  Subclasses must implement the tool's specific functionality through either
  synchronous or asynchronous execution methods, along with JSON schema
  definitions for parameter validation and documentation.
  """

  def __init__(self, name: str, description: str):
    """Initialize the base tool with identification and documentation.

    Args:
        name (str): Unique identifier for the tool, used in tool calls and
          registration. Should be descriptive and follow naming conventions.
        description (str): Human-readable explanation of the tool's purpose and
          functionality, used for agent understanding and documentation.
    """
    self.name = name
    self.description = description

  @abstractmethod
  def get_json_schema(self) -> dict[str, Any]:
    """Generate OpenAI-compatible function metadata for tool registration.

    Provides the schema definition needed for LLM function calling systems
    to understand the tool's interface, parameters, and usage constraints.

    Returns:
        dict[str, Any]: Tool metadata in OpenAI function calling format:
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "What the tool does...",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            }
    """
    pass

  def to_mcp_json(self) -> dict[str, Any]:
    """Generate MCP (Model Context Protocol) compliant tool registration.

    Converts the tool definition to the standardized MCP format for
    cross-platform tool discovery and integration. Uses the inputSchema
    attribute if available, otherwise falls back to empty schema.

    Returns:
        dict[str, Any]: MCP-formatted tool metadata:
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.inputSchema,
                }
            }
    """
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": getattr(self, "inputSchema", {}),
        },
    }

  def apply(self, **kwargs) -> ToolOutput:
    """Execute the tool synchronously with the provided arguments.

    Default implementation raises NotImplementedError. Subclasses should
    override either this method or apply_async() to provide the tool's
    core functionality.

    Args:
        **kwargs: Tool-specific parameters as defined in the JSON schema

    Returns:
        ToolOutput: Standardized result containing output, error, or metadata

    Raises:
        NotImplementedError: If neither apply() nor apply_async() is implemented
    """
    raise NotImplementedError(
        "Tool must implement either `apply()` or `apply_async()`"
    )

  async def apply_async(self, **kwargs) -> ToolOutput:
    """Execute the tool asynchronously with the provided arguments.

    Default implementation delegates to the synchronous apply() method.
    Tools that perform I/O operations or long-running computations should
    override this method to provide true asynchronous execution.

    Args:
        **kwargs: Tool-specific parameters as defined in the JSON schema

    Returns:
        ToolOutput: Standardized result containing output, error, or metadata
    """
    raise NotImplementedError(
        "Tool must implement either `apply()` or `apply_async()`"
    )
