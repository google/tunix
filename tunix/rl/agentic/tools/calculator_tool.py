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

"""A tool for performing basic arithmetic calculations.

This module defines the `CalculatorTool` class, which is a subclass of
`BaseTool`. It provides functionality for addition, subtraction, multiplication,
and division, including error handling for cases like division by zero.
"""

from typing import Any

from tunix.rl.agentic.tools import base_tool

ToolOutput = base_tool.ToolOutput
BaseTool = base_tool.BaseTool


class CalculatorTool(BaseTool):
  """A basic calculator tool that performs arithmetic operations.

  Supports the four fundamental arithmetic operations: addition, subtraction,
  multiplication, and division. Provides proper error handling for edge cases
  such as division by zero and invalid operators. Returns numerical results
  in a standardized ToolOutput format for consistent integration with agent
  systems.
  """

  def get_json_schema(self) -> dict[str, Any]:
    """Generate OpenAI-compatible function schema for the calculator tool.

    Defines the tool's interface with strongly typed parameters and
    enumerated operator values to ensure valid inputs. The schema
    enables LLMs to understand how to properly invoke the calculator
    with appropriate arguments and constraints.

    Returns:
        dict: OpenAI function calling format schema with parameter
            specifications, types, and usage constraints
    """
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first operand"},
                    "b": {
                        "type": "number",
                        "description": "The second operand",
                    },
                    "op": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/"],
                        "description": "Operator, one of: + - * /",
                    },
                },
                "required": ["a", "b", "op"],
            },
        },
    }

  def apply(self, **kwargs: Any) -> ToolOutput:
    """Execute the arithmetic operation with the provided operands and operator.

    Performs the requested calculation while handling edge cases and potential
    errors. Validates the operator and provides specific error messages for
    common failure scenarios like division by zero.

    Args:
        **kwargs: Keyword arguments containing 'a' (float), 'b' (float),
            and 'op' (str).

    Returns:
        ToolOutput: Result containing either the calculated value or
            detailed error information if the operation fails
    """
    a = kwargs.get("a")
    b = kwargs.get("b")
    op = kwargs.get("op")

    if a is None or b is None or op is None:
      return ToolOutput(
          name=self.name,
          error="Missing required arguments: 'a', 'b', and 'op'",
      )
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
      return ToolOutput(
          name=self.name,
          error="Operands 'a' and 'b' must be numbers (int or float)",
      )
    if not isinstance(op, str):
      return ToolOutput(name=self.name, error="Operator 'op' must be a string")

    # pylint: disable=broad-exception-caught
    try:
      if op == "+":
        result = a + b
      elif op == "-":
        result = a - b
      elif op == "*":
        result = a * b
      elif op == "/":
        if b == 0:
          return ToolOutput(
              name=self.name, error="Division by zero is not allowed"
          )
        result = a / b
      else:
        return ToolOutput(name=self.name, error=f"Unsupported operator: {op}")

      return ToolOutput(name=self.name, output=str(result))

    except Exception as e:
      return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")
