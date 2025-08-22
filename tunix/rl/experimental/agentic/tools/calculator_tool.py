"""A tool for performing basic arithmetic calculations.

This module defines the `CalculatorTool` class, which is a subclass of
`BaseTool`. It provides functionality for addition, subtraction, multiplication,
and division, including error handling for cases like division by zero.
"""

from typing import Any

from tunix.rl.experimental.agentic.tools import base_tool

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

  @property
  def json(self) -> dict[str, Any]:
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

  def apply(self, a: float, b: float, op: str) -> ToolOutput:
    """Execute the arithmetic operation with the provided operands and operator.

    Performs the requested calculation while handling edge cases and potential
    errors. Validates the operator and provides specific error messages for
    common failure scenarios like division by zero.

    Args:
        a (float): The first operand for the arithmetic operation
        b (float): The second operand for the arithmetic operation
        op (str): The arithmetic operator ("+", "-", "*", "/")

    Returns:
        ToolOutput: Result containing either the calculated value or
            detailed error information if the operation fails
    """
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

      return ToolOutput(name=self.name, output=result)

    except Exception as e:
      return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")
