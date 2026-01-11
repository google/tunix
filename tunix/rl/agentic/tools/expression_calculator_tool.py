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

"""A tool for evaluating mathematical expressions safely.

This module defines the `ExpressionCalculatorTool` class, which is a subclass of
`BaseTool`. It provides functionality for evaluating mathematical expressions
with support for basic arithmetic operations (+, -, *, /), parentheses, and
negative numbers using Python's AST for safe evaluation.
"""

import ast
import operator
from typing import Any

from tunix.rl.agentic.tools import base_tool


class ExpressionCalculatorTool(base_tool.BaseTool):
  """Calculator that evaluates mathematical expressions safely.

  Takes a single 'expression' argument (string) and returns the result.
  Uses Python's AST module for safe evaluation without using eval().

  Supported operations:
    - Addition: +
    - Subtraction: -
    - Multiplication: *
    - Division: /
    - Parentheses for grouping
    - Unary operators (negative and positive numbers)

  Examples:
      {"expression": "2 + 3 * 4"} -> 14
      {"expression": "(100 - 20) / 4"} -> 20
      {"expression": "15 * 3 + 10"} -> 55
      {"expression": "-5 + 3"} -> -2
  """

  # Mapping of AST node types to their corresponding operators for safe eval.
  _OPERATORS = {
      ast.Add: operator.add,
      ast.Sub: operator.sub,
      ast.Mult: operator.mul,
      ast.Div: operator.truediv,
      ast.USub: operator.neg,
      ast.UAdd: operator.pos,
  }

  def get_json_schema(self) -> dict[str, Any]:
    """Generate OpenAI-compatible function schema for the calculator tool.

    Defines the tool's interface for LLMs to understand how to properly
    invoke the calculator with mathematical expressions.

    Returns:
        dict: OpenAI function calling format schema with parameter
            specifications and usage constraints
    """
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A mathematical expression to evaluate "
                            "(e.g., '2 + 3 * 4', '(100 - 20) / 4')"
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    }

  @classmethod
  def _safe_eval(cls, node: ast.AST) -> float | int:
    """Recursively evaluate an AST node safely.

    Only allows numeric constants and basic arithmetic operations,
    preventing code injection attacks that would be possible with eval().

    Args:
        node: An AST node to evaluate

    Returns:
        The numeric result of evaluating the node

    Raises:
        ValueError: If the node type is unsupported or contains
            non-numeric values
    """
    if isinstance(node, ast.Constant):  # Python 3.8+
      if isinstance(node.value, (int, float)):
        return node.value
      raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
      if type(node.op) not in cls._OPERATORS:
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
      left = cls._safe_eval(node.left)
      right = cls._safe_eval(node.right)
      return cls._OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
      if type(node.op) not in cls._OPERATORS:
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
      operand = cls._safe_eval(node.operand)
      return cls._OPERATORS[type(node.op)](operand)
    elif isinstance(node, ast.Expression):
      return cls._safe_eval(node.body)
    else:
      raise ValueError(f"Unsupported expression type: {type(node).__name__}")

  def apply(self, **kwargs: Any) -> base_tool.ToolOutput:
    """Evaluate a mathematical expression safely.

    Parses the expression into an AST and evaluates it using only
    whitelisted numeric operations, preventing code injection.

    Args:
        **kwargs: Keyword arguments containing 'expression' (str).

    Returns:
        ToolOutput: Result containing either the calculated value or
            an error message if evaluation fails
    """
    expression = kwargs.get("expression")
    if expression is None:
      return base_tool.ToolOutput(
          name=self.name, error="Missing required argument: 'expression'"
      )

    # pylint: disable=broad-exception-caught
    try:
      # Parse the expression into an AST
      tree = ast.parse(str(expression), mode="eval")
      # Safely evaluate using only whitelisted operations
      result = self._safe_eval(tree)
      # Format result (remove trailing .0 for whole numbers)
      if isinstance(result, float) and result.is_integer():
        return base_tool.ToolOutput(name=self.name, output=str(int(result)))
      return base_tool.ToolOutput(name=self.name, output=str(result))
    except ZeroDivisionError:
      return base_tool.ToolOutput(
          name=self.name, error="Division by zero is not allowed"
      )
    except SyntaxError as e:
      return base_tool.ToolOutput(
          name=self.name, error=f"Invalid expression syntax: {e}"
      )
    except Exception as e:
      return base_tool.ToolOutput(
          name=self.name, error=f"{type(e).__name__}: {e}"
      )

