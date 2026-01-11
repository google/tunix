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

"""Tests for ExpressionCalculatorTool."""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic.tools import expression_calculator_tool
from tunix.rl.agentic.tools import tool_manager


class ExpressionCalculatorToolTest(parameterized.TestCase):
  """Test cases for ExpressionCalculatorTool."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tool = expression_calculator_tool.ExpressionCalculatorTool(
        name="calculator",
        description="Evaluates a mathematical expression and returns the result.",
    )

  def test_json_schema(self):
    """Test that JSON schema is correctly generated."""
    schema = self.tool.get_json_schema()
    self.assertEqual(schema["type"], "function")
    self.assertEqual(schema["function"]["name"], "calculator")
    self.assertIn("parameters", schema["function"])
    self.assertIn("expression", schema["function"]["parameters"]["properties"])

  @parameterized.named_parameters(
      ("simple_addition", "2 + 3", "5"),
      ("simple_subtraction", "10 - 4", "6"),
      ("simple_multiplication", "6 * 7", "42"),
      ("simple_division", "20 / 4", "5"),
      ("order_of_operations", "2 + 3 * 4", "14"),
      ("parentheses", "(2 + 3) * 4", "20"),
      ("complex_expression", "(100 - 20) / 4", "20"),
      ("negative_number", "-5 + 8", "3"),
      ("multiple_operations", "15 * 3 + 10", "55"),
      ("nested_parentheses", "((2 + 3) * 4) / 2", "10"),
      ("decimal_result", "7 / 2", "3.5"),
      ("float_input", "3.14 * 2", "6.28"),
  )
  def test_basic_operations(self, expression, expected):
    """Test various basic mathematical operations."""
    result = self.tool.apply(expression=expression)
    self.assertIsNone(result.error)
    self.assertEqual(result.output, expected)

  def test_missing_expression(self):
    """Test error handling when expression is missing."""
    result = self.tool.apply()
    self.assertIsNotNone(result.error)
    self.assertIn("Missing required argument", result.error)

  def test_division_by_zero(self):
    """Test error handling for division by zero."""
    result = self.tool.apply(expression="10 / 0")
    self.assertIsNotNone(result.error)
    self.assertIn("Division by zero", result.error)

  def test_invalid_syntax(self):
    """Test error handling for invalid expression syntax."""
    result = self.tool.apply(expression="2 +")
    self.assertIsNotNone(result.error)
    self.assertIn("Invalid expression syntax", result.error)

  def test_unsupported_operations(self):
    """Test error handling for unsupported operations like exponentiation."""
    result = self.tool.apply(expression="2 ** 3")
    self.assertIsNotNone(result.error)

  def test_non_numeric_expression(self):
    """Test error handling for expressions with function calls."""
    result = self.tool.apply(expression="print('hello')")
    self.assertIsNotNone(result.error)

  def test_tool_output_repr(self):
    """Test that ToolOutput string representation works correctly."""
    result = self.tool.apply(expression="2 + 2")
    self.assertEqual(str(result), "4")

  def test_error_output_repr(self):
    """Test that error ToolOutput string representation works correctly."""
    result = self.tool.apply(expression="10 / 0")
    self.assertTrue(str(result).startswith("Error:"))


class ExpressionCalculatorToolManagerTest(absltest.TestCase):
  """Test cases for ExpressionCalculatorTool with ToolManager."""

  def test_tool_manager_integration(self):
    """Test that ExpressionCalculatorTool works with ToolManager."""
    tool_map = {
        "calculator": expression_calculator_tool.ExpressionCalculatorTool
    }
    manager = tool_manager.ToolManager(tool_map)

    result = manager.run("calculator", expression="3 + 4 * 5")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "23")

  def test_tool_manager_schema(self):
    """Test that JSON schema is properly generated through ToolManager."""
    tool_map = {
        "calculator": expression_calculator_tool.ExpressionCalculatorTool
    }
    manager = tool_manager.ToolManager(tool_map)

    schemas = manager.get_json_schema()
    self.assertLen(schemas, 1)
    self.assertEqual(schemas[0]["function"]["name"], "calculator")


if __name__ == "__main__":
  absltest.main()

