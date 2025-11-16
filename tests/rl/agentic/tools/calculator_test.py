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

from absl.testing import absltest
from tunix.rl.agentic.tools import calculator_tool


class CalculatorToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tool = calculator_tool.CalculatorTool(
        name="calculator", description="A calculator tool."
    )

  def test_get_json_schema(self):
    schema = self.tool.get_json_schema()
    self.assertEqual(schema["type"], "function")
    self.assertEqual(schema["function"]["name"], "calculator")
    self.assertIn("description", schema["function"])
    self.assertIn("a", schema["function"]["parameters"]["properties"])
    self.assertIn("b", schema["function"]["parameters"]["properties"])
    self.assertIn("op", schema["function"]["parameters"]["properties"])
    self.assertEqual(
        schema["function"]["parameters"]["required"], ["a", "b", "op"]
    )

  def test_add(self):
    result = self.tool.apply(a=5, b=3, op="+")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "8")

  def test_subtract(self):
    result = self.tool.apply(a=5, b=3, op="-")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "2")

  def test_multiply(self):
    result = self.tool.apply(a=5, b=3, op="*")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "15")

  def test_divide(self):
    result = self.tool.apply(a=6, b=3, op="/")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "2.0")

  def test_divide_by_zero(self):
    result = self.tool.apply(a=5, b=0, op="/")
    self.assertIsNotNone(result.error)
    self.assertIn("Division by zero", result.error)
    self.assertIsNone(result.output)

  def test_unsupported_operator(self):
    result = self.tool.apply(a=5, b=3, op="^")
    self.assertIsNotNone(result.error)
    self.assertIn("Unsupported operator", result.error)
    self.assertIsNone(result.output)


if __name__ == "__main__":
  absltest.main()
