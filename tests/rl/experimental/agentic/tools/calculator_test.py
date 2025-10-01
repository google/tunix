from absl.testing import absltest
from tunix.rl.experimental.agentic.tools import calculator_tool


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
