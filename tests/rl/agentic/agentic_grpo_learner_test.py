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

"""Tests for AgenticGRPOLearner and AgenticGRPOConfig."""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic import AgenticGRPOConfig
from tunix.rl.agentic.tools import ExpressionCalculatorTool


class AgenticGRPOConfigTest(parameterized.TestCase):
  """Test cases for AgenticGRPOConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    config = AgenticGRPOConfig()
    self.assertEqual(config.num_generations, 2)
    self.assertEqual(config.num_iterations, 1)
    self.assertEqual(config.beta, 0.04)
    self.assertEqual(config.epsilon, 0.2)
    self.assertEqual(config.max_tool_steps, 5)
    self.assertEqual(config.tool_parser_name, "qwen")
    self.assertEqual(config.algo_variant, "agentic_grpo")

  def test_custom_config(self):
    """Test custom configuration values."""
    config = AgenticGRPOConfig(
        num_generations=4,
        num_iterations=2,
        beta=0.08,
        epsilon=0.3,
        max_tool_steps=10,
        tool_parser_name="gemini",
        system_prompt="You are a helpful assistant.",
    )
    self.assertEqual(config.num_generations, 4)
    self.assertEqual(config.num_iterations, 2)
    self.assertEqual(config.beta, 0.08)
    self.assertEqual(config.epsilon, 0.3)
    self.assertEqual(config.max_tool_steps, 10)
    self.assertEqual(config.tool_parser_name, "gemini")
    self.assertEqual(config.system_prompt, "You are a helpful assistant.")

  def test_num_generations_validation(self):
    """Test that num_generations must be greater than 1."""
    with self.assertRaises(ValueError) as context:
      AgenticGRPOConfig(num_generations=1)
    self.assertIn("num_generations must be greater than 1", str(context.exception))

  def test_invalid_loss_algo(self):
    """Test that invalid loss_algo raises an error."""
    with self.assertRaises(ValueError) as context:
      AgenticGRPOConfig(loss_algo="invalid")
    self.assertIn("loss_algo should be either grpo or gspo-token", str(context.exception))

  @parameterized.named_parameters(
      ("grpo", "grpo"),
      ("gspo_token", "gspo-token"),
  )
  def test_valid_loss_algo(self, loss_algo):
    """Test that valid loss_algo values are accepted."""
    config = AgenticGRPOConfig(loss_algo=loss_algo)
    self.assertEqual(config.loss_algo, loss_algo)

  def test_epsilon_high_defaults_to_epsilon(self):
    """Test that epsilon_high defaults to epsilon if not specified."""
    config = AgenticGRPOConfig(epsilon=0.3)
    self.assertEqual(config.epsilon_high, 0.3)

  def test_epsilon_high_custom_value(self):
    """Test that epsilon_high can be set independently."""
    config = AgenticGRPOConfig(epsilon=0.2, epsilon_high=0.28)
    self.assertEqual(config.epsilon, 0.2)
    self.assertEqual(config.epsilon_high, 0.28)


class AgenticGRPOToolMapTest(absltest.TestCase):
  """Test cases for tool map configuration."""

  def test_tool_map_with_expression_calculator(self):
    """Test that ExpressionCalculatorTool can be used in tool_map."""
    tool_map = {"calculator": ExpressionCalculatorTool}
    self.assertIn("calculator", tool_map)
    self.assertEqual(tool_map["calculator"], ExpressionCalculatorTool)

  def test_multiple_tools_in_map(self):
    """Test tool_map with multiple tools."""
    from tunix.rl.agentic.tools import CalculatorTool

    tool_map = {
        "calculator": CalculatorTool,
        "expression_calculator": ExpressionCalculatorTool,
    }
    self.assertLen(tool_map, 2)


if __name__ == "__main__":
  absltest.main()

