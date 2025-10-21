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

from absl.testing import parameterized

from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.rewards import reward_types


class RewardTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Back up the registry to ensure test isolation
    self._original_registry = reward._REGISTRY.copy()

  def tearDown(self):
    super().tearDown()
    # Restore the original registry
    reward._REGISTRY = self._original_registry

  def test_registry(self):
    """Tests the reward function registry mechanism."""
    # A simple reward function for testing registration
    def test_fn(task, action):
      return reward_types.RewardOutput(0.5, {})

    reward.register("test_fn")(test_fn)
    self.assertIs(reward.get_reward_fn("test_fn"), test_fn)

    # Test reregistering raises ValueError
    with self.assertRaises(ValueError):
      reward.register("test_fn")(test_fn)

    # Test unregister
    self.assertTrue(reward.unregister("test_fn"))
    self.assertFalse(reward.unregister("test_fn"))  # Already removed
    with self.assertRaises(KeyError):
      reward.get_reward_fn("test_fn")

  @parameterized.named_parameters(
      ("match", "hello world", "hello world", 1.0),
      ("mismatch", "hello world", "hello", 0.0),
      ("whitespace", "  hello world  ", "hello world", 1.0),
  )
  def test_exact_match(self, ground_truth, action, expected_score):
    """Tests the exact_match reward function."""
    task = {"ground_truth": ground_truth}
    result = reward.exact_match(task, action)
    self.assertEqual(result.reward, expected_score)
    self.assertEqual(result.metadata["exact_match"], expected_score)

  @parameterized.named_parameters(
      ("integer_string", "2", 1.0),
      ("float_string", "2.0", 1.0),
      ("with_whitespace", "  2 ", 1.0),
      ("wrong_number", "3", 0.0),
      ("not_a_number", "two", 0.0),
  )
  def test_is_two_reward(self, action, expected_score):
    """Tests the is_two_reward function."""
    result = reward.is_two_reward({}, action)
    self.assertEqual(result.reward, expected_score)
    self.assertEqual(result.metadata["is_two"], expected_score)

  def test_dummy_reward(self):
    """Tests the dummy_reward function."""
    result = reward.dummy_reward({}, "any action")
    self.assertEqual(result.reward, 0.0)
    self.assertEqual(result.metadata, {})

  @parameterized.named_parameters(
      ("correct", "2 + 2 = ?", "The answer is 4", 1.0),
      ("incorrect", "5 * 3 = ?", "14", 0.0),
      ("parsing_error", "10 / 2 = ?", "five", 0.0),
      # eval() will raise ZeroDivisionError, which is caught by the try-except
      ("eval_error", "10 / 0 = ?", "1", 0.0),
  )
  def test_calculate_reward(self, question, action, expected_score):
    """Tests the calculate_reward function."""
    task = {"question": question}
    result = reward.calculate_reward(task, action)
    self.assertEqual(result.reward, expected_score)
    self.assertEqual(result.metadata["calculate_correct"], expected_score)

  def test_combine_rewards(self):
    """Tests the reward combination logic."""
    weights = {"exact_match": 0.7, "dummy": 0.3}
    combined_fn = reward.combine_rewards(weights)

    # Case 1: Matches exact_match
    task = {"ground_truth": "hello"}
    action = "hello"
    result = combined_fn(task, action)
    self.assertAlmostEqual(result.reward, 0.7)
    self.assertEqual(result.metadata, {"exact_match": 1.0})

    # Case 2: Does not match exact_match
    task = {"ground_truth": "world"}
    action = "hello"
    result = combined_fn(task, action)
    self.assertAlmostEqual(result.reward, 0.0)
    self.assertEqual(result.metadata, {"exact_match": 0.0})
