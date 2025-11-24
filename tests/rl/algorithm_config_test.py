# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl import algorithm_config


class AlgorithmConfigTest(parameterized.TestCase):

  def test_defaults_are_valid(self):
    """Ensures the default constructor values pass validation."""
    try:
      config = algorithm_config.AlgorithmConfig()
      self.assertEqual(config.algo_variant, "grpo")
      self.assertEqual(config.advantage_estimator, "grpo")
      self.assertEqual(config.policy_loss_fn, "grpo")
    except ValueError as e:
      self.fail(f"Default AlgorithmConfig values raised ValueError: {e}")

  @parameterized.named_parameters(
      dict(testcase_name="gspo_gae_ppo", algo="gspo", adv="gae", loss="ppo"),
      dict(
          testcase_name="grpo_grpo_grpo", algo="grpo", adv="grpo", loss="grpo"
      ),
      dict(testcase_name="ppo_gae_ppo", algo="ppo", adv="gae", loss="ppo"),
      dict(testcase_name="gspo_grpo_ppo", algo="gspo", adv="grpo", loss="ppo"),
  )
  def test_valid_combinations(self, algo: str, adv: str, loss: str):
    """Tests various valid combinations of core algorithm parameters."""
    try:
      config = algorithm_config.AlgorithmConfig(
          algo_variant=algo,
          advantage_estimator=adv,
          policy_loss_fn=loss,
      )
      self.assertEqual(config.algo_variant, algo)
      self.assertEqual(config.advantage_estimator, adv)
      self.assertEqual(config.policy_loss_fn, loss)
    except ValueError as e:
      self.fail(
          f"Valid combination {algo}, {adv}, {loss} raised ValueError: {e}"
      )

  @parameterized.named_parameters(
      dict(testcase_name="invalid_algo_dapo", value="dapo"),
      dict(testcase_name="invalid_algo_else", value="something_else"),
  )
  def test_invalid_algo_variant(self, value: str):
    """Tests that invalid algo_variant values raise ValueError."""
    with self.assertRaisesRegex(
        ValueError, f"algo_variant must be one of .* Received: {value!r}"
    ):
      algorithm_config.AlgorithmConfig(algo_variant=value)

  @parameterized.named_parameters(
      dict(testcase_name="invalid_adv_other", value="other"),
      dict(testcase_name="invalid_adv_ppo", value="ppo"),
  )
  def test_invalid_advantage_estimator(self, value: str):
    """Tests that invalid advantage_estimator values raise ValueError."""
    with self.assertRaisesRegex(
        ValueError, f"advantage_estimator must be one of .* Received: .*"
    ):
      algorithm_config.AlgorithmConfig(advantage_estimator=value)

  @parameterized.named_parameters(
      dict(testcase_name="invalid_loss_gspo", value="gspo"),
      dict(testcase_name="invalid_loss_mse", value="mse"),
  )
  def test_invalid_policy_loss_fn(self, value: str):
    """Tests that invalid policy_loss_fn values raise ValueError."""
    with self.assertRaisesRegex(
        ValueError,
        "policy_loss_fn must be one of .* Received: .*",
    ):
      algorithm_config.AlgorithmConfig(policy_loss_fn=value)

  def test_kw_only_enforcement(self):
    """Ensures that positional arguments are not allowed."""
    with self.assertRaises(TypeError):
      # Attempt to initialize with positional arguments
      algorithm_config.AlgorithmConfig("grpo", "grpo", "grpo")

    # Check that standard keyword initialization works
    try:
      algorithm_config.AlgorithmConfig(
          algo_variant="gspo", advantage_estimator="gae", policy_loss_fn="ppo"
      )
    except TypeError:
      self.fail("Keyword arguments failed for kw_only dataclass")

  def test_slots_enabled(self):
    """Checks that slots are active, preventing arbitrary attribute assignment."""
    config = algorithm_config.AlgorithmConfig()
    with self.assertRaises(AttributeError):
      config.new_attribute = "test"

  def test_field_assignment(self):
    """Tests that fields can be set after initialization (since frozen=False)."""
    config = algorithm_config.AlgorithmConfig()
    config.algo_variant = "gspo"
    self.assertEqual(config.algo_variant, "gspo")
    # Note: __post_init__ is NOT called again on field assignment,
    # so we can assign invalid values after creation.
    config.algo_variant = "invalid_after_init"
    self.assertEqual(config.algo_variant, "invalid_after_init")


if __name__ == "__main__":
  absltest.main()
