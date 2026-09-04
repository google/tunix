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
      dict(
          testcase_name="gspo_gae_ppo", algo="gspo-token", adv="gae", loss="ppo"
      ),
      dict(
          testcase_name="grpo_grpo_grpo", algo="grpo", adv="grpo", loss="grpo"
      ),
      dict(testcase_name="ppo_gae_ppo", algo="ppo", adv="gae", loss="ppo"),
      dict(
          testcase_name="gspo_grpo_ppo",
          algo="gspo-token",
          adv="grpo",
          loss="ppo",
      ),
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
      algorithm_config.AlgorithmConfig("grpo-token", "grpo", "grpo")

    # Check that standard keyword initialization works
    try:
      algorithm_config.AlgorithmConfig(
          algo_variant="gspo-token",
          advantage_estimator="gae",
          policy_loss_fn="ppo",
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

  def test_config_logging(self):
    """Tests that configuration is logged correctly upon initialization."""
    # assertLogs catches logs at the specified level or higher
    with self.assertLogs(level="INFO") as log:
      algorithm_config.AlgorithmConfig(
          algo_variant="gspo-token",
          advantage_estimator="gae",
          policy_loss_fn="ppo",
      )

    # log.output is a list of strings like ['INFO:root:message...']
    full_log_output = "\n".join(log.output)

    self.assertIn("Initializing AlgorithmConfig", full_log_output)
    self.assertIn("algo_variant: gspo", full_log_output)
    self.assertIn("advantage_estimator: gae", full_log_output)
    self.assertIn("policy_loss_fn: ppo", full_log_output)

  def test_kl_clamp_value_default_is_none(self):
    """Default `kl_clamp_value` is None (no clamp, prior behavior)."""
    config = algorithm_config.AlgorithmConfig()
    self.assertIsNone(config.kl_clamp_value)

  @parameterized.named_parameters(
      ("ten_thousand", 10000.0),
      ("one", 1.0),
      ("explicit_none", None),
  )
  def test_kl_clamp_value_round_trips(self, value):
    """`kl_clamp_value` is stored as-set on the config."""
    config = algorithm_config.AlgorithmConfig(kl_clamp_value=value)
    self.assertEqual(config.kl_clamp_value, value)


class SamplerIsOptionsTest(parameterized.TestCase):
  """Validation of the sampler-vs-trainer options."""

  def test_all_off_by_default(self):
    config = algorithm_config.AlgorithmConfig()
    self.assertFalse(config.overlong_loss_masking)
    self.assertIsNone(config.seq_logprob_error_threshold)
    self.assertIsNone(config.truncated_importance_sampling_type)
    self.assertIsNone(config.truncated_importance_sampling_ratio_min)
    self.assertIsNone(config.truncated_importance_sampling_ratio)
    self.assertEqual(config.sampler_is_report_bands, ())
    self.assertIsNone(config.sampler_is_length_buckets)

  def test_grpo_loo_is_a_valid_estimator(self):
    config = algorithm_config.AlgorithmConfig(advantage_estimator="grpo-loo")
    self.assertEqual(config.advantage_estimator, "grpo-loo")

  @parameterized.parameters("rloo", "drgrpo")
  def test_registered_estimators_are_selectable(self, name):
    # These are implemented in the registry; the validator used to reject
    # them, making them unreachable.
    self.assertEqual(
        algorithm_config.AlgorithmConfig(
            advantage_estimator=name
        ).advantage_estimator,
        name,
    )

  def test_keep_band_must_be_complete(self):
    for kwargs in (
        {"truncated_importance_sampling_ratio_min": 0.999},
        {"truncated_importance_sampling_ratio": 1.002},
    ):
      with self.assertRaisesRegex(ValueError, "must be set together"):
        algorithm_config.AlgorithmConfig(**kwargs)

  def test_keep_band_must_be_ordered(self):
    with self.assertRaisesRegex(ValueError, "must not exceed"):
      algorithm_config.AlgorithmConfig(
          truncated_importance_sampling_ratio_min=1.002,
          truncated_importance_sampling_ratio=0.999,
      )

  def test_correction_requires_a_band(self):
    with self.assertRaisesRegex(ValueError, "requires a keep-band"):
      algorithm_config.AlgorithmConfig(
          truncated_importance_sampling_type="seq-mask-tis"
      )

  def test_unknown_correction_type_rejected(self):
    with self.assertRaisesRegex(ValueError, "only supports 'seq-mask-tis'"):
      algorithm_config.AlgorithmConfig(
          truncated_importance_sampling_type="tis",
          truncated_importance_sampling_ratio_min=0.5,
          truncated_importance_sampling_ratio=5.0,
      )

  def test_correction_accepts_a_complete_band(self):
    config = algorithm_config.AlgorithmConfig(
        truncated_importance_sampling_type="seq-mask-tis",
        truncated_importance_sampling_ratio_min=0.999,
        truncated_importance_sampling_ratio=1.002,
    )
    self.assertEqual(
        config.truncated_importance_sampling_type, "seq-mask-tis"
    )

  @parameterized.parameters(
      ((0, 512),),  # non-positive edge
      ((512, 512),),  # not strictly increasing
      ((1024, 512),),  # decreasing
      ((),),  # empty means "disabled", which is None, not ()
  )
  def test_invalid_length_buckets_rejected(self, edges):
    with self.assertRaises(ValueError):
      algorithm_config.AlgorithmConfig(sampler_is_length_buckets=edges)

  def test_length_buckets_normalised_to_tuple(self):
    # Config loaders hand us lists; bucket edges end up in metric names, so
    # they need to be a stable, hashable tuple.
    config = algorithm_config.AlgorithmConfig(
        sampler_is_length_buckets=[512, 1024]
    )
    self.assertEqual(config.sampler_is_length_buckets, (512, 1024))

  def test_report_bands_normalised_and_validated(self):
    config = algorithm_config.AlgorithmConfig(
        sampler_is_report_bands=[[0.99, 1.01]]
    )
    self.assertEqual(config.sampler_is_report_bands, ((0.99, 1.01),))
    with self.assertRaisesRegex(ValueError, "ordered"):
      algorithm_config.AlgorithmConfig(
          sampler_is_report_bands=[(1.01, 0.99)]
      )


if __name__ == "__main__":
  absltest.main()
