# Copyright 2026 Google LLC
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

"""Tests that the GRPO adapter refuses configurations it does not implement.

The adapter implements the on-policy subset of the agentic GRPO learner's
postprocess. Every configuration covered here would otherwise run to
completion while computing different math than the learner it mirrors, which
is indistinguishable from a healthy run. Each one must raise instead.

The positive control -- a supported configuration training end to end -- is
`orchestrated_agentic_learner_test`.
"""

import types
from typing import Any

from absl.testing import absltest
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.rl.agentic import agentic_grpo_learner


def _config(**overrides: Any) -> agentic_grpo_learner.GRPOConfig:
  """A supported GRPO config, with the field under test overridden."""
  kwargs: dict[str, Any] = {
      "num_generations": 2,
      "num_iterations": 1,
      "beta": 0.0,
      "max_response_length": 10,
  }
  kwargs.update(overrides)
  return agentic_grpo_learner.GRPOConfig(**kwargs)


def _cluster(max_seq_token_per_tpu: int | None = None) -> Any:
  """A stand-in exposing only what the configuration check reads."""
  return types.SimpleNamespace(
      cluster_config=types.SimpleNamespace(
          training_config=types.SimpleNamespace(
              max_seq_token_per_tpu=max_seq_token_per_tpu
          )
      )
  )


class SupportedConfigTest(absltest.TestCase):
  """The guards must not fire on what the adapter does implement."""

  def test_default_on_policy_config_is_accepted(self):
    adapter = algorithm_adapter.GRPOAdapter(_config())
    adapter.check_supported_config(_cluster())

  def test_kl_penalty_is_accepted(self):
    adapter = algorithm_adapter.GRPOAdapter(_config(beta=0.04))
    adapter.check_supported_config(_cluster())

  def test_recomputed_old_logps_is_accepted(self):
    adapter = algorithm_adapter.GRPOAdapter(_config(use_rollout_logps=False))
    adapter.check_supported_config(_cluster())


class StrictRolloutLogpsTest(absltest.TestCase):
  """Missing sampler log-probabilities are rejected, not substituted."""

  def test_rejects_by_default(self):
    self.assertTrue(
        algorithm_adapter.GRPOAdapter(_config()).strict_rollout_logps
    )

  def test_opt_out_is_honored(self):
    adapter = algorithm_adapter.GRPOAdapter(
        _config(strict_rollout_logps=False)
    )
    self.assertFalse(adapter.strict_rollout_logps)

  def test_explicit_opt_in_is_honored(self):
    adapter = algorithm_adapter.GRPOAdapter(_config(strict_rollout_logps=True))
    self.assertTrue(adapter.strict_rollout_logps)


class UnsupportedConfigTest(absltest.TestCase):

  def test_sampler_importance_sampling_is_accepted(self):
    """Implemented via the same agreement helper the agentic learner uses."""
    adapter = algorithm_adapter.GRPOAdapter(_config(sampler_is="token"))
    adapter.check_supported_config(_cluster())

  def test_multiple_iterations_per_batch_is_accepted(self):
    """Legal once old log-probabilities are guaranteed to be real."""
    adapter = algorithm_adapter.GRPOAdapter(_config(num_iterations=2))
    adapter.check_supported_config(_cluster())

  def test_sequence_packing_is_rejected(self):
    adapter = algorithm_adapter.GRPOAdapter(_config())
    with self.assertRaisesRegex(
        algorithm_adapter.UnsupportedConfigError, "max_seq_token_per_tpu"
    ):
      adapter.check_supported_config(_cluster(max_seq_token_per_tpu=512))

  def test_sequence_packing_is_rejected_before_configuring_the_trainer(self):
    adapter = algorithm_adapter.GRPOAdapter(_config())
    with self.assertRaises(algorithm_adapter.UnsupportedConfigError):
      adapter.configure_trainer(_cluster(max_seq_token_per_tpu=512))

  def test_sequence_packing_is_rejected_before_postprocessing(self):
    adapter = algorithm_adapter.GRPOAdapter(_config())
    with self.assertRaises(algorithm_adapter.UnsupportedConfigError):
      adapter.postprocess_group(
          _cluster(max_seq_token_per_tpu=512),
          trajectories=[],
          compute_rewards=lambda **kwargs: [],
          mode=None,
      )

  def test_packing_is_still_rejected_when_only_the_cluster_knows(self):
    """The knob lives on the cluster, so construction alone cannot catch it."""
    adapter = algorithm_adapter.GRPOAdapter(_config())
    adapter.check_supported_config(_cluster())  # No packing: fine.

    with self.assertRaises(algorithm_adapter.UnsupportedConfigError):
      adapter.check_supported_config(_cluster(max_seq_token_per_tpu=1024))


if __name__ == "__main__":
  absltest.main()
