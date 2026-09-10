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
"""Tests for Tunix Checkpointing Options custom implementation for Orbax v1."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import v1 as ocp
from tunix.sft import checkpoint_options


class CheckpointOptionsTest(parameterized.TestCase):
  def test_resolve_checkpointing_defaults_with_none(self):
    opts = checkpoint_options.resolve_checkpointing_defaults(None)
    self.assertEqual(opts, checkpoint_options.DEFAULT_CHECKPOINTING_OPTIONS)

  def test_resolve_checkpointing_defaults_with_empty_options(self):
    opts = checkpoint_options.resolve_checkpointing_defaults(
        checkpoint_options.TunixCheckpointingOptions()
    )
    self.assertEqual(opts, checkpoint_options.DEFAULT_CHECKPOINTING_OPTIONS)

  def test_resolve_checkpointing_defaults_with_deprecated_options(self):
    legacy_opts = ocp_v0.CheckpointManagerOptions(
        save_interval_steps=100, max_to_keep=5
    )

    with self.assertLogs(level='WARNING') as log:
      opts = checkpoint_options.resolve_checkpointing_defaults(
          legacy_opts
      )

    # Verify deprecation warnings were logged
    v0_warnings = [msg for msg in log.output if 'Using v0' in msg]
    self.assertNotEmpty(v0_warnings)

    # Verify policies were resolved correctly
    self.assertEqual(
        opts.save_decision_policy,
        ocp.training.save_decision_policies.FixedIntervalPolicy(100),
    )

    self.assertEqual(
        opts.preservation_policy,
        ocp.training.preservation_policies.LatestN(5),
    )

  def test_resolve_checkpointing_defaults_with_legacy_options_dataclass(self):
    legacy_opts = ocp_v0.CheckpointManagerOptions(
        save_decision_policy=ocp_v0.checkpoint_managers.ContinuousCheckpointingPolicy(
            minimum_interval_secs=10,
        ),
    )
    opts = checkpoint_options.resolve_checkpointing_defaults(
        legacy_opts
    )
    self.assertIsInstance(
        opts.save_decision_policy,
        ocp.training.save_decision_policies.ContinuousCheckpointingPolicy,
    )
    # pytype: disable=attribute-error
    self.assertEqual(opts.save_decision_policy.minimum_interval_secs, 10)
    # pytype: enable=attribute-error

  def test_resolve_checkpointing_defaults_with_async_timeout(self):
    async_opts = ocp.options.AsyncOptions(timeout_secs=5000)
    options = mock.create_autospec(
        checkpoint_options.TunixCheckpointingOptions, instance=True
    )
    options.async_options = async_opts
    options.save_decision_policy = None
    options.preservation_policy = None
    options.step_name_format = None
    options.enable_async_checkpointing = None

    opts = checkpoint_options.resolve_checkpointing_defaults(options)
    self.assertIsNotNone(opts.async_options)
    assert opts.async_options is not None
    self.assertEqual(opts.async_options.timeout_secs, 5000)

  def test_resolve_checkpointing_defaults_with_modern_options(self):
    modern_opts = checkpoint_options.TunixCheckpointingOptions(
        save_decision_policy=ocp.training.save_decision_policies.FixedIntervalPolicy(
            50
        ),
        preservation_policy=ocp.training.preservation_policies.LatestN(10),
        enable_async_checkpointing=False,
    )
    opts = checkpoint_options.resolve_checkpointing_defaults(
        modern_opts
    )
    self.assertEqual(
        opts.save_decision_policy, modern_opts.save_decision_policy
    )
    self.assertEqual(
        opts.preservation_policy, modern_opts.preservation_policy
    )
    self.assertFalse(opts.enable_async_checkpointing)

  def test_create_checkpointing_options(self):
    opts = checkpoint_options.create_checkpointing_options(
        save_decision_policy=ocp.training.save_decision_policies.FixedIntervalPolicy(
            50
        ),
        preservation_policy=ocp.training.preservation_policies.LatestN(10),
        enable_async_checkpointing=False,
    )
    self.assertIsInstance(opts, checkpoint_options.TunixCheckpointingOptions)
    self.assertEqual(
        opts.save_decision_policy,
        ocp.training.save_decision_policies.FixedIntervalPolicy(50),
    )
    self.assertEqual(
        opts.preservation_policy,
        ocp.training.preservation_policies.LatestN(10),
    )
    self.assertFalse(opts.enable_async_checkpointing)

  def test_checkpointing_options_from_dict(self):
    opts_dict = {
        'max_to_keep': 5,
        'save_interval_steps': 10,
        'enable_async_checkpointing': False,
    }
    opts = checkpoint_options.checkpointing_options_from_dict(opts_dict)
    self.assertIsInstance(opts, checkpoint_options.TunixCheckpointingOptions)

    self.assertIsInstance(
        opts.save_decision_policy,
        ocp.training.save_decision_policies.FixedIntervalPolicy,
    )
    self.assertIsInstance(
        opts.preservation_policy,
        ocp.training.preservation_policies.LatestN,
    )
    self.assertFalse(opts.enable_async_checkpointing)

  def test_checkpointing_options_from_dict_with_keep_every_nth_step(self):
    opts_dict = {
        'max_to_keep': 5,
        'keep_every_nth_step': 100,
    }
    opts = checkpoint_options.checkpointing_options_from_dict(opts_dict)
    self.assertIsInstance(
        opts.preservation_policy,
        ocp.training.preservation_policies.AnyPreservationPolicy,
    )
    # pytype: disable=attribute-error
    self.assertLen(opts.preservation_policy.policies, 2)
    self.assertIsInstance(
        opts.preservation_policy.policies[0],
        ocp.training.preservation_policies.LatestN,
    )
    self.assertIsInstance(
        opts.preservation_policy.policies[1],
        ocp.training.preservation_policies.EveryNSteps,
    )
    # pytype: enable=attribute-error

  def test_to_v1_options_with_timeout_secs_from_dict(self):
    opts_dict = {'timeout_secs': 900}
    opts = checkpoint_options.checkpointing_options_from_dict(opts_dict)
    assert opts.async_options is not None
    self.assertEqual(opts.async_options.timeout_secs, 900)

  def test_checkpointing_options_from_dict_with_async_timeout(self):
    opts_dict = {'enable_async_checkpointing': True, 'timeout_secs': 900}
    opts = checkpoint_options.checkpointing_options_from_dict(opts_dict)
    self.assertTrue(opts.enable_async_checkpointing)
    assert opts.async_options is not None
    self.assertEqual(opts.async_options.timeout_secs, 900)

  def test_checkpointing_options_from_dict_invalid_keys(self):
    opts_dict = {'invalid_key': 5}
    with self.assertRaisesRegex(
        ValueError, "The following options {'invalid_key'} are not supported"
    ):
      checkpoint_options.checkpointing_options_from_dict(opts_dict)


if __name__ == '__main__':
  absltest.main()
