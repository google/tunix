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
"""Tests for Tunix Checkpointing Options custom implementation for Orbax v1."""

from absl.testing import absltest
from absl.testing import parameterized
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import v1 as ocp
from tunix.sft import checkpoint_options


class CheckpointOptionsTest(parameterized.TestCase):
  def test_resolve_checkpointing_defaults_with_none(self):
    opts = checkpoint_options.resolve_checkpointing_defaults(None)
    self.assertEqual(opts, checkpoint_options.DEFAULT_CHECKPOINTING_OPTIONS)

  def test_resolve_checkpointing_defaults_with_deprecated_options(self):
    # Create a mock object that mimics ocp.CheckpointManagerOptions (v0)
    legacy_opts = ocp_v0.CheckpointManagerOptions(
        save_interval_steps=100, max_to_keep=5
    )

    with self.assertLogs(level='WARNING') as log:
      opts = checkpoint_options.resolve_checkpointing_defaults(
          legacy_opts
      )

    # Verify deprecation warnings were logged
    self.assertTrue(any('Using v0' in msg for msg in log.output))

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
    class MockOptions:
      def __init__(self, async_options):
        self.async_options = async_options
        self.save_decision_policy = None
        self.preservation_policy = None
        self.step_name_format = None
        self.enable_async_checkpointing = None

    async_opts = ocp.options.AsyncOptions(timeout_secs=5000)
    options = MockOptions(async_options=async_opts)

    opts = checkpoint_options.resolve_checkpointing_defaults(options)
    self.assertIsNotNone(opts.async_options)
    assert opts.async_options is not None
    self.assertEqual(opts.async_options.timeout_secs, 5000)

  def test_resolve_checkpointing_defaults_with_modern_options(self):
    modern_opts = checkpoint_options._CheckpointingOptions(
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
    self.assertIsInstance(opts, checkpoint_options._CheckpointingOptions)
    self.assertEqual(
        opts.save_decision_policy,
        ocp.training.save_decision_policies.FixedIntervalPolicy(50),
    )
    self.assertEqual(
        opts.preservation_policy,
        ocp.training.preservation_policies.LatestN(10),
    )
    self.assertFalse(opts.enable_async_checkpointing)

if __name__ == '__main__':
  absltest.main()
