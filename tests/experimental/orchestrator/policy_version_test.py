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

"""The weight version and the step counter must move independently."""

import types
from typing import Any

from absl.testing import absltest
from tunix.experimental.orchestrator import policy_version
from tunix.experimental.orchestrator import rl_orchestrator


class _FakeCluster:
  """Mimics the cluster's habit of bumping its step count on every sync."""

  def __init__(self):
    self.global_steps = 0
    self.sync_calls = 0

  def sync_weights(self):
    self.sync_calls += 1
    # The in-process cluster couples these two; the orchestrated path must not
    # inherit that coupling.
    self.global_steps += 1

  def update_actor(self, *args):
    del args


class _FakeAlgorithm:
  algo_config: Any = types.SimpleNamespace()


class PolicyVersionMinterTest(absltest.TestCase):

  def test_starts_at_the_given_version(self):
    self.assertEqual(policy_version.PolicyVersionMinter().current, 0)
    self.assertEqual(
        policy_version.PolicyVersionMinter(7).current, 7
    )

  def test_each_mint_advances_by_one(self):
    minter = policy_version.PolicyVersionMinter()

    self.assertEqual(minter.mint(), 1)
    self.assertEqual(minter.mint(), 2)
    self.assertEqual(minter.current, 2)

  def test_resumes_above_the_version_it_was_given(self):
    minter = policy_version.PolicyVersionMinter(initial_version=5)
    self.assertEqual(minter.mint(), 6)

  def test_rejects_a_negative_starting_version(self):
    with self.assertRaises(ValueError):
      policy_version.PolicyVersionMinter(initial_version=-1)


class OrchestratorVersioningTest(absltest.TestCase):

  def _orchestrator(self, cluster):
    return rl_orchestrator.RLOrchestrator(cluster, _FakeAlgorithm())

  def test_training_does_not_advance_the_weight_version(self):
    cluster = _FakeCluster()
    orch = self._orchestrator(cluster)

    orch.train_step([])
    orch.train_step([])

    self.assertEqual(orch.policy_version, 0)

  def test_syncing_advances_the_version_and_reports_it(self):
    cluster = _FakeCluster()
    orch = self._orchestrator(cluster)

    self.assertEqual(orch.sync_weights(), 1)
    self.assertEqual(orch.sync_weights(), 2)
    self.assertEqual(orch.policy_version, 2)

  def test_the_version_is_not_read_from_the_cluster_step_count(self):
    """The cluster bumps its steps on sync; the version must not mirror it."""
    cluster = _FakeCluster()
    cluster.global_steps = 100
    orch = self._orchestrator(cluster)

    orch.sync_weights()

    self.assertEqual(orch.policy_version, 1)
    self.assertEqual(cluster.global_steps, 101)

  def test_a_resumed_run_continues_its_version_stream(self):
    orch = rl_orchestrator.RLOrchestrator(
        _FakeCluster(), _FakeAlgorithm(), initial_policy_version=4
    )

    self.assertEqual(orch.policy_version, 4)
    self.assertEqual(orch.sync_weights(), 5)


if __name__ == "__main__":
  absltest.main()
