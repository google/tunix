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

"""Tests for ClusterOrchestrator's Trajectory Store construction and shutdown."""

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.trajectory import config as trajectory_config_lib
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import trajectory_testing


def _orchestrator(**kwargs) -> orchestrator.ClusterOrchestrator:
  mock_registry = mock.MagicMock()
  mock_registry.worker_ids.return_value = []
  mock_registry.infos.return_value = []
  mock_registry.group.return_value.members.return_value = []
  return orchestrator.ClusterOrchestrator(
      registry=mock_registry,
      lifecycle_driver=mock.MagicMock(),
      monitor=mock.MagicMock(),
      **kwargs,
  )


class ClusterOrchestratorTrajectoryStoreTest(absltest.TestCase):

  def test_no_config_means_no_store(self):
    orch = _orchestrator()
    self.assertIsNone(orch.trajectory_store)
    orch.shutdown()

  def test_disabled_config_means_no_store(self):
    orch = _orchestrator(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=False
        )
    )
    self.assertIsNone(orch.trajectory_store)
    orch.shutdown()

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _orchestrator(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="cluster_run",
        )
    )
    self.assertIsInstance(orch.trajectory_store, file_store.FileTrajectoryStore)
    orch.shutdown()

  def test_shutdown_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _orchestrator(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="cluster_run",
        )
    )
    store = orch.trajectory_store
    orch.shutdown()
    with self.assertRaises(RuntimeError):
      store.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

  def test_shutdown_without_a_store_does_not_raise(self):
    orch = _orchestrator()
    orch.shutdown()


if __name__ == "__main__":
  absltest.main()
