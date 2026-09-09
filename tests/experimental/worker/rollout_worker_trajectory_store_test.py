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

"""Tests for RolloutWorker's Trajectory Store construction and shutdown."""

import tempfile

from absl.testing import absltest
from etils import epath
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.trajectory import config as trajectory_config_lib
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import trajectory_testing
from tunix.experimental.worker import rollout_worker as rollout_worker_lib


def _worker(config=None):
  return rollout_worker_lib.RolloutWorker(
      worker_id="w0",
      config=config,
      sampler=mocks.MockBaseSamplerImpl(sampler_name="mock_sampler"),
      tokenizer="mock",
      chat_parser="mock",
  )


class RolloutWorkerTrajectoryStoreTest(absltest.TestCase):

  def test_no_config_means_no_store(self):
    worker = _worker()
    self.assertIsNone(worker._trajectory_store)  # pylint: disable=protected-access

  def test_config_without_trajectory_store_config_means_no_store(self):
    worker = _worker(config=rollout_worker_lib.RolloutConfig())
    self.assertIsNone(worker._trajectory_store)  # pylint: disable=protected-access

  def test_disabled_trajectory_store_config_means_no_store(self):
    config = rollout_worker_lib.RolloutConfig(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=False
        )
    )
    worker = _worker(config=config)
    self.assertIsNone(worker._trajectory_store)  # pylint: disable=protected-access

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = rollout_worker_lib.RolloutConfig(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="worker_run",
        )
    )
    worker = _worker(config=config)
    self.assertIsInstance(
        worker._trajectory_store, file_store.FileTrajectoryStore  # pylint: disable=protected-access
    )
    worker.stop()

  def test_two_workers_get_two_independent_store_instances(self):
    # Each process constructs its own store; two RolloutWorker instances in
    # this test process stand in for two separate worker pods, each of which
    # would build its own store exactly once, in its own __init__.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = trajectory_config_lib.TrajectoryStoreConfig(
        enabled=True, backend="file", root_dir=str(tmp_dir), run_id="shared"
    )
    worker_a = _worker(
        config=rollout_worker_lib.RolloutConfig(trajectory_store_config=config)
    )
    worker_b = _worker(
        config=rollout_worker_lib.RolloutConfig(trajectory_store_config=config)
    )
    self.assertIsNot(
        worker_a._trajectory_store, worker_b._trajectory_store  # pylint: disable=protected-access
    )
    worker_a.stop()
    worker_b.stop()

  def test_stop_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = rollout_worker_lib.RolloutConfig(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="worker_run",
        )
    )
    worker = _worker(config=config)
    store = worker._trajectory_store  # pylint: disable=protected-access
    worker.stop()
    with self.assertRaises(RuntimeError):
      store.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

  def test_stop_without_a_store_does_not_raise(self):
    worker = _worker()
    worker.stop()


if __name__ == "__main__":
  absltest.main()
