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

"""Tests for build_trajectory_store."""

import tempfile

from absl.testing import absltest
from etils import epath
from tunix.experimental.trajectory import config as config_lib
from tunix.experimental.trajectory import factory
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store


class BuildTrajectoryStoreTest(absltest.TestCase):

  def test_none_config_returns_none(self):
    self.assertIsNone(factory.build_trajectory_store(None))

  def test_disabled_config_returns_none(self):
    config = config_lib.TrajectoryStoreConfig(enabled=False)
    self.assertIsNone(factory.build_trajectory_store(config))

  def test_file_backend_builds_file_trajectory_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = config_lib.TrajectoryStoreConfig(
        enabled=True, backend="file", root_dir=str(tmp_dir), run_id="run_a"
    )
    store = factory.build_trajectory_store(config)
    try:
      self.assertIsInstance(store, file_store.FileTrajectoryStore)
      self.assertEqual(store.root_dir, tmp_dir / "run_a")
    finally:
      store.close()

  def test_memory_backend_builds_in_memory_trajectory_store(self):
    config = config_lib.TrajectoryStoreConfig(enabled=True, backend="memory")
    store = factory.build_trajectory_store(config)
    self.assertIsInstance(store, in_memory_store.InMemoryTrajectoryStore)

  def test_file_backend_logs_owner_and_resolved_path(self):
    # This log line is how a run_id mismatch between the orchestrator and its
    # workers becomes visible; without it the symptom is a silent empty read.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = config_lib.TrajectoryStoreConfig(
        enabled=True, backend="file", root_dir=str(tmp_dir), run_id="run_c"
    )
    with self.assertLogs(level="INFO") as logs:
      store = factory.build_trajectory_store(config, owner="rollout-0")
    try:
      line = "\n".join(logs.output)
      self.assertIn("owner=rollout-0", line)
      self.assertIn(str(tmp_dir / "run_c"), line)
    finally:
      store.close()

  def test_two_calls_build_two_independent_instances(self):
    # There is no cross-process (or even cross-call) singleton: the guard
    # against double construction is that callers build the store exactly
    # once per process, not that this factory deduplicates for them.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = config_lib.TrajectoryStoreConfig(
        enabled=True, backend="file", root_dir=str(tmp_dir), run_id="run_b"
    )
    store_1 = factory.build_trajectory_store(config)
    store_2 = factory.build_trajectory_store(config)
    try:
      self.assertIsNot(store_1, store_2)
    finally:
      store_1.close()
      store_2.close()


if __name__ == "__main__":
  absltest.main()
