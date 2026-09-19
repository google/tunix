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

"""Tests for TrajectoryStore.from_config and to_config."""

import tempfile

from absl.testing import absltest
from etils import epath
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory import trajectory_testing


class FromConfigTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))

  def _file_config(self, **overrides):
    config = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(self.tmp_dir),
        "run_id": "run_1",
    }
    config.update(overrides)
    return config

  def test_none_config_disables_the_store(self):
    self.assertIsNone(store_lib.TrajectoryStore.from_config(None))

  def test_missing_enabled_key_disables_the_store(self):
    self.assertIsNone(
        store_lib.TrajectoryStore.from_config({"backend": "memory"})
    )

  def test_enabled_false_disables_the_store(self):
    # The rest of the config stays valid and unused, so a run can be turned
    # off without unsetting its root_dir and run_id.
    self.assertIsNone(
        store_lib.TrajectoryStore.from_config(self._file_config(enabled=False))
    )

  def test_file_backend_is_scoped_by_run_id(self):
    store = store_lib.TrajectoryStore.from_config(self._file_config())
    self.assertIsInstance(store, file_store.FileTrajectoryStore)
    self.assertEqual(store.root_dir, self.tmp_dir / "run_1")
    store.close()

  def test_memory_backend_needs_no_other_keys(self):
    store = store_lib.TrajectoryStore.from_config(
        {"enabled": True, "backend": "memory"}
    )
    self.assertIsInstance(store, in_memory_store.InMemoryTrajectoryStore)

  def test_unknown_backend_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown Trajectory Store"):
      store_lib.TrajectoryStore.from_config(
          {"enabled": True, "backend": "sqlite"}
      )

  def test_missing_backend_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown Trajectory Store"):
      store_lib.TrajectoryStore.from_config({"enabled": True})

  def test_file_backend_without_root_dir_raises(self):
    with self.assertRaisesRegex(ValueError, "'root_dir'"):
      store_lib.TrajectoryStore.from_config(self._file_config(root_dir=""))

  def test_file_backend_without_run_id_raises(self):
    # Optional on FileTrajectoryStore.__init__, required here: a configured
    # run is shared across processes and restarts, and run_id is what scopes
    # their common directory.
    with self.assertRaisesRegex(ValueError, "'run_id'"):
      store_lib.TrajectoryStore.from_config(self._file_config(run_id=None))

  def test_file_backend_rejects_a_run_id_that_is_not_a_path_segment(self):
    with self.assertRaisesRegex(ValueError, "unsupported characters"):
      store_lib.TrajectoryStore.from_config(self._file_config(run_id="a/b"))

  def test_two_calls_build_two_independent_stores(self):
    # Nothing here is a singleton: the guard against a second store (and, for
    # the file backend, a second writer thread) is that each process calls
    # this once and holds the result.
    config = self._file_config()
    first = store_lib.TrajectoryStore.from_config(config)
    second = store_lib.TrajectoryStore.from_config(config)
    self.assertIsNot(first, second)
    self.assertEqual(first.root_dir, second.root_dir)
    first.close()
    second.close()

  def test_custom_subclass_registers_automatically(self):
    class CustomStore(store_lib.TrajectoryStore):
      BACKEND = "custom_test_backend"

      @classmethod
      def _from_config(cls, config):
        return cls()

      def to_config(self):
        return {"enabled": True, "backend": self.BACKEND}

      def get_trajectories_metadata(self, trajectory_ids=None):
        return []

      def get_trajectories(self, trajectory_ids):
        return []

      def add_step(self, step, metadata):
        pass

      def update_metadata(self, metadata):
        pass

      def flush(self):
        pass

      def close(self):
        pass

    try:
      store = store_lib.TrajectoryStore.from_config(
          {"enabled": True, "backend": "custom_test_backend"}
      )
      self.assertIsInstance(store, CustomStore)
    finally:
      store_lib.TrajectoryStore._REGISTRY.pop("custom_test_backend", None)


class ToConfigTest(absltest.TestCase):

  def test_file_store_round_trips_through_its_own_config(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    original = file_store.FileTrajectoryStore(
        root_dir=str(tmp_dir), run_id="run_1"
    )
    rebuilt = store_lib.TrajectoryStore.from_config(original.to_config())
    self.assertEqual(rebuilt.root_dir, original.root_dir)

    original.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    original.close()
    self.assertEqual(
        [m.trajectory_id for m in rebuilt.get_trajectories_metadata()],
        [trajectory_testing.METADATA_1.trajectory_id],
    )
    rebuilt.close()

  def test_in_memory_store_round_trips_through_its_own_config(self):
    original = in_memory_store.InMemoryTrajectoryStore()
    self.assertEqual(
        original.to_config(), {"enabled": True, "backend": "memory"}
    )
    rebuilt = store_lib.TrajectoryStore.from_config(original.to_config())
    self.assertIsInstance(rebuilt, in_memory_store.InMemoryTrajectoryStore)

  def test_config_reports_the_backend_it_was_built_from(self):
    # What to_config is for: two processes in one run that report different
    # dicts are reading and writing different data.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "run_1",
    }
    store = store_lib.TrajectoryStore.from_config(config)
    self.assertEqual(store.to_config(), config)
    store.close()

  def test_file_store_without_run_id_raises_on_to_config(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    store = file_store.FileTrajectoryStore(root_dir=str(tmp_dir))
    with self.assertRaisesRegex(ValueError, "run_id"):
      store.to_config()
    store.close()


if __name__ == "__main__":
  absltest.main()
