import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib


class FileTrajectoryReaderTest(store_testing.TrajectoryReaderTestCase):
  """Contract tests for FileTrajectoryStore's TrajectoryReader implementation."""

  def _create_reader(
      self,
      initial_data: (
          list[
              tuple[
                  trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]
              ]
          ]
          | None
      ) = None,
  ) -> store.TrajectoryReader:
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    file_s = file_store.FileTrajectoryStore(
        root_dir=tmp_dir, run_id="test_reader_run"
    )
    if initial_data:
      for meta, steps in initial_data:
        for step in steps:
          file_s.add_step(step, meta)
      file_s.flush()
    return file_s


class FileTrajectoryWriterTest(store_testing.TrajectoryWriterTestCase):
  """Contract tests for FileTrajectoryStore's TrajectoryWriter implementation."""

  def _create_reader_and_writer(
      self,
  ) -> tuple[store.TrajectoryReader, store.TrajectoryWriter]:
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    file_s = file_store.FileTrajectoryStore(
        root_dir=tmp_dir, run_id="test_writer_run"
    )
    return file_s, file_s


class FileTrajectoryStoreTest(parameterized.TestCase):
  """Unit tests for FileTrajectoryStore property behavior and extra-file handling."""

  def setUp(self) -> None:
    super().setUp()
    self.tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    self.file_s = file_store.FileTrajectoryStore(root_dir=self.tmp_dir)

  def test_root_dir_without_run_id(self) -> None:
    """Verifies root_dir directly returns base directory when run_id is omitted."""
    self.assertEqual(self.file_s.root_dir, self.tmp_dir)

  def test_root_dir_with_run_id(self) -> None:
    """Verifies root_dir is scoped under root_dir / run_id when run_id is provided."""
    file_s_with_run = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id="my_run_123"
    )
    self.assertEqual(file_s_with_run.root_dir, self.tmp_dir / "my_run_123")

  def test_skips_unrelated_directories_and_files_in_root_dir(self) -> None:
    """Verifies unrelated root files and non-trajectory directories are skipped during metadata listing."""
    self.file_s.add_step(store_testing.STEP_1_1, store_testing.METADATA_1)
    self.file_s.flush()

    # Create non-trajectory files and directories in root_dir.
    (self.file_s.root_dir / "README.md").write_text("Documentation")
    (self.file_s.root_dir / "tb_logs").mkdir()
    (self.file_s.root_dir / ".git").mkdir()

    # Verify only valid trajectory metadata is returned.
    metas = self.file_s.get_trajectories_metadata()
    self.assertEqual(metas, [store_testing.METADATA_1])

  def test_skips_files_matching_trajectory_dir_prefix(self) -> None:
    """Verifies files matching the trajectory directory prefix are skipped during metadata listing."""
    self.file_s.add_step(store_testing.STEP_1_1, store_testing.METADATA_1)
    self.file_s.flush()

    # Create a regular file whose name matches the trajectory directory prefix.
    file_name = f"{file_store._TRAJECTORY_DIR_PREFIX}_notes.txt"
    (self.file_s.root_dir / file_name).write_text("Notes file")

    metas = self.file_s.get_trajectories_metadata()
    self.assertEqual(metas, [store_testing.METADATA_1])

  def test_skips_unrelated_files_in_trajectory_dir(self) -> None:
    """Verifies unrelated files inside a trajectory directory are skipped during trajectory loading."""
    self.file_s.add_step(store_testing.STEP_1_1, store_testing.METADATA_1)
    self.file_s.flush()

    # Simulate non-trajectory files placed inside the trajectory directory.
    traj_dir = self.file_s.get_trajectory_dir(store_testing.TRAJECTORY_ID_1)
    (traj_dir / "worker_log.txt").write_text("Worker execution details")
    (traj_dir / "lock_file.tmp").write_text("LOCK")

    # Verify trajectory loading ignores unrelated files.
    (traj,) = self.file_s.get_trajectories([store_testing.TRAJECTORY_ID_1])
    self.assertEqual(traj, store_testing.TRAJECTORY_1)

  def test_missing_metadata_in_trajectory_dir_raises_error(self) -> None:
    """Verifies missing metadata.json in a trajectory directory raises error."""
    self.file_s.add_step(store_testing.STEP_1_1, store_testing.METADATA_1)
    self.file_s.flush()
    meta_path = self.file_s.get_trajectory_metadata_path(
        store_testing.TRAJECTORY_ID_1
    )
    meta_path.unlink()

    with self.assertRaises(store.TrajectoryMetadataNotFoundError):
      self.file_s.get_trajectories_metadata()

  @parameterized.named_parameters(
      ("with_slash", "traj/1001"),
      ("with_dot", "traj.1001"),
      ("with_space", "traj 1001"),
      ("with_colon", "traj:1001"),
  )
  def test_add_step_rejects_invalid_trajectory_id(
      self, bad_trajectory_id: str
  ) -> None:
    """Verifies add_step rejects trajectory_ids that would not round-trip."""
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=bad_trajectory_id,
        agent=trajectory_lib.Agent(name="agent", version="1.0"),
    )

    with self.assertRaises(ValueError):
      self.file_s.add_step(store_testing.STEP_1_1, meta)

  def test_add_step_and_get_trajectory_id_with_allowed_characters(self) -> None:
    """Verifies ids with hyphens/underscores are written and read back."""
    traj_id = "traj-100_A1"
    meta = store_testing.METADATA_1.model_copy(
        update={"trajectory_id": traj_id}
    )
    self.file_s.add_step(store_testing.STEP_1_1, meta)
    self.file_s.flush()

    (recovered_traj,) = self.file_s.get_trajectories([traj_id])
    expected_traj = store_testing.TRAJECTORY_1.model_copy(
        update={"trajectory_id": traj_id}
    )
    self.assertEqual(recovered_traj, expected_traj)

  def test_store_recovery_and_persistence_across_instances(self) -> None:
    """Simulates process restart by initializing a new FileTrajectoryStore instance on existing directory."""
    run_id = "persistent_run_42"

    # Instance 1: write initial steps
    store_instance_1 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )

    meta = store_testing.METADATA_2
    store_instance_1.add_step(store_testing.STEP_2_1, meta)
    store_instance_1.flush()

    # Instance 2: new process reading and appending to same run_id
    store_instance_2 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )

    metas_2 = store_instance_2.get_trajectories_metadata()
    self.assertEqual(metas_2, [meta])

    store_instance_2.add_step(store_testing.STEP_2_2, meta)
    store_instance_2.flush()

    # Instance 3: verify complete recovered state
    store_instance_3 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )
    (recovered_traj,) = store_instance_3.get_trajectories(
        [store_testing.TRAJECTORY_ID_2]
    )
    self.assertEqual(recovered_traj, store_testing.TRAJECTORY_2)


if __name__ == "__main__":
  absltest.main()
