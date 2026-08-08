import tempfile
from unittest import mock

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

    # Create a regular file whose name matches the trajectory directory prefix.
    file_name = f"{file_store._TRAJECTORY_DIR_PREFIX}_notes.txt"
    (self.file_s.root_dir / file_name).write_text("Notes file")

    metas = self.file_s.get_trajectories_metadata()
    self.assertEqual(metas, [store_testing.METADATA_1])

  def test_skips_unrelated_files_in_trajectory_dir(self) -> None:
    """Verifies unrelated files inside a trajectory directory are skipped during trajectory loading."""
    self.file_s.add_step(store_testing.STEP_1_1, store_testing.METADATA_1)

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

  def test_mkdir_called_only_once_per_trajectory_across_multiple_steps(
      self,
  ) -> None:
    """Verifies mkdir is called only once per trajectory across multiple steps."""
    traj_1_dir = self.file_s.get_trajectory_dir(store_testing.TRAJECTORY_ID_1)
    traj_2_dir = self.file_s.get_trajectory_dir(store_testing.TRAJECTORY_ID_2)

    with mock.patch.object(
        self.tmp_dir.__class__,
        "mkdir",
        autospec=True,
        side_effect=self.tmp_dir.__class__.mkdir,
    ) as mock_mkdir:
      # Trajectory 1, Step 1: mkdir should be called.
      self.file_s.add_step(store_testing.STEP_1_1, store_testing.METADATA_1)
      traj_1_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_1_dir
      ]
      self.assertLen(traj_1_calls, 1)

      # Trajectory 2, Step 1: mkdir should be called for new trajectory.
      self.file_s.add_step(store_testing.STEP_2_1, store_testing.METADATA_2)
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

      # Trajectory 2, Step 2: mkdir should be skipped.
      self.file_s.add_step(store_testing.STEP_2_2, store_testing.METADATA_2)
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

      # Trajectory 2, Step 3: mkdir still skipped for initialized trajectory 2.
      self.file_s.add_step(store_testing.STEP_2_3, store_testing.METADATA_2)
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

    self.assertIn(
        store_testing.TRAJECTORY_ID_1,
        self.file_s._metadata_hash_by_trajectory_id,
    )
    self.assertIn(
        store_testing.TRAJECTORY_ID_2,
        self.file_s._metadata_hash_by_trajectory_id,
    )

  def test_metadata_written_on_first_step_and_skipped_when_unchanged(
      self,
  ) -> None:
    """Verifies metadata.json is written on step 1 and skipped for unchanged steps."""
    step_1 = store_testing.STEP_1_1
    step_2 = store_testing.STEP_2_1

    path_cls = type(self.tmp_dir)
    meta_path = self.file_s.get_trajectory_metadata_path(
        store_testing.TRAJECTORY_ID_1
    )
    with mock.patch.object(
        path_cls, "write_text", autospec=True, side_effect=path_cls.write_text
    ) as mock_write:
      self.file_s.add_step(step_1, store_testing.METADATA_1)
      self.file_s.add_step(step_2, store_testing.METADATA_1)

      meta_write_calls = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_write_calls, 1)

    self.assertTrue(meta_path.exists())
    saved_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
        meta_path.read_text()
    )
    self.assertEqual(saved_meta, store_testing.METADATA_1)

  def test_metadata_updated_when_metadata_changes(self) -> None:
    """Verifies metadata.json is updated when metadata content changes."""
    step_1 = store_testing.STEP_2_1
    step_2 = store_testing.STEP_2_2
    step_3 = store_testing.STEP_2_3
    step_4 = store_testing.STEP_2_4
    step_5 = store_testing.STEP_2_5

    meta_initial = store_testing.METADATA_1
    meta_completed = store_testing.METADATA_1.model_copy(
        update={"extra": {"status": "COMPLETED"}}
    )
    meta_failed = store_testing.METADATA_1.model_copy(
        update={"extra": {"status": "FAILED"}}
    )

    path_cls = type(self.tmp_dir)
    meta_path = self.file_s.get_trajectory_metadata_path(
        store_testing.TRAJECTORY_ID_1
    )
    with mock.patch.object(
        path_cls, "write_text", autospec=True, side_effect=path_cls.write_text
    ) as mock_write:
      # Step 1: Initial metadata written.
      self.file_s.add_step(step_1, meta_initial)
      # Step 2: Unchanged metadata skipped.
      self.file_s.add_step(step_2, meta_initial)
      # Step 3: Metadata updated to COMPLETED -> written.
      self.file_s.add_step(step_3, meta_completed)
      # Step 4: Metadata unchanged with COMPLETED -> skipped.
      self.file_s.add_step(step_4, meta_completed)
      # Step 5: Metadata updated to FAILED -> written.
      self.file_s.add_step(step_5, meta_failed)

      meta_write_calls = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_write_calls, 3)

    # Verify latest metadata is reflected on disk.
    saved_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
        meta_path.read_text()
    )
    self.assertEqual(saved_meta, meta_failed)


if __name__ == "__main__":
  absltest.main()
