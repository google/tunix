import tempfile
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing


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
    self.file_s.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.file_s.flush()

    # Create non-trajectory files and directories in root_dir.
    (self.file_s.root_dir / "README.md").write_text("Documentation")
    (self.file_s.root_dir / "tb_logs").mkdir()
    (self.file_s.root_dir / ".git").mkdir()

    # Verify only valid trajectory metadata is returned.
    metas = self.file_s.get_trajectories_metadata()
    self.assertEqual(metas, [trajectory_testing.METADATA_1])

  def test_skips_files_matching_trajectory_dir_prefix(self) -> None:
    """Verifies files matching the trajectory directory prefix are skipped during metadata listing."""
    self.file_s.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.file_s.flush()

    # Create a regular file whose name matches the trajectory directory prefix.
    file_name = f"{file_store._TRAJECTORY_DIR_PREFIX}_notes.txt"
    (self.file_s.root_dir / file_name).write_text("Notes file")

    metas = self.file_s.get_trajectories_metadata()
    self.assertEqual(metas, [trajectory_testing.METADATA_1])

  def test_skips_unrelated_files_in_trajectory_dir(self) -> None:
    """Verifies unrelated files inside a trajectory directory are skipped during trajectory loading."""
    self.file_s.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.file_s.flush()

    # Simulate non-trajectory files placed inside the trajectory directory.
    traj_dir = self.file_s.get_trajectory_dir(
        trajectory_testing.TRAJECTORY_ID_1
    )
    (traj_dir / "worker_log.txt").write_text("Worker execution details")
    (traj_dir / "lock_file.tmp").write_text("LOCK")

    # Verify trajectory loading ignores unrelated files.
    (traj,) = self.file_s.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(traj, trajectory_testing.TRAJECTORY_1)

  def test_missing_metadata_in_trajectory_dir_raises_error(self) -> None:
    """Verifies missing metadata.json in a trajectory directory raises error."""
    self.file_s.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.file_s.flush()
    meta_path = self.file_s.get_trajectory_metadata_path(
        trajectory_testing.TRAJECTORY_ID_1
    )
    meta_path.unlink()

    with self.assertRaises(store.TrajectoryMetadataNotFoundError):
      self.file_s.get_trajectories_metadata()

  @parameterized.named_parameters(
      ("with_slash", "traj/1001"),
      ("with_dot", "traj.1001"),
      ("with_space", "traj 1001"),
      ("with_colon", "traj:1001"),
      ("empty", ""),
      ("none", None),
  )
  def test_rejects_invalid_trajectory_id(
      self, bad_trajectory_id: str | None
  ) -> None:
    """Verifies add_step, update_metadata, and helper reject invalid trajectory_ids."""
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=bad_trajectory_id,
        agent=trajectory_lib.Agent(name="agent", version="1.0"),
    )

    with self.subTest("add_step"):
      with self.assertRaises(ValueError):
        self.file_s.add_step(trajectory_testing.STEP_1_1, meta)

    with self.subTest("update_metadata"):
      with self.assertRaises(ValueError):
        self.file_s.update_metadata(meta)

    with self.subTest("_validate_trajectory_id"):
      with self.assertRaises(ValueError):
        file_store._validate_trajectory_id(bad_trajectory_id)

  def test_validate_trajectory_id_valid(self) -> None:
    """Verifies _validate_trajectory_id returns valid trajectory_ids unchanged."""
    self.assertEqual(
        file_store._validate_trajectory_id("traj-100_A1"), "traj-100_A1"
    )

  def test_add_step_and_get_trajectory_id_with_allowed_characters(self) -> None:
    """Verifies ids with hyphens/underscores are written and read back."""
    traj_id = "traj-100_A1"
    meta = trajectory_testing.METADATA_1.model_copy(
        update={"trajectory_id": traj_id}
    )
    self.file_s.add_step(trajectory_testing.STEP_1_1, meta)
    self.file_s.flush()

    (recovered_traj,) = self.file_s.get_trajectories([traj_id])
    expected_traj = trajectory_testing.TRAJECTORY_1.model_copy(
        update={"trajectory_id": traj_id}
    )
    self.assertEqual(recovered_traj, expected_traj)

  def test_add_step_is_non_blocking(self) -> None:
    """Verifies that add_step returns immediately without waiting for disk I/O."""
    block_event = threading.Event()
    original_process_task = self.file_s._writer._process_task

    def blocking_process_task(task):
      block_event.wait()
      original_process_task(task)

    step_path = self.file_s.get_step_path(
        trajectory_testing.TRAJECTORY_ID_1,
        trajectory_testing.STEP_1_1.step_id,
    )

    with mock.patch.object(
        self.file_s._writer, "_process_task", side_effect=blocking_process_task
    ):
      # add_step should enqueue task and return immediately while worker loop is blocked.
      self.file_s.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )
      self.assertFalse(step_path.exists())

      # Unblock worker thread and flush pending task.
      block_event.set()
      self.file_s.flush()
      self.assertTrue(step_path.exists())

  def test_store_recovery_and_persistence_across_instances(self) -> None:
    """Simulates process restart by initializing a new FileTrajectoryStore instance on existing directory."""
    run_id = "persistent_run_42"

    # Instance 1: write initial steps
    store_instance_1 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )

    meta = trajectory_testing.METADATA_2
    store_instance_1.add_step(trajectory_testing.STEP_2_1, meta)
    store_instance_1.flush()

    # Instance 2: new process reading and appending to same run_id
    store_instance_2 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )

    metas_2 = store_instance_2.get_trajectories_metadata()
    self.assertEqual(metas_2, [meta])

    store_instance_2.add_step(trajectory_testing.STEP_2_2, meta)
    store_instance_2.add_step(trajectory_testing.STEP_2_3, meta)
    store_instance_2.add_step(trajectory_testing.STEP_2_4, meta)
    store_instance_2.add_step(trajectory_testing.STEP_2_5, meta)
    store_instance_2.flush()

    # Instance 3: verify complete recovered state
    store_instance_3 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )
    (recovered_traj,) = store_instance_3.get_trajectories(
        [trajectory_testing.TRAJECTORY_ID_2]
    )
    self.assertEqual(recovered_traj, trajectory_testing.TRAJECTORY_2)

  def test_mkdir_called_only_once_per_trajectory_across_multiple_steps(
      self,
  ) -> None:
    """Verifies mkdir is called only once per trajectory across multiple steps."""
    traj_1_dir = self.file_s.get_trajectory_dir(
        trajectory_testing.TRAJECTORY_ID_1
    )
    traj_2_dir = self.file_s.get_trajectory_dir(
        trajectory_testing.TRAJECTORY_ID_2
    )

    self.assertFalse(traj_1_dir.exists())
    self.assertFalse(traj_2_dir.exists())

    path_cls = type(self.tmp_dir)
    with mock.patch.object(
        path_cls, "mkdir", autospec=True, side_effect=path_cls.mkdir
    ) as mock_mkdir:
      # Trajectory 1, Step 1: mkdir should be called.
      self.file_s.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )
      self.file_s.flush()
      traj_1_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_1_dir
      ]
      self.assertLen(traj_1_calls, 1)

      # Trajectory 2, Step 1: mkdir should be called for new trajectory.
      self.file_s.add_step(
          trajectory_testing.STEP_2_1, trajectory_testing.METADATA_2
      )
      self.file_s.flush()
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

      # Trajectory 2, Step 2: mkdir should be skipped.
      self.file_s.add_step(
          trajectory_testing.STEP_2_2, trajectory_testing.METADATA_2
      )
      self.file_s.flush()
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

      # Trajectory 2, Step 3: mkdir still skipped for initialized trajectory 2.
      self.file_s.add_step(
          trajectory_testing.STEP_2_3, trajectory_testing.METADATA_2
      )
      self.file_s.flush()
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

    metas = self.file_s.get_trajectories_metadata()
    self.assertEqual(
        {m.trajectory_id for m in metas},
        {
            trajectory_testing.TRAJECTORY_ID_1,
            trajectory_testing.TRAJECTORY_ID_2,
        },
    )

  def test_metadata_written_on_first_step_and_skipped_when_unchanged(
      self,
  ) -> None:
    """Verifies metadata.json is written on step 1 and skipped for unchanged steps."""
    step_1 = trajectory_testing.STEP_1_1
    step_2 = trajectory_testing.STEP_2_1

    path_cls = type(self.tmp_dir)
    meta_path = self.file_s.get_trajectory_metadata_path(
        trajectory_testing.TRAJECTORY_ID_1
    )
    with mock.patch.object(
        path_cls, "write_text", autospec=True, side_effect=path_cls.write_text
    ) as mock_write:
      # Step 1: metadata should be written.
      self.file_s.add_step(step_1, trajectory_testing.METADATA_1)
      self.file_s.flush()
      meta_write_calls = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_write_calls, 1)

      # Step 2: unchanged metadata should not be rewritten.
      self.file_s.add_step(step_2, trajectory_testing.METADATA_1)
      self.file_s.flush()
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
    self.assertEqual(saved_meta, trajectory_testing.METADATA_1)

  def test_metadata_updated_when_metadata_changes(self) -> None:
    """Verifies metadata.json is updated when metadata content changes."""
    meta_initial = trajectory_testing.METADATA_1
    meta_completed = trajectory_testing.METADATA_1.model_copy(
        update={"extra": {"status": "COMPLETED"}}
    )
    meta_failed = trajectory_testing.METADATA_1.model_copy(
        update={"extra": {"status": "FAILED"}}
    )

    path_cls = type(self.tmp_dir)
    meta_path = self.file_s.get_trajectory_metadata_path(
        trajectory_testing.TRAJECTORY_ID_1
    )
    with mock.patch.object(
        path_cls, "write_text", autospec=True, side_effect=path_cls.write_text
    ) as mock_write:
      # Step 1: Initial metadata written.
      self.file_s.add_step(trajectory_testing.STEP_2_1, meta_initial)
      self.file_s.flush()
      meta_write_calls = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_write_calls, 1)

      # Step 2: Unchanged metadata skipped.
      self.file_s.add_step(trajectory_testing.STEP_2_2, meta_initial)
      self.file_s.flush()
      meta_write_calls = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_write_calls, 1)

      # Step 3: Metadata updated to COMPLETED -> written.
      self.file_s.add_step(trajectory_testing.STEP_2_3, meta_completed)
      self.file_s.flush()
      meta_write_calls = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_write_calls, 2)

      # Step 4: Metadata unchanged with COMPLETED -> skipped.
      self.file_s.add_step(trajectory_testing.STEP_2_4, meta_completed)
      self.file_s.flush()
      meta_write_calls = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_write_calls, 2)

      # Step 5: Metadata updated to FAILED -> written.
      self.file_s.add_step(trajectory_testing.STEP_2_5, meta_failed)
      self.file_s.flush()
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

  def test_close_shuts_down_writer_and_rejects_further_writes(self) -> None:
    """Verifies close() drains the background writer and seals the store."""
    self.file_s.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.file_s.close()

    step_path = self.file_s.get_step_path(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    self.assertTrue(step_path.exists())
    with self.assertRaises(RuntimeError):
      self.file_s.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

  def test_context_manager_closes_store_on_exit(self) -> None:
    """Verifies the store persists pending writes when used as a context manager."""
    with file_store.FileTrajectoryStore(root_dir=self.tmp_dir) as file_s:
      file_s.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )
      step_path = file_s.get_step_path(
          trajectory_testing.TRAJECTORY_ID_1,
          trajectory_testing.STEP_1_1.step_id,
      )

    self.assertTrue(step_path.exists())

  def test_get_trajectories_metadata_nonexistent_root_dir_returns_empty(
      self,
  ) -> None:
    """Verifies get_trajectories_metadata on a non-existent root_dir gracefully returns empty list."""
    nonexistent_root = self.tmp_dir / "does_not_exist"
    store_instance = file_store.FileTrajectoryStore(root_dir=nonexistent_root)
    self.assertEmpty(store_instance.get_trajectories_metadata())


if __name__ == "__main__":
  absltest.main()
