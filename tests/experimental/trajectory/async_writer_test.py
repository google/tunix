"""Tests for AsyncFileWriter."""

import atexit
import concurrent.futures
import tempfile
import threading
import time
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from tunix.experimental.trajectory import async_writer
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing


class AsyncFileWriterTest(parameterized.TestCase):
  """Unit tests for AsyncFileWriter."""

  def setUp(self) -> None:
    super().setUp()
    self.tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    self.writer = async_writer.AsyncFileWriter()

  def _get_traj_paths(
      self, traj_id: str, step_id: int
  ) -> tuple[epath.Path, epath.Path, epath.Path]:
    """Helper returning (traj_dir, meta_path, step_path)."""
    traj_dir = self.tmp_dir / f"traj_{traj_id}"
    meta_path = traj_dir / "metadata.json"
    step_path = traj_dir / f"step_{step_id:06d}.json"
    return traj_dir, meta_path, step_path

  # ============================================================================
  # Core Writing Functionality
  # ============================================================================

  def test_write_step_non_blocking_and_flush_persists(self) -> None:
    """Verifies write_step enqueues asynchronously and flush blocks until files are on disk."""
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )

    self.writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    self.writer.flush()

    self.assertTrue(traj_dir.exists())
    self.assertTrue(meta_path.exists())
    self.assertTrue(step_path.exists())

    saved_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
        meta_path.read_text()
    )
    self.assertEqual(saved_meta, trajectory_testing.METADATA_1)

    saved_step = trajectory_lib.Step.model_validate_json(step_path.read_text())
    self.assertEqual(saved_step, trajectory_testing.STEP_1_1)

  def test_write_step_metadata_only(self) -> None:
    """Verifies write_step enqueues and writes metadata.json without a step payload."""
    traj_dir, meta_path, _ = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, 1
    )
    self.writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        metadata=trajectory_testing.METADATA_1,
    )
    self.writer.flush()

    self.assertTrue(traj_dir.exists())
    self.assertTrue(meta_path.exists())
    saved_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
        meta_path.read_text()
    )
    self.assertEqual(saved_meta, trajectory_testing.METADATA_1)

  def test_sequential_fifo_order_across_steps(self) -> None:
    """Verifies that multiple steps are written sequentially in order."""
    traj_id = trajectory_testing.TRAJECTORY_ID_2
    meta = trajectory_testing.METADATA_2
    steps = [
        trajectory_testing.STEP_2_1,
        trajectory_testing.STEP_2_2,
        trajectory_testing.STEP_2_3,
        trajectory_testing.STEP_2_4,
        trajectory_testing.STEP_2_5,
    ]

    for step in steps:
      traj_dir, meta_path, step_path = self._get_traj_paths(
          traj_id, step.step_id
      )
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_path,
          metadata=meta,
          step=step,
      )

    self.writer.flush()

    for step in steps:
      _, _, step_path = self._get_traj_paths(traj_id, step.step_id)
      self.assertTrue(step_path.exists())
      saved_step = trajectory_lib.Step.model_validate_json(
          step_path.read_text()
      )
      self.assertEqual(saved_step, step)

  # ============================================================================
  # Snapshot-on-Enqueue Ownership
  # ============================================================================

  def test_mutating_step_after_write_step_does_not_affect_file(self) -> None:
    """Verifies an enqueued step is snapshotted, not shared with the caller."""
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    meta = trajectory_testing.METADATA_1.model_copy(deep=True)
    step = trajectory_testing.STEP_1_1.model_copy(deep=True)

    block_event = threading.Event()
    worker_thread = None
    original_process_task = self.writer._process_task

    def blocking_process_task(task):
      block_event.wait()
      original_process_task(task)

    try:
      with mock.patch.object(
          self.writer, "_process_task", side_effect=blocking_process_task
      ):
        self.writer.write_step(
            traj_dir=traj_dir,
            meta_path=meta_path,
            step_path=step_path,
            metadata=meta,
            step=step,
        )
        worker_thread = self.writer._worker_thread

        # Mutate while the task is still queued: the worker has not serialized
        # anything yet, so an unsnapshotted task would pick these up.
        step.message = "mutated after enqueueing"
        meta.notes = "mutated after enqueueing"

        block_event.set()
        self.writer.flush()
    finally:
      block_event.set()
      if worker_thread is not None:
        worker_thread.join(timeout=5.0)

    saved_step = trajectory_lib.Step.model_validate_json(step_path.read_text())
    self.assertEqual(saved_step, trajectory_testing.STEP_1_1)
    saved_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
        meta_path.read_text()
    )
    self.assertEqual(saved_meta, trajectory_testing.METADATA_1)

  def test_mutating_step_after_write_step_does_not_affect_later_step(
      self,
  ) -> None:
    """Verifies a caller can reuse one mutable step object across writes."""
    step = trajectory_testing.STEP_2_1.model_copy(deep=True)
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_2, step.step_id
    )
    self.writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=trajectory_testing.METADATA_2,
        step=step,
    )

    # Reuse the same object for the next step, as a rollout loop might.
    step.step_id = trajectory_testing.STEP_2_2.step_id
    step.source = trajectory_testing.STEP_2_2.source
    step.message = trajectory_testing.STEP_2_2.message
    _, _, next_step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_2, step.step_id
    )
    self.writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=next_step_path,
        metadata=trajectory_testing.METADATA_2,
        step=step,
    )
    self.writer.flush()

    for expected_step in (
        trajectory_testing.STEP_2_1,
        trajectory_testing.STEP_2_2,
    ):
      _, _, step_path = self._get_traj_paths(
          trajectory_testing.TRAJECTORY_ID_2, expected_step.step_id
      )
      saved_step = trajectory_lib.Step.model_validate_json(
          step_path.read_text()
      )
      self.assertEqual(saved_step, expected_step)

  # ============================================================================
  # Lazy Worker Thread Initialization
  # ============================================================================

  def test_lazy_worker_thread_initialization(self) -> None:
    """Verifies worker thread is not started in __init__ and starts on first write_step."""
    fresh_writer = async_writer.AsyncFileWriter()
    self.assertIsNone(fresh_writer._worker_thread)

    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    fresh_writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    self.assertIsNotNone(fresh_writer._worker_thread)
    self.assertTrue(fresh_writer._worker_thread.is_alive())
    fresh_writer.flush()
    fresh_writer.close()

  def test_lazy_worker_thread_initialization_concurrent(self) -> None:
    """Verifies concurrent writes to a fresh writer safely start exactly one worker thread."""
    fresh_writer = async_writer.AsyncFileWriter()
    self.assertIsNone(fresh_writer._worker_thread)

    num_threads = 8
    num_steps_per_thread = 4

    def write_worker(thread_idx: int) -> None:
      traj_id = f"lazy_init_traj_{thread_idx}"
      meta = trajectory_testing.METADATA_1.model_copy(
          update={"trajectory_id": traj_id}
      )
      for step_id in range(1, num_steps_per_thread + 1):
        step = trajectory_testing.STEP_1_1.model_copy(
            update={"step_id": step_id}
        )
        traj_dir, meta_path, step_path = self._get_traj_paths(traj_id, step_id)
        fresh_writer.write_step(
            traj_dir=traj_dir,
            meta_path=meta_path,
            step_path=step_path,
            metadata=meta,
            step=step,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      futures = [executor.submit(write_worker, i) for i in range(num_threads)]
      for f in futures:
        f.result()

    fresh_writer.flush()
    self.assertIsNotNone(fresh_writer._worker_thread)
    self.assertTrue(fresh_writer._worker_thread.is_alive())

    for thread_idx in range(num_threads):
      traj_id = f"lazy_init_traj_{thread_idx}"
      for step_id in range(1, num_steps_per_thread + 1):
        _, _, step_path = self._get_traj_paths(traj_id, step_id)
        self.assertTrue(step_path.exists())

    fresh_writer.close()
    self.assertFalse(fresh_writer._worker_thread.is_alive())

  # ============================================================================
  # Optimizations & Caching
  # ============================================================================

  def test_mkdir_called_only_once_per_trajectory(self) -> None:
    """Verifies mkdir is called only once per unique trajectory ID."""
    traj_1_dir, meta_1_path, step_1_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    traj_2_dir, meta_2_path, step_2_1_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_2, trajectory_testing.STEP_2_1.step_id
    )
    _, _, step_2_2_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_2, trajectory_testing.STEP_2_2.step_id
    )

    path_cls = type(self.tmp_dir)
    with mock.patch.object(
        path_cls, "mkdir", autospec=True, side_effect=path_cls.mkdir
    ) as mock_mkdir:
      # Step 1 in traj 1 -> mkdir called
      self.writer.write_step(
          traj_dir=traj_1_dir,
          meta_path=meta_1_path,
          step_path=step_1_path,
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )
      self.writer.flush()
      traj_1_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_1_dir
      ]
      self.assertLen(traj_1_calls, 1)

      # Step 1 in traj 2 -> mkdir called for new trajectory
      self.writer.write_step(
          traj_dir=traj_2_dir,
          meta_path=meta_2_path,
          step_path=step_2_1_path,
          metadata=trajectory_testing.METADATA_2,
          step=trajectory_testing.STEP_2_1,
      )
      self.writer.flush()
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

      # Step 2 in traj 2 -> mkdir skipped
      self.writer.write_step(
          traj_dir=traj_2_dir,
          meta_path=meta_2_path,
          step_path=step_2_2_path,
          metadata=trajectory_testing.METADATA_2,
          step=trajectory_testing.STEP_2_2,
      )
      self.writer.flush()
      traj_2_calls = [
          c
          for c in mock_mkdir.call_args_list
          if c.args and c.args[0] == traj_2_dir
      ]
      self.assertLen(traj_2_calls, 1)

  def test_metadata_written_on_first_step_and_skipped_when_unchanged(
      self,
  ) -> None:
    """Verifies metadata.json is written on step 1 and skipped when unchanged."""
    traj_dir, meta_path, step_1_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    _, _, step_2_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_2_1.step_id
    )

    path_cls = type(self.tmp_dir)
    with mock.patch.object(
        path_cls, "write_text", autospec=True, side_effect=path_cls.write_text
    ) as mock_write:
      # Step 1 -> metadata written
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_1_path,
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )
      self.writer.flush()
      meta_writes = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_writes, 1)

      # Step 2 -> unchanged metadata skipped
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_2_path,
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_2_1,
      )
      self.writer.flush()
      meta_writes = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_writes, 1)

  def test_metadata_updated_when_metadata_changes(self) -> None:
    """Verifies metadata.json is rewritten when metadata content changes."""
    traj_id = trajectory_testing.TRAJECTORY_ID_1
    meta_initial = trajectory_testing.METADATA_1
    meta_completed = trajectory_testing.METADATA_1.model_copy(
        update={"extra": {"status": "COMPLETED"}}
    )
    meta_failed = trajectory_testing.METADATA_1.model_copy(
        update={"extra": {"status": "FAILED"}}
    )

    traj_dir, meta_path, step_1_path = self._get_traj_paths(traj_id, 1)
    _, _, step_2_path = self._get_traj_paths(traj_id, 2)
    _, _, step_3_path = self._get_traj_paths(traj_id, 3)
    _, _, step_4_path = self._get_traj_paths(traj_id, 4)
    _, _, step_5_path = self._get_traj_paths(traj_id, 5)

    path_cls = type(self.tmp_dir)
    with mock.patch.object(
        path_cls, "write_text", autospec=True, side_effect=path_cls.write_text
    ) as mock_write:
      # Step 1: Initial -> written
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_1_path,
          metadata=meta_initial,
          step=trajectory_testing.STEP_1_1,
      )
      self.writer.flush()
      meta_writes = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_writes, 1)

      # Step 2: Unchanged -> skipped
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_2_path,
          metadata=meta_initial,
          step=trajectory_testing.STEP_2_1,
      )
      self.writer.flush()
      meta_writes = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_writes, 1)

      # Step 3: Updated to COMPLETED -> written
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_3_path,
          metadata=meta_completed,
          step=trajectory_testing.STEP_2_2,
      )
      self.writer.flush()
      meta_writes = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_writes, 2)

      # Step 4: Unchanged with COMPLETED -> skipped
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_4_path,
          metadata=meta_completed,
          step=trajectory_testing.STEP_2_3,
      )
      self.writer.flush()
      meta_writes = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_writes, 2)

      # Step 5: Updated to FAILED -> written
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_5_path,
          metadata=meta_failed,
          step=trajectory_testing.STEP_2_4,
      )
      self.writer.flush()
      meta_writes = [
          c
          for c in mock_write.call_args_list
          if c.args and c.args[0] == meta_path
      ]
      self.assertLen(meta_writes, 3)

    saved_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
        meta_path.read_text()
    )
    self.assertEqual(saved_meta, meta_failed)

  # ============================================================================
  # Barrier Synchronization
  # ============================================================================

  def test_flush_idempotent_and_empty(self) -> None:
    """Verifies flush on empty writer is safe and multiple flushes are idempotent."""
    self.writer.flush()
    self.writer.flush()

    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    self.writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    self.writer.flush()
    self.writer.flush()
    self.assertTrue(step_path.exists())

  # ============================================================================
  # Error Handling & Best-Effort Resilience
  # ============================================================================

  def test_error_handling_suppresses_exceptions_and_logs(self) -> None:
    """Verifies that background write errors are logged and suppressed without raising on flush."""
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )

    path_cls = type(self.tmp_dir)
    with mock.patch.object(
        path_cls,
        "write_text",
        autospec=True,
        side_effect=IOError("Simulated disk write failure"),
    ), mock.patch.object(logging, "exception", autospec=True) as mock_log_exc:
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_path,
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )
      self.writer.flush()
      mock_log_exc.assert_called_once()

  def test_error_handling_mkdir_failure_suppressed_and_logged(self) -> None:
    """Verifies that directory creation errors are logged and suppressed without failing flush."""
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )

    path_cls = type(self.tmp_dir)
    with mock.patch.object(
        path_cls,
        "mkdir",
        autospec=True,
        side_effect=PermissionError("Permission denied to create dir"),
    ), mock.patch.object(logging, "exception", autospec=True) as mock_log_exc:
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_path,
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )
      self.writer.flush()
      mock_log_exc.assert_called_once()

  def test_subsequent_writes_continue_after_error(self) -> None:
    """Verifies that worker continues processing subsequent writes after an error."""
    traj_dir_1, meta_1_path, step_1_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    traj_dir_2, meta_2_path, step_2_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_2, trajectory_testing.STEP_2_1.step_id
    )

    path_cls = type(self.tmp_dir)
    original_write_text = path_cls.write_text

    def failing_write_text(self_path: epath.Path, text: str) -> int:
      if self_path == step_1_path:
        raise IOError("Simulated step 1 write failure")
      return original_write_text(self_path, text)

    with mock.patch.object(
        path_cls, "write_text", autospec=True, side_effect=failing_write_text
    ):
      # Step 1 fails in background
      self.writer.write_step(
          traj_dir=traj_dir_1,
          meta_path=meta_1_path,
          step_path=step_1_path,
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )
      # Step 2 succeeds
      self.writer.write_step(
          traj_dir=traj_dir_2,
          meta_path=meta_2_path,
          step_path=step_2_path,
          metadata=trajectory_testing.METADATA_2,
          step=trajectory_testing.STEP_2_1,
      )
      self.writer.flush()

    # Step 2 was written successfully despite step 1 failing
    self.assertTrue(step_2_path.exists())
    saved_step_2 = trajectory_lib.Step.model_validate_json(
        step_2_path.read_text()
    )
    self.assertEqual(saved_step_2, trajectory_testing.STEP_2_1)

  # ============================================================================
  # Concurrency
  # ============================================================================

  def test_concurrent_writes(self) -> None:
    """Verifies concurrent writes from multiple threads across trajectories."""
    num_threads = 4
    num_steps_per_thread = 5

    def worker_thread(thread_idx: int) -> None:
      traj_id = f"concurrent_traj_{thread_idx}"
      meta = trajectory_testing.METADATA_1.model_copy(
          update={"trajectory_id": traj_id}
      )
      for step_id in range(1, num_steps_per_thread + 1):
        step = trajectory_testing.STEP_1_1.model_copy(
            update={"step_id": step_id}
        )
        traj_dir, meta_path, step_path = self._get_traj_paths(traj_id, step_id)
        self.writer.write_step(
            traj_dir=traj_dir,
            meta_path=meta_path,
            step_path=step_path,
            metadata=meta,
            step=step,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
      for f in futures:
        f.result()

    self.writer.flush()

    for thread_idx in range(num_threads):
      traj_id = f"concurrent_traj_{thread_idx}"
      traj_dir, meta_path, _ = self._get_traj_paths(traj_id, 1)
      self.assertTrue(meta_path.exists())
      for step_id in range(1, num_steps_per_thread + 1):
        _, _, step_path = self._get_traj_paths(traj_id, step_id)
        self.assertTrue(step_path.exists())

  # ============================================================================
  # Shutdown & Destructor Teardown
  # ============================================================================

  def test_close_shuts_down_worker_and_prevents_further_writes(self) -> None:
    """Verifies close drains pending items, terminates worker, and rejects future writes."""
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    self.writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    self.writer.close()

    # Step should have been written before close finished
    self.assertTrue(step_path.exists())

    # Further writes must be rejected
    with self.assertRaisesRegex(
        RuntimeError, "Cannot write to a closed AsyncFileWriter."
    ):
      self.writer.write_step(
          traj_dir=traj_dir,
          meta_path=meta_path,
          step_path=step_path,
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )

    # Calling close again is safe and idempotent
    self.writer.close()

  def test_close_concurrent_with_writes(self) -> None:
    """Verifies write_step calls during close are either persisted or cleanly rejected."""
    num_writers = 8
    num_steps = 10
    accepted_steps: list[tuple[epath.Path, trajectory_lib.Step]] = []
    lock = threading.Lock()

    def write_worker(thread_idx: int) -> None:
      traj_id = f"concurrent_close_traj_{thread_idx}"
      meta = trajectory_testing.METADATA_1.model_copy(
          update={"trajectory_id": traj_id}
      )
      for step_id in range(1, num_steps + 1):
        step = trajectory_testing.STEP_1_1.model_copy(
            update={"step_id": step_id}
        )
        traj_dir, meta_path, step_path = self._get_traj_paths(traj_id, step_id)
        try:
          self.writer.write_step(
              traj_dir=traj_dir,
              meta_path=meta_path,
              step_path=step_path,
              metadata=meta,
              step=step,
          )
          with lock:
            accepted_steps.append((step_path, step))
        except RuntimeError as e:
          if "Cannot write to a closed AsyncFileWriter." in str(e):
            break
          raise

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_writers + 1
    ) as executor:
      write_futures = [
          executor.submit(write_worker, i) for i in range(num_writers)
      ]
      time.sleep(0.005)
      close_future = executor.submit(self.writer.close)

      for f in write_futures:
        f.result()
      close_future.result()

    # Verify that every step that was accepted before closure was written to disk.
    for step_path, expected_step in accepted_steps:
      self.assertTrue(step_path.exists(), f"Missing file: {step_path}")
      saved_step = trajectory_lib.Step.model_validate_json(
          step_path.read_text()
      )
      self.assertEqual(saved_step, expected_step)

  def test_multiple_concurrent_close_calls_safe(self) -> None:
    """Verifies that multiple threads calling close() concurrently terminate cleanly."""
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    self.writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
      futures = [executor.submit(self.writer.close) for _ in range(5)]
      for f in futures:
        f.result()

    self.assertTrue(step_path.exists())

  def test_close_timeout_logs_warning(self) -> None:
    """Verifies that close() logs a warning with discarded trajectory IDs if worker thread does not terminate within timeout."""
    block_event = threading.Event()
    worker_thread = None

    def blocking_process_task(task):
      del task
      block_event.wait()

    traj_dir_1, meta_path_1, step_path_1 = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    traj_dir_2, meta_path_2, step_path_2 = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_2, trajectory_testing.STEP_2_1.step_id
    )

    try:
      with mock.patch.object(
          self.writer, "_process_task", side_effect=blocking_process_task
      ):
        # Enqueue step 1: worker thread starts and blocks on task 1.
        self.writer.write_step(
            traj_dir=traj_dir_1,
            meta_path=meta_path_1,
            step_path=step_path_1,
            metadata=trajectory_testing.METADATA_1,
            step=trajectory_testing.STEP_1_1,
        )
        time.sleep(0.05)

        # Enqueue step 2: stays in queue while worker is blocked on task 1.
        self.writer.write_step(
            traj_dir=traj_dir_2,
            meta_path=meta_path_2,
            step_path=step_path_2,
            metadata=trajectory_testing.METADATA_2,
            step=trajectory_testing.STEP_2_1,
        )

        worker_thread = self.writer._worker_thread
        self.assertIsNotNone(worker_thread)

        with mock.patch.object(
            worker_thread, "is_alive", return_value=True
        ), mock.patch.object(logging, "warning", autospec=True) as mock_warning:
          self.writer.close(timeout=0.01)
          mock_warning.assert_called_once()
          call_args = mock_warning.call_args[0]
          self.assertIn("did not finish within timeout", call_args[0])
          self.assertIn("Discarded remaining tasks for trajectory IDs", call_args[0])
          self.assertEqual(call_args[2], [trajectory_testing.TRAJECTORY_ID_2])
    finally:
      block_event.set()
      if worker_thread is not None:
        worker_thread.join(timeout=5.0)

  def test_destructor_on_unstarted_writer(self) -> None:
    """Verifies that __del__ on an unstarted AsyncFileWriter executes cleanly without errors."""
    unstarted_writer = async_writer.AsyncFileWriter()
    self.assertIsNone(unstarted_writer._worker_thread)
    # Should not raise any exception.
    unstarted_writer.__del__()
    self.assertTrue(unstarted_writer._closed)

  def test_destructor_closes_worker_gracefully(self) -> None:
    """Verifies that __del__ closes the worker thread."""
    fresh_writer = async_writer.AsyncFileWriter()
    traj_dir, meta_path, step_path = self._get_traj_paths(
        trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.STEP_1_1.step_id
    )
    fresh_writer.write_step(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    worker_thread = fresh_writer._worker_thread
    self.assertIsNotNone(worker_thread)
    self.assertTrue(worker_thread.is_alive())

    fresh_writer.__del__()
    self.assertFalse(worker_thread.is_alive())
    self.assertTrue(step_path.exists())


class AsyncFileWriterShutdownHookTest(parameterized.TestCase):
  """Tests the atexit hook that drains writers still live at process exit."""

  def setUp(self) -> None:
    super().setUp()
    self.tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))

  def _write_one_step(
      self, writer: async_writer.AsyncFileWriter
  ) -> epath.Path:
    """Enqueues a single step without flushing, returning its file path."""
    traj_dir = self.tmp_dir / f"traj_{trajectory_testing.TRAJECTORY_ID_1}"
    step_path = (
        traj_dir / f"step_{trajectory_testing.STEP_1_1.step_id:06d}.json"
    )
    writer.write_step(
        traj_dir=traj_dir,
        meta_path=traj_dir / "metadata.json",
        step_path=step_path,
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    return step_path

  def test_hook_is_registered_with_atexit(self) -> None:
    """Verifies the module registers its shutdown hook on import."""
    callbacks_before = atexit._ncallbacks()
    atexit.unregister(async_writer._close_live_writers)
    self.addCleanup(atexit.register, async_writer._close_live_writers)

    self.assertEqual(atexit._ncallbacks(), callbacks_before - 1)

  def test_live_writer_is_registered_and_unregistered_on_close(self) -> None:
    """Verifies writers track their liveness for the shutdown hook."""
    writer = async_writer.AsyncFileWriter()
    self.assertIn(writer, async_writer._LIVE_WRITERS)

    writer.close()
    self.assertNotIn(writer, async_writer._LIVE_WRITERS)

  def test_pending_writes_persisted_by_shutdown_hook(self) -> None:
    """Verifies queued steps reach disk when the hook runs, without a flush()."""
    writer = async_writer.AsyncFileWriter()
    step_path = self._write_one_step(writer)

    async_writer._close_live_writers()

    self.assertTrue(step_path.exists())
    self.assertTrue(writer._closed)

  def test_shutdown_hook_suppresses_close_errors(self) -> None:
    """Verifies one failing writer neither propagates nor blocks the others."""
    failing_writer = async_writer.AsyncFileWriter()
    healthy_writer = async_writer.AsyncFileWriter()
    step_path = self._write_one_step(healthy_writer)

    with mock.patch.object(
        failing_writer, "close", side_effect=RuntimeError("close failed")
    ):
      with mock.patch.object(logging, "exception") as mock_log_exception:
        async_writer._close_live_writers()

    mock_log_exception.assert_called_once()
    self.assertTrue(step_path.exists())
    self.assertTrue(healthy_writer._closed)


if __name__ == "__main__":
  absltest.main()
