"""Unit tests verifying lifecycle and queue mechanics of AsyncWriter."""

import concurrent.futures
import dataclasses
import threading
from typing import Final
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory import async_writer
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing

# Bounds for barriers that must never be hit on a healthy run; generous enough
# to stay reliable on a loaded test machine.
_WORKER_START_TIMEOUT_S: Final[float] = 10.0
_WORKER_BLOCK_TIMEOUT_S: Final[float] = 30.0
# Deliberately short so close() gives up on the stalled worker.
_CLOSE_TIMEOUT_S: Final[float] = 0.05
_NUM_RACING_THREADS: Final[int] = 10


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TestWriteTask(async_writer.WriteTask):
  """Minimal concrete write task for testing AsyncWriter."""


class _TestAsyncWriter(async_writer.AsyncWriter[_TestWriteTask]):
  """In-memory AsyncWriter recording the tasks its worker thread processes."""

  def __init__(
      self,
      thread_name: str | None = None,
      fail_on_step_id: int | None = None,
  ):
    super().__init__(thread_name=thread_name)
    self.processed_tasks: list[_TestWriteTask] = []
    self.processing_thread_ids: list[int] = []
    self.fail_on_step_id = fail_on_step_id

  def enqueue(
      self,
      trajectory_id: str = trajectory_testing.TRAJECTORY_ID_1,
      step_id: int | None = None,
      run_id: str | None = None,
  ) -> _TestWriteTask:
    """Builds a task and submits it through the engine's `_enqueue` entry."""
    task = _TestWriteTask(
        metadata=trajectory_testing.make_metadata(trajectory_id=trajectory_id),
        step=(
            trajectory_testing.make_step(step_id=step_id)
            if step_id is not None
            else None
        ),
        run_id=run_id,
    )
    self._enqueue(task)
    return task

  def _process_task(self, task: _TestWriteTask) -> None:
    if (
        self.fail_on_step_id is not None
        and task.step is not None
        and task.step.step_id == self.fail_on_step_id
    ):
      raise ValueError(f"Intentional test failure on step {task.step.step_id}")
    self.processing_thread_ids.append(threading.get_ident())
    self.processed_tasks.append(task)


class AsyncWriterTest(parameterized.TestCase):
  """Unit tests for AsyncWriter lifecycle and queue mechanics."""

  def _create_writer(
      self,
      thread_name: str | None = None,
      fail_on_step_id: int | None = None,
  ) -> _TestAsyncWriter:
    """Creates a test writer with guaranteed cleanup on test completion."""
    writer = _TestAsyncWriter(
        thread_name=thread_name, fail_on_step_id=fail_on_step_id
    )
    self.addCleanup(writer.close)
    return writer

  # ============================================================================
  # 1. Worker Thread Initialization & Lazy Startup
  # ============================================================================

  def test_enqueue_on_unstarted_writer_spawns_worker_thread_lazily(
      self,
  ) -> None:
    writer = self._create_writer(thread_name="LazyWorkerThread")
    self.assertIsNone(writer._worker_thread)

    writer.enqueue(step_id=1)

    self.assertIsNotNone(writer._worker_thread)
    self.assertTrue(writer._worker_thread.is_alive())
    self.assertTrue(writer._worker_thread.daemon)

  def test_writer_without_thread_name_uses_default_class_worker_name(
      self,
  ) -> None:
    writer = self._create_writer()
    writer.enqueue(step_id=1)

    self.assertIsNotNone(writer._worker_thread)
    self.assertEqual(writer._worker_thread.name, "_TestAsyncWriterWorker")

  def test_writer_with_thread_name_uses_custom_worker_name(self) -> None:
    writer = self._create_writer(thread_name="CustomWorkerThread")
    writer.enqueue(step_id=1)

    self.assertIsNotNone(writer._worker_thread)
    self.assertEqual(writer._worker_thread.name, "CustomWorkerThread")

  def test_enqueue_from_simultaneous_threads_spawns_exactly_one_worker_thread(
      self,
  ) -> None:
    worker_name = "SimultaneousWorkerThread"
    writer = self._create_writer(thread_name=worker_name)
    self.assertIsNone(writer._worker_thread)
    barrier = threading.Barrier(_NUM_RACING_THREADS)

    def racing_enqueue(idx: int) -> None:
      # All threads are released at once, so they race on the lazy spawn.
      barrier.wait()
      writer.enqueue(step_id=idx + 1, run_id=f"run_{idx}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_NUM_RACING_THREADS
    ) as executor:
      futures = [
          executor.submit(racing_enqueue, i) for i in range(_NUM_RACING_THREADS)
      ]
      for f in futures:
        f.result()

    active_workers = [t for t in threading.enumerate() if t.name == worker_name]
    self.assertLen(active_workers, 1)
    worker_thread = writer._worker_thread
    self.assertIsNotNone(worker_thread)
    self.assertIs(active_workers[0], worker_thread)
    self.assertTrue(worker_thread.is_alive())

  def test_enqueue_after_worker_started_reuses_existing_worker_thread(
      self,
  ) -> None:
    worker_name = "ReusedWorkerThread"
    writer = self._create_writer(thread_name=worker_name)
    writer.enqueue(step_id=1)
    initial_worker = writer._worker_thread
    self.assertIsNotNone(initial_worker)

    def later_enqueue(idx: int) -> None:
      writer.enqueue(step_id=idx + 2)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_NUM_RACING_THREADS
    ) as executor:
      futures = [
          executor.submit(later_enqueue, i) for i in range(_NUM_RACING_THREADS)
      ]
      for f in futures:
        f.result()

    active_workers = [t for t in threading.enumerate() if t.name == worker_name]
    self.assertLen(active_workers, 1)
    self.assertIs(writer._worker_thread, initial_worker)

  # ============================================================================
  # 2. Task Enqueue & FIFO Queue Processing
  # ============================================================================

  def test_flush_after_enqueues_processes_all_tasks_in_fifo_order(self) -> None:
    writer = self._create_writer()
    step_ids = [1, 2, 3, 4, 5]
    for step_id in step_ids:
      writer.enqueue(step_id=step_id)
    writer.flush()

    processed_step_ids = [
        t.step.step_id for t in writer.processed_tasks if t.step is not None
    ]
    self.assertEqual(processed_step_ids, step_ids)

  def test_enqueue_preserves_all_task_fields_through_the_queue(self) -> None:
    writer = self._create_writer()
    enqueued_task = writer.enqueue(
        trajectory_id=trajectory_testing.TRAJECTORY_ID_1,
        step_id=1,
        run_id="run_100",
    )
    writer.flush()

    self.assertLen(writer.processed_tasks, 1)
    processed_task = writer.processed_tasks[0]
    self.assertEqual(processed_task, enqueued_task)
    self.assertEqual(processed_task.run_id, "run_100")
    self.assertEqual(
        processed_task.trajectory_id, trajectory_testing.TRAJECTORY_ID_1
    )

  def test_trajectory_id_is_derived_from_task_metadata(self) -> None:
    task = _TestWriteTask(
        metadata=trajectory_testing.make_metadata(trajectory_id="traj_derived")
    )

    self.assertEqual(task.trajectory_id, "traj_derived")

  def test_write_task_projects_and_deep_copies_payload(self) -> None:
    meta = trajectory_testing.TUNIX_METADATA_1.model_copy(deep=True)
    step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(deep=True)
    task = _TestWriteTask(metadata=meta, step=step)

    meta.agent.tool_definitions[0]["name"] = "mutated"
    step.tool_calls[0].arguments["query"] = "mutated"

    self.assertIs(type(task.metadata), trajectory_lib.TrajectoryMetadata)
    self.assertEqual(
        task.metadata, trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()
    )
    self.assertIs(type(task.step), trajectory_lib.Step)
    self.assertEqual(
        task.step, trajectory_testing.TUNIX_AGENT_STEP_1.to_atif_step()
    )

  # ============================================================================
  # 3. Barrier Synchronization (flush)
  # ============================================================================

  def test_flush_on_empty_writer_succeeds(self) -> None:
    writer = self._create_writer()
    writer.flush()

    self.assertIsNone(writer._worker_thread)
    self.assertTrue(writer._queue.empty())
    self.assertEmpty(writer.processed_tasks)
    self.assertFalse(writer.is_closed)

  def test_flush_when_called_multiple_times_is_idempotent(self) -> None:
    writer = self._create_writer()
    writer.enqueue(step_id=1)
    writer.flush()
    writer.flush()

    self.assertLen(writer.processed_tasks, 1)

  def test_flush_when_closed_skips_queue_join(self) -> None:
    writer = self._create_writer()
    writer.close()

    # Flushing a closed writer must return immediately without joining queue.
    with mock.patch.object(writer._queue, "join") as mock_join:
      writer.flush()
      mock_join.assert_not_called()
    self.assertTrue(writer.is_closed)

  # ============================================================================
  # 4. Fault Tolerance & Error Suppression
  # ============================================================================

  def test_process_task_on_failure_continues_processing_remaining_queue(
      self,
  ) -> None:
    writer = self._create_writer(fail_on_step_id=2)

    with mock.patch.object(logging, "exception") as mock_log_exception:
      writer.enqueue(step_id=1)
      writer.enqueue(step_id=2)
      writer.enqueue(step_id=3)
      writer.flush()

    # Steps 1 and 3 should be processed despite failure on step 2.
    processed_step_ids = [
        t.step.step_id for t in writer.processed_tasks if t.step is not None
    ]
    self.assertEqual(processed_step_ids, [1, 3])
    mock_log_exception.assert_called_once_with(
        "%s failed to process task for trajectory_id=%s, step_id=%s.",
        "_TestAsyncWriterWorker",
        trajectory_testing.TRAJECTORY_ID_1,
        2,
    )

  def test_log_task_error_with_step_logs_trajectory_and_step_id(self) -> None:
    writer = self._create_writer()
    task = _TestWriteTask(
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )

    with mock.patch.object(logging, "exception") as mock_log_exc:
      writer._log_task_error(task)

      mock_log_exc.assert_called_once_with(
          "%s failed to process task for trajectory_id=%s, step_id=%s.",
          "_TestAsyncWriterWorker",
          trajectory_testing.TRAJECTORY_ID_1,
          trajectory_testing.STEP_1_1.step_id,
      )

  def test_log_task_error_without_step_logs_none_step_id(self) -> None:
    writer = self._create_writer()
    task = _TestWriteTask(metadata=trajectory_testing.METADATA_1, step=None)

    with mock.patch.object(logging, "exception") as mock_log_exc:
      writer._log_task_error(task)

      mock_log_exc.assert_called_once_with(
          "%s failed to process task for trajectory_id=%s, step_id=%s.",
          "_TestAsyncWriterWorker",
          trajectory_testing.TRAJECTORY_ID_1,
          None,
      )

  def test_worker_loop_when_log_task_error_raises_continues_processing(
      self,
  ) -> None:
    writer = self._create_writer(fail_on_step_id=1)

    with mock.patch.object(
        writer,
        "_log_task_error",
        side_effect=RuntimeError("Logging handler crashed!"),
    ):
      with mock.patch.object(logging, "exception") as mock_log_exc:
        writer.enqueue(step_id=1)
        writer.enqueue(step_id=2)
        writer.flush()

    # Step 2 must be processed even when _log_task_error raised on step 1.
    processed_step_ids = [
        t.step.step_id for t in writer.processed_tasks if t.step is not None
    ]
    self.assertEqual(processed_step_ids, [2])
    mock_log_exc.assert_called_once_with(
        "%s failed to log a task error.", "_TestAsyncWriterWorker"
    )

  # ============================================================================
  # 5. Multi-Threaded Concurrency
  # ============================================================================

  def test_enqueue_from_multiple_threads_processes_all_tasks(self) -> None:
    writer = self._create_writer()
    num_threads = 5
    tasks_per_thread = 20

    def concurrent_enqueue(thread_idx: int) -> None:
      for i in range(tasks_per_thread):
        writer.enqueue(
            trajectory_id=f"{trajectory_testing.TRAJECTORY_ID_1}_{thread_idx}",
            step_id=i + 1,
            run_id=f"run_{thread_idx}",
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      futures = [
          executor.submit(concurrent_enqueue, t) for t in range(num_threads)
      ]
      for f in futures:
        f.result()
    writer.flush()

    self.assertLen(writer.processed_tasks, num_threads * tasks_per_thread)
    self.assertIsNotNone(writer._worker_thread)
    self.assertEqual(
        set(writer.processing_thread_ids), {writer._worker_thread.ident}
    )

  # ============================================================================
  # 6. Shutdown & Destructor Teardown
  # ============================================================================

  def test_close_shuts_down_worker_and_prevents_further_writes(self) -> None:
    writer = self._create_writer()
    writer.enqueue(step_id=1)
    writer.close()

    self.assertLen(writer.processed_tasks, 1)
    self.assertTrue(writer.is_closed)
    self.assertIsNotNone(writer._worker_thread)
    self.assertFalse(writer._worker_thread.is_alive())

    with self.assertRaisesRegex(RuntimeError, r"Cannot write to a closed"):
      writer.enqueue(step_id=2)

    # Calling close again is safe and idempotent.
    writer.close()
    self.assertTrue(writer.is_closed)

  def test_close_concurrent_with_enqueues_processes_every_accepted_task(
      self,
  ) -> None:
    num_enqueue_threads = 8
    num_steps = 10
    accepted_steps: list[tuple[str, int]] = []
    lock = threading.Lock()
    first_task_accepted = threading.Event()
    writer = self._create_writer()

    def enqueue_worker(thread_idx: int) -> None:
      traj_id = f"concurrent_close_traj_{thread_idx}"
      for step_id in range(1, num_steps + 1):
        try:
          writer.enqueue(trajectory_id=traj_id, step_id=step_id)
        except RuntimeError as e:
          if "Cannot write to a closed" in str(e):
            break
          raise
        with lock:
          accepted_steps.append((traj_id, step_id))
        first_task_accepted.set()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_enqueue_threads + 1
    ) as executor:
      enqueue_futures = [
          executor.submit(enqueue_worker, i) for i in range(num_enqueue_threads)
      ]
      # Close once writes are genuinely in flight, without relying on sleeps.
      self.assertTrue(first_task_accepted.wait(_WORKER_START_TIMEOUT_S))
      close_future = executor.submit(writer.close)

      for f in enqueue_futures:
        f.result()
      close_future.result()

    processed_keys = [
        (t.trajectory_id, t.step.step_id)
        for t in writer.processed_tasks
        if t.step is not None
    ]
    self.assertCountEqual(processed_keys, accepted_steps)

  def test_multiple_concurrent_close_calls_safe(self) -> None:
    writer = self._create_writer()
    writer.enqueue(step_id=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
      futures = [executor.submit(writer.close) for _ in range(5)]
      for f in futures:
        f.result()

    self.assertLen(writer.processed_tasks, 1)
    self.assertTrue(writer.is_closed)

  def test_close_when_worker_exceeds_timeout_logs_discarded_trajectories(
      self,
  ) -> None:
    writer = self._create_writer()
    worker_started = threading.Event()
    unblock_worker = threading.Event()
    self.addCleanup(unblock_worker.set)

    def blocking_process_task(task: _TestWriteTask) -> None:
      del task  # Stalls the worker instead of recording the task.
      worker_started.set()
      unblock_worker.wait(timeout=_WORKER_BLOCK_TIMEOUT_S)

    with mock.patch.object(
        writer, "_process_task", side_effect=blocking_process_task
    ):
      writer.enqueue(trajectory_id="traj_in_flight")
      # Barrier: the first task is dequeued and stalled before the remaining
      # tasks are queued, so the discarded set is deterministic.
      self.assertTrue(worker_started.wait(timeout=_WORKER_START_TIMEOUT_S))
      writer.enqueue(trajectory_id=trajectory_testing.TRAJECTORY_ID_2)
      writer.enqueue(trajectory_id="traj_discarded_3")

      with mock.patch.object(logging, "warning") as mock_log_warning:
        writer.close(timeout=_CLOSE_TIMEOUT_S)

    mock_log_warning.assert_called_once_with(
        "%s did not finish within the %s second timeout. Discarded"
        " remaining tasks for trajectory IDs: %s",
        "_TestAsyncWriterWorker",
        _CLOSE_TIMEOUT_S,
        sorted([trajectory_testing.TRAJECTORY_ID_2, "traj_discarded_3"]),
    )
    self.assertTrue(writer.is_closed)

  def test_del_on_unstarted_writer_marks_writer_closed(self) -> None:
    unstarted_writer = _TestAsyncWriter()
    self.assertIsNone(unstarted_writer._worker_thread)

    unstarted_writer.__del__()

    self.assertTrue(unstarted_writer.is_closed)

  def test_del_on_started_writer_drains_queue_and_stops_worker(self) -> None:
    fresh_writer = _TestAsyncWriter()
    fresh_writer.enqueue(step_id=1)
    worker_thread = fresh_writer._worker_thread
    self.assertIsNotNone(worker_thread)
    self.assertTrue(worker_thread.is_alive())

    fresh_writer.__del__()

    self.assertFalse(worker_thread.is_alive())
    self.assertLen(fresh_writer.processed_tasks, 1)


class AsyncWriterShutdownHookTest(parameterized.TestCase):
  """Tests the atexit hook that drains writers still live at process exit."""

  def test_live_writer_is_registered_and_unregistered_on_close(self) -> None:
    writer = _TestAsyncWriter()
    self.assertIn(writer, async_writer._LIVE_WRITERS)

    writer.close()

    self.assertNotIn(writer, async_writer._LIVE_WRITERS)

  def test_pending_writes_persisted_by_shutdown_hook(self) -> None:
    writer = _TestAsyncWriter()
    self.addCleanup(async_writer._LIVE_WRITERS.discard, writer)
    self.addCleanup(writer.close)
    writer.enqueue(step_id=1)

    async_writer._close_live_writers()

    self.assertTrue(writer.is_closed)
    self.assertLen(writer.processed_tasks, 1)

  def test_shutdown_hook_suppresses_close_errors(self) -> None:
    failing_writer = _TestAsyncWriter()
    healthy_writer = _TestAsyncWriter()
    self.addCleanup(async_writer._LIVE_WRITERS.discard, failing_writer)
    self.addCleanup(async_writer._LIVE_WRITERS.discard, healthy_writer)
    self.addCleanup(failing_writer.close)
    self.addCleanup(healthy_writer.close)
    failing_writer.enqueue(trajectory_id=trajectory_testing.TRAJECTORY_ID_1)
    healthy_writer.enqueue(trajectory_id=trajectory_testing.TRAJECTORY_ID_2)

    with mock.patch.object(
        failing_writer, "close", side_effect=RuntimeError("close failed")
    ):
      with mock.patch.object(logging, "exception") as mock_log_exception:
        async_writer._close_live_writers()

    mock_log_exception.assert_called_once()
    self.assertTrue(healthy_writer.is_closed)
    self.assertLen(healthy_writer.processed_tasks, 1)


if __name__ == "__main__":
  absltest.main()
