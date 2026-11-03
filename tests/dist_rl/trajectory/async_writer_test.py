"""Unit tests verifying lifecycle and queue mechanics of AsyncWriter."""

from collections.abc import Sequence
import concurrent.futures
import dataclasses
import threading
from typing import Final
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.dist_rl.trajectory import async_writer
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing

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
      batchable_error_step_ids: Sequence[int] = (),
      batch_error_on_call: int | None = None,
      block_on_batch_call: int | None = None,
      max_queue_size: int = async_writer.DEFAULT_MAX_QUEUE_SIZE,
      batching: async_writer.BatchConfig | None = (
          async_writer.DEFAULT_BATCH_CONFIG
      ),
  ):
    super().__init__(
        thread_name=thread_name,
        max_queue_size=max_queue_size,
        batching=batching,
    )
    self.processed_tasks: list[_TestWriteTask] = []
    self.processed_batches: list[tuple[_TestWriteTask, ...]] = []
    self.processing_thread_ids: list[int] = []
    self.fail_on_step_id = fail_on_step_id
    self.batchable_error_step_ids: set[int] = set(batchable_error_step_ids)
    self.batch_error_on_call = batch_error_on_call
    self.block_on_batch_call = block_on_batch_call
    self.batch_started_event = threading.Event()
    self.unblock_batch_event = threading.Event()
    self._batch_call_count = 0

  def enqueue(
      self,
      trajectory_id: str = trajectory_testing.TRAJECTORY_ID_1,
      step_id: int | None = None,
      run_id: str | None = None,
      metadata: trajectory_lib.TrajectoryMetadata | None = None,
      metrics: trajectory_lib.TrajectoryMetrics | None = None,
  ) -> _TestWriteTask:
    """Builds a task and submits it through the engine's `_enqueue` entry."""
    if metrics is not None:
      task = _TestWriteTask(
          trajectory_id=trajectory_id,
          metrics=metrics,
          run_id=run_id,
      )
    else:
      task = _TestWriteTask(
          metadata=(
              metadata
              if metadata is not None
              else trajectory_testing.make_metadata(trajectory_id=trajectory_id)
          ),
          step=(
              trajectory_testing.make_step(step_id=step_id)
              if step_id is not None
              else None
          ),
          run_id=run_id,
      )
    self._enqueue(task)
    return task

  def _process_batch(self, tasks: Sequence[_TestWriteTask]) -> None:
    self._batch_call_count += 1
    if self._batch_call_count == self.block_on_batch_call:
      self.batch_started_event.set()
      self.unblock_batch_event.wait(timeout=_WORKER_BLOCK_TIMEOUT_S)
    if self._batch_call_count == self.batch_error_on_call:
      raise RuntimeError("429 ResourceExhausted: batch rate limit exceeded")
    self.processed_batches.append(tuple(tasks))
    super()._process_batch(tasks)

  def _should_batch_on_error(self, exc: BaseException) -> bool:
    return "429" in str(exc)

  def _process_task(self, task: _TestWriteTask) -> None:
    if task.step is not None and task.step.step_id is not None:
      if task.step.step_id in self.batchable_error_step_ids:
        raise RuntimeError(f"429 ResourceExhausted on step {task.step.step_id}")
      if (
          self.fail_on_step_id is not None
          and task.step.step_id == self.fail_on_step_id
      ):
        raise ValueError(
            f"Intentional test failure on step {task.step.step_id}"
        )
    self.processing_thread_ids.append(threading.get_ident())
    self.processed_tasks.append(task)


class AsyncWriterTest(trajectory_testing.TrajectoryTestCase):
  """Unit tests for AsyncWriter lifecycle and queue mechanics."""

  def _create_writer(
      self,
      thread_name: str | None = None,
      fail_on_step_id: int | None = None,
      max_queue_size: int = async_writer.DEFAULT_MAX_QUEUE_SIZE,
      batching: async_writer.BatchConfig | None = (
          async_writer.DEFAULT_BATCH_CONFIG
      ),
  ) -> _TestAsyncWriter:
    """Creates a test writer with guaranteed cleanup on test completion."""
    writer = _TestAsyncWriter(
        thread_name=thread_name,
        fail_on_step_id=fail_on_step_id,
        max_queue_size=max_queue_size,
        batching=batching,
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

  def test_write_task_deep_copies_on_init_and_projects_on_worker(self) -> None:
    writer = self._create_writer()
    meta = trajectory_testing.TUNIX_METADATA_1.model_copy(deep=True)
    step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(deep=True)
    task = _TestWriteTask(metadata=meta, step=step)

    # Mutate base ATIF containers.
    meta.agent.tool_definitions[0]["name"] = "mutated_tool"
    meta.agent.tool_definitions.append({"name": "extra_tool"})
    meta.final_metrics.extra["final_key"] = "mutated_final"
    meta.extra["user_meta"] = "mutated_meta"
    step.tool_calls[0].arguments["query"] = "mutated_query"
    step.observation.results[0].extra["obs_res_key"] = "mutated_obs"
    step.metrics.prompt_token_ids.extend([101, 202, 303])
    step.extra["user_key"] = "mutated_step"

    # Mutate Tunix subclass containers (packed into TUNIX_EXTENSIONS_KEY).
    meta.target_policy_versions.extend([4, 5, 6])
    meta.hyperparams["temperature"] = 1.5
    meta.env_time["step_0"] = 12.3
    meta.reward_time["step_1"] = 45.6
    step.assistant_tokens[0] = 777
    step.assistant_masks[0] = 0
    step.logprobs[0] = -9.9

    # Caller thread only deep-copies; Tunix subclass types remain unprojected.
    self.assertIs(type(task.metadata), trajectory_lib.TunixTrajectoryMetadata)
    self.assertEqual(task.metadata, trajectory_testing.TUNIX_METADATA_1)
    self.assertStepEqual(task.step, trajectory_testing.TUNIX_AGENT_STEP_1)

    # Worker thread projects to base ATIF when draining the queue.
    writer._enqueue(task)
    writer.flush()

    self.assertLen(writer.processed_tasks, 1)
    processed_task = writer.processed_tasks[0]
    self.assertIs(
        type(processed_task.metadata), trajectory_lib.TrajectoryMetadata
    )
    self.assertEqual(
        processed_task.metadata,
        trajectory_testing.TUNIX_METADATA_1.to_atif_metadata(),
    )
    self.assertIs(type(processed_task.step), trajectory_lib.Step)
    self.assertEqual(
        processed_task.step,
        trajectory_testing.TUNIX_AGENT_STEP_1.to_atif_step(),
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
        "%s failed to write %s for trajectory_id=%s.",
        "_TestAsyncWriterWorker",
        "step (step_id=2)",
        trajectory_testing.TRAJECTORY_ID_1,
    )

  def test_log_task_error_with_step_logs_step_task_type(self) -> None:
    writer = self._create_writer()
    task = _TestWriteTask(
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )

    with mock.patch.object(logging, "exception") as mock_log_exc:
      writer._log_task_error(task)

      mock_log_exc.assert_called_once_with(
          "%s failed to write %s for trajectory_id=%s.",
          "_TestAsyncWriterWorker",
          f"step (step_id={trajectory_testing.STEP_1_1.step_id})",
          trajectory_testing.TRAJECTORY_ID_1,
      )

  def test_log_task_error_without_step_logs_metadata_task_type(self) -> None:
    writer = self._create_writer()
    task = _TestWriteTask(metadata=trajectory_testing.METADATA_1, step=None)

    with mock.patch.object(logging, "exception") as mock_log_exc:
      writer._log_task_error(task)

      mock_log_exc.assert_called_once_with(
          "%s failed to write %s for trajectory_id=%s.",
          "_TestAsyncWriterWorker",
          "metadata",
          trajectory_testing.TRAJECTORY_ID_1,
      )

  def test_log_task_error_with_metrics_logs_metrics_task_type(self) -> None:
    writer = self._create_writer()
    task = _TestWriteTask(
        trajectory_id=trajectory_testing.TRAJECTORY_ID_1,
        metrics=trajectory_lib.TrajectoryMetrics(extra={"score": 100}),
    )

    with mock.patch.object(logging, "exception") as mock_log_exc:
      writer._log_task_error(task)

      mock_log_exc.assert_called_once_with(
          "%s failed to write %s for trajectory_id=%s.",
          "_TestAsyncWriterWorker",
          "metrics",
          trajectory_testing.TRAJECTORY_ID_1,
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

  # ============================================================================
  # 7. Bounded Queue Capacity & Metadata Reserve
  # ============================================================================

  def test_writer_without_max_queue_size_uses_default_queue_limits(
      self,
  ) -> None:
    writer = self._create_writer()

    self.assertEqual(writer.max_queue_size, async_writer.DEFAULT_MAX_QUEUE_SIZE)
    self.assertEqual(writer.step_queue_limit, 8_000)
    self.assertEqual(writer.metrics_queue_limit, 9_500)

  @parameterized.parameters((10, 8, 9), (1, 1, 1))
  def test_writer_with_custom_max_queue_size_sets_queue_limits(
      self,
      max_queue_size: int,
      expected_step_limit: int,
      expected_metrics_limit: int,
  ) -> None:
    writer = self._create_writer(max_queue_size=max_queue_size)

    self.assertEqual(writer.max_queue_size, max_queue_size)
    self.assertEqual(writer.step_queue_limit, expected_step_limit)
    self.assertEqual(writer.metrics_queue_limit, expected_metrics_limit)

  def test_write_task_metrics_only(self) -> None:
    metrics = trajectory_lib.TrajectoryMetrics(extra={"scalar": 42.0})
    task = async_writer.WriteTask(
        trajectory_id="metrics_traj",
        metrics=metrics,
    )
    self.assertEqual(task.trajectory_id, "metrics_traj")
    self.assertIsNone(task.metadata)
    self.assertIsNone(task.step)
    self.assertEqual(task.metrics, metrics)
    task.to_atif()
    self.assertEqual(task.trajectory_id, "metrics_traj")
    self.assertEqual(task.metrics, metrics)

  def test_write_task_missing_trajectory_id_raises(self) -> None:
    metrics = trajectory_lib.TrajectoryMetrics(extra={})
    with self.assertRaises(ValueError):
      async_writer.WriteTask(metrics=metrics)

  def test_enqueue_metrics_task_enqueues_and_processes(self) -> None:
    writer = self._create_writer()
    metrics = trajectory_lib.TrajectoryMetrics(extra={"reward": 1.0})
    writer.enqueue(
        trajectory_id=trajectory_testing.TRAJECTORY_ID_1, metrics=metrics
    )
    writer.flush()

    self.assertLen(writer.processed_tasks, 1)
    task = writer.processed_tasks[0]
    self.assertEqual(task.trajectory_id, trajectory_testing.TRAJECTORY_ID_1)
    self.assertEqual(task.metrics, metrics)
    self.assertIsNone(task.metadata)
    self.assertIsNone(task.step)

  def test_write_task_snapshots_nested_dicts_and_numpy_arrays(self) -> None:
    writer = self._create_writer()
    arr = np.array([1.0, 2.0, 3.0])
    metrics = trajectory_lib.TrajectoryMetrics(
        extra={"arr": arr, "nested": {"val": 10}}
    )
    writer.enqueue(
        trajectory_id=trajectory_testing.TRAJECTORY_ID_1, metrics=metrics
    )

    # Mutate caller's objects
    arr[0] = 999.0
    self.assertIsNotNone(metrics.extra)
    metrics.extra["nested"]["val"] = 999
    writer.flush()

    self.assertLen(writer.processed_tasks, 1)
    task_metrics = writer.processed_tasks[0].metrics
    self.assertIsNotNone(task_metrics)
    stored_extra = task_metrics.extra
    self.assertIsNotNone(stored_extra)
    self.assertEqual(stored_extra["nested"]["val"], 10)
    np.testing.assert_array_equal(
        stored_extra["arr"], np.array([1.0, 2.0, 3.0])
    )

  def test_write_task_invalid_metrics_type_raises_type_error(self) -> None:
    with self.assertRaises(TypeError):
      async_writer.WriteTask(
          trajectory_id=trajectory_testing.TRAJECTORY_ID_1,
          metrics={"not": "metrics"},  # type: ignore
      )

  @parameterized.parameters("", "   ")
  def test_write_task_invalid_trajectory_id_raises_value_error(
      self, invalid_id: str
  ) -> None:
    metrics = trajectory_lib.TrajectoryMetrics(extra={})
    with self.assertRaises(ValueError):
      async_writer.WriteTask(trajectory_id=invalid_id, metrics=metrics)

  def test_enqueue_on_closed_writer_raises_runtime_error(self) -> None:
    writer = self._create_writer()
    writer.close()
    metrics = trajectory_lib.TrajectoryMetrics(extra={})
    with self.assertRaisesRegex(
        RuntimeError, r"Cannot write to a closed writer"
    ):
      writer.enqueue(
          trajectory_id=trajectory_testing.TRAJECTORY_ID_1, metrics=metrics
      )

  def test_three_tier_watermark_queue_admission(self) -> None:
    writer = self._create_writer(max_queue_size=10)
    worker_started = threading.Event()
    unblock_worker = threading.Event()
    self.addCleanup(unblock_worker.set)
    original_process_task = writer._process_task

    def stalled_process_task(task: _TestWriteTask) -> None:
      if not worker_started.is_set():
        worker_started.set()
        unblock_worker.wait(timeout=_WORKER_BLOCK_TIMEOUT_S)
      original_process_task(task)

    with mock.patch.object(
        writer, "_process_task", side_effect=stalled_process_task
    ):
      # Task 1 dequeued and stalls worker; tasks 2-9 fill 8 step slots.
      writer.enqueue(step_id=1)
      self.assertTrue(worker_started.wait(timeout=_WORKER_START_TIMEOUT_S))
      for i in range(2, 10):
        writer.enqueue(step_id=i)

      # 9th slot in queue (current_size=8 >= step_queue_limit=8):
      with mock.patch.object(logging, "warning") as mock_log_warning:
        writer.enqueue(step_id=10)
      mock_log_warning.assert_called_once_with(
          "%s step queue reached capacity (%d/%d); dropped step_id=%s"
          " for trajectory_id=%s.",
          "_TestAsyncWriterWorker",
          8,
          8,
          10,
          trajectory_testing.TRAJECTORY_ID_1,
      )

      # Metrics task is admitted into the 9th slot (current_size=8 < 9).
      metrics = trajectory_lib.TrajectoryMetrics(extra={"score": 100})
      writer.enqueue(
          trajectory_id=trajectory_testing.TRAJECTORY_ID_1, metrics=metrics
      )

      # At current_size=9 (>= metrics_queue_limit=9), metrics task is dropped.
      with mock.patch.object(logging, "warning") as mock_log_warning:
        writer.enqueue(
            trajectory_id=trajectory_testing.TRAJECTORY_ID_2, metrics=metrics
        )
      mock_log_warning.assert_called_once_with(
          "%s metrics queue reached capacity (%d/%d); dropped metrics"
          " for trajectory_id=%s.",
          "_TestAsyncWriterWorker",
          9,
          9,
          trajectory_testing.TRAJECTORY_ID_2,
      )

      # Metadata task is admitted into the 10th slot (current_size=9 < 10).
      completed_meta = trajectory_testing.make_metadata(
          trajectory_id="completed_traj",
          session_id="session_end",
      )
      writer.enqueue(metadata=completed_meta)

      # At current_size=10 (>= max_queue_size=10), metadata task is dropped.
      with mock.patch.object(logging, "warning") as mock_log_warning:
        writer.enqueue(metadata=completed_meta)
      mock_log_warning.assert_called_once_with(
          "%s queue is full (%d/%d); dropped metadata task for"
          " trajectory_id=%s.",
          "_TestAsyncWriterWorker",
          10,
          10,
          "completed_traj",
      )

      unblock_worker.set()
      writer.flush()

    step_ids = [
        t.step.step_id for t in writer.processed_tasks if t.step is not None
    ]
    self.assertEqual(step_ids, list(range(1, 10)))
    metrics_tasks = [t for t in writer.processed_tasks if t.metrics is not None]
    self.assertLen(metrics_tasks, 1)
    metadata_tasks = [
        t
        for t in writer.processed_tasks
        if t.metadata is not None and t.step is None and t.metrics is None
    ]
    self.assertLen(metadata_tasks, 1)
    self.assertIsNotNone(metadata_tasks[0].metadata)
    self.assertEqual(metadata_tasks[0].metadata.session_id, "session_end")

  @parameterized.parameters(0, -1, -100)
  def test_writer_with_non_positive_max_queue_size_raises_value_error(
      self, invalid_max_queue_size: int
  ) -> None:
    with self.assertRaisesRegex(ValueError, r"max_queue_size must be positive"):
      _TestAsyncWriter(max_queue_size=invalid_max_queue_size)

  def test_enqueue_when_step_queue_is_full_drops_step_and_admits_metadata(
      self,
  ) -> None:
    # With max_queue_size=4, step_queue_limit is int(4 * 0.8) = 3, leaving 1
    # reserved slot for metadata-only tasks (step=None).
    writer = self._create_writer(max_queue_size=4)
    worker_started = threading.Event()
    unblock_worker = threading.Event()
    self.addCleanup(unblock_worker.set)
    original_process_task = writer._process_task

    def stalled_process_task(task: _TestWriteTask) -> None:
      if not worker_started.is_set():
        worker_started.set()
        unblock_worker.wait(timeout=_WORKER_BLOCK_TIMEOUT_S)
      original_process_task(task)

    with mock.patch.object(
        writer, "_process_task", side_effect=stalled_process_task
    ):
      # Task 1 is dequeued immediately and stalls the worker; tasks 2-4 fill
      # the 3 step slots in the queue.
      writer.enqueue(step_id=1)
      self.assertTrue(worker_started.wait(timeout=_WORKER_START_TIMEOUT_S))
      writer.enqueue(step_id=2)
      writer.enqueue(step_id=3)
      writer.enqueue(step_id=4)

      with mock.patch.object(logging, "warning") as mock_log_warning:
        writer.enqueue(step_id=5)

      # A metadata-only update (step_id=None) is still admitted into the
      # reserved 4th slot.
      completed_meta = trajectory_testing.make_metadata(
          trajectory_id=trajectory_testing.TRAJECTORY_ID_1,
          session_id="completed_session",
      )
      writer.enqueue(metadata=completed_meta)
      unblock_worker.set()
      writer.flush()

    mock_log_warning.assert_called_once_with(
        "%s step queue reached capacity (%d/%d); dropped step_id=%s"
        " for trajectory_id=%s.",
        "_TestAsyncWriterWorker",
        3,
        3,
        5,
        trajectory_testing.TRAJECTORY_ID_1,
    )
    processed_step_ids = [
        t.step.step_id for t in writer.processed_tasks if t.step is not None
    ]
    self.assertEqual(processed_step_ids, [1, 2, 3, 4])
    self.assertIsNone(writer.processed_tasks[-1].step)
    self.assertEqual(
        writer.processed_tasks[-1].metadata.session_id, "completed_session"
    )

  def test_enqueue_when_queue_is_full_drops_metadata_task(self) -> None:
    writer = self._create_writer(max_queue_size=2)
    worker_started = threading.Event()
    unblock_worker = threading.Event()
    self.addCleanup(unblock_worker.set)
    original_process_task = writer._process_task

    def stalled_process_task(task: _TestWriteTask) -> None:
      if not worker_started.is_set():
        worker_started.set()
        unblock_worker.wait(timeout=_WORKER_BLOCK_TIMEOUT_S)
      original_process_task(task)

    with mock.patch.object(
        writer, "_process_task", side_effect=stalled_process_task
    ):
      writer.enqueue(step_id=1)
      self.assertTrue(worker_started.wait(timeout=_WORKER_START_TIMEOUT_S))
      # Fill queue to max_queue_size=2 (1 step slot + 1 metadata reserve slot).
      writer.enqueue(step_id=2)
      writer.enqueue(trajectory_id=trajectory_testing.TRAJECTORY_ID_1)

      with mock.patch.object(logging, "warning") as mock_log_warning:
        writer.enqueue(trajectory_id=trajectory_testing.TRAJECTORY_ID_2)

      unblock_worker.set()
      writer.flush()

    mock_log_warning.assert_called_once_with(
        "%s queue is full (%d/%d); dropped metadata task for trajectory_id=%s.",
        "_TestAsyncWriterWorker",
        2,
        2,
        trajectory_testing.TRAJECTORY_ID_2,
    )
    self.assertEqual(
        [t.trajectory_id for t in writer.processed_tasks],
        [trajectory_testing.TRAJECTORY_ID_1] * 3,
    )


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


class AsyncWriterBatchingTest(trajectory_testing.TrajectoryTestCase):
  """Unit tests for BatchConfig, TrajectoryBatch, and adaptive batching."""

  def _create_writer(
      self,
      batching: async_writer.BatchConfig | None = (
          async_writer.DEFAULT_BATCH_CONFIG
      ),
      fail_on_step_id: int | None = None,
      batchable_error_step_ids: Sequence[int] = (),
      batch_error_on_call: int | None = None,
      block_on_batch_call: int | None = None,
  ) -> _TestAsyncWriter:
    """Creates a test writer with guaranteed cleanup on test completion."""
    writer = _TestAsyncWriter(
        batching=batching,
        fail_on_step_id=fail_on_step_id,
        batchable_error_step_ids=batchable_error_step_ids,
        batch_error_on_call=batch_error_on_call,
        block_on_batch_call=block_on_batch_call,
    )
    self.addCleanup(writer.close)
    self.addCleanup(writer.unblock_batch_event.set)
    return writer

  @parameterized.named_parameters(
      dict(
          testcase_name="defaults",
          kwargs={},
          expected_max_items=32,
          expected_max_wait_s=0.25,
      ),
      dict(
          testcase_name="custom_limits",
          kwargs={"max_items": 8, "max_wait_s": 0.5},
          expected_max_items=8,
          expected_max_wait_s=0.5,
      ),
      dict(
          testcase_name="zero_max_wait_s",
          kwargs={"max_items": 16, "max_wait_s": 0.0},
          expected_max_items=16,
          expected_max_wait_s=0.0,
      ),
  )
  def test_batch_config_init_with_valid_limits_sets_expected_fields(
      self,
      kwargs: dict[str, int | float],
      expected_max_items: int,
      expected_max_wait_s: float,
  ) -> None:
    config = async_writer.BatchConfig(**kwargs)

    self.assertEqual(config.max_items, expected_max_items)
    self.assertEqual(config.max_wait_s, expected_max_wait_s)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_max_items",
          max_items=0,
          max_wait_s=0.25,
          expected_error=r"BatchConfig\.max_items must be positive",
      ),
      dict(
          testcase_name="negative_max_items",
          max_items=-1,
          max_wait_s=0.25,
          expected_error=r"BatchConfig\.max_items must be positive",
      ),
      dict(
          testcase_name="negative_max_wait_s",
          max_items=8,
          max_wait_s=-0.1,
          expected_error=r"BatchConfig\.max_wait_s must be non-negative",
      ),
  )
  def test_batch_config_init_with_invalid_limits_raises_value_error(
      self,
      max_items: int,
      max_wait_s: float,
      expected_error: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, expected_error):
      async_writer.BatchConfig(
          max_items=max_items,
          max_wait_s=max_wait_s,
      )

  def test_writer_init_with_none_batching_uses_default_batch_config(
      self,
  ) -> None:
    writer = self._create_writer(batching=None)

    self.assertEqual(writer.batching, async_writer.DEFAULT_BATCH_CONFIG)

  def test_group_tasks_by_trajectory_with_empty_sequence_returns_empty_tuple(
      self,
  ) -> None:
    self.assertEmpty(async_writer.group_tasks_by_trajectory([]))

  def test_group_tasks_by_trajectory_with_mixed_tasks_groups_by_run_and_id(
      self,
  ) -> None:
    meta_1_v1 = trajectory_testing.make_metadata(
        trajectory_id="traj_1", session_id="s1_v1"
    )
    meta_1_v2 = trajectory_testing.make_metadata(
        trajectory_id="traj_1", session_id="s1_v2"
    )
    meta_2_v1 = trajectory_testing.make_metadata(
        trajectory_id="traj_2", session_id="s2_v1"
    )
    meta_1_run_b = trajectory_testing.make_metadata(
        trajectory_id="traj_1", session_id="s1_run_b"
    )

    t1_step1 = _TestWriteTask(
        metadata=meta_1_v1,
        step=trajectory_testing.make_step(step_id=1),
        run_id="run_a",
    )
    t2_step1 = _TestWriteTask(
        metadata=meta_2_v1,
        step=trajectory_testing.make_step(step_id=1),
        run_id="run_a",
    )
    t1_step2 = _TestWriteTask(
        metadata=meta_1_v2,
        step=trajectory_testing.make_step(step_id=2),
        run_id="run_a",
    )
    t1_metrics1 = _TestWriteTask(
        trajectory_id="traj_1",
        metrics=trajectory_lib.TrajectoryMetrics(
            extra={"reward": 0.5, "kl": 0.1}
        ),
        run_id="run_a",
    )
    t1_metrics2 = _TestWriteTask(
        trajectory_id="traj_1",
        metrics=trajectory_lib.TrajectoryMetrics(
            extra={"reward": 1.0, "entropy": 0.2}
        ),
        run_id="run_a",
    )
    t2_metrics1 = _TestWriteTask(
        trajectory_id="traj_2",
        metrics=trajectory_lib.TrajectoryMetrics(extra={"reward": -1.0}),
        run_id="run_a",
    )
    t1_run_b_step1 = _TestWriteTask(
        metadata=meta_1_run_b,
        step=trajectory_testing.make_step(step_id=1),
        run_id="run_b",
    )

    trajectory_batches = async_writer.group_tasks_by_trajectory([
        t1_step1,
        t2_step1,
        t1_step2,
        t1_metrics1,
        t1_metrics2,
        t2_metrics1,
        t1_run_b_step1,
    ])

    self.assertLen(trajectory_batches, 3)

    traj_batch_1, traj_batch_2, traj_batch_1_run_b = trajectory_batches
    self.assertEqual(traj_batch_1.trajectory_id, "traj_1")
    self.assertEqual(traj_batch_1.run_id, "run_a")
    self.assertIsNotNone(traj_batch_1.latest_metadata_task)
    self.assertEqual(
        traj_batch_1.latest_metadata_task.metadata.session_id, "s1_v2"
    )
    self.assertEqual(
        [t.step.step_id for t in traj_batch_1.step_tasks if t.step is not None],
        [1, 2],
    )
    self.assertIsNotNone(traj_batch_1.merged_queued_metrics_task)
    self.assertEqual(
        traj_batch_1.merged_queued_metrics_task.metrics.extra,
        {"reward": 1.0, "kl": 0.1, "entropy": 0.2},
    )

    # Distinct trajectory `traj_2` retains its own latest metadata and metrics.
    self.assertEqual(traj_batch_2.trajectory_id, "traj_2")
    self.assertEqual(traj_batch_2.run_id, "run_a")
    self.assertIsNotNone(traj_batch_2.latest_metadata_task)
    self.assertEqual(
        traj_batch_2.latest_metadata_task.metadata.session_id, "s2_v1"
    )
    self.assertEqual(
        [t.step.step_id for t in traj_batch_2.step_tasks if t.step is not None],
        [1],
    )
    self.assertIsNotNone(traj_batch_2.merged_queued_metrics_task)
    self.assertEqual(
        traj_batch_2.merged_queued_metrics_task.metrics.extra, {"reward": -1.0}
    )

    # Same trajectory_id `traj_1` under a different `run_id` stays isolated.
    self.assertEqual(traj_batch_1_run_b.trajectory_id, "traj_1")
    self.assertEqual(traj_batch_1_run_b.run_id, "run_b")
    self.assertIsNotNone(traj_batch_1_run_b.latest_metadata_task)
    self.assertEqual(
        traj_batch_1_run_b.latest_metadata_task.metadata.session_id,
        "s1_run_b",
    )

  def test_group_tasks_by_trajectory_with_metrics_and_step_only_sets_fields(
      self,
  ) -> None:
    metrics_none_1 = _TestWriteTask(
        trajectory_id="traj_metrics_none",
        metrics=trajectory_lib.TrajectoryMetrics(extra=None),
    )
    metrics_none_2 = _TestWriteTask(
        trajectory_id="traj_metrics_none",
        metrics=trajectory_lib.TrajectoryMetrics(extra=None),
    )
    step_only = _TestWriteTask(
        metadata=trajectory_testing.make_metadata(
            trajectory_id="traj_step_only"
        ),
        step=trajectory_testing.make_step(step_id=1),
    )

    none_batch, step_batch = async_writer.group_tasks_by_trajectory(
        [metrics_none_1, metrics_none_2, step_only]
    )

    self.assertIsNone(none_batch.latest_metadata_task)
    self.assertEmpty(none_batch.step_tasks)
    self.assertIsNotNone(none_batch.merged_queued_metrics_task)
    self.assertIsNone(none_batch.merged_queued_metrics_task.metrics.extra)

    self.assertIsNotNone(step_batch.latest_metadata_task)
    self.assertLen(step_batch.step_tasks, 1)
    self.assertIsNone(step_batch.merged_queued_metrics_task)

  def test_group_tasks_by_trajectory_merges_queued_metrics_and_preserves_empty_extra(
      self,
  ) -> None:
    class _CustomTrajectoryMetrics(trajectory_lib.TrajectoryMetrics):
      reward: float | None = None
      kl_divergence: float | None = None

    empty_extra_1 = _TestWriteTask(
        trajectory_id="traj_empty_extra",
        metrics=trajectory_lib.TrajectoryMetrics(extra={}),
    )
    empty_extra_2 = _TestWriteTask(
        trajectory_id="traj_empty_extra",
        metrics=trajectory_lib.TrajectoryMetrics(extra=None),
    )
    custom_metrics_1 = _TestWriteTask(
        trajectory_id="traj_custom_metrics",
        metrics=_CustomTrajectoryMetrics(
            reward=1.5, kl_divergence=0.1, extra={"a": 1}
        ),
    )
    custom_metrics_2 = _TestWriteTask(
        trajectory_id="traj_custom_metrics",
        metrics=_CustomTrajectoryMetrics(reward=2.0, extra={"b": 2}),
    )

    empty_extra_batch, custom_batch = async_writer.group_tasks_by_trajectory([
        empty_extra_1,
        empty_extra_2,
        custom_metrics_1,
        custom_metrics_2,
    ])

    self.assertIsNotNone(empty_extra_batch.merged_queued_metrics_task)
    self.assertEqual(
        empty_extra_batch.merged_queued_metrics_task.metrics.extra, {}
    )
    self.assertIsNotNone(custom_batch.merged_queued_metrics_task)
    self.assertEqual(
        custom_batch.merged_queued_metrics_task.metrics,
        _CustomTrajectoryMetrics(
            reward=2.0, kl_divergence=0.1, extra={"a": 1, "b": 2}
        ),
    )

  def test_should_batch_on_error_with_non_batchable_error_keeps_zero_wait(
      self,
  ) -> None:
    writer = self._create_writer(fail_on_step_id=1)

    self.assertFalse(
        async_writer.AsyncWriter._should_batch_on_error(
            writer, RuntimeError("429 ResourceExhausted")
        )
    )

    with self.assertLogs(level=logging.ERROR) as log_output:
      writer.enqueue(step_id=1)
      writer.flush()

    self.assertTrue(
        any(
            "failed to write step (step_id=1)" in msg
            for msg in log_output.output
        )
    )
    # Non-batchable ValueError on step 1 leaves `batch_wait_s` at 0.0.
    self.assertEqual(writer.batch_wait_s, 0.0)

  def test_enqueue_when_batch_wait_is_zero_writes_single_task_immediately(
      self,
  ) -> None:
    writer = self._create_writer(block_on_batch_call=1)
    self.assertEqual(writer.batching, async_writer.DEFAULT_BATCH_CONFIG)
    self.assertEqual(writer.batch_wait_s, 0.0)

    writer.enqueue(step_id=1)
    self.assertTrue(
        writer.batch_started_event.wait(timeout=_WORKER_START_TIMEOUT_S)
    )
    # Enqueue 3 more tasks while task 1 is executing at `batch_wait_s == 0.0`.
    writer.enqueue(step_id=2)
    writer.enqueue(step_id=3)
    writer.enqueue(step_id=4)
    writer.unblock_batch_event.set()
    writer.flush()

    # In default single-task mode (`batch_wait_s == 0.0`), each task is
    # processed 1-by-1 immediately without multi-item batching or waiting.
    self.assertEqual(writer.batch_wait_s, 0.0)
    self.assertLen(writer.processed_batches, 4)
    self.assertEqual([len(b) for b in writer.processed_batches], [1] * 4)
    self.assertEqual(
        [t.step.step_id for t in writer.processed_tasks if t.step is not None],
        [1, 2, 3, 4],
    )

  def test_enqueue_after_batchable_error_batches_tasks_and_decays_wait_to_zero(
      self,
  ) -> None:
    config = async_writer.BatchConfig(max_items=4, max_wait_s=0.25)
    writer = self._create_writer(
        batching=config,
        batch_error_on_call=1,
        block_on_batch_call=2,
    )

    with self.assertLogs(level=logging.ERROR) as log_output:
      # 1. Step 1 triggers a batch-level 429 error (`batch_wait_s` -> 0.05s).
      writer.enqueue(step_id=1)
      writer.flush()
      self.assertAlmostEqual(writer.batch_wait_s, 0.05)

      # 2. Step 2 starts batch 2; while it runs, enqueue steps 3..6 so the
      # worker collects all 4 together up to `max_items=4` on batch 3.
      writer.enqueue(step_id=2)
      self.assertTrue(
          writer.batch_started_event.wait(timeout=_WORKER_START_TIMEOUT_S)
      )
      for step_id in (3, 4, 5, 6):
        writer.enqueue(step_id=step_id)
      writer.unblock_batch_event.set()
      writer.flush()

      # 2 clean batches completed (batch 2 and batch 3) -> `0.03s`.
      self.assertAlmostEqual(writer.batch_wait_s, 0.03)
      self.assertEqual([len(b) for b in writer.processed_batches], [1, 4])

      # 3. Complete 3 more clean batches to decay `batch_wait_s` to `0.0`.
      for step_id in (7, 8, 9):
        writer.enqueue(step_id=step_id)
        writer.flush()

    self.assertTrue(
        any(
            "failed to process a write batch" in msg
            for msg in log_output.output
        )
    )
    self.assertEqual(writer.batch_wait_s, 0.0)

  def test_increase_and_decrease_batch_wait_adjust_wait_s_and_cap_at_max(
      self,
  ) -> None:
    writer = self._create_writer(
        batching=async_writer.BatchConfig(max_items=8, max_wait_s=0.25)
    )

    # 1 batchable error -> +0.05s (`batch_wait_s == 0.05s`).
    writer._increase_batch_wait()
    self.assertAlmostEqual(writer.batch_wait_s, 0.05)

    # 3 clean batches -> -0.03s (`batch_wait_s == 0.02s`).
    for _ in range(3):
      writer._decrease_batch_wait()
    self.assertAlmostEqual(writer.batch_wait_s, 0.02)

    # Another error before reaching 0.0 -> adds +0.05s (`0.07s`),
    # requiring 7 clean batches (instead of 5) to decay back to 0.0.
    writer._increase_batch_wait()
    self.assertAlmostEqual(writer.batch_wait_s, 0.07)

    # 5 more errors saturate and cap `batch_wait_s` at `max_wait_s` (`0.25s`).
    for _ in range(5):
      writer._increase_batch_wait()
    self.assertAlmostEqual(writer.batch_wait_s, 0.25)

    # 25 clean batches decay `0.25s` by `0.01s` per batch back to `0.0s`.
    for _ in range(25):
      writer._decrease_batch_wait()
    self.assertEqual(writer.batch_wait_s, 0.0)

    # Extra `_decrease_batch_wait()` at `0.0s` is a no-op.
    writer._decrease_batch_wait()
    self.assertEqual(writer.batch_wait_s, 0.0)

  def test_increase_batch_wait_with_zero_max_wait_s_keeps_zero_wait(
      self,
  ) -> None:
    writer = self._create_writer(
        batching=async_writer.BatchConfig(max_items=8, max_wait_s=0.0)
    )

    writer._increase_batch_wait()

    self.assertEqual(writer.batch_wait_s, 0.0)

  def test_collect_next_batch_when_max_items_reached_splits_batches(
      self,
  ) -> None:
    writer = self._create_writer(
        batching=async_writer.BatchConfig(max_items=2, max_wait_s=0.25),
        block_on_batch_call=1,
    )
    writer._increase_batch_wait()

    writer.enqueue(step_id=1)
    self.assertTrue(
        writer.batch_started_event.wait(timeout=_WORKER_START_TIMEOUT_S)
    )
    for step_id in (2, 3, 4, 5):
      writer.enqueue(step_id=step_id)
    writer.unblock_batch_event.set()
    writer.flush()

    # Task 1 runs in batch 1; tasks 2..5 split into two 2-task batches by
    # `max_items=2`.
    self.assertEqual([len(b) for b in writer.processed_batches], [1, 2, 2])

  def test_close_during_batch_collection_flushes_partial_batch_and_exits(
      self,
  ) -> None:
    writer = self._create_writer(
        batching=async_writer.BatchConfig(max_items=8, max_wait_s=5.0)
    )
    writer._increase_batch_wait()

    # Enqueue 1 task so the worker enters `queue.get(timeout=remaining_s)`
    # waiting for up to 7 more tasks, then call `close()` to enqueue `None`.
    writer.enqueue(step_id=1)
    writer.close(timeout=_WORKER_START_TIMEOUT_S)

    self.assertTrue(writer.is_closed)
    self.assertIsNotNone(writer._worker_thread)
    self.assertFalse(writer._worker_thread.is_alive())
    (processed_task,) = writer.processed_tasks
    self.assertEqual(processed_task.step.step_id, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_batchable_error_decays_wait",
          fail_on_step_id=3,
          batchable_error_step_ids=(),
          expected_batch_wait_s=0.03,
          expected_processed_step_ids=[1, 2, 4, 5],
      ),
      dict(
          testcase_name="batchable_error_increases_wait",
          fail_on_step_id=None,
          batchable_error_step_ids=(3,),
          expected_batch_wait_s=0.09,
          expected_processed_step_ids=[1, 2, 4, 5],
      ),
      dict(
          testcase_name="multiple_batchable_errors_increase_wait_once",
          fail_on_step_id=None,
          batchable_error_step_ids=(2, 3, 4),
          expected_batch_wait_s=0.09,
          expected_processed_step_ids=[1, 5],
      ),
  )
  def test_process_batch_when_tasks_fail_isolates_errors_and_updates_wait(
      self,
      fail_on_step_id: int | None,
      batchable_error_step_ids: Sequence[int],
      expected_batch_wait_s: float,
      expected_processed_step_ids: Sequence[int],
  ) -> None:
    writer = self._create_writer(
        batching=async_writer.BatchConfig(max_items=4, max_wait_s=0.25),
        fail_on_step_id=fail_on_step_id,
        batchable_error_step_ids=batchable_error_step_ids,
        block_on_batch_call=1,
    )
    writer._increase_batch_wait()

    with self.assertLogs(level=logging.ERROR) as log_output:
      writer.enqueue(step_id=1)
      self.assertTrue(
          writer.batch_started_event.wait(timeout=_WORKER_START_TIMEOUT_S)
      )
      for step_id in (2, 3, 4, 5):
        writer.enqueue(step_id=step_id)
      writer.unblock_batch_event.set()
      writer.flush()

    self.assertTrue(
        any(
            "_TestAsyncWriterWorker failed to write step (step_id=3)" in msg
            for msg in log_output.output
        )
    )
    self.assertAlmostEqual(writer.batch_wait_s, expected_batch_wait_s)
    self.assertEqual(
        [t.step.step_id for t in writer.processed_tasks if t.step is not None],
        expected_processed_step_ids,
    )

  def test_worker_loop_when_to_atif_raises_logs_error_and_processes_siblings(
      self,
  ) -> None:
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _BrokenProjectionTask(_TestWriteTask):
      fail_projection: bool = False

      def to_atif(self) -> None:
        if self.fail_projection:
          raise ValueError("Custom step projection failed")
        super().to_atif()

    writer = self._create_writer(
        batching=async_writer.BatchConfig(max_items=4, max_wait_s=0.25),
        block_on_batch_call=1,
    )
    writer._increase_batch_wait()

    with self.assertLogs(level=logging.ERROR) as log_output:
      writer.enqueue(step_id=1)
      self.assertTrue(
          writer.batch_started_event.wait(timeout=_WORKER_START_TIMEOUT_S)
      )
      writer.enqueue(step_id=2)
      writer._enqueue(
          _BrokenProjectionTask(
              metadata=trajectory_testing.METADATA_1,
              step=trajectory_testing.make_step(step_id=3),
              fail_projection=True,
          )
      )
      writer.enqueue(step_id=4)
      writer.unblock_batch_event.set()
      writer.flush()
      self.assertAlmostEqual(writer.batch_wait_s, 0.03)

      # A batch where every task fails `to_atif()` (`ready_tasks` is empty)
      # still decays `batch_wait_s` by one step (`0.03s -> 0.02s`).
      writer._enqueue(
          _BrokenProjectionTask(
              metadata=trajectory_testing.METADATA_1,
              step=trajectory_testing.make_step(step_id=5),
              fail_projection=True,
          )
      )
      writer.flush()

    self.assertAlmostEqual(writer.batch_wait_s, 0.02)
    self.assertTrue(
        any(
            "_TestAsyncWriterWorker failed to write step (step_id=3)" in msg
            for msg in log_output.output
        )
    )
    self.assertEqual(
        [t.step.step_id for t in writer.processed_tasks if t.step is not None],
        [1, 2, 4],
    )


if __name__ == "__main__":
  absltest.main()
