"""Unit tests verifying lifecycle and queue mechanics of BaseAsyncWriter."""

import atexit
import importlib
import threading
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory import base_writer
from tunix.experimental.trajectory import trajectory as trajectory_lib


class _TestWriter(base_writer.BaseAsyncWriter):
  """Concrete test implementation of BaseAsyncWriter."""

  def __init__(self, should_fail_on: int | None = None):
    super().__init__()
    self.processed: list[int] = []
    self.should_fail_on = should_fail_on

  def write_item(self, item: int) -> None:
    task = base_writer.WriteTask(
        metadata=trajectory_lib.TrajectoryMetadata(
            trajectory_id=str(item),
            agent=trajectory_lib.Agent(name="test_agent", version="0.1.0"),
        )
    )
    self._enqueue(task, "TestWriterWorker")

  def _process_task(self, task: base_writer.WriteTask) -> None:
    if task.metadata.trajectory_id is None:
      raise ValueError("trajectory_id must not be None.")
    item = int(task.metadata.trajectory_id)
    if self.should_fail_on is not None and item == self.should_fail_on:
      raise ValueError(f"Intentional test failure on item {item}")
    self.processed.append(item)


class BaseAsyncWriterTest(parameterized.TestCase):
  """Unit tests for BaseAsyncWriter lifecycle and queue mechanics."""

  # ============================================================================
  # Worker Thread Initialization & Enqueue
  # ============================================================================

  def test_lazy_worker_thread_startup(self) -> None:
    writer = _TestWriter()
    self.assertIsNone(writer._worker_thread)

    writer.write_item(1)
    self.assertIsNotNone(writer._worker_thread)
    self.assertTrue(writer._worker_thread.is_alive())

    writer.flush()
    writer.close()

  def test_enqueue_spawns_worker_thread_with_default_name(self) -> None:
    writer = _TestWriter()
    task = base_writer.WriteTask(
        metadata=trajectory_lib.TrajectoryMetadata(
            trajectory_id="default_thread",
            agent=trajectory_lib.Agent(name="test_agent", version="0.1.0"),
        )
    )
    writer._enqueue(task)
    self.assertIsNotNone(writer._worker_thread)
    self.assertEqual(writer._worker_thread.name, "_TestWriterWorker")
    writer.flush()
    writer.close()

  def test_concurrent_enqueue_spawns_single_worker_thread(self) -> None:
    writer = _TestWriter()
    barrier = threading.Barrier(10)

    def worker_task(idx: int) -> None:
      barrier.wait()
      writer.write_item(idx)

    threads = [
        threading.Thread(target=worker_task, args=(i,)) for i in range(10)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertIsNotNone(writer._worker_thread)
    writer.flush()
    writer.close()
    self.assertLen(writer.processed, 10)

  # ============================================================================
  # Barrier Synchronization & Ordering
  # ============================================================================

  def test_write_and_flush_processes_all_items_in_order(self) -> None:
    writer = _TestWriter()
    items = [10, 20, 30, 40, 50]
    for item in items:
      writer.write_item(item)
    writer.flush()

    self.assertEqual(writer.processed, items)
    writer.close()

  def test_flush_on_empty_and_closed_writer(self) -> None:
    writer = _TestWriter()
    # Flushing an empty writer must not hang or error.
    writer.flush()
    writer.close()
    # Flushing a closed writer must return immediately without joining queue.
    with mock.patch.object(writer._queue, "join") as mock_join:
      writer.flush()
      mock_join.assert_not_called()
    self.assertTrue(writer.is_closed)

  def test_flush_when_closed_skips_queue_join(self) -> None:
    writer = _TestWriter()
    writer.close()

    with mock.patch.object(writer._queue, "join") as mock_join:
      writer.flush()
      mock_join.assert_not_called()

  # ============================================================================
  # Error Handling & Fault Tolerance
  # ============================================================================

  def test_worker_error_suppression_does_not_halt_queue(self) -> None:
    writer = _TestWriter(should_fail_on=2)
    with mock.patch.object(logging, "exception") as mock_log_exception:
      writer.write_item(1)
      writer.write_item(2)  # Will raise ValueError inside _process_task.
      writer.write_item(3)
      writer.flush()

    # 1 and 3 should be processed despite failure on 2.
    self.assertEqual(writer.processed, [1, 3])
    mock_log_exception.assert_called_once()
    writer.close()

  # ============================================================================
  # Lifecycle & Shutdown (close, __del__)
  # ============================================================================

  def test_close_drains_queue_and_joins_thread(self) -> None:
    writer = _TestWriter()
    items = list(range(10))
    for item in items:
      writer.write_item(item)

    writer.close()

    self.assertEqual(writer.processed, items)
    self.assertTrue(writer.is_closed)
    if writer._worker_thread is not None:
      self.assertFalse(writer._worker_thread.is_alive())

  def test_close_timeout_logs_warning_and_discards_tasks(self) -> None:
    writer = _TestWriter()
    unblock_worker = threading.Event()

    def slow_process(task: base_writer.WriteTask) -> None:
      unblock_worker.wait(timeout=5.0)
      if task.metadata.trajectory_id is None:
        raise ValueError("trajectory_id must not be None.")
      writer.processed.append(int(task.metadata.trajectory_id))

    with mock.patch.object(writer, "_process_task", side_effect=slow_process):
      with mock.patch.object(logging, "warning") as mock_log_warning:
        writer.write_item(1)
        writer.write_item(2)
        writer.close(timeout=0.01)

    unblock_worker.set()
    mock_log_warning.assert_called_once()
    self.assertEqual(mock_log_warning.call_args[0][1], "_TestWriter")
    self.assertTrue(writer.is_closed)

  def test_write_to_closed_writer_raises_runtime_error(self) -> None:
    writer = _TestWriter()
    writer.close()

    with self.assertRaisesRegex(RuntimeError, r"Cannot write to a closed"):
      writer.write_item(1)

  def test_idempotent_close(self) -> None:
    writer = _TestWriter()
    writer.write_item(1)
    writer.close()
    writer.close()
    self.assertTrue(writer.is_closed)

  def test_destructor_triggers_close(self) -> None:
    writer = _TestWriter()
    with mock.patch.object(writer, "close", wraps=writer.close) as mock_close:
      writer.__del__()
      mock_close.assert_called_once_with(timeout=1.0)
    self.assertTrue(writer.is_closed)

  # ============================================================================
  # Process Teardown & atexit Hook
  # ============================================================================

  def test_hook_is_registered_with_atexit(self) -> None:
    """Verifies the module registers its shutdown hook on import."""
    with mock.patch.object(atexit, "register") as mock_register:
      importlib.reload(base_writer)
      mock_register.assert_called_with(base_writer._close_live_writers)
    self.addCleanup(atexit.register, base_writer._close_live_writers)

  def test_shutdown_hook_suppresses_close_errors(self) -> None:
    """Verifies that an error in one writer does not break shutdown."""
    failing_writer = _TestWriter()
    healthy_writer = _TestWriter()
    self.addCleanup(base_writer._LIVE_WRITERS.discard, failing_writer)
    self.addCleanup(base_writer._LIVE_WRITERS.discard, healthy_writer)
    failing_writer.write_item(1)
    healthy_writer.write_item(2)

    with mock.patch.object(
        failing_writer, "close", side_effect=RuntimeError("close failed")
    ):
      with mock.patch.object(logging, "exception") as mock_log_exception:
        base_writer._close_live_writers()

    mock_log_exception.assert_called_once()
    self.assertTrue(healthy_writer.is_closed)

  def test_live_writers_tracking_and_atexit_drain(self) -> None:
    writer1 = _TestWriter()
    writer2 = _TestWriter()
    self.addCleanup(base_writer._LIVE_WRITERS.discard, writer1)
    self.addCleanup(base_writer._LIVE_WRITERS.discard, writer2)
    self.assertIn(writer1, base_writer._LIVE_WRITERS)
    self.assertIn(writer2, base_writer._LIVE_WRITERS)

    writer1.write_item(10)
    writer2.write_item(20)

    base_writer._close_live_writers()

    self.assertTrue(writer1.is_closed)
    self.assertTrue(writer2.is_closed)
    self.assertEqual(writer1.processed, [10])
    self.assertEqual(writer2.processed, [20])
    self.assertNotIn(writer1, base_writer._LIVE_WRITERS)
    self.assertNotIn(writer2, base_writer._LIVE_WRITERS)


if __name__ == "__main__":
  absltest.main()
