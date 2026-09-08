"""Base asynchronous queue writer for Trajectory Stores."""

import abc
import atexit
import dataclasses
import queue
import threading
import weakref

from absl import logging
from etils import epath
from tunix.experimental.trajectory import trajectory as trajectory_lib


@dataclasses.dataclass(frozen=True, kw_only=True)
class WriteTask:
  """Container for an asynchronous trajectory write operation.

  Encapsulates all necessary data transferred across the thread boundary from
  the frontend calling thread (e.g. rollout worker) to the background worker
  thread. `metadata` and `step` are private deep copies owned by the task.
  """

  metadata: trajectory_lib.TrajectoryMetadata
  step: trajectory_lib.Step | None = None
  run_id: str | None = None
  traj_dir: epath.Path | None = None
  meta_path: epath.Path | None = None
  step_path: epath.Path | None = None


# Every live (i.e. not garbage collected) BaseAsyncWriter, so that
# `_close_live_writers` can drain them at interpreter shutdown. Weak references
# are used so registration does not keep writers alive.
_LIVE_WRITERS: "weakref.WeakSet[BaseAsyncWriter]" = weakref.WeakSet()


def _close_live_writers() -> None:
  """Closes every live BaseAsyncWriter, draining its pending writes.

  Registered with `atexit`, which runs while daemon threads are still alive but
  before the interpreter kills them. Without this, write operations still
  sitting in a writer's queue when the process ends are silently lost, because
  the worker is a daemon thread and `__del__` is not guaranteed to run for
  objects that are still referenced at shutdown.
  """
  for writer in list(_LIVE_WRITERS):
    try:
      writer.close()
    except Exception:  # pylint: disable=broad-exception-caught
      # Best-effort error handling: a failure to persist diagnostic data must
      # not turn into a non-zero exit status for training jobs.
      logging.exception("Failed to close BaseAsyncWriter at interpreter exit.")


atexit.register(_close_live_writers)


class BaseAsyncWriter(abc.ABC):
  """Abstract asynchronous queue writer managing worker thread lifecycle.

  Architectural Decisions & Invariants:
    1. Single Dedicated Background Worker Thread:
       A single background daemon thread processes write tasks sequentially
       from an unbounded FIFO queue (`queue.Queue`). Using a single sequential
       worker ensures chronological order per entity without requiring complex
       per-record locking, while keeping memory and thread overhead minimal in
       distributed training environments.

    2. Lazy Worker Thread Initialization:
       The worker thread is not spawned during `__init__`. Instead, it is
       lazily initialized under a thread lock on the first enqueue operation.
       This prevents unnecessary OS thread allocation in read-only processes.

    3. Best-Effort Fault Tolerance:
       Persistence errors (e.g. disk full, transient network database errors)
       must never crash distributed training loops. The worker loop catches all
       task processing exceptions, logs them with full tracebacks via
       `_log_task_error`, and continues draining subsequent tasks. Errors are
       suppressed and never propagated back to callers or `flush()`.

    4. Strict Barrier Synchronization:
       `flush()` blocks on `_queue.join()`. When `flush()` returns, all tasks
       enqueued prior to the call are guaranteed to have completed execution.

    5. Safe Lifecycle & Shutdown Hook:
       `close()` enqueues a sentinel None task, joins the worker thread with a
       configurable timeout, and deregisters from `_LIVE_WRITERS`. The `atexit`
       hook drains all live instances when the interpreter exits.
  """

  def __init__(self):
    """Initializes BaseAsyncWriter without starting the background worker."""
    # Unbounded FIFO queue for passing write tasks to the worker thread.
    self._queue: queue.Queue[WriteTask | None] = queue.Queue()
    # Lock protecting lazy thread spawning and closed state transitions.
    self._lock = threading.Lock()
    # Users are not expected to explicitly call close() on the writer, as its
    # lifecycle is managed automatically.
    self._closed: bool = False
    self._worker_thread: threading.Thread | None = None
    # Drained by `_close_live_writers` at interpreter exit.
    _LIVE_WRITERS.add(self)

  @property
  def is_closed(self) -> bool:
    """Returns True if the writer has been closed."""
    with self._lock:
      return self._closed

  def _enqueue(self, task: WriteTask, thread_name: str | None = None) -> None:
    """Enqueues a task for asynchronous processing by the worker thread.

    Lazily spawns the background worker thread under lock if not already
    running.

    Args:
      task: Container holding task payload.
      thread_name: Optional descriptive name for the worker thread. Defaults to
        '<ClassName>Worker'.

    Raises:
      RuntimeError: If the writer has already been closed.
    """
    resolved_thread_name = thread_name or f"{self.__class__.__name__}Worker"
    with self._lock:
      if self._closed:
        raise RuntimeError(
            f"Cannot write to a closed {self.__class__.__name__}."
        )
      if self._worker_thread is None:
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=resolved_thread_name,
            daemon=True,
        )
        self._worker_thread.start()
      self._queue.put(task)

  def _worker_loop(self) -> None:
    """Worker loop processing tasks sequentially from the queue.

    Catches and logs all task processing exceptions without propagating them
    to callers or breaking the loop, ensuring callers are never failed by
    background write errors. Uses `task_done()` in a `finally` block to ensure
    queue join barriers (`flush()`) unblock even when tasks fail.
    """
    try:
      while True:
        task = self._queue.get()
        # None is the shutdown sentinel enqueued by close().
        if task is None:
          self._queue.task_done()
          break

        try:
          self._process_task(task)
        except Exception:  # pylint: disable=broad-exception-caught
          # Best-effort error suppression: log full traceback but never crash
          # the worker.
          try:
            self._log_task_error(task)
          except Exception:  # pylint: disable=broad-exception-caught
            logging.exception(
                "Failed to log task error in %s.", self.__class__.__name__
            )
        finally:
          # Crucial: always mark task as done so `flush()` barrier does not hang
          # on failure.
          self._queue.task_done()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception(
          "Fatal unhandled error in %s worker loop.", self.__class__.__name__
      )

  @abc.abstractmethod
  def _process_task(self, task: WriteTask) -> None:
    """Processes a single write task dequeued from the queue.

    Args:
      task: Container holding task payload.
    """
    ...

  def _log_task_error(self, task: WriteTask) -> None:
    """Logs an exception that occurred while processing a task.

    Can be overridden by subclasses to provide domain-specific error details.

    Args:
      task: The task whose processing raised an exception.
    """
    logging.exception(
        "Failed to process task in %s: %s",
        self.__class__.__name__,
        task,
    )

  def flush(self) -> None:
    """Blocks until all queued write operations have been processed.

    Users do not need to call flush() in normal usage; it is primarily for
    testing.

    Provides strict barrier synchronization: when this method returns, all
    tasks enqueued prior to the call have been executed by the worker thread.
    """
    if self._closed:
      return
    self._queue.join()

  def close(self, timeout: float | None = 5.0) -> None:
    """Flushes pending writes and shuts down the background worker thread.

    Users are not expected to explicitly call close() on the writer, as its
    lifecycle is managed automatically.

    Enqueues a sentinel `None` task to signal the worker thread to exit after
    draining all previously queued tasks, then joins the worker thread.

    The timeout is a hard limit that discards any remaining unfinished tasks if
    the worker thread fails to complete within the specified duration.

    Calling `close()` more than once is safe; subsequent calls are no-ops.

    Args:
      timeout: Maximum time in seconds to wait for the worker thread to
        terminate. Defaults to 5.0 seconds.
    """
    _LIVE_WRITERS.discard(self)
    with self._lock:
      if not self._closed:
        self._closed = True
        if self._worker_thread is not None:
          self._queue.put(None)

    if self._worker_thread is not None and self._worker_thread.is_alive():
      self._worker_thread.join(timeout=timeout)
      if self._worker_thread.is_alive():
        discarded_traj_ids = set()
        while True:
          try:
            task = self._queue.get_nowait()
            self._queue.task_done()
            if (
                task is not None
                and task.metadata
                and task.metadata.trajectory_id
            ):
              discarded_traj_ids.add(task.metadata.trajectory_id)
          except queue.Empty:
            break
        self._queue.put(None)
        logging.warning(
            "%s worker thread did not finish within timeout of"
            " %s seconds. Discarded remaining tasks for trajectory IDs: %s",
            self.__class__.__name__,
            timeout,
            sorted(discarded_traj_ids),
        )

  def __del__(self) -> None:
    """Destructor to ensure worker thread shutdown is signaled.

    Runs upon garbage collection to signal worker thread termination.
    """
    try:
      self.close(timeout=1.0)
    except BaseException:  # pylint: disable=broad-exception-caught
      # Suppress exceptions during GC/interpreter teardown where modules or
      # locks may already be partially destroyed.
      pass
