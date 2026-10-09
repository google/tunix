"""Asynchronous queue and worker thread engine for Trajectory Store."""

import abc
import atexit
import dataclasses
import queue
import threading
from typing import Any, Final, Generic, TypeVar
import weakref

from absl import logging
from tunix.experimental.trajectory import trajectory as trajectory_lib

# Default maximum number of pending WriteTasks buffered in AsyncWriter's queue.
# Each WriteTask holds a deep copy of TrajectoryMetadata and Step (~24 KB/task
# on average across 600k production steps, or ~11.5-36 KB across 8k-32k token
# context windows). Bounding the queue at 10,000 tasks (~240 MB, <= 0.5% of a
# 70 GB container limit) prevents unbounded heap growth, pymalloc arena
# fragmentation, and Gen-2 GC pauses during storage stalls while providing ~10x
# headroom over peak per-worker queue depths (~730-1,010 tasks across 8-16
# workers) observed in 9,856-trajectory load testing.
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 10_000

# Fraction of `max_queue_size` available for `Step` writes (90%, or 9,000 slots
# at the default 10,000 limit). The remaining 10% (1,000 slots, < 1 MB RAM) is
# reserved for metadata-only updates (`task.step is None`) so trajectory
# records and final episode statuses are preserved when step writes fill the
# queue.
_STEP_QUEUE_CAPACITY_RATIO: Final[float] = 0.9


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

  def __post_init__(self) -> None:
    """Deep copies the payload so later caller mutations cannot leak in."""
    object.__setattr__(
        self, "metadata", trajectory_lib.deep_copy(self.metadata)
    )
    if self.step is not None:
      object.__setattr__(self, "step", trajectory_lib.deep_copy(self.step))

  def to_atif(self) -> None:
    """Projects `metadata` and `step` to base ATIF models in-place.

    Called by the background worker thread before processing the task. Mutates
    the frozen instance in-place to avoid a second deep copy.
    """
    object.__setattr__(self, "metadata", self.metadata.to_atif_metadata())
    if self.step is not None:
      object.__setattr__(self, "step", self.step.to_atif_step())

  @property
  def trajectory_id(self) -> str:
    """Returns the trajectory this task writes, derived from `metadata`.

    Raises:
      ValueError: If `metadata` carries no trajectory_id.
    """
    traj_id = self.metadata.trajectory_id
    if traj_id is None:
      raise ValueError("WriteTask requires metadata.trajectory_id.")
    return traj_id


_TaskT = TypeVar("_TaskT", bound=WriteTask)


# Every live (i.e. not garbage collected) AsyncWriter, so that
# `_close_live_writers` can drain them at interpreter shutdown. Weak references
# are used so registration does not keep writers alive.
_LIVE_WRITERS: "weakref.WeakSet[AsyncWriter[Any]]" = weakref.WeakSet()


def _close_live_writers() -> None:
  """Closes every live AsyncWriter, draining its pending writes.

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
      logging.exception(
          "%s failed to close at interpreter exit.", writer.worker_name
      )


atexit.register(_close_live_writers)


class AsyncWriter(abc.ABC, Generic[_TaskT]):
  """Abstract asynchronous queue writer managing worker thread lifecycle.

  Architectural Decisions & Invariants:
    1. Single Dedicated Background Worker Thread:
       A single background daemon thread processes write tasks sequentially
       from a bounded FIFO queue (`queue.Queue`). Using a single sequential
       worker ensures chronological order per entity without requiring complex
       per-record locking, while keeping memory and thread overhead minimal in
       distributed training environments.

    2. Bounded Queue Capacity with Metadata Reserve:
       To prevent unbounded memory growth or caller-thread blocking during
       storage latency spikes, the queue enforces a hard capacity of
       `max_queue_size`:
         - Step tasks (`task.step is not None`) are admitted below
           `step_queue_limit` (90% of `max_queue_size`) and dropped
           non-blockingly with a warning once `step_queue_limit` is reached.
         - Metadata-only tasks (`task.step is None`) may use the remaining 10%
           reserve up to `max_queue_size` so parent trajectory records and
           final episode statuses are preserved when step writes fill the
           queue. At `max_queue_size`, all incoming tasks are dropped
           non-blockingly with a warning.

    3. Lazy Worker Thread Initialization:
       The worker thread is not spawned during `__init__`. Instead, it is
       lazily initialized under a thread lock on the first enqueue operation.
       This prevents unnecessary OS thread allocation in read-only processes.

    4. Best-Effort Fault Tolerance:
       Persistence errors (e.g. disk full, transient network database errors)
       must never crash distributed training loops. The worker loop catches all
       task processing exceptions, logs them with full tracebacks via
       `_log_task_error`, and continues draining subsequent tasks. Errors are
       suppressed and never propagated back to callers or `flush()`.

    5. Strict Barrier Synchronization:
       `flush()` blocks on `_queue.join()`. When `flush()` returns, all tasks
       enqueued prior to the call are guaranteed to have completed execution.

    6. Safe Lifecycle & Shutdown Hook:
       `close()` enqueues a sentinel None task, joins the worker thread with a
       configurable timeout, and deregisters from `_LIVE_WRITERS`. The `atexit`
       hook drains all live instances when the interpreter exits.
  """

  def __init__(
      self,
      thread_name: str | None = None,
      max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
  ):
    """Initializes AsyncWriter without starting the background worker.

    Args:
      thread_name: Descriptive name for the background worker thread. Defaults
        to '<ClassName>Worker'.
      max_queue_size: Maximum number of pending tasks buffered in the queue
        before dropping incoming writes. Defaults to `DEFAULT_MAX_QUEUE_SIZE`
        (10,000).

    Raises:
      ValueError: If `max_queue_size` is not positive.
    """
    if max_queue_size <= 0:
      raise ValueError(
          f"max_queue_size must be positive, got {max_queue_size}."
      )
    self._thread_name = thread_name or f"{type(self).__name__}Worker"
    self._max_queue_size = max_queue_size
    self._step_queue_limit = max(
        1, int(max_queue_size * _STEP_QUEUE_CAPACITY_RATIO)
    )
    # FIFO queue for passing write tasks to the worker thread. Capacity limits
    # (`_step_queue_limit` and `_max_queue_size`) are enforced under
    # `self._lock` in `_enqueue` so that `close()` can always append the `None`
    # shutdown sentinel without blocking when the queue is full.
    self._queue: queue.Queue[_TaskT | None] = queue.Queue()
    # Lock protecting lazy thread spawning, queue admission, and closed state.
    self._lock = threading.Lock()
    # Users are not expected to explicitly call close() on the writer, as its
    # lifecycle is managed automatically.
    self._closed: bool = False
    self._worker_thread: threading.Thread | None = None
    # Drained by `_close_live_writers` at interpreter exit.
    _LIVE_WRITERS.add(self)

  @property
  def worker_name(self) -> str:
    """Returns the name of the background worker thread."""
    return self._thread_name

  @property
  def max_queue_size(self) -> int:
    """Returns the hard maximum capacity of the write queue."""
    return self._max_queue_size

  @property
  def step_queue_limit(self) -> int:
    """Returns the queue limit above which step tasks are dropped."""
    return self._step_queue_limit

  @property
  def is_closed(self) -> bool:
    """Returns True if the writer has been closed."""
    with self._lock:
      return self._closed

  def _enqueue(self, task: _TaskT) -> None:
    """Enqueues a task for asynchronous processing by the worker thread.

    Lazily spawns the background worker thread under lock if not already
    running, and drops incoming tasks non-blockingly when queue capacity is
    reached.

    Args:
      task: Container holding task payload.

    Raises:
      RuntimeError: If the writer has already been closed.
    """
    traj_id = task.trajectory_id
    with self._lock:
      if self._closed:
        raise RuntimeError(
            f"Cannot write to a closed writer ({self._thread_name})."
        )

      qsize = self._queue.qsize()
      if task.step is not None and qsize >= self._step_queue_limit:
        logging.warning(
            "%s step queue reached capacity (%d/%d); dropped step_id=%s"
            " for trajectory_id=%s.",
            self._thread_name,
            qsize,
            self._step_queue_limit,
            task.step.step_id,
            traj_id,
        )
        return

      if qsize >= self._max_queue_size:
        logging.warning(
            "%s queue is full (%d/%d); dropped metadata task for"
            " trajectory_id=%s.",
            self._thread_name,
            qsize,
            self._max_queue_size,
            traj_id,
        )
        return

      if self._worker_thread is None:
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=self._thread_name,
            daemon=True,
        )
        self._worker_thread.start()
      self._queue.put_nowait(task)

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
          task.to_atif()
          self._process_task(task)
        except Exception:  # pylint: disable=broad-exception-caught
          # Best-effort error suppression: log full traceback but never crash
          # the worker.
          try:
            self._log_task_error(task)
          except Exception:  # pylint: disable=broad-exception-caught
            logging.exception(
                "%s failed to log a task error.", self._thread_name
            )
        finally:
          # Crucial: always mark task as done so `flush()` barrier does not hang
          # on failure.
          self._queue.task_done()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("%s exited on an unhandled error.", self._thread_name)

  @abc.abstractmethod
  def _process_task(self, task: _TaskT) -> None:
    """Processes a single write task dequeued from the queue.

    Args:
      task: Container holding task payload.
    """
    ...

  def _log_task_error(self, task: _TaskT) -> None:
    """Logs an exception that occurred while processing a task.

    Can be overridden by subclasses to provide domain-specific error details.

    Args:
      task: The write task that failed to process.
    """
    step_id = task.step.step_id if task.step is not None else None
    logging.exception(
        "%s failed to process task for trajectory_id=%s, step_id=%s.",
        self._thread_name,
        task.trajectory_id,
        step_id,
    )

  def flush(self) -> None:
    """Blocks until all queued write operations have been processed.

    Users do not need to call flush() in normal usage; it is primarily for
    testing.

    Provides strict barrier synchronization: when this method returns, all
    tasks enqueued prior to the call have been executed by the worker thread.
    The barrier does not apply to a closed writer, for which this is a no-op;
    `close()` already drains the queue.
    """
    if self.is_closed:
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
            if task is not None and task.trajectory_id:
              discarded_traj_ids.add(task.trajectory_id)
          except queue.Empty:
            break
        self._queue.put(None)
        logging.warning(
            "%s did not finish within the %s second timeout. Discarded"
            " remaining tasks for trajectory IDs: %s",
            self._thread_name,
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
      pass
