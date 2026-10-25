"""Asynchronous queue and worker thread engine for Trajectory Store."""

import abc
import atexit
import dataclasses
import queue
import threading
from typing import Any, Final, Generic, TypeVar, cast, overload
import weakref

from absl import logging
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

# Default maximum number of pending WriteTasks buffered in AsyncWriter's queue.
# Each WriteTask holds a deep copy of TrajectoryMetadata and Step (~24 KB/task
# on average across 600k production steps, or ~11.5-36 KB across 8k-32k token
# context windows). Bounding the queue at 10,000 tasks (~240 MB, <= 0.5% of a
# 70 GB container limit) prevents unbounded heap growth, pymalloc arena
# fragmentation, and Gen-2 GC pauses during storage stalls while providing ~10x
# headroom over peak per-worker queue depths (~730-1,010 tasks across 8-16
# workers) observed in 9,856-trajectory load testing.
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 10_000

# Fraction of `max_queue_size` available for `Step` writes (80%, or 8,000 slots
# at the default 10,000 limit).
_STEP_QUEUE_CAPACITY_RATIO: Final[float] = 0.8

# Fraction of `max_queue_size` available for `TrajectoryMetrics` writes (95%, or
# 9,500 slots at the default 10,000 limit). The remaining 5% (500 slots) is
# reserved for metadata-only updates (`task.step is None and task.metrics is None`)
# so trajectory records and final episode statuses are preserved when metrics
# and step writes fill the queue.
_METRICS_QUEUE_CAPACITY_RATIO: Final[float] = 0.95


@dataclasses.dataclass(frozen=True, kw_only=True)
class StepArrayFilter:
  """Configuration for filtering heavy per-token array fields from Step objects.

  Used before enqueuing or deep-copying steps to avoid buffering or writing
  massive NumPy arrays (assistant_tokens, assistant_masks, logprobs, env_tokens,
  env_masks, prefill_routed_experts) in observability-only runs.
  """

  filter_tokens: bool = False
  filter_masks: bool = False
  filter_logprobs: bool = False
  filter_routed_experts: bool = False

  @property
  def has_filters(self) -> bool:
    """Returns True if any array filter is enabled."""
    return (
        self.filter_tokens
        or self.filter_masks
        or self.filter_logprobs
        or self.filter_routed_experts
    )


_StepModelT = TypeVar("_StepModelT", bound=trajectory_lib.Step)


@overload
def filter_step_arrays(
    step: None,
    array_filter: StepArrayFilter | None = None,
) -> None:
  ...


@overload
def filter_step_arrays(
    step: _StepModelT,
    array_filter: StepArrayFilter | None = None,
) -> _StepModelT:
  ...


def filter_step_arrays(
    step: _StepModelT | None,
    array_filter: StepArrayFilter | None = None,
) -> _StepModelT | None:
  """Returns a new Step with specified array fields stripped, or step if no filters.

  Never mutates the caller's live `step` object in place.

  Args:
    step: The step instance to filter, or None.
    array_filter: Filter options specifying which arrays to strip.

  Returns:
    A new Step instance of the same type with filtered arrays set to None, or
    the original step if step is None, array_filter is None, or no filters are
    active.
  """
  if step is None or array_filter is None or not array_filter.has_filters:
    return step

  updates: dict[str, Any] = {}
  if isinstance(step, trajectory_lib.TunixAgentStep):
    if array_filter.filter_tokens:
      updates["assistant_tokens"] = None
    if array_filter.filter_masks:
      updates["assistant_masks"] = None
    if array_filter.filter_logprobs:
      updates["logprobs"] = None
    if array_filter.filter_routed_experts:
      updates["prefill_routed_experts"] = None
      updates["prefill_start"] = None
      updates["prefill_num_context"] = None
  elif isinstance(step, trajectory_lib.TunixEnvStep):
    if array_filter.filter_tokens:
      updates["env_tokens"] = None
    if array_filter.filter_masks:
      updates["env_masks"] = None

  if step.metrics is not None:
    metrics_updates: dict[str, Any] = {}
    if (
        array_filter.filter_tokens
        and step.metrics.completion_token_ids is not None
    ):
      metrics_updates["completion_token_ids"] = None
    if array_filter.filter_logprobs and step.metrics.logprobs is not None:
      metrics_updates["logprobs"] = None
    if metrics_updates:
      updates["metrics"] = step.metrics.model_copy(update=metrics_updates)

  if not updates:
    return step

  return step.model_copy(update=updates)


@dataclasses.dataclass(frozen=True, kw_only=True)
class WriteTask:
  """Container for an asynchronous trajectory write operation.

  Encapsulates all necessary data transferred across the thread boundary from
  the frontend calling thread (e.g. rollout worker) to the background worker
  thread. `metadata`, `step`, and `metrics` are private deep copies owned by
  the task.
  """

  metadata: trajectory_lib.TrajectoryMetadata | None = None
  step: trajectory_lib.Step | None = None
  run_id: str | None = None
  trajectory_id: str = ""
  metrics: trajectory_lib.TrajectoryMetrics | None = None
  step_array_filter: StepArrayFilter | None = None

  def __post_init__(self) -> None:
    """Deep copies the payload so later caller mutations cannot leak in."""
    if self.metadata is not None:
      object.__setattr__(
          self, "metadata", trajectory_lib.deep_copy(self.metadata)
      )
      if not self.trajectory_id and self.metadata.trajectory_id:
        object.__setattr__(self, "trajectory_id", self.metadata.trajectory_id)
    if self.step is not None:
      filtered_step = filter_step_arrays(self.step, self.step_array_filter)
      object.__setattr__(self, "step", trajectory_lib.deep_copy(filtered_step))
    if self.metrics is not None:
      object.__setattr__(
          self, "metrics", trajectory_lib.deep_copy(self.metrics)
      )
    if not self.trajectory_id:
      if self.metadata is not None:
        raise ValueError("WriteTask requires metadata.trajectory_id.")
      raise ValueError("WriteTask requires trajectory_id.")

  def to_atif(self) -> None:
    """Projects `metadata` and `step` to base ATIF models in-place.

    Called by the background worker thread before processing the task. Mutates
    the frozen instance in-place to avoid a second deep copy.
    """
    if self.metadata is not None:
      object.__setattr__(self, "metadata", self.metadata.to_atif_metadata())
    if self.step is not None:
      object.__setattr__(self, "step", self.step.to_atif_step())


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

    2. Bounded Queue Capacity with Three-Tier Admission:
       To prevent unbounded memory growth or caller-thread blocking during
       storage latency spikes, the queue enforces a hard capacity of
       `max_queue_size` with a 3-tier admission policy:
         - Step tasks (`task.step is not None`) are admitted below
           `step_queue_limit` (80% of `max_queue_size`) and dropped
           non-blockingly with a warning once `step_queue_limit` is reached.
         - Metrics tasks (`task.metrics is not None`) are admitted below
           `metrics_queue_limit` (95% of `max_queue_size`) and dropped
           non-blockingly with a warning once `metrics_queue_limit` is reached.
         - Metadata-only tasks (`task.step is None and task.metrics is None`)
           may use the remaining 5% reserve up to `max_queue_size` so parent
           trajectory records and final episode statuses are preserved when
           step or metrics writes fill the queue. At `max_queue_size`, all
           incoming tasks are dropped non-blockingly with a warning.

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
      step_array_filter: StepArrayFilter | None = None,
  ):
    """Initializes AsyncWriter without starting the background worker.

    Args:
      thread_name: Descriptive name for the background worker thread. Defaults
        to '<ClassName>Worker'.
      max_queue_size: Maximum number of pending tasks buffered in the queue
        before dropping incoming writes. Defaults to `DEFAULT_MAX_QUEUE_SIZE`
        (10,000).
      step_array_filter: Optional configuration for stripping per-token array
        fields before enqueuing or writing steps.

    Raises:
      ValueError: If `max_queue_size` is not positive.
    """
    if max_queue_size <= 0:
      raise ValueError(
          f"max_queue_size must be positive, got {max_queue_size}."
      )
    self._thread_name = thread_name or f"{type(self).__name__}Worker"
    self._max_queue_size = max_queue_size
    self._step_array_filter = step_array_filter
    self._step_queue_limit = max(
        1, int(max_queue_size * _STEP_QUEUE_CAPACITY_RATIO)
    )
    self._metrics_queue_limit = max(
        1, int(max_queue_size * _METRICS_QUEUE_CAPACITY_RATIO)
    )
    # FIFO queue for passing write tasks to the worker thread. Capacity limits
    # (`_step_queue_limit`, `_metrics_queue_limit`, and `_max_queue_size`) are
    # enforced under `self._lock` in `_enqueue` so that `close()` can always
    # append the `None` shutdown sentinel without blocking when the queue is full.
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
  def metrics_queue_limit(self) -> int:
    """Returns the queue limit above which metrics tasks are dropped."""
    return self._metrics_queue_limit

  @property
  def step_array_filter(self) -> StepArrayFilter | None:
    """Returns the configured step array filter, or None."""
    return self._step_array_filter

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

      if task.metrics is not None and qsize >= self._metrics_queue_limit:
        logging.warning(
            "%s metrics queue reached capacity (%d/%d); dropped metrics"
            " for trajectory_id=%s.",
            self._thread_name,
            qsize,
            self._metrics_queue_limit,
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

  def _create_metrics_task(
      self,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> _TaskT:
    """Factory creating a WriteTask instance for metrics updates.

    Subclasses can override this to construct backend-specific WriteTask
    types if needed.

    Args:
      trajectory_id: Identifier of the trajectory to update.
      metrics: TrajectoryMetrics instance.

    Returns:
      A WriteTask (or subclass) instance.
    """
    return cast(_TaskT, WriteTask(trajectory_id=trajectory_id, metrics=metrics))

  def update_metrics(
      self,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> None:
    """Enqueues a TrajectoryMetrics update operation.

    Identifiers and types are validated on the calling thread so malformed
    writes fail fast with actionable errors.

    Args:
      trajectory_id: Identifier of trajectory to update.
      metrics: TrajectoryMetrics containing final summary metrics to merge.

    Raises:
      TypeError: If metrics is not an instance of TrajectoryMetrics.
      ValueError: If trajectory_id is empty or whitespace.
      RuntimeError: If the writer has already been closed.
    """
    if not isinstance(metrics, trajectory_lib.TrajectoryMetrics):
      raise TypeError(
          "metrics must be an instance of TrajectoryMetrics, got"
          f" {type(metrics)!r}."
      )
    if not trajectory_id or not trajectory_id.strip():
      raise ValueError("trajectory_id must be a non-empty string.")

    task = self._create_metrics_task(
        trajectory_id=trajectory_id.strip(),
        metrics=metrics,
    )
    self._enqueue(task)

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
