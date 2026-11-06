"""Asynchronous queue and worker thread engine for Trajectory Store."""

import abc
import atexit
from collections.abc import Sequence
import dataclasses
import queue
import threading
import time
from typing import Any, Final, Generic, TypeVar
import weakref

from absl import logging
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

# Default maximum number of pending WriteTasks buffered in AsyncWriter's queue.
# Each WriteTask holds deep copies of TrajectoryMetadata and TrajectoryMetrics
# plus a reference to Step by default (or a deep copy when `copy_step=True`),
# retaining ~24 KB/task in memory until drained (~11.5-36 KB across 8k-32k token
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
# reserved for metadata-only updates (`task.step is None and task.metrics is
# None`) so trajectory records and final episode statuses are preserved when
# metrics and step writes fill the queue.
_METRICS_QUEUE_CAPACITY_RATIO: Final[float] = 0.95

# Fraction of `BatchConfig.max_wait_s` added to `_batch_wait_s` when a backend
# reports a batchable error (+20%, or +50 ms when `max_wait_s == 0.25`).
_BATCH_WAIT_STEP_UP_FRACTION: Final[float] = 0.2

# Fraction of `BatchConfig.max_wait_s` subtracted from `_batch_wait_s` on each
# successful batch (-4%, or -10 ms when `max_wait_s == 0.25`). Because
# step-down is 5x smaller than step-up, 1 error decays in 5 clean batches, 2
# errors decay in 10 clean batches, and full saturation decays in 25 clean
# batches.
_BATCH_WAIT_STEP_DOWN_FRACTION: Final[float] = 0.04


@dataclasses.dataclass(frozen=True)
class BatchConfig:
  """Configuration for adaptive worker write batching.

  In normal operation (`batch_wait_s == 0.0`), the worker writes each task
  immediately (`1` task per write) with zero waiting and zero batching. When a
  backend reports a batchable contention or rate-limit error via
  `_should_batch_on_error`, the worker increases `batch_wait_s` up to
  `max_wait_s` and collects queued tasks into a batch up to `max_items`.

  Attributes:
    max_items: Maximum number of `WriteTask` items collected into one batch when
      batching is active. Defaults to 32.
    max_wait_s: Upper bound in seconds on how long the worker waits for
      additional tasks to accumulate when batching is active. Normal single-task
      operation waits `0.0` seconds. Defaults to 0.25 seconds.
  """

  max_items: int = 32
  max_wait_s: float = 0.25

  def __post_init__(self) -> None:
    """Validates batch item and wait bounds."""
    if self.max_items <= 0:
      raise ValueError(
          f"BatchConfig.max_items must be positive, got {self.max_items}."
      )
    if self.max_wait_s < 0.0:
      raise ValueError(
          f"BatchConfig.max_wait_s must be non-negative, got {self.max_wait_s}."
      )


DEFAULT_BATCH_CONFIG: Final[BatchConfig] = BatchConfig()


@dataclasses.dataclass(frozen=True, kw_only=True)
class WriteTask:
  """Container for an asynchronous trajectory write operation.

  Encapsulates all necessary data transferred across the thread boundary from
  the frontend calling thread (e.g. rollout worker) to the background worker
  thread. `metadata` and `metrics` are always deep copied so caller mutations
  cannot leak in. `step` is transferred with zero-copy ownership by default, and
  deep copied when `copy_step` is True.
  """

  metadata: trajectory_lib.TrajectoryMetadata | None = None
  step: trajectory_lib.Step | None = None
  run_id: str | None = None
  trajectory_id: str = ""
  metrics: trajectory_lib.TrajectoryMetrics | None = None
  copy_step: bool = False
  routed_experts_payload: bytes | None = dataclasses.field(
      default=None, init=False
  )

  def __post_init__(self) -> None:
    """Deep copies metadata/metrics and optionally step to isolate mutations."""
    if self.metadata is not None:
      object.__setattr__(
          self, "metadata", trajectory_lib.deep_copy(self.metadata)
      )
      if not self.trajectory_id and self.metadata.trajectory_id:
        object.__setattr__(self, "trajectory_id", self.metadata.trajectory_id)
    if self.step is not None and self.copy_step:
      object.__setattr__(self, "step", trajectory_lib.deep_copy(self.step))
    if self.metrics is not None:
      if not isinstance(self.metrics, trajectory_lib.TrajectoryMetrics):
        raise TypeError(
            "WriteTask.metrics must be an instance of TrajectoryMetrics, got"
            f" {type(self.metrics)!r}."
        )
      object.__setattr__(
          self, "metrics", trajectory_lib.deep_copy(self.metrics)
      )
    if not self.trajectory_id or not self.trajectory_id.strip():
      if self.metadata is not None:
        raise ValueError("WriteTask requires metadata.trajectory_id.")
      raise ValueError("WriteTask requires trajectory_id.")

  def to_atif(self) -> None:
    """Projects `metadata` and `step` to base ATIF models in-place.

    Called by the background worker thread before processing the task. Mutates
    the frozen instance in-place to avoid a second deep copy. Replaces the task
    references with projected models; the caller's input turn object is never
    mutated.
    """
    if self.metadata is not None:
      object.__setattr__(self, "metadata", self.metadata.to_atif_metadata())
    if self.step is not None:
      atif_step, routed_experts_payload = (
          self.step.to_atif_step_with_routed_experts()
      )
      object.__setattr__(self, "step", atif_step)
      if routed_experts_payload is not None:
        object.__setattr__(
            self, "routed_experts_payload", routed_experts_payload
        )


_TaskT = TypeVar("_TaskT", bound=WriteTask)


def _merge_queued_metrics_tasks(
    metrics_tasks: Sequence[_TaskT],
) -> _TaskT | None:
  """Merges `TrajectoryMetrics` across queued `metrics_tasks` in arrival order."""
  if not metrics_tasks:
    return None
  if len(metrics_tasks) == 1:
    return metrics_tasks[0]

  merged_fields: dict[str, Any] = {}
  merged_extra: dict[str, Any] | None = None
  latest_metrics: trajectory_lib.TrajectoryMetrics | None = None
  for task in metrics_tasks:
    if task.metrics is None:
      continue
    latest_metrics = task.metrics
    for field, value in latest_metrics.__dict__.items():
      if field != "extra" and value is not None:
        merged_fields[field] = value
    if latest_metrics.extra is not None:
      if merged_extra is None:
        merged_extra = dict(latest_metrics.extra)
      else:
        merged_extra.update(latest_metrics.extra)

  if latest_metrics is None:
    return metrics_tasks[-1]
  merged_fields["extra"] = merged_extra
  return dataclasses.replace(
      metrics_tasks[-1],
      metadata=None,
      step=None,
      metrics=latest_metrics.model_copy(update=merged_fields),
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrajectoryBatch(Generic[_TaskT]):
  """Batched write tasks grouped for a single `(run_id, trajectory_id)` pair.

  When multiple tasks for the same trajectory arrive in one batch,
  `TrajectoryBatch` keeps only the newest `metadata` task
  (`latest_metadata_task`), collects all `step` tasks in arrival order
  (`step_tasks`), and merges all queued `metrics` tasks into a single
  `merged_queued_metrics_task` so backends write metadata and metrics at most
  once per trajectory per batch.

  Attributes:
    run_id: Optional run identifier shared by tasks in this trajectory batch.
    trajectory_id: Trajectory identifier shared by tasks in this batch.
    latest_metadata_task: Latest task carrying non-None `metadata` for this
      trajectory, or `None` if all tasks in the group are metrics-only.
    step_tasks: Ordered tuple of tasks carrying a `Step` payload (`step is not
      None`), in arrival order.
    merged_queued_metrics_task: Single task carrying the merged
      `TrajectoryMetrics` (`extra` dict shallow-merged in arrival order) across
      all queued tasks with `metrics is not None` for this trajectory, or `None`
      if no task carries `metrics`.
  """

  run_id: str | None
  trajectory_id: str
  latest_metadata_task: _TaskT | None = None
  step_tasks: tuple[_TaskT, ...] = ()
  merged_queued_metrics_task: _TaskT | None = None


def group_tasks_by_trajectory(
    tasks: Sequence[_TaskT],
) -> tuple[TrajectoryBatch[_TaskT], ...]:
  """Groups `tasks` by `(run_id, trajectory_id)` into `TrajectoryBatch` items.

  Preserves first-seen `(run_id, trajectory_id)` order so storage backends can
  write each trajectory's latest metadata, ordered steps, and merged metrics in
  one operation while isolating distinct trajectories.

  Args:
    tasks: Sequence of `WriteTask` instances collected from the queue.

  Returns:
    Ordered tuple of `TrajectoryBatch` items in first-seen trajectory order.
  """
  tasks_by_trajectory_key: dict[tuple[str | None, str], list[_TaskT]] = {}
  for task in tasks:
    trajectory_key = (task.run_id, task.trajectory_id)
    tasks_by_trajectory_key.setdefault(trajectory_key, []).append(task)

  trajectory_batches: list[TrajectoryBatch[_TaskT]] = []
  for (
      run_id,
      trajectory_id,
  ), trajectory_tasks in tasks_by_trajectory_key.items():
    latest_metadata_task: _TaskT | None = None
    step_tasks: list[_TaskT] = []
    metrics_tasks: list[_TaskT] = []
    for task in trajectory_tasks:
      if task.metadata is not None:
        latest_metadata_task = task
      if task.step is not None:
        step_tasks.append(task)
      if task.metrics is not None:
        metrics_tasks.append(task)

    trajectory_batches.append(
        TrajectoryBatch(
            run_id=run_id,
            trajectory_id=trajectory_id,
            latest_metadata_task=latest_metadata_task,
            step_tasks=tuple(step_tasks),
            merged_queued_metrics_task=_merge_queued_metrics_tasks(
                metrics_tasks
            ),
        )
    )

  return tuple(trajectory_batches)


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

    4. Adaptive Contention-Driven Batching (AIMD on `batch_wait_s`):
       `BatchConfig` is active by default (`DEFAULT_BATCH_CONFIG`). In normal
       operation (`batch_wait_s == 0.0`), the worker executes each task
       immediately (`1` task per write) with zero waiting and zero batching
       latency. When a backend subclass reports a batchable error via
       `_should_batch_on_error`, `_increase_batch_wait` steps `batch_wait_s`
       up by 20% of `max_wait_s` (up to `max_wait_s`), activating batch
       collection via `queue.get(timeout=...)` up to `max_items`. Each
       successful batch decays `batch_wait_s` by 4% of `max_wait_s` down to
       `0.0` (5 clean batches per step-up, up to 25 clean batches from full
       saturation).

    5. Best-Effort Fault Tolerance:
       Persistence errors (e.g. disk full, transient network database errors)
       must never crash distributed training loops. The worker loop catches all
       task processing exceptions, logs them with full tracebacks via
       `_log_task_error`, and continues draining subsequent tasks. Errors are
       suppressed and never propagated back to callers or `flush()`.

    6. Strict Barrier Synchronization:
       `flush()` blocks on `_queue.join()`. When `flush()` returns, all tasks
       enqueued prior to the call are guaranteed to have completed execution.

    7. Safe Lifecycle & Shutdown Hook:
       `close()` enqueues a sentinel None task, joins the worker thread with a
       configurable timeout, and deregisters from `_LIVE_WRITERS`. The `atexit`
       hook drains all live instances when the interpreter exits.
  """

  def __init__(
      self,
      thread_name: str | None = None,
      max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
      batching: BatchConfig | None = DEFAULT_BATCH_CONFIG,
      copy_step: bool = False,
  ):
    """Initializes AsyncWriter without starting the background worker.

    Args:
      thread_name: Descriptive name for the background worker thread. Defaults
        to '<ClassName>Worker'.
      max_queue_size: Maximum number of pending tasks buffered in the queue
        before dropping incoming writes. Defaults to `DEFAULT_MAX_QUEUE_SIZE`
        (10,000).
      batching: Configuration governing batch item limits and maximum batch wait
        time. Defaults to `DEFAULT_BATCH_CONFIG`.
      copy_step: If True, deep copies step objects before enqueuing to prevent
        caller mutations from leaking. Defaults to False (zero-copy step
        handover) to avoid synchronous GIL contention on rollout workers.

    Raises:
      ValueError: If `max_queue_size` is not positive.
    """
    if max_queue_size <= 0:
      raise ValueError(
          f"max_queue_size must be positive, got {max_queue_size}."
      )
    self._thread_name = thread_name or f"{type(self).__name__}Worker"
    self._max_queue_size = max_queue_size
    self._copy_step = copy_step
    self._step_queue_limit = max(
        1, int(max_queue_size * _STEP_QUEUE_CAPACITY_RATIO)
    )
    self._metrics_queue_limit = max(
        1, int(max_queue_size * _METRICS_QUEUE_CAPACITY_RATIO)
    )
    self._batching: BatchConfig = (
        batching if batching is not None else DEFAULT_BATCH_CONFIG
    )
    # FIFO queue for passing write tasks to the worker thread. Capacity limits
    # (`_step_queue_limit`, `_metrics_queue_limit`, and `_max_queue_size`) are
    # enforced under `self._lock` in `_enqueue` so that `close()` can always
    # append the `None` shutdown sentinel without blocking when the queue is
    # full.
    self._queue: queue.Queue[_TaskT | None] = queue.Queue()
    # Lock protecting lazy thread spawning, queue admission, and closed state.
    self._lock = threading.Lock()
    # Current adaptive batch wait time in seconds (`0.0` in normal single-task
    # mode; scales up to `_batching.max_wait_s` on batchable errors and decays
    # back to `0.0` across clean batches). Mutated only by the worker thread.
    self._batch_wait_s: float = 0.0
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
  def batching(self) -> BatchConfig:
    """Returns the active `BatchConfig` for this writer."""
    return self._batching

  @property
  def batch_wait_s(self) -> float:
    """Returns the current adaptive batch wait duration in seconds."""
    return self._batch_wait_s

  @property
  def copy_step(self) -> bool:
    """Returns True if step objects are deep copied on enqueue."""
    return self._copy_step

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
    trajectory_id = task.trajectory_id
    with self._lock:
      if self._closed:
        raise RuntimeError(
            f"Cannot write to a closed writer ({self._thread_name})."
        )

      queue_size = self._queue.qsize()
      if task.step is not None and queue_size >= self._step_queue_limit:
        logging.warning(
            "%s step queue reached capacity (%d/%d); dropped step_id=%s"
            " for trajectory_id=%s.",
            self._thread_name,
            queue_size,
            self._step_queue_limit,
            task.step.step_id,
            trajectory_id,
        )
        return

      if task.metrics is not None and queue_size >= self._metrics_queue_limit:
        logging.warning(
            "%s metrics queue reached capacity (%d/%d); dropped metrics"
            " for trajectory_id=%s.",
            self._thread_name,
            queue_size,
            self._metrics_queue_limit,
            trajectory_id,
        )
        return

      if queue_size >= self._max_queue_size:
        logging.warning(
            "%s queue is full (%d/%d); dropped metadata task for"
            " trajectory_id=%s.",
            self._thread_name,
            queue_size,
            self._max_queue_size,
            trajectory_id,
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
    """Collects task batches from the queue and passes them to `_process_batch`.

    In normal operation (`_batch_wait_s == 0.0`), collects a 1-task batch
    immediately without waiting. When batching is active (`_batch_wait_s > 0`),
    collects up to `max_items` or `_batch_wait_s`. Always calls `task_done()`
    in a `finally` block for every dequeued item so `flush()` barriers never
    hang.
    """
    try:
      while True:
        tasks = self._collect_next_batch()
        if not tasks:
          break

        try:
          if ready_tasks := [t for t in tasks if self._project_task_to_atif(t)]:
            self._process_batch(ready_tasks)
          else:
            self._decrease_batch_wait()
        except Exception as exc:  # pylint: disable=broad-exception-caught
          if self._should_batch_on_error(exc):
            self._increase_batch_wait()
          logging.exception(
              "%s failed to process a write batch.", self._thread_name
          )
        finally:
          for _ in tasks:
            self._queue.task_done()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("%s exited on an unhandled error.", self._thread_name)

  def _project_task_to_atif(self, task: _TaskT) -> bool:
    """Projects `task` to ATIF in-place, logging and returning False on error."""
    try:
      task.to_atif()
      return True
    except Exception:  # pylint: disable=broad-exception-caught
      self._safe_log_task_error(task)
      return False

  def _collect_next_batch(self) -> list[_TaskT]:
    """Waits for a task and collects up to `max_items` within `batch_wait_s`."""
    # Wait until at least one task (or the `None` shutdown signal) arrives so
    # the worker thread sleeps when the queue is empty.
    first_task = self._queue.get()
    if first_task is None:
      self._queue.task_done()
      return []

    collected_tasks: list[_TaskT] = [first_task]
    batch_wait_s = self._batch_wait_s
    max_items = self._batching.max_items
    if batch_wait_s > 0.0 and max_items > 1:
      batch_deadline = time.monotonic() + batch_wait_s
      while len(collected_tasks) < max_items:
        remaining_wait_s = batch_deadline - time.monotonic()
        if remaining_wait_s <= 0.0:
          break
        try:
          next_task = self._queue.get(timeout=remaining_wait_s)
        except queue.Empty:
          break
        if next_task is None:
          # Re-queue the shutdown signal so this partial batch flushes first
          # and the next loop iteration exits cleanly.
          self._queue.put_nowait(None)
          self._queue.task_done()
          break
        collected_tasks.append(next_task)

    return collected_tasks

  def _safe_log_task_error(self, task: _TaskT) -> None:
    """Calls `_log_task_error(task)` while suppressing any logging failure."""
    try:
      self._log_task_error(task)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("%s failed to log a task error.", self._thread_name)

  def _process_batch(self, tasks: Sequence[_TaskT]) -> None:
    """Processes a batch of `WriteTask` items collected from the queue.

    Stores that write a batch as a whole (such as a single SQL transaction or
    a chunked JSONL file) override this method. The default implementation
    iterates through `tasks` and calls `_process_task` for each task.
    """
    had_batchable_error = False
    for task in tasks:
      try:
        self._process_task(task)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if self._should_batch_on_error(exc):
          had_batchable_error = True
        self._safe_log_task_error(task)
    if had_batchable_error:
      self._increase_batch_wait()
    else:
      self._decrease_batch_wait()

  def _should_batch_on_error(self, exc: BaseException) -> bool:
    """Returns True if `exc` should trigger adaptive write batching.

    Overridden by concrete store writers (such as `SqlTrajectoryStore` and
    `FileTrajectoryStore`) using their domain-specific exception types.
    """
    del exc
    return False

  def _increase_batch_wait(self) -> None:
    """Steps `_batch_wait_s` up by 20% of `max_wait_s` (up to `max_wait_s`)."""
    step_up_s = self._batching.max_wait_s * _BATCH_WAIT_STEP_UP_FRACTION
    self._batch_wait_s = min(
        self._batching.max_wait_s,
        round(self._batch_wait_s + step_up_s, 6),
    )

  def _decrease_batch_wait(self) -> None:
    """Decays `_batch_wait_s` by 4% of `max_wait_s` down to `0.0`."""
    if self._batch_wait_s <= 0.0:
      return
    step_down_s = self._batching.max_wait_s * _BATCH_WAIT_STEP_DOWN_FRACTION
    self._batch_wait_s = max(0.0, round(self._batch_wait_s - step_down_s, 6))

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
    if task.metrics is not None:
      task_desc = "metrics"
    elif task.step is not None:
      task_desc = f"step (step_id={task.step.step_id})"
    else:
      task_desc = "metadata"

    logging.exception(
        "%s failed to write %s for trajectory_id=%s.",
        self._thread_name,
        task_desc,
        task.trajectory_id,
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
