"""Asynchronous queue and worker thread engine for Trajectory Store."""

import abc
import atexit
from collections.abc import Mapping, Sequence
import dataclasses
import queue
import sys
import threading
import time
from typing import Any, Final, Generic, TypeVar
import weakref

from absl import logging
import numpy as np
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

# ==============================================================================
# --- Queue Capacity & Three-Tier Admission Constants ---
# ==============================================================================

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
# reserved for metadata-only updates (`task.step is None and task.metrics is
# None`) so trajectory records and final episode statuses are preserved when
# metrics and step writes fill the queue.
_METRICS_QUEUE_CAPACITY_RATIO: Final[float] = 0.95

# ==============================================================================
# --- Adaptive Batching & AIMD Backoff Constants ---
# ==============================================================================

# Fraction of `BatchConfig.max_wait_s` added to `_wait_s` on each throttle
# error (+20%, or +50 ms when `max_wait_s == 0.25`).
_WAIT_STEP_UP_FRACTION: Final[float] = 0.2

# Fraction of `BatchConfig.max_wait_s` subtracted from `_wait_s` on each clean
# batch (-4%, or -10 ms when `max_wait_s == 0.25`). Because step-down is 5x
# smaller than step-up, 1 throttle error decays in 5 clean batches, 2 throttle
# errors decay in 10 clean batches, and full saturation decays in 25 clean
# batches.
_WAIT_STEP_DOWN_FRACTION: Final[float] = 0.04


@dataclasses.dataclass(frozen=True)
class BatchConfig:
  """Configuration for adaptive worker write batching under contention.

  In normal unthrottled operation (`current_wait_s == 0.0`), the worker writes
  each task immediately (`1` task per write) with zero waiting and zero
  batching. When a backend reports a throttle or contention error via
  `_is_throttle_error`, the worker increases `current_wait_s` up to `max_wait_s`
  and drains queued tasks into a batch up to `max_items` or `max_bytes`.

  Attributes:
    max_items: Maximum number of `WriteTask` items drained into one batch when
      throttling is active. Defaults to 32.
    max_bytes: Maximum cumulative estimated payload bytes drained into one batch
      when throttling is active (`first_task` is always admitted even if it
      exceeds `max_bytes`). Defaults to 4 MiB (4,194,304 bytes).
    max_wait_s: Upper bound in seconds on how long the worker waits for
      additional tasks to accumulate when storage is throttled. Normal
      unthrottled operation waits `0.0` seconds. Defaults to 0.25 seconds.
  """

  max_items: int = 32
  max_bytes: int = 4 * 1024 * 1024
  max_wait_s: float = 0.25

  def __post_init__(self) -> None:
    """Validates batch sizing and wait bounds."""
    if self.max_items <= 0:
      raise ValueError(
          f"BatchConfig.max_items must be positive, got {self.max_items}."
      )
    if self.max_bytes <= 0:
      raise ValueError(
          f"BatchConfig.max_bytes must be positive, got {self.max_bytes}."
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
  thread. `metadata`, `step`, and `metrics` are private deep copies owned by
  the task.
  """

  metadata: trajectory_lib.TrajectoryMetadata | None = None
  step: trajectory_lib.Step | None = None
  run_id: str | None = None
  trajectory_id: str = ""
  metrics: trajectory_lib.TrajectoryMetrics | None = None

  def __post_init__(self) -> None:
    """Deep copies the payload so later caller mutations cannot leak in."""
    if self.metadata is not None:
      object.__setattr__(
          self, "metadata", trajectory_lib.deep_copy(self.metadata)
      )
      if not self.trajectory_id and self.metadata.trajectory_id:
        object.__setattr__(self, "trajectory_id", self.metadata.trajectory_id)
    if self.step is not None:
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
    the frozen instance in-place to avoid a second deep copy.
    """
    if self.metadata is not None:
      object.__setattr__(self, "metadata", self.metadata.to_atif_metadata())
    if self.step is not None:
      object.__setattr__(self, "step", self.step.to_atif_step())


_TaskT = TypeVar("_TaskT", bound=WriteTask)


def _estimate_task_bytes(task: WriteTask) -> int:
  """Estimates the in-memory byte size of `task` when draining a batch."""
  total_bytes = 0
  seen_ids: set[int] = set()
  stack: list[Any] = [
      task.metadata,
      task.step,
      task.metrics,
      task.run_id,
      task.trajectory_id,
  ]

  while stack:
    current = stack.pop()
    if current is None:
      continue
    if isinstance(current, (int, float, bool, str, bytes, bytearray)):
      total_bytes += sys.getsizeof(current)
      continue

    obj_id = id(current)
    if obj_id in seen_ids:
      continue
    seen_ids.add(obj_id)

    if isinstance(current, np.ndarray):
      total_bytes += int(current.nbytes)
    elif isinstance(current, Mapping):
      total_bytes += sys.getsizeof(current)
      stack.extend(current.keys())
      stack.extend(current.values())
    elif isinstance(current, (list, tuple)):
      total_bytes += sys.getsizeof(current)
      if current and isinstance(current[0], (int, float, bool)):
        total_bytes += len(current) * sys.getsizeof(current[0])
      else:
        stack.extend(current)
    elif hasattr(current, "__dict__"):
      total_bytes += sys.getsizeof(current)
      stack.extend(vars(current).values())
    else:
      total_bytes += sys.getsizeof(current)

  return total_bytes


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrajectoryBatch(Generic[_TaskT]):
  """Batched write tasks grouped for a single `(run_id, trajectory_id)` pair.

  When multiple tasks for the same trajectory arrive in one batch,
  `TrajectoryBatch` keeps only the newest `metadata` task
  (`latest_metadata_task`),
  collects all `step` tasks in arrival order (`step_tasks`), and shallow-merges
  all `metrics` tasks into a single `merged_metrics_task` so backends write
  metadata and metrics at most once per trajectory per batch.

  Attributes:
    run_id: Optional run identifier shared by tasks in this trajectory batch.
    trajectory_id: Trajectory identifier shared by tasks in this batch.
    latest_metadata_task: Latest task carrying non-None `metadata` for this
      trajectory, or `None` if all tasks in the group are metrics-only.
    step_tasks: Ordered tuple of tasks carrying a `Step` payload (`step is not
      None`), in arrival order.
    merged_metrics_task: Single task carrying the shallow-merged
      `TrajectoryMetrics` (`extra` dict merged in arrival order) across all
      tasks with `metrics is not None` for this trajectory, or `None` if no task
      carries `metrics`.
    tasks: Complete ordered tuple of all raw tasks for this trajectory.
  """

  run_id: str | None
  trajectory_id: str
  latest_metadata_task: _TaskT | None = None
  step_tasks: tuple[_TaskT, ...] = ()
  merged_metrics_task: _TaskT | None = None
  tasks: tuple[_TaskT, ...] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class WriteBatch(Generic[_TaskT]):
  """A batch of `WriteTask` items drained from `AsyncWriter` in one cycle.

  Groups tasks by `(run_id, trajectory_id)` in first-seen trajectory order so
  storage backends can write each trajectory's latest metadata, ordered steps,
  and merged metrics in one operation while isolating distinct trajectories.

  Attributes:
    tasks: Ordered tuple of all raw `WriteTask` items in the batch.
    trajectory_batches: Ordered tuple of per-trajectory `TrajectoryBatch` items
      in first-seen `(run_id, trajectory_id)` order.
  """

  tasks: tuple[_TaskT, ...]
  trajectory_batches: tuple[TrajectoryBatch[_TaskT], ...]

  @property
  def trajectory_groups(self) -> tuple[TrajectoryBatch[_TaskT], ...]:
    """Alias for `trajectory_batches`."""
    return self.trajectory_batches

  @classmethod
  def from_tasks(cls, tasks: Sequence[_TaskT]) -> "WriteBatch[_TaskT]":
    """Constructs a `WriteBatch` grouped by `(run_id, trajectory_id)`.

    Args:
      tasks: Non-empty sequence of `WriteTask` instances drained from the queue.

    Returns:
      A `WriteBatch` containing `tasks` and per-trajectory `trajectory_batches`.

    Raises:
      ValueError: If `tasks` is empty.
    """
    if not tasks:
      raise ValueError("WriteBatch.from_tasks requires at least one task.")

    tasks_by_traj_key: dict[tuple[str | None, str], list[_TaskT]] = {}
    for task in tasks:
      traj_key = (task.run_id, task.trajectory_id)
      if traj_key not in tasks_by_traj_key:
        tasks_by_traj_key[traj_key] = []
      tasks_by_traj_key[traj_key].append(task)

    batches_by_traj: list[TrajectoryBatch[_TaskT]] = []
    for (run_id, trajectory_id), traj_tasks in tasks_by_traj_key.items():
      latest_metadata_task: _TaskT | None = None
      step_tasks: list[_TaskT] = []
      metrics_tasks: list[_TaskT] = []
      for task in traj_tasks:
        if task.metadata is not None:
          latest_metadata_task = task
        if task.step is not None:
          step_tasks.append(task)
        if task.metrics is not None:
          metrics_tasks.append(task)

      merged_metrics_task: _TaskT | None = None
      if len(metrics_tasks) == 1:
        merged_metrics_task = metrics_tasks[0]
      elif len(metrics_tasks) > 1:
        merged_extra: dict[str, Any] = {}
        for metrics_task in metrics_tasks:
          if (
              metrics_task.metrics is not None
              and metrics_task.metrics.extra is not None
          ):
            merged_extra.update(metrics_task.metrics.extra)
        last_metrics_task = metrics_tasks[-1]
        merged_metrics = trajectory_lib.TrajectoryMetrics(
            extra=merged_extra if merged_extra else None
        )
        merged_metrics_task = dataclasses.replace(
            last_metrics_task,
            metadata=None,
            step=None,
            metrics=merged_metrics,
        )

      batches_by_traj.append(
          TrajectoryBatch(
              run_id=run_id,
              trajectory_id=trajectory_id,
              latest_metadata_task=latest_metadata_task,
              step_tasks=tuple(step_tasks),
              merged_metrics_task=merged_metrics_task,
              tasks=tuple(traj_tasks),
          )
      )

    return cls(
        tasks=tuple(tasks),
        trajectory_batches=tuple(batches_by_traj),
    )


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

    4. Adaptive Contention-Driven Batching (AIMD on `_wait_s`):
       `BatchConfig` is active by default (`DEFAULT_BATCH_CONFIG`). In normal
       operation (`_wait_s == 0.0`), the worker executes each task immediately
       (`1` task per write) with zero waiting and zero batching latency. When a
       backend subclass reports a throttle error via `_is_throttle_error`,
       `_record_throttle_error` steps `_wait_s` up by 20% of `max_wait_s` (up
       to `max_wait_s`), activating batch draining via `queue.get(timeout=...)`
       up to `max_items` or `max_bytes`. Each clean batch decays `_wait_s` by
       4% of `max_wait_s` down to `0.0` (5 clean batches per step-up, up to 25
       clean batches from full saturation).

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
  ):
    """Initializes AsyncWriter without starting the background worker.

    Args:
      thread_name: Descriptive name for the background worker thread. Defaults
        to '<ClassName>Worker'.
      max_queue_size: Maximum number of pending tasks buffered in the queue
        before dropping incoming writes. Defaults to `DEFAULT_MAX_QUEUE_SIZE`
        (10,000).
      batching: Configuration governing batch item/byte limits and maximum
        throttle wait time. Defaults to `DEFAULT_BATCH_CONFIG`.

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
    self._metrics_queue_limit = max(
        1, int(max_queue_size * _METRICS_QUEUE_CAPACITY_RATIO)
    )
    self._batching: BatchConfig = (
        batching if isinstance(batching, BatchConfig) else DEFAULT_BATCH_CONFIG
    )
    # FIFO queue for passing write tasks to the worker thread. Capacity limits
    # (`_step_queue_limit`, `_metrics_queue_limit`, and `_max_queue_size`) are
    # enforced under `self._lock` in `_enqueue` so that `close()` can always
    # append the `None` shutdown sentinel without blocking when the queue is
    # full.
    self._queue: queue.Queue[_TaskT | None] = queue.Queue()
    # Lock protecting lazy thread spawning, queue admission, `_wait_s`, and
    # closed state.
    self._lock = threading.Lock()
    # Current adaptive batch wait time in seconds (`0.0` in normal unthrottled
    # mode; scales up to `_batching.max_wait_s` on throttle errors and decays
    # back to `0.0` across clean batches).
    self._wait_s: float = 0.0
    self._shutdown_sentinel_pending: bool = False
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
  def current_wait_s(self) -> float:
    """Returns the current adaptive batch wait duration in seconds."""
    with self._lock:
      return self._wait_s

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
    """Worker loop draining and processing tasks or batches from the queue.

    In normal operation (`_wait_s == 0.0`), processes each task immediately
    without waiting or batching. When throttled (`_wait_s > 0.0`), drains tasks
    into a `WriteBatch` up to `max_items`, `max_bytes`, or `_wait_s`. Always
    calls `task_done()` in a `finally` block for every dequeued item so
    `flush()` barriers never hang.
    """
    try:
      while True:
        batch_tasks = self._drain_next_batch()
        if not batch_tasks:
          break

        try:
          for task in batch_tasks:
            task.to_atif()
          batch = WriteBatch.from_tasks(batch_tasks)
          self._process_batch(batch)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          if self._is_throttle_error(exc):
            self._record_throttle_error()
          if len(batch_tasks) == 1:
            try:
              self._log_task_error(batch_tasks[0])
            except Exception:  # pylint: disable=broad-exception-caught
              logging.exception(
                  "%s failed to log a task error.", self._thread_name
              )
          else:
            logging.warning(
                "%s batch write of %d tasks failed; falling back to sequential"
                " per-task writes.",
                self._thread_name,
                len(batch_tasks),
                exc_info=True,
            )
            self._replay_tasks_individually(batch_tasks)
        finally:
          for _ in batch_tasks:
            self._queue.task_done()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("%s exited on an unhandled error.", self._thread_name)

  def _drain_next_batch(self) -> list[_TaskT]:
    """Drains the next single task (`_wait_s == 0.0`) or throttled batch."""
    if self._shutdown_sentinel_pending:
      self._shutdown_sentinel_pending = False
      self._queue.task_done()
      return []

    first_task = self._queue.get()
    if first_task is None:
      self._queue.task_done()
      return []

    wait_s = self.current_wait_s
    # In the default unthrottled state (`_wait_s == 0.0`), execute immediately
    # with zero batching and zero waiting.
    if wait_s <= 0.0:
      return [first_task]

    max_items = self._batching.max_items
    max_bytes = self._batching.max_bytes
    batch_tasks: list[_TaskT] = [first_task]
    if max_items == 1:
      return batch_tasks

    batch_bytes = _estimate_task_bytes(first_task)
    if batch_bytes >= max_bytes:
      return batch_tasks

    deadline = time.monotonic() + wait_s
    while len(batch_tasks) < max_items:
      remaining_s = deadline - time.monotonic()
      try:
        if remaining_s > 0.0:
          next_task = self._queue.get(timeout=remaining_s)
        else:
          next_task = self._queue.get_nowait()
      except queue.Empty:
        break

      if next_task is None:
        self._shutdown_sentinel_pending = True
        break

      batch_tasks.append(next_task)
      batch_bytes += _estimate_task_bytes(next_task)
      if batch_bytes >= max_bytes:
        break

    return batch_tasks

  def _replay_tasks_individually(self, tasks: Sequence[_TaskT]) -> None:
    """Replays each task in `tasks` sequentially via `_process_task`."""
    for task in tasks:
      try:
        self._process_task(task)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if self._is_throttle_error(exc):
          self._record_throttle_error()
        try:
          self._log_task_error(task)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.exception("%s failed to log a task error.", self._thread_name)

  def _process_batch(self, batch: WriteBatch[_TaskT]) -> None:
    """Processes a drained `WriteBatch` and updates adaptive backoff state.

    Subclasses with bulk storage operations (such as single-transaction SQL
    upserts or chunked file writes) override this method and call
    `self._record_batch_success()` upon completion. The default implementation
    dispatches each task sequentially to `_process_task`.

    Args:
      batch: `WriteBatch` containing one or more `WriteTask` items.
    """
    throttled_in_batch = False
    for task in batch.tasks:
      try:
        self._process_task(task)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if self._is_throttle_error(exc):
          self._record_throttle_error()
          throttled_in_batch = True
        try:
          self._log_task_error(task)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.exception("%s failed to log a task error.", self._thread_name)
    if not throttled_in_batch:
      self._record_batch_success()

  def _is_throttle_error(self, exc: BaseException) -> bool:
    """Returns True if `exc` indicates storage rate-limiting or lock contention.

    Overridden by concrete store writers (such as `SqlTrajectoryStore` and
    `FileTrajectoryStore`) using their domain-specific exception types.

    Args:
      exc: Exception raised during task or batch processing.
    """
    del exc
    return False

  def _record_throttle_error(self) -> None:
    """Steps `_wait_s` up by 20% of `max_wait_s` (capped at `max_wait_s`)."""
    with self._lock:
      step_up_s = self._batching.max_wait_s * _WAIT_STEP_UP_FRACTION
      self._wait_s = min(
          self._batching.max_wait_s,
          round(self._wait_s + step_up_s, 6),
      )

  def _record_batch_success(self) -> None:
    """Decays `_wait_s` by 4% of `max_wait_s` down to `0.0` on a clean batch."""
    if self._wait_s <= 0.0:
      return
    with self._lock:
      step_down_s = self._batching.max_wait_s * _WAIT_STEP_DOWN_FRACTION
      self._wait_s = max(0.0, round(self._wait_s - step_down_s, 6))

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
