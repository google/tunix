"""Asynchronous file writer for Trajectory Store."""

import atexit
import dataclasses
import queue
import threading
import weakref

from absl import logging
from etils import epath
import pydantic
from tunix.experimental.trajectory import trajectory as trajectory_lib


def _dump_json(model: pydantic.BaseModel) -> str:
  """Serializes a Pydantic model to indented, human-readable JSON excluding None values."""
  return model.model_dump_json(indent=2, exclude_none=True)


@dataclasses.dataclass(frozen=True)
class _WriteTask:
  """Container for an asynchronous step write operation.

  Encapsulates all necessary data transferred across the thread boundary from
  the frontend calling thread (e.g. rollout worker) to the background worker
  thread executing disk I/O. `metadata` and `step` are private deep copies
  owned by the task, never the caller's live objects; see `write_step`.
  """

  traj_dir: epath.Path
  meta_path: epath.Path
  step_path: epath.Path | None
  metadata: trajectory_lib.TrajectoryMetadata
  step: trajectory_lib.Step | None


# Every live (i.e. not garbage collected) AsyncFileWriter, so that
# `_close_live_writers` can drain them at interpreter shutdown. Weak references
# are used so registration does not keep writers alive.
_LIVE_WRITERS: "weakref.WeakSet[AsyncFileWriter]" = weakref.WeakSet()


def _close_live_writers() -> None:
  """Closes every live AsyncFileWriter, draining its pending writes.

  Registered with `atexit`, which runs while daemon threads are still alive but
  before the interpreter kills them. Without this, steps still sitting in a
  writer's queue when the process ends are silently lost, because the worker is
  a daemon thread and `__del__` is not guaranteed to run for objects that are
  still referenced at shutdown.
  """
  for writer in list(_LIVE_WRITERS):
    try:
      writer.close()
    except Exception:  # pylint: disable=broad-exception-caught
      # Best-effort, consistent with the writer's error handling: a failure to
      # persist diagnostic data must not turn into a non-zero exit status.
      logging.exception("Failed to close AsyncFileWriter at interpreter exit.")


atexit.register(_close_live_writers)


class AsyncFileWriter:
  """Asynchronously writes trajectory metadata and step files to disk.

  Architectural Decisions & Design Trade-offs:
    1. Single Background Worker Thread:
       A dedicated single background daemon thread processes write tasks
       sequentially from an unbounded FIFO queue (`queue.Queue`). Using a single
       sequential worker ensures:
       - Strict chronological ordering of steps per trajectory without needing
         complex per-file or per-trajectory locks.
       - Elimination of concurrent file write races or corruptions.
       - Minimal memory and thread overhead, which is critical in distributed
         reinforcement learning (RL) training where dozens of rollout worker
         processes run concurrently on each host.

    2. Lazy Worker Thread Initialization:
       The worker thread is NOT spawned during `__init__`. Instead, it is
       lazily initialized on the first invocation of `write_step()` under a
       thread lock (`_lock`). This prevents unnecessary OS thread allocation
       and resource waste in read-heavy or read-only processes (such as offline
       evaluators, visualizers, or analysis scripts) that instantiate a store
       solely to query trajectories.

    3. Best-Effort Error Handling for Rollout Worker Resilience:
       In distributed RL environments (e.g., Tunix rollout workers), trajectory
       persistence is non-critical diagnostic and telemetry data compared to the
       primary training loop and policy rollout generation. If disk I/O fails
       (e.g., disk full, transient network filesystem error, or permission
       issues), raising an exception would crash the entire distributed training
       job. Therefore, `_worker_loop()` catches all exceptions during task
       processing, logs them with full traceback via `logging.exception`, and
       continues draining subsequent tasks. Errors are suppressed and never
       propagated back to `write_step()` or `flush()`.

    4. Strict Barrier Synchronization via `flush()`:
       `flush()` provides strict barrier synchronization by blocking on
       `_queue.join()`. When `flush()` returns, all write tasks enqueued prior
       to the call are guaranteed to have been processed by the worker thread.
       This enables deterministic testing, reliable step inspection, and clean
       synchronization at episode or checkpoint boundaries.

    5. Snapshot-on-Enqueue Ownership:
       Because writes are serialized on the worker thread rather than on the
       caller thread, `write_step()` deep copies the metadata and step it is
       given. The queued task then owns data no other thread can mutate, so a
       caller that keeps updating a step or trajectory after logging it cannot
       corrupt the file that is about to be written.

    6. Daemon Thread Lifecycle, Destructor & Shutdown Hook:
       The background worker thread is marked as a daemon (`daemon=True`) so it
       never blocks Python process termination if an unhandled signal or exit
       occurs. The `__del__` destructor provides a best-effort graceful shutdown
       signal and joins the worker with a short timeout during garbage
       collection. Because a writer that is still referenced at process exit is
       never garbage collected, and because daemon threads are killed outright
       once the interpreter shuts down, every instance also registers itself in
       `_LIVE_WRITERS`; the `atexit` hook `_close_live_writers` drains them all
       so queued steps are not lost when a rollout worker exits normally.
  """

  def __init__(self) -> None:
    """Initializes AsyncFileWriter without starting the background worker."""
    # Unbounded FIFO queue for passing write tasks to the worker thread.
    self._queue: queue.Queue[_WriteTask | None] = queue.Queue()
    # In-memory cache mapping trajectory_id to the hash of its last written
    # metadata JSON. Used by the worker thread to skip redundant metadata.json
    # disk writes across steps.
    self._metadata_hash_by_trajectory_id: dict[str, int] = {}
    # Lock protecting lazy thread spawning and closed state transitions.
    self._lock = threading.Lock()
    # Users are not expected to explicitly call close() on the writer, as its
    # lifecycle is managed automatically.
    self._closed: bool = False
    self._worker_thread: threading.Thread | None = None
    # Drained by `_close_live_writers` at interpreter exit.
    _LIVE_WRITERS.add(self)

  def write_step(
      self,
      traj_dir: epath.Path,
      meta_path: epath.Path,
      metadata: trajectory_lib.TrajectoryMetadata,
      step_path: epath.Path | None = None,
      step: trajectory_lib.Step | None = None,
  ) -> None:
    """Enqueues a step and/or trajectory metadata for asynchronous writing.

    This operation is non-blocking and returns on the caller thread without
    waiting for any disk I/O. The worker thread is lazily spawned on the first
    invocation if not already running.

    `metadata` and `step` are deep copied before being enqueued, so what lands
    on disk is exactly what the caller passed in. Serialization happens on the
    worker thread, possibly long after this call returns, and callers routinely
    keep mutating the objects they hand over (a rollout worker appending tokens
    to the step it just logged, or flipping trajectory status from RUNNING to
    COMPLETED). Without the copy, those later mutations would leak into the
    already enqueued write, producing files that never matched any state the
    trajectory actually had. The copy makes the caller-side cost proportional
    to the payload size rather than O(1), which is a deliberate trade for
    correctness; the expensive part, serialization and I/O, remains off the
    caller thread.

    Args:
      traj_dir: Directory path for the trajectory.
      meta_path: File path for the trajectory metadata.json.
      step_path: Optional file path for the step JSON.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write.

    Raises:
      RuntimeError: If the writer has already been closed.
    """
    task = _WriteTask(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=metadata.model_copy(deep=True),
        step=step.model_copy(deep=True) if step is not None else None,
    )
    with self._lock:
      if self._closed:
        raise RuntimeError("Cannot write to a closed AsyncFileWriter.")
      if self._worker_thread is None:
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncFileWriterWorker",
            daemon=True,
        )
        self._worker_thread.start()
      self._queue.put(task)

  def _worker_loop(self) -> None:
    """Worker loop processing write tasks sequentially from the queue.

    Catches and logs all task processing exceptions without propagating them
    to callers or breaking the loop, ensuring rollout workers are never failed
    by write errors. Uses `task_done()` in a `finally` block to ensure queue
    join barriers (`flush()`) unblock even when tasks fail.
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
          # Best-effort error suppression: log full traceback but never crash the worker.
          step_info = (
              f"step {task.step.step_id}"
              if task.step is not None
              else "metadata"
          )
          target_path = (
              task.step_path if task.step_path is not None else task.meta_path
          )
          logging.exception(
              "Failed to write trajectory %s (trajectory_id=%s) to %s",
              step_info,
              task.metadata.trajectory_id,
              target_path,
          )
        finally:
          # Crucial: always mark task as done so flush() barrier does not hang on failure.
          self._queue.task_done()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Fatal unhandled error in AsyncFileWriter worker loop.")

  def _process_task(self, task: _WriteTask) -> None:
    """Processes a single write task by writing metadata and step files.

    Optimizations:
      - Directory Creation: `mkdir` is executed only once per trajectory on the
        first step, tracked by `_metadata_hash_by_trajectory_id`.
      - Metadata Caching: `metadata.json` is only written when its serialized
        content changes, minimizing redundant writes across multi-step turns.

    Args:
      task: Container holding directory paths, metadata, and step payload.

    Raises:
      ValueError: If metadata.trajectory_id is None.
    """
    traj_id = task.metadata.trajectory_id
    if traj_id is None:
      raise ValueError("TrajectoryMetadata.trajectory_id cannot be None.")

    # Create directory on first step of this trajectory.
    if traj_id not in self._metadata_hash_by_trajectory_id:
      task.traj_dir.mkdir(parents=True, exist_ok=True)

    # Only write metadata.json if metadata content has changed.
    meta_json = _dump_json(task.metadata)
    meta_hash = hash(meta_json)
    if self._metadata_hash_by_trajectory_id.get(traj_id) != meta_hash:
      task.meta_path.write_text(meta_json)
      self._metadata_hash_by_trajectory_id[traj_id] = meta_hash

    # Write step file if provided.
    if task.step_path is not None and task.step is not None:
      task.step_path.write_text(_dump_json(task.step))

  def flush(self) -> None:
    """Blocks until all queued write operations have been processed to disk.

    Users do not need to call flush() in normal usage; it is primarily for
    testing.

    Provides strict barrier synchronization: when this method returns, all
    tasks enqueued prior to the call have been executed by the worker thread.
    """
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
            "AsyncFileWriter worker thread did not finish within timeout of"
            " %s seconds. Discarded remaining tasks for trajectory IDs: %s",
            timeout,
            sorted(discarded_traj_ids),
        )

  def __del__(self) -> None:
    """Destructor to ensure worker thread shutdown is signaled upon garbage collection."""
    try:
      self.close(timeout=1.0)
    except BaseException:  # pylint: disable=broad-exception-caught
      pass
