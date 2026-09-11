"""Asynchronous file writer for Trajectory Store."""

from absl import logging
from etils import epath
import pydantic
from tunix.experimental.trajectory import base_writer
from tunix.experimental.trajectory import trajectory as trajectory_lib


def _dump_json(model: pydantic.BaseModel) -> str:
  """Serializes a Pydantic model to indented JSON excluding None values."""
  return model.model_dump_json(indent=2, exclude_none=True)


class AsyncFileWriter(base_writer.BaseAsyncWriter):
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
       occurs. `AsyncFileWriter` inherits `close()` and `atexit` draining from
       `BaseAsyncWriter`.
  """

  def __init__(self):
    """Initializes AsyncFileWriter without starting the background worker."""
    super().__init__()
    # In-memory cache mapping trajectory_id to the hash of its last written
    # metadata JSON. Used by the worker thread to skip redundant metadata.json
    # disk writes across steps.
    self._metadata_hash_by_trajectory_id: dict[str, int] = {}

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
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step_path: Optional file path for the step JSON.
      step: Optional Step object to write.

    Raises:
      RuntimeError: If the writer has already been closed.
    """
    task = base_writer.WriteTask(
        traj_dir=traj_dir,
        meta_path=meta_path,
        step_path=step_path,
        metadata=metadata.model_copy(deep=True),
        step=step.model_copy(deep=True) if step is not None else None,
    )
    self._enqueue(task, "AsyncFileWriterWorker")

  def _process_task(self, task: base_writer.WriteTask) -> None:
    """Processes a single write task by writing metadata and step files.

    Optimizations:
      - Directory Creation: `mkdir` is executed only once per trajectory on the
        first step, tracked by `_metadata_hash_by_trajectory_id`.
      - Metadata Caching: `metadata.json` is only written when its serialized
        content changes, minimizing redundant writes across multi-step turns.

    Args:
      task: Container holding directory paths, metadata, and step payload.

    Raises:
      ValueError: If metadata.trajectory_id, traj_dir, or meta_path is None.
    """
    traj_id = task.metadata.trajectory_id
    if traj_id is None:
      raise ValueError("TrajectoryMetadata.trajectory_id cannot be None.")

    if task.traj_dir is None or task.meta_path is None:
      raise ValueError(
          "traj_dir and meta_path must not be None for AsyncFileWriter."
      )

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

  def _log_task_error(self, task: base_writer.WriteTask) -> None:
    """Logs detailed task error with trajectory and step path context.

    Args:
      task: The write task that failed to process.
    """
    step_info = (
        f"step {task.step.step_id}" if task.step is not None else "metadata"
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
