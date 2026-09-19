"""File-based implementation for Trajectory Store."""

import dataclasses
import functools
import re
import types
from typing import Any, ClassVar, Final, Mapping

from absl import logging
from etils import epath
import pydantic
from tunix.experimental.trajectory import async_writer
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib

_METADATA_FILENAME: Final[str] = "metadata.json"
_TRAJECTORY_DIR_PREFIX: Final[str] = "traj_"
# Characters allowed in a trajectory_id: ASCII letters, digits, underscores, and
# hyphens. Shared between `_TRAJECTORY_ID_REGEX` and `_TRAJECTORY_DIR_REGEX` so
# that every ID written to disk is discoverable when listing trajectory
# directories.
_TRAJECTORY_ID_PATTERN: Final[str] = r"[a-zA-Z0-9_\-]+"
_TRAJECTORY_ID_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"^{_TRAJECTORY_ID_PATTERN}$"
)
_TRAJECTORY_DIR_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"^{_TRAJECTORY_DIR_PREFIX}(?P<trajectory_id>{_TRAJECTORY_ID_PATTERN})$"
)
_STEP_FILENAME_TEMPLATE: Final[str] = "step_{step_id:06d}.json"
_STEP_FILENAME_REGEX: Final[re.Pattern[str]] = re.compile(r"^step_\d+\.json$")


def _validate_trajectory_id(trajectory_id: str | None) -> str:
  """Validates that trajectory_id is non-empty and contains supported characters.

  Args:
    trajectory_id: The trajectory identifier to validate.

  Returns:
    The validated trajectory_id string.

  Raises:
    ValueError: If trajectory_id is None, empty, or contains characters that
      cannot be encoded in a trajectory directory name.
  """
  if not trajectory_id:
    raise ValueError("TrajectoryMetadata must have a non-empty trajectory_id.")
  if not _TRAJECTORY_ID_REGEX.match(trajectory_id):
    raise ValueError(
        f"trajectory_id {trajectory_id!r} contains unsupported characters; only"
        " letters, digits, underscores, and hyphens are allowed."
    )
  return trajectory_id


def _dump_json(model: pydantic.BaseModel) -> str:
  """Serializes a Pydantic model to indented JSON excluding None values."""
  return model.model_dump_json(indent=2, exclude_none=True)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FileWriteTask(async_writer.WriteTask):
  """File-specific write task containing filesystem destination paths."""

  traj_dir: epath.Path
  meta_path: epath.Path
  step_path: epath.Path | None = None


class _AsyncFileWriter(async_writer.AsyncWriter[_FileWriteTask]):
  """Asynchronously writes trajectory metadata and step files to disk."""

  def __init__(self):
    """Initializes _AsyncFileWriter without starting the background worker."""
    super().__init__()
    # In-memory cache mapping trajectory_id to the hash of its last written
    # metadata JSON. Used by the worker thread to skip redundant metadata.json
    # disk writes across steps.
    self._metadata_hash_by_trajectory_id: dict[str, int] = {}

  def enqueue_write(
      self,
      *,
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

    `metadata` and `step` are deep copied by `WriteTask`, so what lands
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
    task = _FileWriteTask(
        traj_dir=traj_dir,
        meta_path=meta_path,
        metadata=metadata,
        step_path=step_path,
        step=step,
    )
    self._enqueue(task)

  def _process_task(self, task: _FileWriteTask) -> None:
    """Processes a single write task by writing metadata and step files.

    Optimizations:
      - Directory Creation: `mkdir` is executed only once per trajectory on the
        first step, tracked by `_metadata_hash_by_trajectory_id`.
      - Metadata Caching: `metadata.json` is only written when its serialized
        content changes, minimizing redundant writes across multi-step turns.

    Args:
      task: Container holding directory paths, metadata, and step payload.
    """
    traj_id = task.trajectory_id

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

  def _log_task_error(self, task: _FileWriteTask) -> None:
    """Logs detailed task error with trajectory and filesystem path context.

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
        "%s failed to write trajectory %s (trajectory_id=%s) to %s.",
        self._thread_name,
        step_info,
        task.trajectory_id,
        target_path,
    )


class FileTrajectoryStore(
    store.TrajectoryStore, store.TrajectoryReader, store.TrajectoryWriter
):
  """File-based implementation satisfying TrajectoryReader and TrajectoryWriter.

  Architectural Separation of Responsibilities:
    `FileTrajectoryStore` acts as a lightweight frontend responsible for:
    1. Filesystem path layout, directory hierarchy, and naming conventions.
    2. Frontend input validation (trajectory ID format and presence).
    3. Synchronous trajectory reading and metadata queries (`get_trajectories`,
       `get_trajectories_metadata`).
    4. Forwarding step write tasks and flush barriers to `_AsyncFileWriter`.

    All asynchronous queuing, background worker thread lifecycle, error
    suppression for rollout resilience, and physical disk I/O are handled
    by `_AsyncFileWriter`.

  Directory Structure:
    <root_dir>/[<run_id>/]/
        └── traj_<trajectory_id>/
            ├── metadata.json
            ├── step_000001.json
            ├── step_000002.json
            └── ...
  """

  BACKEND: ClassVar[str] = "file"

  def __init__(
      self, root_dir: epath.PathLike, run_id: str | None = None
  ) -> None:
    """Initializes FileTrajectoryStore.

    Args:
      root_dir: Base directory path for storage (supports local paths and GCS
        uris e.g., 'gs://bucket/path').
      run_id: Optional unique identifier for the RL run. If provided, paths are
        scoped under root_dir / run_id. This ID MUST stay the same when
        recovering from failures or process restarts as long as the same RL
        process is being continued.

    Raises:
      ValueError: If root_dir is empty, or run_id is given but cannot be used
        as a directory name.
    """
    if not root_dir or not str(root_dir).strip():
      raise ValueError("FileTrajectoryStore requires a non-empty root_dir.")
    if run_id is not None and not _TRAJECTORY_ID_REGEX.match(run_id):
      # run_id becomes a path segment under root_dir, so it is held to the
      # same character rule as the trajectory directories nested beneath it.
      # A run_id carrying a path separator would nest those directories a
      # level deeper than `get_trajectories_metadata` looks for them, which
      # fails silently as an empty store rather than as an error.
      raise ValueError(
          f"run_id {run_id!r} contains unsupported characters; only letters,"
          " digits, underscores, and hyphens are allowed."
      )
    self._raw_root_dir = epath.Path(root_dir)
    self._run_id = run_id
    self._writer = _AsyncFileWriter()

  @classmethod
  def _from_config(cls, config: Mapping[str, Any]) -> "FileTrajectoryStore":
    """Builds a file-backed store from `config`.

    Args:
      config: Requires "root_dir" and "run_id".

    Returns:
      A new FileTrajectoryStore.

    Raises:
      ValueError: If "root_dir" or "run_id" is missing or empty.
    """
    root_dir = config.get("root_dir")
    if not root_dir or not str(root_dir).strip():
      raise ValueError(
          "Trajectory Store backend 'file' requires a non-empty 'root_dir'."
      )
    if not config.get("run_id"):
      # Optional on __init__, required here: a configured run is shared by the
      # orchestrator and every rollout worker, and every process restart of
      # them, and run_id is what scopes their common directory. Without it
      # every run writes straight into root_dir, and since trajectory ids
      # restart from the first prompt each run, a later run silently
      # overwrites an earlier one's trajectories.
      raise ValueError(
          "Trajectory Store backend 'file' requires a non-empty 'run_id'."
          " Generate it once, wherever the run is launched, and pass the same"
          " value to every process; a per-process default would split one run"
          " across as many directory trees as there are processes."
      )
    return cls(root_dir=config["root_dir"], run_id=config["run_id"])

  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent store.

    Raises:
      ValueError: If this store was constructed without a run_id.
    """
    if not self._run_id:
      raise ValueError(
          "FileTrajectoryStore without a run_id cannot export a valid"
          " distributed config."
      )
    return {
        "enabled": True,
        "backend": self.BACKEND,
        "root_dir": str(self._raw_root_dir),
        "run_id": self._run_id,
    }

  @functools.cached_property
  def root_dir(self) -> epath.Path:
    """Returns the effective root directory path."""
    return (
        self._raw_root_dir / self._run_id
        if self._run_id
        else self._raw_root_dir
    )

  def get_trajectory_dir(self, trajectory_id: str) -> epath.Path:
    """Returns the directory path for a given trajectory ID."""
    return self.root_dir / f"{_TRAJECTORY_DIR_PREFIX}{trajectory_id}"

  def get_trajectory_metadata_path(self, trajectory_id: str) -> epath.Path:
    """Returns the file path for a given trajectory ID's metadata."""
    return self.get_trajectory_dir(trajectory_id) / _METADATA_FILENAME

  def get_step_path(self, trajectory_id: str, step_id: int) -> epath.Path:
    """Returns the file path for a given trajectory ID and step ID."""
    step_filename = _STEP_FILENAME_TEMPLATE.format(step_id=step_id)
    return self.get_trajectory_dir(trajectory_id) / step_filename

  def get_trajectories_metadata(
      self, trajectory_ids: list[str] | None = None
  ) -> list[trajectory_lib.TrajectoryMetadata]:
    """Retrieves metadata for trajectories in the run.

    Args:
      trajectory_ids: Optional list of unique trajectory identifiers. If
        specified, only metadata for these IDs is returned. If None, metadata
        for all trajectories in the run is returned.

    Returns:
      A list of TrajectoryMetadata objects for the requested trajectories.

    Raises:
      store.TrajectoryMetadataNotFoundError: If any requested trajectory ID does
        not exist.
    """
    metas: list[trajectory_lib.TrajectoryMetadata] = []
    if trajectory_ids is None:
      if not self.root_dir.exists():
        return metas
      trajectory_ids = []
      for entry in self.root_dir.iterdir():
        if not entry.is_dir():
          continue
        if not (match := _TRAJECTORY_DIR_REGEX.match(entry.name)):
          continue
        trajectory_ids.append(match.group("trajectory_id"))

    for traj_id in trajectory_ids:
      meta_path = self.get_trajectory_metadata_path(traj_id)
      if not meta_path.exists():
        raise store.TrajectoryMetadataNotFoundError(traj_id)
      meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
          meta_path.read_text()
      )
      metas.append(meta)

    return metas

  def get_trajectories(
      self, trajectory_ids: list[str]
  ) -> list[trajectory_lib.Trajectory]:
    """Retrieves full trajectories for a list of trajectory IDs.

    Args:
      trajectory_ids: List of unique trajectory identifiers to load.

    Returns:
      A list of full Trajectory objects corresponding to the requested IDs.

    Raises:
      store.TrajectoryNotFoundError: If any requested trajectory ID does not
      exist.
    """
    trajs: list[trajectory_lib.Trajectory] = []

    for traj_id in trajectory_ids:
      traj_dir = self.get_trajectory_dir(traj_id)
      meta_path = self.get_trajectory_metadata_path(traj_id)
      if not meta_path.exists():
        raise store.TrajectoryNotFoundError(traj_id)

      meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
          meta_path.read_text()
      )
      steps: list[trajectory_lib.Step] = []

      for file_entry in traj_dir.iterdir():
        if not _STEP_FILENAME_REGEX.match(file_entry.name):
          continue
        step = trajectory_lib.Step.model_validate_json(file_entry.read_text())
        steps.append(step)

      traj_data = meta.model_dump()
      traj_data["steps"] = steps
      trajs.append(trajectory_lib.Trajectory(**traj_data))

    return trajs

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Asynchronously logs a turn step and its trajectory metadata.

    Performs synchronous frontend validation of the trajectory ID on the
    calling thread so invalid IDs fail fast with actionable errors, then
    delegates asynchronous queuing and non-blocking background I/O to the
    `_AsyncFileWriter`.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty, None, or contains
        characters that cannot be encoded in a trajectory directory name.
    """
    self.update_metadata(metadata, step=step)

  def update_metadata(
      self,
      metadata: trajectory_lib.TrajectoryMetadata,
      step: trajectory_lib.Step | None = None,
  ) -> None:
    """Updates or creates trajectory metadata asynchronously.

    Performs synchronous frontend validation of the trajectory ID on the
    calling thread so invalid IDs fail fast with actionable errors, then
    delegates asynchronous queuing and non-blocking background I/O to the
    `_AsyncFileWriter`.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write alongside metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty, None, or contains
        characters that cannot be encoded in a trajectory directory name.
    """
    traj_id = _validate_trajectory_id(metadata.trajectory_id)
    traj_dir = self.get_trajectory_dir(traj_id)
    meta_path = self.get_trajectory_metadata_path(traj_id)
    step_path = self.get_step_path(traj_id, step.step_id) if step else None
    self._writer.enqueue_write(
        traj_dir=traj_dir,
        meta_path=meta_path,
        metadata=metadata,
        step_path=step_path,
        step=step,
    )

  def flush(self) -> None:
    """Flushes any pending or asynchronous writes to persistent storage.

    Users do not need to call `flush()` in normal usage; it is primarily for
    testing.

    Delegates directly to `_AsyncFileWriter.flush()` to provide strict barrier
    synchronization. The barrier does not apply to a closed store, for which
    this is a no-op.
    """
    self._writer.flush()

  def close(self) -> None:
    """Flushes pending writes and shuts down the background writer thread.

    Calling `close()` is optional: the underlying `_AsyncFileWriter` also drains
    itself at interpreter exit. It is worth calling explicitly for a store that
    becomes garbage well before the process ends, so its worker thread is
    released promptly. Closing is idempotent, but the store must not be written
    to afterwards; reads remain available.

    The writer is given a bounded window to drain (see `AsyncWriter.close`); any
    writes still queued when that window expires are discarded and the affected
    trajectory IDs are logged.
    """
    self._writer.close()

  def __enter__(self) -> "FileTrajectoryStore":
    """Returns this store, for use as a context manager."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_value: BaseException | None,
      traceback: types.TracebackType | None,
  ) -> None:
    """Closes the store on exiting the context manager."""
    self.close()
