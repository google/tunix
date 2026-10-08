"""File-based implementation for Trajectory Store."""

import collections
from concurrent import futures
import dataclasses
import functools
import re
import types
from typing import Any, ClassVar, Final, Mapping, TypeVar

from absl import logging
from etils import epath
import pydantic
from tunix.experimental.trajectory import async_writer
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)

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
# Maximum number of threads that each read call uses to access files in
# parallel, which hides the per-file latency of remote filesystems such as GCS.
# Only file access runs on these threads: parsing holds the GIL, so the calling
# thread parses the files in order while later files are still being read. On a
# local disk, where file access is fast, starting the threads and handing the
# GIL between them and the calling thread make reads somewhat slower than
# reading the files one by one.
_MAX_READ_WORKERS: Final[int] = 16
# Name prefix of the threads that read files.
_READ_THREAD_NAME_PREFIX: Final[str] = "FileTrajectoryStoreReader"


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
  """Serializes a Pydantic model to compact JSON excluding None values.

  Indentation would put every element of the token, mask, and logprob arrays on
  its own line behind a newline and leading spaces, nearly doubling step files.

  Args:
    model: The Pydantic model to serialize.

  Returns:
    The compact JSON string representation of `model`.
  """
  return model.model_dump_json(exclude_none=True)


def _get_step_path(traj_dir: epath.Path, atif_step_id: int) -> epath.Path:
  """Returns the step file path for a 1-indexed ATIF step_id under traj_dir."""
  return traj_dir / _STEP_FILENAME_TEMPLATE.format(step_id=atif_step_id)


def _read_text_or_none(path: epath.Path) -> str | None:
  """Returns the contents of the file at `path`, or None if it is missing."""
  if not path.exists():
    return None
  return path.read_text()


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FileWriteTask(async_writer.WriteTask):
  """File-specific write task containing filesystem destination paths."""

  traj_dir: epath.Path
  meta_path: epath.Path
  step_path: epath.Path | None = dataclasses.field(default=None, init=False)

  def to_atif(self) -> None:
    """Projects the payload to ATIF and records the resulting step file path."""
    super().to_atif()
    if self.step is not None:
      object.__setattr__(
          self, "step_path", _get_step_path(self.traj_dir, self.step.step_id)
      )


def _resolve_target_path(task: _FileWriteTask) -> epath.Path:
  """Returns the most specific filesystem path a task was writing to.

  If `to_atif()` failed, `task.step` is still an unprojected subclass whose
  step_id may not match the on-disk id, so the trajectory directory is returned
  instead of a guessed step file path.

  Args:
    task: The write task.
  """
  if task.step is None:
    return task.meta_path
  if task.step_path is None:
    return task.traj_dir
  return task.step_path


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
      step: Optional Step object to write.

    Raises:
      RuntimeError: If the writer has already been closed.
    """
    task = _FileWriteTask(
        traj_dir=traj_dir,
        meta_path=meta_path,
        metadata=metadata,
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

    Raises:
      RuntimeError: If the task has a step but `to_atif()` was not called first,
        so its step file path is unknown.
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

    # Write step file if provided. `step_path` is set by `to_atif()`.
    if task.step is not None:
      if task.step_path is None:
        raise RuntimeError("to_atif() must run before _process_task().")
      task.step_path.write_text(_dump_json(task.step))

  def _log_task_error(self, task: _FileWriteTask) -> None:
    """Logs detailed task error with trajectory and filesystem path context.

    Args:
      task: The write task that failed to process.
    """
    step_info = (
        f"step {task.step.step_id}" if task.step is not None else "metadata"
    )
    logging.exception(
        "%s failed to write trajectory %s (trajectory_id=%s) to %s.",
        self._thread_name,
        step_info,
        task.trajectory_id,
        _resolve_target_path(task),
    )


class FileTrajectoryStore(store.TrajectoryStore[MetadataT]):
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
      self,
      root_dir: epath.PathLike,
      run_id: str | None = None,
      *,
      metadata_cls: type[MetadataT],
  ) -> None:
    """Initializes FileTrajectoryStore.

    Args:
      root_dir: Base directory path for storage (supports local paths and GCS
        uris e.g., 'gs://bucket/path').
      run_id: Optional unique identifier for the RL run. If provided, paths are
        scoped under root_dir / run_id. This ID MUST stay the same when
        recovering from failures or process restarts as long as the same RL
        process is being continued.
      metadata_cls: The TrajectoryMetadata subclass to read stored metadata back
        as; the type checker infers `MetadataT` from it. See
        `store.TrajectoryStore`.

    Raises:
      TypeError: If metadata_cls is not a TrajectoryMetadata subclass.
      ValueError: If metadata_cls is not registered in
        TrajectoryMetadata._REGISTRY, root_dir is empty, or run_id is given but
        cannot be used as a directory name.
    """
    super().__init__(metadata_cls=metadata_cls)
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
  def _from_config(
      cls,
      config: Mapping[str, Any],
      *,
      metadata_cls: type[trajectory_lib.TrajectoryMetadata],
  ) -> "FileTrajectoryStore[Any]":
    """Builds a file-backed store from `config`.

    Requires "root_dir" and "run_id".

    Raises:
      ValueError: If "root_dir" or "run_id" is missing or empty.
    Returns:
      A new FileTrajectoryStore.
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
    return cls(
        root_dir=config["root_dir"],
        run_id=config["run_id"],
        metadata_cls=metadata_cls,
    )

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
        "metadata_type": self._metadata_type,
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
    """Returns the file path for a given trajectory ID and ATIF step ID.

    Args:
      trajectory_id: The trajectory identifier.
      step_id: The 1-indexed ATIF step_id, as serialized on disk.

    Returns:
      The path of the step JSON file.
    """
    return _get_step_path(self.get_trajectory_dir(trajectory_id), step_id)

  def get_trajectories_metadata(
      self, trajectory_ids: list[str] | None = None
  ) -> list[MetadataT]:
    """Retrieves metadata for trajectories in the run.

    Files are read in parallel to hide the latency of remote filesystems such
    as GCS. Errors are raised for the first failing trajectory in order, as if
    the files were read one by one.

    Args:
      trajectory_ids: Optional list of unique trajectory identifiers. If
        specified, only metadata for these IDs is returned. If None, metadata
        for all trajectories in the run is returned.

    Returns:
      A list of TrajectoryMetadata objects for the requested trajectories.

    Raises:
      store.TrajectoryMetadataNotFoundError: If any requested trajectory ID does
        not exist.
      RuntimeError: If the reader threads cannot be started, e.g. during
        interpreter shutdown.
    """
    if trajectory_ids is None and not self.root_dir.exists():
      return []
    executor = futures.ThreadPoolExecutor(
        max_workers=_MAX_READ_WORKERS,
        thread_name_prefix=_READ_THREAD_NAME_PREFIX,
    )
    try:
      if trajectory_ids is None:
        trajectory_ids = self._list_trajectory_ids(executor)
      # Iterates over `trajectory_ids` only once, as reading the files one by
      # one did, so that any iterable of IDs still works. Each entry pairs an ID
      # with the future of its metadata file contents. Entries are popped as
      # they are consumed, so that each file's contents can be freed once
      # parsed.
      pending_reads = collections.deque(
          (
              traj_id,
              executor.submit(
                  _read_text_or_none, self.get_trajectory_metadata_path(traj_id)
              ),
          )
          for traj_id in trajectory_ids
      )
      metas: list[MetadataT] = []
      while pending_reads:
        traj_id, meta_text_future = pending_reads.popleft()
        meta_text = meta_text_future.result()
        if meta_text is None:
          raise store.TrajectoryMetadataNotFoundError(traj_id)
        metas.append(self._parse_metadata(meta_text))
      return metas
    finally:
      # Skips the reads that have not started and waits for the running ones,
      # so that no reader thread outlives the call.
      executor.shutdown(cancel_futures=True)

  def get_trajectories(
      self, trajectory_ids: list[str]
  ) -> list[trajectory_lib.Trajectory[Any]]:
    """Retrieves full trajectories for a list of trajectory IDs.

    Files are read in parallel to hide the latency of remote filesystems such
    as GCS. Errors are raised for the first failing trajectory in order, as if
    the files were read one by one.

    Args:
      trajectory_ids: List of unique trajectory identifiers to load.

    Returns:
      A list of full Trajectory objects corresponding to the requested IDs.

    Raises:
      store.TrajectoryNotFoundError: If any requested trajectory ID does not
      exist.
      RuntimeError: If the reader threads cannot be started, e.g. during
        interpreter shutdown.
    """
    executor = futures.ThreadPoolExecutor(
        max_workers=_MAX_READ_WORKERS,
        thread_name_prefix=_READ_THREAD_NAME_PREFIX,
    )
    try:
      # Iterates over `trajectory_ids` only once, as reading the files one by
      # one did, so that any iterable of IDs still works. Each entry holds an
      # ID, the future of its metadata file contents, and a future that
      # resolves to the futures of its step file contents. Entries and step
      # futures are popped as they are consumed, so that each file's contents
      # can be freed once parsed.
      pending_reads = collections.deque(
          (
              traj_id,
              executor.submit(
                  _read_text_or_none, self.get_trajectory_metadata_path(traj_id)
              ),
              executor.submit(self._submit_step_reads, traj_id, executor),
          )
          for traj_id in trajectory_ids
      )
      trajs: list[trajectory_lib.Trajectory[Any]] = []
      while pending_reads:
        traj_id, meta_text_future, step_reads_future = pending_reads.popleft()
        meta_text = meta_text_future.result()
        if meta_text is None:
          raise store.TrajectoryNotFoundError(traj_id)
        meta = self._parse_metadata(meta_text)
        step_text_futures = step_reads_future.result()
        steps: list[trajectory_lib.Step] = []
        while step_text_futures:
          steps.append(
              trajectory_lib.Step.model_validate_json(
                  step_text_futures.popleft().result()
              )
          )
        trajs.append(meta.create_trajectory(steps=steps))
      return trajs
    finally:
      # Skips the reads that have not started and waits for the running ones,
      # so that no reader thread outlives the call.
      executor.shutdown(cancel_futures=True)

  def _list_trajectory_ids(self, executor: futures.Executor) -> list[str]:
    """Returns the IDs of the trajectory directories in `root_dir`.

    Args:
      executor: The executor to check on, in parallel, whether each entry of
        `root_dir` is a directory.

    Returns:
      The trajectory IDs, in the order that `root_dir.iterdir()` lists them.
    """
    entries = list(self.root_dir.iterdir())
    is_dir_futures = [executor.submit(entry.is_dir) for entry in entries]
    return [
        match.group("trajectory_id")
        for entry, is_dir_future in zip(entries, is_dir_futures, strict=True)
        if is_dir_future.result()
        and (match := _TRAJECTORY_DIR_REGEX.match(entry.name))
    ]

  def _parse_metadata(self, meta_text: str) -> MetadataT:
    """Parses the contents of a metadata file as `metadata_cls`."""
    base_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(meta_text)
    return self._metadata_cls.from_atif_metadata(base_meta)

  def _submit_step_reads(
      self, trajectory_id: str, executor: futures.Executor
  ) -> collections.deque[futures.Future[str]]:
    """Lists a trajectory's step files and submits their reads to `executor`.

    Returns without waiting for the reads, so that no task on `executor` waits
    for another one.

    Args:
      trajectory_id: The trajectory identifier.
      executor: The executor to read the step files on.

    Returns:
      Futures of the contents of the step files, in the order that `iterdir()`
      lists the files.
    """
    return collections.deque(
        executor.submit(entry.read_text)
        for entry in self.get_trajectory_dir(trajectory_id).iterdir()
        if _STEP_FILENAME_REGEX.match(entry.name)
    )

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: MetadataT,
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
      metadata: MetadataT,
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
    self._writer.enqueue_write(
        traj_dir=traj_dir,
        meta_path=meta_path,
        metadata=metadata,
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

  def __enter__(self) -> "FileTrajectoryStore[MetadataT]":
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
