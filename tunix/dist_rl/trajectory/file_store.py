"""File-based implementation for Trajectory Store."""

import collections
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent import futures
import contextlib
import dataclasses
import functools
import re
import types
from typing import Any, ClassVar, Final, TypeVar

from absl import logging
from etils import epath
import pydantic
from tunix.dist_rl.trajectory import async_writer
from tunix.dist_rl.trajectory import base_store as store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)
_T = TypeVar("_T")

_METADATA_FILENAME: Final[str] = "metadata.json"
_METRICS_FILENAME: Final[str] = "metrics.json"
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
_STEP_FILENAME_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^step_(?P<step_id>\d+)\.json$"
)
_STEP_ROUTED_EXPERTS_TEMPLATE: Final[str] = (
    "step_{step_id:06d}_routed_experts.bin"
)
_STEP_ROUTED_EXPERTS_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^step_(?P<step_id>\d+)_routed_experts\.bin$"
)
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
# Maximum number of trajectory metrics entries retained in the worker's LRU
# cache to avoid re-reading `metrics.json` across repeated updates.
_MAX_CACHED_METRICS: Final[int] = 1024


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


def _natural_sort_key(s: str) -> list[int | str]:
  """Splits text into alternating string and integer tokens for natural order."""
  return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


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


def _get_step_routed_experts_path(
    traj_dir: epath.Path, atif_step_id: int
) -> epath.Path:
  """Returns the routed-experts binary sidecar path for a 1-indexed step_id."""
  return traj_dir / _STEP_ROUTED_EXPERTS_TEMPLATE.format(step_id=atif_step_id)


def _is_local_filesystem(path: epath.Path) -> bool:
  """Returns whether `path` is on a local POSIX filesystem."""
  # Local POSIX paths are either relative or resolve to a `file://` URI, whereas
  # remote object stores (such as `gs://` or `s3://`) preserve their cloud URI
  # scheme in `as_uri()` even when `str(path)` normalizes to a mount path.
  return not path.is_absolute() or path.as_uri().startswith("file://")


def _write_atomically(
    path: epath.Path, write_fn: Callable[[epath.Path], Any]
) -> None:
  """Writes `path` via `write_fn`, using local atomic temp-file replacement."""
  if not _is_local_filesystem(path):
    write_fn(path)
    return

  # Stage the payload in a hidden sibling file on the same filesystem so
  # `replace()` performs an atomic POSIX rename and concurrent readers never
  # observe a truncated or partially written file.
  tmp_path = path.parent / f".{path.name}.tmp"
  try:
    write_fn(tmp_path)
    tmp_path.replace(path)
  except BaseException:
    try:
      tmp_path.unlink(missing_ok=True)
    except OSError:
      pass
    raise


def _write_text(path: epath.Path, content: str) -> None:
  """Writes `content` to `path`, using atomic temp-file replacement locally."""
  _write_atomically(path, lambda target: target.write_text(content))


def _write_bytes(path: epath.Path, content: bytes) -> None:
  """Writes raw `content` bytes to `path` atomically on local filesystems."""
  _write_atomically(path, lambda target: target.write_bytes(content))


def _read_or_none(
    path: epath.Path, read_fn: Callable[[epath.Path], _T]
) -> _T | None:
  """Reads `path` via `read_fn`, returning None when `path` does not exist."""
  try:
    return read_fn(path)
  except FileNotFoundError:
    return None
  except OSError:
    if not path.exists():
      return None
    raise


def _read_text_or_none(path: epath.Path) -> str | None:
  """Returns the contents of the file at `path`, or None if it is missing.

  Reads the file before checking that it exists, so that a file that exists
  costs a single round trip on remote filesystems such as GCS. Existence is
  checked only once the read fails.

  Args:
    path: The path of the file to read.
  """
  return _read_or_none(path, lambda target: target.read_text())


def _read_bytes_or_none(path: epath.Path) -> bytes | None:
  """Returns the raw bytes of `path`, or None if it does not exist."""
  return _read_or_none(path, lambda target: target.read_bytes())


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FileWriteTask(async_writer.WriteTask):
  """File-specific write task containing filesystem destination paths."""

  traj_dir: epath.Path
  step_path: epath.Path | None = dataclasses.field(default=None, init=False)
  routed_experts_path: epath.Path | None = dataclasses.field(
      default=None, init=False
  )

  @property
  def meta_path(self) -> epath.Path:
    return self.traj_dir / _METADATA_FILENAME

  @property
  def metrics_path(self) -> epath.Path:
    return self.traj_dir / _METRICS_FILENAME

  def to_atif(self) -> None:
    """Projects the payload to ATIF and records the resulting step file path."""
    super().to_atif()
    if self.step is not None:
      object.__setattr__(
          self, "step_path", _get_step_path(self.traj_dir, self.step.step_id)
      )
      object.__setattr__(
          self,
          "routed_experts_path",
          _get_step_routed_experts_path(self.traj_dir, self.step.step_id),
      )


def _resolve_target_path(task: _FileWriteTask) -> epath.Path:
  """Returns the most specific filesystem path a task was writing to.

  If `to_atif()` failed, `task.step` is still an unprojected subclass whose
  step_id may not match the on-disk id, so the trajectory directory is returned
  instead of a guessed step file path.

  Args:
    task: The write task.
  """
  if task.metrics is not None:
    return task.metrics_path
  if task.step is None:
    return task.meta_path
  if task.step_path is None:
    return task.traj_dir
  return task.step_path


def _merge_metrics(
    existing: trajectory_lib.TrajectoryMetrics | None,
    incoming: trajectory_lib.TrajectoryMetrics,
) -> trajectory_lib.TrajectoryMetrics:
  """Returns `incoming` shallow-merged onto `existing`."""
  if existing is None:
    return incoming
  if incoming.extra is None or (
      not incoming.extra and existing.extra is not None
  ):
    return existing
  return existing.model_copy(
      update={"extra": {**(existing.extra or {}), **incoming.extra}}
  )


class _AsyncFileWriter(async_writer.AsyncWriter[_FileWriteTask]):
  """Asynchronously writes trajectory metadata, step, and metrics files."""

  def __init__(self, copy_step: bool = False):
    """Initializes _AsyncFileWriter without starting the background worker."""
    super().__init__(copy_step=copy_step)
    # Worker-owned caches of persisted state used to skip redundant disk writes.
    self._metadata_hash_by_trajectory_id: dict[str, int] = {}
    self._metrics_by_trajectory_id: collections.OrderedDict[
        str, trajectory_lib.TrajectoryMetrics
    ] = collections.OrderedDict()

  def enqueue_write(
      self,
      *,
      traj_dir: epath.Path,
      metadata: trajectory_lib.TrajectoryMetadata,
      step: trajectory_lib.Step | None = None,
  ) -> None:
    """Enqueues a step and/or trajectory metadata for asynchronous writing.

    This operation is non-blocking and returns on the caller thread without
    waiting for any disk I/O. The worker thread is lazily spawned on the first
    invocation if not already running.

    `metadata` is always deep copied by `WriteTask` so status/reward mutations
    across turns do not leak. `step` is handed over with zero copy by default to
    eliminate synchronous GIL contention on rollout workers. When `copy_step`
    is True, `step` is deep copied defensively.

    Args:
      traj_dir: Directory path for the trajectory.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write.

    Raises:
      RuntimeError: If the writer has already been closed.
    """
    self._enqueue(
        _FileWriteTask(
            traj_dir=traj_dir,
            metadata=metadata,
            step=step,
            copy_step=self._copy_step,
        )
    )

  def enqueue_metrics_write(
      self,
      *,
      traj_dir: epath.Path,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> None:
    """Enqueues trajectory metrics for asynchronous shallow-merge writing."""
    self._enqueue(
        _FileWriteTask(
            traj_dir=traj_dir, trajectory_id=trajectory_id, metrics=metrics
        )
    )

  def _process_task(self, task: _FileWriteTask) -> None:
    """Processes a single write task by writing metadata, step, or metrics.

    Optimizations:
      - Directory Creation: `mkdir` is executed once per trajectory on local
        (`file://`) filesystems, tracked by cached metadata and metrics state.
      - Write Deduplication: `metadata.json` and `metrics.json` are written only
        when their content changes.
      - Atomic Writes: Local files are staged via a hidden temporary file and
        swapped atomically with `replace()`. Cloud object stores write directly
        since object uploads are atomic on stream close.

    Args:
      task: Container holding destination paths and write payload.

    Raises:
      RuntimeError: If the task has a step but `to_atif()` was not called first,
        so its step file path is unknown.
    """
    traj_id = task.trajectory_id

    # Create directory once per trajectory on local filesystems. Remote object
    # stores use a flat object namespace where writing a file materializes the
    # prefix without a 0-byte directory marker.
    has_dir = (
        traj_id in self._metadata_hash_by_trajectory_id
        or traj_id in self._metrics_by_trajectory_id
    )
    if not has_dir and _is_local_filesystem(task.traj_dir):
      task.traj_dir.mkdir(parents=True, exist_ok=True)

    # Write step file before metadata.json so a crash or I/O error mid-write
    # never leaves metadata.json advertising a newer trajectory state (such as
    # COMPLETED) while the corresponding step file is missing on disk.
    if task.step is not None:
      if task.step_path is None or task.routed_experts_path is None:
        raise RuntimeError("to_atif() must run before _process_task().")
      if task.routed_experts_payload is not None:
        _write_bytes(task.routed_experts_path, task.routed_experts_payload)
      elif _is_local_filesystem(task.routed_experts_path):
        task.routed_experts_path.unlink(missing_ok=True)
      _write_text(task.step_path, _dump_json(task.step))

    # Only write metadata.json if metadata content has changed.
    if task.metadata is not None:
      meta_json = _dump_json(task.metadata)
      meta_hash = hash(meta_json)
      if self._metadata_hash_by_trajectory_id.get(traj_id) != meta_hash:
        _write_text(task.meta_path, meta_json)
        self._metadata_hash_by_trajectory_id[traj_id] = meta_hash

    if task.metrics is not None:
      self._write_metrics(traj_id, task)

  def _write_metrics(self, traj_id: str, task: _FileWriteTask) -> None:
    """Merges incoming trajectory metrics with cached state and writes."""
    if task.metrics is None:
      return
    existing = self._metrics_by_trajectory_id.get(traj_id)
    if (
        existing is None
        and (disk_text := _read_text_or_none(task.metrics_path)) is not None
    ):
      existing = trajectory_lib.TrajectoryMetrics.model_validate_json(disk_text)
    merged = _merge_metrics(existing, task.metrics)
    if merged != existing:
      _write_text(task.metrics_path, merged.model_dump_json())
    self._metrics_by_trajectory_id[traj_id] = merged
    self._metrics_by_trajectory_id.move_to_end(traj_id)
    if len(self._metrics_by_trajectory_id) > _MAX_CACHED_METRICS:
      self._metrics_by_trajectory_id.popitem(last=False)

  def _log_task_error(self, task: _FileWriteTask) -> None:
    """Logs detailed task error with trajectory and filesystem path context.

    Args:
      task: The write task that failed to process.
    """
    if task.metrics is not None:
      payload_desc = "metrics"
    elif task.step is not None:
      payload_desc = f"step {task.step.step_id}"
    else:
      payload_desc = "metadata"
    logging.exception(
        "%s failed to write trajectory %s (trajectory_id=%s) to %s.",
        self._thread_name,
        payload_desc,
        task.trajectory_id,
        _resolve_target_path(task),
    )


class FileTrajectoryStore(store.TrajectoryStore[MetadataT]):
  """File-based implementation satisfying TrajectoryReader and TrajectoryWriter.

  Architectural Separation of Responsibilities:
    `FileTrajectoryStore` acts as a lightweight frontend responsible for:
    1. Filesystem path layout, directory hierarchy, and naming conventions.
    2. Frontend input validation (trajectory ID format and presence).
    3. Synchronous trajectory, metadata, and metrics queries
       (`get_trajectories`, `get_trajectories_metadata`,
       `get_trajectories_metrics`).
    4. Forwarding step, metadata, and metrics write tasks and flush barriers to
       `_AsyncFileWriter`.

    All asynchronous queuing, background worker thread lifecycle, error
    suppression for rollout resilience, and physical disk I/O are handled
    by `_AsyncFileWriter`.

  Directory Structure:
    <root_dir>/[<run_id>/]/
        └── traj_<trajectory_id>/
            ├── metadata.json
            ├── metrics.json
            ├── step_000001.json
            ├── step_000001_routed_experts.bin  (optional)
            ├── step_000002.json
            └── ...
  """

  BACKEND: ClassVar[str] = "file"

  def __init__(
      self,
      root_dir: epath.PathLike,
      run_id: str | None = None,
      copy_step: bool = False,
  ) -> None:
    """Initializes FileTrajectoryStore.

    Args:
      root_dir: Base directory path for storage (supports local paths and GCS
        uris e.g., 'gs://bucket/path').
      run_id: Optional unique identifier for the RL run. If provided, paths are
        scoped under root_dir / run_id. This ID MUST stay the same when
        recovering from failures or process restarts as long as the same RL
        process is being continued.
      copy_step: If True, deep copies step objects before writing. Defaults to
        False (zero-copy step handover) to avoid synchronous GIL contention on
        rollout workers.

    Raises:
      ValueError: If root_dir is empty, or if run_id is given but cannot be
        used as a directory name.
    """
    super().__init__()
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
    self._writer = _AsyncFileWriter(copy_step=copy_step)

  @classmethod
  def _from_config(
      cls,
      config: Mapping[str, Any],
  ) -> "FileTrajectoryStore[MetadataT]":
    """Builds a file-backed store from `config`.

    Requires "root_dir" and "run_id", with optional boolean "copy_step"
    (defaults to False).

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
        copy_step=bool(config.get("copy_step", False)),
    )

  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent store.

    Writer-local tuning flags such as `copy_step` are omitted so `to_config()`
    strictly represents the shared storage location across processes.

    Raises:
      ValueError: If this store was constructed without a run_id.
    """
    if not self._run_id:
      raise ValueError(
          "FileTrajectoryStore without a run_id cannot export a valid"
          " distributed config."
      )
    return {
        "backend": self.BACKEND,
        "root_dir": str(self._raw_root_dir),
        "run_id": self._run_id,
    }

  @property
  def copy_step(self) -> bool:
    """Returns True if step objects are deep copied before writing."""
    return self._writer.copy_step

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

  def get_trajectory_metrics_path(self, trajectory_id: str) -> epath.Path:
    """Returns the file path for a given trajectory ID's metrics."""
    return self.get_trajectory_dir(trajectory_id) / _METRICS_FILENAME

  def get_step_path(self, trajectory_id: str, step_id: int) -> epath.Path:
    """Returns the file path for a given trajectory ID and ATIF step ID.

    Args:
      trajectory_id: The trajectory identifier.
      step_id: The 1-indexed ATIF step_id, as serialized on disk.

    Returns:
      The path of the step JSON file.
    """
    return _get_step_path(self.get_trajectory_dir(trajectory_id), step_id)

  def get_step_routed_experts_path(
      self, trajectory_id: str, step_id: int
  ) -> epath.Path:
    """Returns the binary routed-experts sidecar path for a 1-indexed step ID.

    Args:
      trajectory_id: The trajectory identifier.
      step_id: The 1-indexed ATIF step_id, as serialized on disk.

    Returns:
      The path of the step routed-experts binary sidecar file.
    """
    return _get_step_routed_experts_path(
        self.get_trajectory_dir(trajectory_id), step_id
    )

  @contextlib.contextmanager
  def _open_parallel_reads(
      self,
      trajectory_ids: Iterable[str] | None,
      get_path: Callable[[str], epath.Path],
  ) -> Iterator[Iterator[tuple[str, str | None]]]:
    """Opens parallel file reads yielding `(trajectory_id, text_or_none)`."""
    if trajectory_ids is None and not self.root_dir.exists():
      yield iter(())
      return
    executor = futures.ThreadPoolExecutor(
        max_workers=_MAX_READ_WORKERS,
        thread_name_prefix=_READ_THREAD_NAME_PREFIX,
    )
    try:
      ids_to_read: Iterable[str] = (
          self._list_trajectory_ids(executor)
          if trajectory_ids is None
          else trajectory_ids
      )
      # Iterates over `ids_to_read` only once, as reading the files one by one
      # did, so that any iterable of IDs still works, including the lazy one
      # from `_list_trajectory_ids`. Entries are popped as they are consumed so
      # each file's contents can be freed once parsed.
      pending_reads = collections.deque(
          (traj_id, executor.submit(_read_text_or_none, get_path(traj_id)))
          for traj_id in ids_to_read
      )

      def _iter_results() -> Iterator[tuple[str, str | None]]:
        while pending_reads:
          traj_id, text_future = pending_reads.popleft()
          yield traj_id, text_future.result()

      yield _iter_results()
    finally:
      # Skips the reads that have not started and waits for the running ones,
      # so that no reader thread outlives the call.
      executor.shutdown(cancel_futures=True)

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
    metas: list[MetadataT] = []
    with self._open_parallel_reads(
        trajectory_ids, self.get_trajectory_metadata_path
    ) as reads:
      for traj_id, meta_text in reads:
        if meta_text is None:
          if trajectory_ids is None:
            if not any(
                _STEP_FILENAME_REGEX.match(entry.name)
                for entry in self.get_trajectory_dir(traj_id).iterdir()
            ):
              continue
            meta_text = _read_text_or_none(
                self.get_trajectory_metadata_path(traj_id)
            )
          if meta_text is None:
            raise store.TrajectoryMetadataNotFoundError(traj_id)
        metas.append(self._parse_metadata(meta_text))
    return metas

  def get_trajectories_metrics(
      self, trajectory_ids: list[str] | None = None
  ) -> dict[str, trajectory_lib.TrajectoryMetrics]:
    """Retrieves metrics for trajectories in the run.

    Files are read in parallel to hide the latency of remote filesystems such
    as GCS. Errors are raised for the first failing trajectory in order, as if
    the files were read one by one.

    Args:
      trajectory_ids: Optional list of unique trajectory identifiers. If
        specified, only metrics for these IDs are returned. If None, metrics for
        all trajectories in the run with recorded metrics are returned.

    Returns:
      A dictionary mapping trajectory IDs to TrajectoryMetrics objects, in the
      order of `trajectory_ids` if specified, or natural ID order if None.

    Raises:
      store.TrajectoryMetricsNotFoundError: If any explicitly requested
        trajectory ID does not have recorded metrics.
      RuntimeError: If the reader threads cannot be started, e.g. during
        interpreter shutdown.
    """
    metrics_by_id: dict[str, trajectory_lib.TrajectoryMetrics] = {}
    with self._open_parallel_reads(
        trajectory_ids, self.get_trajectory_metrics_path
    ) as reads:
      for traj_id, metrics_text in reads:
        if metrics_text is None:
          if trajectory_ids is None:
            continue
          raise store.TrajectoryMetricsNotFoundError(traj_id)
        metrics_by_id[traj_id] = (
            trajectory_lib.TrajectoryMetrics.model_validate_json(metrics_text)
        )
    return metrics_by_id

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
        step_read_futures = step_reads_future.result()
        steps: list[trajectory_lib.Step] = []
        while step_read_futures:
          text_future, bytes_future = step_read_futures.popleft()
          step = trajectory_lib.Step.model_validate_json(text_future.result())
          routed_experts_bytes = (
              bytes_future.result() if bytes_future is not None else None
          )
          steps.append(
              trajectory_lib.resolve_step_routed_experts(
                  step, routed_experts_bytes
              )
          )
        trajs.append(meta.create_trajectory(steps=steps))
      return trajs
    finally:
      # Skips the reads that have not started and waits for the running ones,
      # so that no reader thread outlives the call.
      executor.shutdown(cancel_futures=True)

  def _list_trajectory_ids(self, executor: futures.Executor) -> Iterator[str]:
    """Returns the IDs of the trajectory directories in `root_dir`.

    Entries are filtered by name before checking whether they are directories,
    so that entries that cannot be trajectory directories cost no round trip on
    remote filesystems such as GCS. All the checks are submitted before this
    returns, but the IDs are yielded lazily, so that callers can start working
    on the leading IDs while the remaining checks are still running.

    Args:
      executor: The executor to check on, in parallel, whether each entry of
        `root_dir` named like a trajectory directory is a directory. It must
        stay alive until the returned iterator is exhausted.

    Returns:
      An iterator over the trajectory IDs, sorted naturally by trajectory ID.
    """
    matched_entries = sorted(
        (
            (match.group("trajectory_id"), entry)
            for entry in self.root_dir.iterdir()
            if (match := _TRAJECTORY_DIR_REGEX.match(entry.name))
        ),
        key=lambda pair: _natural_sort_key(pair[0]),
    )
    candidates = [
        (trajectory_id, executor.submit(entry.is_dir))
        for trajectory_id, entry in matched_entries
    ]
    return (
        trajectory_id
        for trajectory_id, is_dir_future in candidates
        if is_dir_future.result()
    )

  def _parse_metadata(self, meta_text: str) -> MetadataT:
    """Parses the contents of a metadata file as `metadata_cls`."""
    base_meta = trajectory_lib.TrajectoryMetadata.model_validate_json(meta_text)
    return self._rehydrate_metadata(base_meta)

  def _submit_step_reads(
      self,
      trajectory_id: str,
      executor: futures.Executor,
  ) -> collections.deque[
      tuple[futures.Future[str], futures.Future[bytes | None] | None]
  ]:
    """Lists a trajectory's step files and submits their reads to `executor`.

    Returns without waiting for the reads, so that no task on `executor` waits
    for another one.

    Args:
      trajectory_id: The trajectory identifier.
      executor: The executor to read the step files on.

    Returns:
      Pairs of `(step_json_future, optional_routed_experts_bytes_future)` in
      the order that `iterdir()` lists the step JSON files.
    """
    json_entries_with_id: list[tuple[int, epath.Path]] = []
    bin_entry_by_step_id: dict[int, epath.Path] = {}
    for entry in self.get_trajectory_dir(trajectory_id).iterdir():
      if match := _STEP_FILENAME_REGEX.match(entry.name):
        json_entries_with_id.append((int(match.group("step_id")), entry))
      elif match := _STEP_ROUTED_EXPERTS_REGEX.match(entry.name):
        bin_entry_by_step_id[int(match.group("step_id"))] = entry

    step_futures: collections.deque[
        tuple[futures.Future[str], futures.Future[bytes | None] | None]
    ] = collections.deque()
    for step_id, json_entry in json_entries_with_id:
      bin_entry = bin_entry_by_step_id.get(step_id)
      bytes_future = (
          executor.submit(_read_bytes_or_none, bin_entry)
          if bin_entry is not None
          else None
      )
      step_futures.append((executor.submit(json_entry.read_text), bytes_future))
    return step_futures

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

    `metadata` is always snapshotted at call time. By default
    (`copy_step=False`), ownership of `step` transfers to the background writer,
    so callers must not mutate `step` or its nested containers after calling
    `add_step()`. Pass `copy_step=True` if the caller mutates `step` afterwards.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      TypeError: If metadata is not a TrajectoryMetadata instance or its type
        conflicts with the store's bound metadata type.
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

    `metadata` is always snapshotted at call time. When `step` is provided and
    `copy_step=False` (the default), ownership of `step` transfers to the
    background writer and callers must not mutate `step` afterwards.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write alongside metadata.

    Raises:
      TypeError: If metadata is not a TrajectoryMetadata instance or its type
        conflicts with the store's bound metadata type.
      ValueError: If metadata.trajectory_id is empty, None, or contains
        characters that cannot be encoded in a trajectory directory name.
    """
    self._validate_and_bind_metadata(metadata)
    traj_id = _validate_trajectory_id(metadata.trajectory_id)
    self._writer.enqueue_write(
        traj_dir=self.get_trajectory_dir(traj_id),
        metadata=metadata,
        step=step,
    )

  def update_metrics(
      self,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> None:
    """Updates or creates trajectory metrics asynchronously.

    Top-level keys in `metrics.extra` are shallow-merged onto any existing
    `extra` dictionary persisted for `trajectory_id` (assuming single-writer
    ownership per trajectory at a time). `metrics` is snapshotted on the calling
    thread before enqueueing.

    Args:
      trajectory_id: Unique identifier for the trajectory.
      metrics: TrajectoryMetrics containing trajectory-level metrics to record.

    Raises:
      TypeError: If metrics is not a TrajectoryMetrics instance.
      ValueError: If trajectory_id is empty, None, or contains characters that
        cannot be encoded in a trajectory directory name.
      RuntimeError: If the store has already been closed.
    """
    if not isinstance(metrics, trajectory_lib.TrajectoryMetrics):
      raise TypeError(
          f"Expected TrajectoryMetrics instance, got {type(metrics).__name__}."
      )
    traj_id = _validate_trajectory_id(trajectory_id)
    self._writer.enqueue_metrics_write(
        traj_dir=self.get_trajectory_dir(traj_id),
        trajectory_id=traj_id,
        metrics=metrics,
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
