"""File-based implementation for Trajectory Store."""

import collections
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent import futures
import contextlib
import dataclasses
import functools
import itertools
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
    rf"^{_TRAJECTORY_DIR_PREFIX}({_TRAJECTORY_ID_PATTERN})$"
)
_STEP_FILENAME_TEMPLATE: Final[str] = "step_{step_id:06d}.json"
_STEP_FILENAME_REGEX: Final[re.Pattern[str]] = re.compile(r"^step_(\d+)\.json$")
_STEPS_CHUNK_FILENAME_TEMPLATE: Final[str] = (
    "steps_{first_step_id:06d}_{last_step_id:06d}.jsonl"
)
_STEPS_CHUNK_FILENAME_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^steps_(\d+)_(\d+)\.jsonl$"
)
# Canonical status codes (`grpc.StatusCode`) on `gfile.GOSError.status.code` or
# `tf.errors.OpError.error_code` that indicate object write rate limits,
# contention, or backend backpressure.
_BATCHABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({
    8,  # RESOURCE_EXHAUSTED.
    10,  # ABORTED.
    14,  # UNAVAILABLE.
})
# HTTP status codes on cloud storage exceptions (`HttpError.code` /
# `GoogleAPICallError.code`) for the same rate-limit and contention conditions.
_BATCHABLE_HTTP_STATUS_CODES: Final[frozenset[int]] = frozenset({
    409,  # Conflict.
    429,  # Too Many Requests.
    503,  # Service Unavailable.
})
# Canonical `grpc.StatusCode.NOT_FOUND` (5) and HTTP 404 status codes raised by
# cloud storage backends (`gfile.GOSError`, `tf.errors.NotFoundError`,
# `GoogleAPICallError`) when an object or prefix does not exist.
_NOT_FOUND_STATUS_CODE: Final[int] = 5
_NOT_FOUND_HTTP_STATUS_CODE: Final[int] = 404
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


def _get_steps_chunk_path(
    traj_dir: epath.Path, first_step_id: int, last_step_id: int
) -> epath.Path:
  """Returns the JSONL chunk path for an ATIF step_id range under traj_dir."""
  return traj_dir / _STEPS_CHUNK_FILENAME_TEMPLATE.format(
      first_step_id=first_step_id,
      last_step_id=last_step_id,
  )


def _step_file_sort_key(filename: str) -> tuple[int, int, int, str] | None:
  """Returns the read-order sort key for a step or chunk file, or None.

  Orders files by starting `step_id` so earlier steps are read first. When a
  single-step `step_XXXXXX.json` revision overlaps a `steps_XXXXXX_YYYYYY.jsonl`
  chunk (`step_id >= first_step_id`), the chunk (`tier=0`) sorts before the
  single-step revision (`tier=1`) so the revision overwrites the chunk entry.
  """
  if match := _STEP_FILENAME_REGEX.match(filename):
    (step_id_str,) = match.groups()
    step_id = int(step_id_str)
    return (step_id, 1, step_id, filename)
  if match := _STEPS_CHUNK_FILENAME_REGEX.match(filename):
    first_id_str, last_id_str = match.groups()
    return (int(first_id_str), 0, int(last_id_str), filename)
  return None


def _group_contiguous_step_ids(
    step_ids: Sequence[int],
) -> Iterator[list[int]]:
  """Yields maximal runs of consecutive ascending step IDs."""
  counter = itertools.count()
  for _, group in itertools.groupby(
      step_ids, key=lambda step_id: step_id - next(counter)
  ):
    yield list(group)


def _parse_step_file(filename: str, text: str) -> list[trajectory_lib.Step]:
  """Parses a single-step .json file or a multi-step .jsonl chunk file."""
  if filename.endswith(".jsonl"):
    steps = [
        trajectory_lib.Step.model_validate_json(line)
        for line in text.splitlines()
        if line and not line.isspace()
    ]
    if steps:
      return steps
  return [trajectory_lib.Step.model_validate_json(text)]


def _is_local_filesystem(path: epath.Path) -> bool:
  """Returns whether `path` is on a local POSIX filesystem."""
  # Local POSIX paths are either relative or resolve to a `file://` URI, whereas
  # remote object stores (such as `gs://` or `s3://`) preserve their cloud URI
  # scheme in `as_uri()` even when `str(path)` normalizes to a mount path.
  return not path.is_absolute() or path.as_uri().startswith("file://")


def _write_text(path: epath.Path, content: str) -> None:
  """Writes `content` to `path`, using atomic temp-file replacement locally."""
  if not _is_local_filesystem(path):
    path.write_text(content)
    return

  # Stage the payload in a hidden sibling file on the same filesystem so
  # `replace()` performs an atomic POSIX rename and concurrent readers never
  # observe a truncated or partially written JSON file.
  tmp_path = path.parent / f".{path.name}.tmp"
  try:
    tmp_path.write_text(content)
    tmp_path.replace(path)
  except BaseException:
    try:
      tmp_path.unlink(missing_ok=True)
    except OSError:
      pass
    raise


def _is_not_found_error(exc: BaseException) -> bool:
  """Returns whether `exc` indicates a missing file, directory, or object."""
  return (
      isinstance(exc, (FileNotFoundError, NotADirectoryError))
      or getattr(getattr(exc, "status", None), "code", None)
      == _NOT_FOUND_STATUS_CODE
      or getattr(exc, "error_code", None) == _NOT_FOUND_STATUS_CODE
      or getattr(exc, "code", None) == _NOT_FOUND_HTTP_STATUS_CODE
  )


def _read_text_or_none(path: epath.Path) -> str | None:
  """Returns the contents of the file at `path`, or None if it is missing.

  Reads the file before checking that it exists, so that a file that exists
  costs a single round trip on remote filesystems such as GCS. Existence is
  checked only when the read fails with an ambiguous `OSError`.

  Args:
    path: The path of the file to read.
  """
  try:
    return path.read_text()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if _is_not_found_error(exc):
      return None
    if isinstance(exc, OSError) and not path.exists():
      return None
    raise


def _read_max_persisted_step_id(traj_dir: epath.Path) -> int:
  """Returns the maximum ATIF step_id already present under `traj_dir`."""
  max_step_id = 0
  try:
    for entry in traj_dir.iterdir():
      if (sort_key := _step_file_sort_key(entry.name)) is not None:
        _, _, last_step_id, _ = sort_key
        max_step_id = max(max_step_id, last_step_id)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if _is_not_found_error(exc) or isinstance(exc, OSError):
      return 0
    raise
  return max_step_id


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FileWriteTask(async_writer.WriteTask):
  """File-specific write task containing filesystem destination paths."""

  traj_dir: epath.Path
  step_path: epath.Path | None = dataclasses.field(default=None, init=False)

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

  def __init__(
      self,
      batching: async_writer.BatchConfig | None = None,
      copy_step: bool = False,
      exclude_step_arrays: Collection[str] = (),
  ):
    """Initializes _AsyncFileWriter without starting the background worker.

    Args:
      batching: Optional BatchConfig governing how queued writes are coalesced.
      copy_step: If True, deep copies step objects before enqueuing.
      exclude_step_arrays: Step array fields dropped before queueing step
        writes (see `AsyncWriter`).
    """
    super().__init__(
        batching=batching,
        copy_step=copy_step,
        exclude_step_arrays=exclude_step_arrays,
    )
    # Worker-owned caches of persisted state used to skip redundant disk writes
    # and route step revisions to single-step JSON files.
    self._max_step_id_by_trajectory_id: dict[str, int] = {}
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

  def _should_batch_on_error(self, exc: BaseException) -> bool:
    """Returns True for rate-limit, contention, or backpressure errors."""
    status = getattr(exc, "status", None)
    status_code = getattr(status, "code", None) or getattr(
        exc, "error_code", None
    )
    http_code = getattr(exc, "code", None)
    return (
        status_code in _BATCHABLE_STATUS_CODES
        or http_code in _BATCHABLE_HTTP_STATUS_CODES
    )

  def _ensure_traj_dir(self, traj_id: str, traj_dir: epath.Path) -> None:
    """Creates `traj_dir` once per trajectory on local filesystems."""
    # Create directory once per trajectory on local filesystems. Remote object
    # stores use a flat object namespace where writing a file materializes the
    # prefix without a 0-byte directory marker.
    has_dir = (
        traj_id in self._max_step_id_by_trajectory_id
        or traj_id in self._metadata_hash_by_trajectory_id
        or traj_id in self._metrics_by_trajectory_id
    )
    if not has_dir and _is_local_filesystem(traj_dir):
      traj_dir.mkdir(parents=True, exist_ok=True)

  def _write_steps(self, step_tasks: Sequence[_FileWriteTask]) -> None:
    """Writes revised steps as `.json` and contiguous new runs as `.jsonl`."""
    if not step_tasks:
      return
    latest_step_by_id: dict[int, trajectory_lib.Step] = {}
    for task in step_tasks:
      if task.step is None or task.step_path is None:
        raise RuntimeError("to_atif() must run before _write_steps().")
      latest_step_by_id[task.step.step_id] = task.step

    traj_id = step_tasks[-1].trajectory_id
    traj_dir = step_tasks[-1].traj_dir
    if (
        traj_id not in self._max_step_id_by_trajectory_id
        and len(latest_step_by_id) > 1
    ):
      self._max_step_id_by_trajectory_id[traj_id] = _read_max_persisted_step_id(
          traj_dir
      )
    max_persisted_step_id = self._max_step_id_by_trajectory_id.get(traj_id, 0)
    new_step_ids: list[int] = []
    for step_id in sorted(latest_step_by_id):
      if step_id <= max_persisted_step_id:
        _write_text(
            _get_step_path(traj_dir, step_id),
            _dump_json(latest_step_by_id[step_id]),
        )
        continue
      new_step_ids.append(step_id)

    for run in _group_contiguous_step_ids(new_step_ids):
      if len(run) == 1:
        (step_id,) = run
        _write_text(
            _get_step_path(traj_dir, step_id),
            _dump_json(latest_step_by_id[step_id]),
        )
        last_step_id = step_id
      else:
        first_step_id, *_, last_step_id = run
        chunk_path = _get_steps_chunk_path(
            traj_dir, first_step_id, last_step_id
        )
        chunk_lines = [
            _dump_json(latest_step_by_id[step_id]) for step_id in run
        ]
        _write_text(chunk_path, "\n".join(chunk_lines))
      self._max_step_id_by_trajectory_id[traj_id] = last_step_id

  def _write_metadata(self, task: _FileWriteTask | None) -> None:
    """Writes `metadata.json` when the serialized metadata has changed."""
    if task is None or task.metadata is None:
      return
    traj_id = task.trajectory_id
    meta_json = _dump_json(task.metadata)
    meta_hash = hash(meta_json)
    if self._metadata_hash_by_trajectory_id.get(traj_id) != meta_hash:
      _write_text(task.meta_path, meta_json)
      self._metadata_hash_by_trajectory_id[traj_id] = meta_hash

  def _write_metrics(self, task: _FileWriteTask | None) -> None:
    """Shallow-merges and writes `metrics.json` when metrics have changed."""
    if task is None or task.metrics is None:
      return
    traj_id = task.trajectory_id
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

  def _write_single_task(self, task: _FileWriteTask) -> None:
    """Writes directory, step, metadata, and metrics for a single `task`."""
    self._ensure_traj_dir(task.trajectory_id, task.traj_dir)
    # Write step file before metadata.json so a crash or I/O error mid-write
    # never leaves metadata.json advertising a newer trajectory state (such as
    # COMPLETED) while the corresponding step file is missing on disk.
    if task.step is not None:
      self._write_steps((task,))
    self._write_metadata(task)
    self._write_metrics(task)

  def _replay_failed_tasks(
      self, tasks: Sequence[_FileWriteTask], trajectory_id: str
  ) -> bool:
    """Replays tasks for `trajectory_id` one by one after a batch failure."""
    had_batchable_error = False
    for task in tasks:
      if task.trajectory_id != trajectory_id:
        continue
      try:
        self._write_single_task(task)
      except Exception as replay_exc:  # pylint: disable=broad-exception-caught
        replay_exc.__context__ = None
        self._safe_log_task_error(task)
        if self._should_batch_on_error(replay_exc):
          had_batchable_error = True
    return had_batchable_error

  def _process_batch(self, tasks: Sequence[_FileWriteTask]) -> None:
    """Writes coalesced step, metadata, and metrics files for task batches."""
    if len(tasks) <= 1:
      super()._process_batch(tasks)
      return

    had_batchable_error = False
    traj_dir_by_id = {task.trajectory_id: task.traj_dir for task in tasks}
    for batch in async_writer.group_tasks_by_trajectory(tasks):
      traj_dir = traj_dir_by_id[batch.trajectory_id]
      try:
        self._ensure_traj_dir(batch.trajectory_id, traj_dir)
        self._write_steps(batch.step_tasks)
        self._write_metadata(batch.latest_metadata_task)
        self._write_metrics(batch.merged_queued_metrics_task)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if self._should_batch_on_error(exc):
          logging.exception(
              "%s failed to write trajectory batch (trajectory_id=%s) to %s.",
              self._thread_name,
              batch.trajectory_id,
              traj_dir,
          )
          had_batchable_error = True
          continue
        logging.warning(
            "%s trajectory batch write failed (trajectory_id=%s) at %s;"
            " replaying trajectory tasks individually.",
            self._thread_name,
            batch.trajectory_id,
            traj_dir,
            exc_info=True,
        )
        if self._replay_failed_tasks(tasks, batch.trajectory_id):
          had_batchable_error = True

    if had_batchable_error:
      self._increase_batch_wait()
    else:
      self._decrease_batch_wait()

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
    self._write_single_task(task)

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
            ├── step_000002.json
            ├── steps_000003_000005.jsonl
            └── ...
  """

  BACKEND: ClassVar[str] = "file"

  def __init__(
      self,
      root_dir: epath.PathLike,
      run_id: str | None = None,
      batching: async_writer.BatchConfig | None = None,
      copy_step: bool = False,
      exclude_step_arrays: Collection[str] = (),
  ) -> None:
    """Initializes FileTrajectoryStore.

    Args:
      root_dir: Base directory path for storage (supports local paths and GCS
        uris e.g., 'gs://bucket/path').
      run_id: Optional unique identifier for the RL run. If provided, paths are
        scoped under root_dir / run_id. This ID MUST stay the same when
        recovering from failures or process restarts as long as the same RL
        process is being continued.
      batching: Optional BatchConfig governing how queued step and metadata
        writes are coalesced and chunked on disk.
      copy_step: If True, deep copies step objects before writing. Defaults to
        False (zero-copy step handover) to avoid synchronous GIL contention on
        rollout workers.
      exclude_step_arrays: Heavy step array fields to drop before steps are
        queued and written, e.g. `("assistant_tokens", "logprobs")` for
        observability-only runs. Names must be declared in
        `EXCLUDABLE_ARRAY_FIELDS` by the step classes of the bound metadata
        type (checked on the first write, or at startup via a schema-bound
        `from_config`). Defaults to keeping all arrays.

    Raises:
      ValueError: If root_dir is empty, or if run_id is given but cannot be
        used as a directory name.
      TypeError: If exclude_step_arrays is a single string or contains
        non-string entries.
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
    self._writer = _AsyncFileWriter(
        batching=batching,
        copy_step=copy_step,
        exclude_step_arrays=exclude_step_arrays,
    )

  @property
  def batching(self) -> async_writer.BatchConfig:
    """Returns the active BatchConfig for this store."""
    return self._writer.batching

  @classmethod
  def _from_config(
      cls,
      config: Mapping[str, Any],
  ) -> "FileTrajectoryStore[MetadataT]":
    """Builds a file-backed store from `config`.

    Requires "root_dir" and "run_id", with optional boolean "copy_step"
    (defaults to False). Accepts an optional "exclude_step_arrays" list of
    step array fields to drop before writing.

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
        exclude_step_arrays=(
            ()
            if (exclude := config.get("exclude_step_arrays")) is None
            else exclude
        ),
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
    config: dict[str, Any] = {
        "backend": self.BACKEND,
        "root_dir": str(self._raw_root_dir),
        "run_id": self._run_id,
    }
    if self._writer.exclude_step_arrays:
      config["exclude_step_arrays"] = sorted(self._writer.exclude_step_arrays)
    return config

  @property
  def exclude_step_arrays(self) -> frozenset[str]:
    """Returns the step array fields dropped before writing steps."""
    return self._writer.exclude_step_arrays

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

  def get_steps_chunk_path(
      self,
      trajectory_id: str,
      first_step_id: int,
      last_step_id: int,
  ) -> epath.Path:
    """Returns the JSONL chunk file path for a trajectory and ATIF step range.

    Args:
      trajectory_id: The trajectory identifier.
      first_step_id: First 1-indexed ATIF step_id in the chunk.
      last_step_id: Last 1-indexed ATIF step_id in the chunk.

    Returns:
      The path of the step chunk JSONL file.
    """
    return _get_steps_chunk_path(
        self.get_trajectory_dir(trajectory_id), first_step_id, last_step_id
    )

  @contextlib.contextmanager
  def _open_parallel_reads(
      self,
      trajectory_ids: Iterable[str] | None,
      get_path: Callable[[str], epath.Path],
  ) -> Iterator[Iterator[tuple[str, str | None]]]:
    """Opens parallel file reads yielding `(trajectory_id, text_or_none)`."""
    if (
        trajectory_ids is None
        and _is_local_filesystem(self.root_dir)
        and not self.root_dir.exists()
    ):
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
            if not _read_max_persisted_step_id(
                self.get_trajectory_dir(traj_id)
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
        step_text_futures = step_reads_future.result()
        step_by_id: dict[int, trajectory_lib.Step] = {}
        while step_text_futures:
          filename, step_text_future = step_text_futures.popleft()
          for step in _parse_step_file(filename, step_text_future.result()):
            step_by_id[step.step_id] = step
        steps = [step_by_id[step_id] for step_id in sorted(step_by_id)]
        trajs.append(meta.create_trajectory(steps=steps))
      return trajs
    finally:
      # Skips the reads that have not started and waits for the running ones,
      # so that no reader thread outlives the call.
      executor.shutdown(cancel_futures=True)

  def _list_trajectory_ids(self, executor: futures.Executor) -> Iterator[str]:
    """Returns the IDs of the trajectory directories in `root_dir`.

    Entries are filtered by name first. On remote object stores (where prefixes
    only exist when objects are written beneath them), name-matched entries are
    returned directly without per-entry `is_dir` RPCs. On local filesystems,
    `is_dir` checks are submitted in parallel before returning and yielded
    lazily in natural ID order.

    Args:
      executor: The executor to check on, in parallel, whether each entry of
        `root_dir` named like a trajectory directory is a directory on local
        filesystems. It must stay alive until the returned iterator is
        exhausted.

    Returns:
      An iterator over the trajectory IDs, sorted naturally by trajectory ID.
    """

    def _entry_sort_key(item: tuple[str, epath.Path]) -> list[int | str]:
      trajectory_id, _ = item
      return _natural_sort_key(trajectory_id)

    try:
      entries = list(self.root_dir.iterdir())
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if _is_not_found_error(exc) or (
          isinstance(exc, OSError) and not self.root_dir.exists()
      ):
        return iter(())
      raise

    matched_entries = sorted(
        (
            (trajectory_id, entry)
            for entry in entries
            if (match := _TRAJECTORY_DIR_REGEX.match(entry.name))
            for (trajectory_id,) in (match.groups(),)
        ),
        key=_entry_sort_key,
    )
    if not _is_local_filesystem(self.root_dir):
      return (trajectory_id for trajectory_id, _ in matched_entries)
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
      self, trajectory_id: str, executor: futures.Executor
  ) -> collections.deque[tuple[str, futures.Future[str]]]:
    """Lists a trajectory's step files and submits their reads to `executor`.

    Returns without waiting for the reads, so that no task on `executor` waits
    for another one.

    Args:
      trajectory_id: The trajectory identifier.
      executor: The executor to read the step files on.

    Returns:
      Pairs of `(filename, content_future)` ordered by step range so chunked
      `.jsonl` files are applied before single-step `.json` revisions.
    """
    matched_entries: list[tuple[tuple[int, int, int, str], epath.Path]] = []
    for entry in self.get_trajectory_dir(trajectory_id).iterdir():
      sort_key = _step_file_sort_key(entry.name)
      if sort_key is not None:
        matched_entries.append((sort_key, entry))
    matched_entries.sort()
    return collections.deque(
        (entry.name, executor.submit(entry.read_text))
        for _, entry in matched_entries
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
        characters that cannot be encoded in a trajectory directory name, or
        if `exclude_step_arrays` names fields that the bound metadata type's
        step classes cannot exclude.
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
        characters that cannot be encoded in a trajectory directory name, or
        if `exclude_step_arrays` names fields that the bound metadata type's
        step classes cannot exclude.
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
