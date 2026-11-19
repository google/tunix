# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging utilities for trajectory data, saving as CSV, JSONL, or JSON."""

import atexit
from collections.abc import Callable
from concurrent import futures
import csv
import dataclasses
import json
import os
import pathlib
import queue
import shutil
import signal
import tempfile
import threading
import time
import types
from typing import Any, Literal, TypeVar

from absl import logging
from etils import epath
from google.protobuf import json_format
from google.protobuf import message
import numpy as np

try:
  from tunix.utils import _trajectory_logger_ext  # pylint: disable=g-import-not-at-top
except ImportError:
  _trajectory_logger_ext = None

csv.field_size_limit(2**31 - 1)


class _PySerializedArray(str):
  """Pure-Python `SerializedArray` used when the C++ extension is unavailable.

  `repr()` returns the unquoted bracketed array, so arrays nested in containers
  format like Python lists.
  """

  def __repr__(self) -> str:
    return str(self)


if _trajectory_logger_ext is None:
  SerializedArray = _PySerializedArray
else:
  SerializedArray = _trajectory_logger_ext.SerializedArray

_T = TypeVar('_T')
TrajectoryFileFormat = Literal['csv', 'jsonl', 'json']

_DEFAULT_GCS_TIMEOUT_SEC = 10.0
_DEFAULT_STOP_TIMEOUT_SEC = 15.0
_DEFAULT_MAX_QUEUE_SIZE = 1000
_DEFAULT_NUM_WORKERS = 4


class _AbandonedOperationError(TimeoutError):
  """Raised when a timed-out worker thread may still be running."""


def _run_with_timeout(
    fn: Callable[[], _T],
    timeout_sec: float | None,
    operation_name: str = 'GCS I/O operation',
) -> _T:
  """Runs `fn` in a daemon thread and raises TimeoutError if deadline expires."""
  if timeout_sec is None or timeout_sec <= 0:
    return fn()

  result_box: list[_T] = []
  error_box: list[BaseException] = []

  def _target():
    try:
      result_box.append(fn())
    except BaseException as exc:  # pylint: disable=broad-except
      error_box.append(exc)

  worker = threading.Thread(target=_target, daemon=True)
  worker.start()
  worker.join(timeout=timeout_sec)
  if worker.is_alive():
    raise _AbandonedOperationError(
        f'{operation_name} timed out after {timeout_sec:.1f}s.'
    )
  if error_box:
    raise error_box[0]
  return result_box[0]


def _collect_csv_fieldnames(
    rows: list[dict[str, Any]],
    existing_fieldnames: list[str] | None = None,
) -> list[str]:
  """Returns ordered CSV column names preserving existing and first-seen order."""
  fieldnames: list[str] = list(existing_fieldnames or [])
  seen = set(fieldnames)
  for row in rows:
    for key in row:
      if key not in seen:
        seen.add(key)
        fieldnames.append(key)
  return fieldnames


def _read_csv_rows(f: Any) -> tuple[list[str], list[dict[str, Any]]]:
  """Reads column headers and row dicts from an open CSV text stream."""
  reader = csv.DictReader(f)
  fieldnames = list(reader.fieldnames or [])
  rows: list[dict[str, Any]] = list(reader)
  return fieldnames, rows


def _write_csv_rows(
    f: Any,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    *,
    write_header: bool = True,
) -> None:
  """Writes row dicts to an open CSV text stream."""
  writer = csv.DictWriter(
      f, fieldnames=fieldnames, lineterminator='\n', extrasaction='ignore'
  )
  if write_header:
    writer.writeheader()
  writer.writerows(rows)


def _read_gcs_csv(
    file_path: Any, gcs_timeout_sec: float | None
) -> tuple[list[str], list[dict[str, Any]]] | None:
  """Reads an existing CSV from GCS with a timeout."""

  def _do_read() -> tuple[list[str], list[dict[str, Any]]]:
    with file_path.open('r', encoding='utf-8') as f:
      return _read_csv_rows(f)

  try:
    return _run_with_timeout(
        _do_read, gcs_timeout_sec, f'GCS read({file_path})'
    )
  except TimeoutError:
    raise
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        'Could not read existing GCS file (possibly partial write): %s',
        e,
    )
    return None


def _cleanup_tmp_gcs_file(
    tmp_file_path: Any, gcs_timeout_sec: float | None
) -> None:
  """Attempts best-effort cleanup of a temporary GCS file with a timeout."""

  def _do_cleanup():
    if tmp_file_path.exists():
      tmp_file_path.unlink()

  try:
    _run_with_timeout(
        _do_cleanup, gcs_timeout_sec, f'GCS cleanup({tmp_file_path})'
    )
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        'Failed to clean up temporary GCS file %s: %s', tmp_file_path, e
    )


def _make_serializable(item: Any) -> Any:
  """Makes an object serializable."""
  if item is None:
    return None
  if isinstance(item, dict):
    return {key: _make_serializable(value) for key, value in item.items()}
  elif isinstance(item, list):
    return [_make_serializable(x) for x in item]
  elif isinstance(item, tuple):
    return tuple(_make_serializable(x) for x in item)
  elif dataclasses.is_dataclass(item) and not isinstance(item, type):
    return _make_serializable(dataclasses.asdict(item))
  elif isinstance(item, message.Message):
    return json_format.MessageToDict(item)
  elif isinstance(item, np.ndarray):
    if item.ndim == 0:
      return _make_serializable(item.item())
    if item.dtype.kind in ('i', 'u', 'f', 'b', 'U'):
      return item.tolist()
    return _make_serializable(item.tolist())
  elif isinstance(item, np.integer):
    return int(item)
  elif isinstance(item, np.floating):
    return float(item)
  elif isinstance(item, np.bool_):
    return bool(item)
  elif isinstance(item, np.str_):
    return str(item)
  elif isinstance(item, (float, int, bool, str)):
    return item
  else:
    # Serialize other types by stringifying them.
    logging.log_first_n(
        logging.WARNING,
        'Could not serialize item of type %s, turning to string',
        1,
        type(item),
    )
    return str(item)


def _format_ndarray_for_csv(arr: np.ndarray) -> SerializedArray:
  """Formats a 1D+ numeric or boolean ndarray for CSV, releasing the GIL in C++."""
  if arr.dtype.kind == 'f' and arr.dtype.itemsize < 4:
    arr = arr.astype(np.float32)
  if _trajectory_logger_ext is None:
    return _PySerializedArray(str(arr.tolist()))
  return _trajectory_logger_ext.format_ndarray(arr)


def _make_csv_serializable(item: Any) -> Any:
  """Makes an object serializable for DataFrame/CSV with GIL-free ndarray formatting."""
  if item is None:
    return None
  if isinstance(item, dict):
    return {key: _make_csv_serializable(value) for key, value in item.items()}
  elif isinstance(item, list):
    return [_make_csv_serializable(x) for x in item]
  elif isinstance(item, tuple):
    return tuple(_make_csv_serializable(x) for x in item)
  elif dataclasses.is_dataclass(item) and not isinstance(item, type):
    return _make_csv_serializable(dataclasses.asdict(item))
  elif isinstance(item, message.Message):
    return json_format.MessageToDict(item)
  elif isinstance(item, np.ndarray):
    if item.ndim == 0:
      return _make_csv_serializable(item.item())
    if item.dtype.kind in ('i', 'u', 'f', 'b'):
      return _format_ndarray_for_csv(item)
    if item.dtype.kind == 'U':
      return item.tolist()
    return _make_csv_serializable(item.tolist())
  elif isinstance(item, np.integer):
    return int(item)
  elif isinstance(item, np.floating):
    return float(item)
  elif isinstance(item, np.bool_):
    return bool(item)
  elif isinstance(item, np.str_):
    return str(item)
  elif isinstance(item, (float, int, bool, str)):
    return item
  else:
    logging.log_first_n(
        logging.WARNING,
        'Could not serialize item of type %s, turning to string',
        1,
        type(item),
    )
    return str(item)


def dumps_json(item: Any, *, indent: int | None = None) -> str:
  """Serializes a trajectory item to a JSON string, releasing the GIL in C++."""
  if _trajectory_logger_ext is None:
    return json.dumps(_make_serializable(item), indent=indent)
  return _trajectory_logger_ext.dumps_json(
      item, -1 if indent is None else indent
  )


def _get_item_name(item: Any) -> str | None:
  """Returns item class name if it's a dataclass, else None."""
  if dataclasses.is_dataclass(item):
    return item.__class__.__name__
  return None


def _is_gcs_path(path: Any) -> bool:
  """Returns True if `path` points to Google Cloud Storage."""
  return str(path).startswith('gs://')


def _serialize_items(
    items_list: list[Any],
    serializer_fn: Callable[[Any], _T],
    *,
    num_workers: int = 1,
    executor: futures.Executor | None = None,
) -> list[_T]:
  """Serializes a batch of items, parallelizing across worker threads when >1."""
  if len(items_list) <= 1 or (executor is None and num_workers <= 1):
    return [serializer_fn(x) for x in items_list]
  if executor is not None:
    return list(executor.map(serializer_fn, items_list))
  with futures.ThreadPoolExecutor(
      max_workers=min(num_workers, len(items_list))
  ) as pool:
    return list(pool.map(serializer_fn, items_list))


def _read_gcs_text(file_path: Any, gcs_timeout_sec: float | None) -> str | None:
  """Reads an existing UTF-8 text file from GCS with a timeout."""

  def _do_read() -> str:
    with file_path.open('r') as f:
      return f.read()

  try:
    return _run_with_timeout(
        _do_read, gcs_timeout_sec, f'GCS read({file_path})'
    )
  except TimeoutError:
    raise
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        'Could not read existing GCS file (possibly partial write): %s',
        e,
    )
    return None


def _append_json_array_text(
    existing_text: str | None, json_lines: list[str]
) -> str:
  """Appends pre-serialized JSON object strings into a valid JSON array document."""
  new_body = ',\n'.join(json_lines)
  if existing_text is not None:
    stripped = existing_text.strip()
    if stripped.startswith('[') and stripped.endswith(']'):
      inner = stripped[1:-1].strip()
      if inner:
        return f'[\n{inner},\n{new_body}\n]\n'
  return f'[\n{new_body}\n]\n'


def _write_json_or_jsonl(
    file_path: Any,
    filename: str,
    json_lines: list[str],
    *,
    file_format: Literal['jsonl', 'json'],
    gcs_timeout_sec: float | None,
    local_staging_dir: str | os.PathLike[str] | None,
) -> None:
  """Writes pre-serialized JSON lines to a local or GCS `.jsonl` / `.json` file."""
  if _is_gcs_path(file_path):
    tmp_file_path = file_path.parent / f'{file_path.name}.{time.time_ns()}.tmp'
    if local_staging_dir is not None:
      staging_file = pathlib.Path(local_staging_dir) / filename
      staging_file.parent.mkdir(parents=True, exist_ok=True)
      existing_text: str | None = None
      if not staging_file.exists():
        try:
          remote_exists = _run_with_timeout(
              file_path.exists, gcs_timeout_sec, f'GCS exists({file_path})'
          )
        except TimeoutError as e:
          logging.warning(
              'Timed out checking existing GCS file %s; skipping flush to avoid'
              ' overwriting remote state: %s',
              file_path,
              e,
          )
          return
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              'Could not check existing GCS file %s: %s', file_path, e
          )
          remote_exists = False
        if remote_exists:
          try:
            existing_text = _read_gcs_text(file_path, gcs_timeout_sec)
          except TimeoutError as e:
            logging.warning(
                'Timed out reading existing GCS file %s; skipping flush to'
                ' avoid overwriting remote state: %s',
                file_path,
                e,
            )
            return
        if file_format == 'jsonl':
          with staging_file.open('w', encoding='utf-8', newline='') as f:
            if existing_text:
              f.write(existing_text)
              if not existing_text.endswith('\n'):
                f.write('\n')
            f.write(''.join(f'{line}\n' for line in json_lines))
        else:
          with staging_file.open('w', encoding='utf-8', newline='') as f:
            f.write(_append_json_array_text(existing_text, json_lines))
      else:
        if file_format == 'jsonl':
          with staging_file.open('a', encoding='utf-8', newline='') as f:
            f.write(''.join(f'{line}\n' for line in json_lines))
        else:
          existing_text = staging_file.read_text(encoding='utf-8')
          with staging_file.open('w', encoding='utf-8', newline='') as f:
            f.write(_append_json_array_text(existing_text, json_lines))

      aborted = threading.Event()

      def _upload_staged_to_gcs():
        with (
            staging_file.open('r', encoding='utf-8') as src,
            tmp_file_path.open('w') as dst,
        ):
          shutil.copyfileobj(src, dst)
        if aborted.is_set():
          return
        tmp_file_path.replace(file_path)

      try:
        _run_with_timeout(
            _upload_staged_to_gcs, gcs_timeout_sec, f'GCS write({file_path})'
        )
      except _AbandonedOperationError as e:
        aborted.set()
        logging.error(
            'Timed out finalizing write to %s; leaving %s for lifecycle'
            ' cleanup: %s',
            file_path,
            tmp_file_path,
            e,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.error('Failed to finalize write to %s: %s', file_path, e)
        _cleanup_tmp_gcs_file(tmp_file_path, gcs_timeout_sec)
    else:
      try:
        remote_exists = _run_with_timeout(
            file_path.exists, gcs_timeout_sec, f'GCS exists({file_path})'
        )
      except TimeoutError as e:
        logging.warning(
            'Timed out checking existing GCS file %s; skipping flush to avoid'
            ' overwriting remote state: %s',
            file_path,
            e,
        )
        return
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Could not check existing GCS file %s: %s', file_path, e
        )
        remote_exists = False

      existing_text = None
      if remote_exists:
        try:
          existing_text = _read_gcs_text(file_path, gcs_timeout_sec)
        except TimeoutError as e:
          logging.warning(
              'Timed out reading existing GCS file %s; skipping flush to avoid'
              ' overwriting remote state: %s',
              file_path,
              e,
          )
          return

      if file_format == 'jsonl':
        prefix = ''
        if existing_text:
          prefix = (
              existing_text
              if existing_text.endswith('\n')
              else f'{existing_text}\n'
          )
        full_text = prefix + ''.join(f'{line}\n' for line in json_lines)
      else:
        full_text = _append_json_array_text(existing_text, json_lines)

      aborted = threading.Event()

      def _write_and_replace():
        with tmp_file_path.open('w') as f:
          f.write(full_text)
        if aborted.is_set():
          return
        tmp_file_path.replace(file_path)

      try:
        _run_with_timeout(
            _write_and_replace, gcs_timeout_sec, f'GCS write({file_path})'
        )
      except _AbandonedOperationError as e:
        aborted.set()
        logging.error(
            'Timed out finalizing write to %s; leaving %s for lifecycle'
            ' cleanup: %s',
            file_path,
            tmp_file_path,
            e,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.error('Failed to finalize write to %s: %s', file_path, e)
        _cleanup_tmp_gcs_file(tmp_file_path, gcs_timeout_sec)
  else:
    if file_format == 'jsonl':
      with file_path.open('a') as f:
        f.write(''.join(f'{line}\n' for line in json_lines))
    else:
      existing_text = None
      if file_path.exists():
        with file_path.open('r') as f:
          existing_text = f.read()
      with file_path.open('w') as f:
        f.write(_append_json_array_text(existing_text, json_lines))


def log_item(
    log_path: str,
    item: dict[str, Any] | Any,
    suffix: str | None = None,
    *,
    file_format: TrajectoryFileFormat = 'json',
    num_workers: int = 1,
    executor: futures.Executor | None = None,
    gcs_timeout_sec: float | None = _DEFAULT_GCS_TIMEOUT_SEC,
    local_staging_dir: str | os.PathLike[str] | None = None,
):
  """Logs a dictionary, dataclass or list to a CSV, JSONL, or JSON file.

  The filename is determined by item type if it is a dataclass, otherwise
  it defaults to `trajectory_log.<file_format>`. If item is a list, the type of
  the first element is used.

  Args:
    log_path: Directory to log to.
    item: Item (or list of items) to log.
    suffix: Optional suffix to add to filename before the file extension.
    file_format: Output format, one of `'csv'`, `'jsonl'`, or `'json'`.
    num_workers: Number of worker threads for parallel C++ `nogil` serialization
      when `item` is a batch (`list`) and `executor` is not provided.
    executor: Optional shared `Executor` for parallel batch serialization.
    gcs_timeout_sec: Timeout in seconds for GCS read/write operations.
    local_staging_dir: Optional local directory used to stage incremental
      appends before uploading to GCS, avoiding quadratic re-reads from GCS.
  """

  if log_path is None:
    raise ValueError('No directory for logging provided.')
  if file_format not in ('csv', 'jsonl', 'json'):
    raise ValueError(
        f'Unsupported file_format {file_format!r}; expected one of'
        " ('csv', 'jsonl', 'json')."
    )
  if num_workers < 1:
    raise ValueError(f'num_workers must be >= 1, got {num_workers}.')

  if isinstance(item, list) and not item:
    logging.warning('Trying to log an empty list, skipping.')
    return

  if not (dataclasses.is_dataclass(item) or isinstance(item, (dict, list))):
    raise ValueError(f'Item {item} is not a dataclass, dictionary or list.')

  items_list = item if isinstance(item, list) else [item]

  log_path = epath.Path(log_path)  # pyrefly: ignore[bad-assignment]
  log_path.mkdir(parents=True, exist_ok=True)  # pyrefly: ignore[missing-attribute]

  # GCS has no real directories: `mkdir()` is a no-op and `is_dir()` stays
  # False until an object exists under the prefix, so only enforce the
  # directory check on filesystems that model directories.
  assert (
      _is_gcs_path(log_path) or log_path.is_dir()  # pyrefly: ignore[missing-attribute]
  ), f'log_path `{log_path}` must be a directory.'

  item_name = _get_item_name(items_list[0])
  file_stem = item_name if item_name else 'trajectory_log'
  filename = (
      f'{file_stem}_{suffix}.{file_format}'
      if suffix
      else f'{file_stem}.{file_format}'
  )
  file_path = log_path / filename  # pyrefly: ignore[unsupported-operation]
  logging.log_first_n(logging.INFO, f'Logging item to {file_path}', 1)

  if file_format in ('jsonl', 'json'):
    json_lines = _serialize_items(
        items_list,
        dumps_json,
        num_workers=num_workers,
        executor=executor,
    )
    _write_json_or_jsonl(
        file_path,
        filename,
        json_lines,
        file_format=file_format,
        gcs_timeout_sec=gcs_timeout_sec,
        local_staging_dir=local_staging_dir,
    )
    return

  serialized_items: list[dict[str, Any]] = _serialize_items(
      items_list,
      _make_csv_serializable,
      num_workers=num_workers,
      executor=executor,
  )
  fieldnames = _collect_csv_fieldnames(serialized_items)
  if _is_gcs_path(file_path):
    tmp_file_path = (
        file_path.parent / f'{file_path.name}.{time.time_ns()}.tmp'
    )
    if local_staging_dir is not None:
      staging_file = pathlib.Path(local_staging_dir) / filename
      staging_file.parent.mkdir(parents=True, exist_ok=True)
      if not staging_file.exists():
        try:
          remote_exists = _run_with_timeout(
              file_path.exists, gcs_timeout_sec, f'GCS exists({file_path})'
          )
        except TimeoutError as e:
          logging.warning(
              'Timed out checking existing GCS file %s; skipping flush to avoid'
              ' overwriting remote state: %s',
              file_path,
              e,
          )
          return
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              'Could not check existing GCS file %s: %s', file_path, e
          )
          remote_exists = False
        rows_to_write = serialized_items
        if remote_exists:
          try:
            old_csv = _read_gcs_csv(file_path, gcs_timeout_sec)
          except TimeoutError as e:
            logging.warning(
                'Timed out reading existing GCS file %s; skipping flush to'
                ' avoid overwriting remote state: %s',
                file_path,
                e,
            )
            return
          if old_csv is not None:
            old_cols, old_rows = old_csv
            fieldnames = _collect_csv_fieldnames(serialized_items, old_cols)
            rows_to_write = old_rows + serialized_items
        with staging_file.open('w', encoding='utf-8', newline='') as f:
          _write_csv_rows(f, fieldnames, rows_to_write, write_header=True)
      else:
        with staging_file.open('r', encoding='utf-8', newline='') as f:
          existing_cols = list(next(csv.reader(f), []))
        if set(fieldnames).issubset(set(existing_cols)):
          with staging_file.open('a', encoding='utf-8', newline='') as f:
            _write_csv_rows(
                f, existing_cols, serialized_items, write_header=False
            )
        else:
          with staging_file.open('r', encoding='utf-8', newline='') as f:
            staged_cols, staged_rows = _read_csv_rows(f)
          combined_cols = _collect_csv_fieldnames(
              serialized_items, staged_cols
          )
          with staging_file.open('w', encoding='utf-8', newline='') as f:
            _write_csv_rows(
                f,
                combined_cols,
                staged_rows + serialized_items,
                write_header=True,
            )

      aborted = threading.Event()

      def _upload_staged_to_gcs():
        with (
            staging_file.open('r', encoding='utf-8') as src,
            tmp_file_path.open('w', encoding='utf-8') as dst,
        ):
          shutil.copyfileobj(src, dst)
        if aborted.is_set():
          return
        tmp_file_path.replace(file_path)

      try:
        _run_with_timeout(
            _upload_staged_to_gcs, gcs_timeout_sec, f'GCS write({file_path})'
        )
      except _AbandonedOperationError as e:
        aborted.set()
        # Writer may still be live; unlinking now would race it.
        logging.error(
            'Timed out finalizing write to %s; leaving %s for lifecycle'
            ' cleanup: %s',
            file_path,
            tmp_file_path,
            e,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.error('Failed to finalize write to %s: %s', file_path, e)
        _cleanup_tmp_gcs_file(tmp_file_path, gcs_timeout_sec)
    else:
      try:
        remote_exists = _run_with_timeout(
            file_path.exists, gcs_timeout_sec, f'GCS exists({file_path})'
        )
      except TimeoutError as e:
        logging.warning(
            'Timed out checking existing GCS file %s; skipping flush to avoid'
            ' overwriting remote state: %s',
            file_path,
            e,
        )
        return
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Could not check existing GCS file %s: %s', file_path, e
        )
        remote_exists = False

      rows_to_write = serialized_items
      if remote_exists:
        try:
          old_csv = _read_gcs_csv(file_path, gcs_timeout_sec)
        except TimeoutError as e:
          logging.warning(
              'Timed out reading existing GCS file %s; skipping flush to avoid'
              ' overwriting remote state: %s',
              file_path,
              e,
          )
          return
        if old_csv is not None:
          old_cols, old_rows = old_csv
          fieldnames = _collect_csv_fieldnames(serialized_items, old_cols)
          rows_to_write = old_rows + serialized_items

      aborted = threading.Event()

      def _write_and_replace():
        with tmp_file_path.open('w', encoding='utf-8') as f:
          _write_csv_rows(f, fieldnames, rows_to_write, write_header=True)
        if aborted.is_set():
          return
        # epath.Path.replace() handles the GCS 'rename' (copy + delete)
        tmp_file_path.replace(file_path)

      try:
        _run_with_timeout(
            _write_and_replace, gcs_timeout_sec, f'GCS write({file_path})'
        )
      except _AbandonedOperationError as e:
        aborted.set()
        # Writer may still be live; unlinking now would race it.
        logging.error(
            'Timed out finalizing write to %s; leaving %s for lifecycle'
            ' cleanup: %s',
            file_path,
            tmp_file_path,
            e,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.error('Failed to finalize write to %s: %s', file_path, e)
        _cleanup_tmp_gcs_file(tmp_file_path, gcs_timeout_sec)
  else:
    write_header = not file_path.exists()
    with file_path.open('a', encoding='utf-8') as f:
      _write_csv_rows(
          f, fieldnames, serialized_items, write_header=write_header
      )


class AsyncTrajectoryLogger:
  """A logger that logs trajectories asynchronously in a background thread."""

  def __init__(
      self,
      log_dir: str,
      *,
      file_format: TrajectoryFileFormat = 'json',
      num_workers: int = _DEFAULT_NUM_WORKERS,
      max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
      stop_timeout_sec: float = _DEFAULT_STOP_TIMEOUT_SEC,
      gcs_timeout_sec: float | None = _DEFAULT_GCS_TIMEOUT_SEC,
  ):
    self._stopped = True
    self._stop_lock = threading.RLock()
    if file_format not in ('csv', 'jsonl', 'json'):
      raise ValueError(
          f'Unsupported file_format {file_format!r}; expected one of'
          " ('csv', 'jsonl', 'json')."
      )
    if num_workers < 1:
      raise ValueError(f'num_workers must be >= 1, got {num_workers}.')
    self._log_dir = log_dir
    self._file_format: TrajectoryFileFormat = file_format
    self._num_workers = num_workers
    self._file_suffix = str(int(time.time()))
    self._stop_timeout_sec = stop_timeout_sec
    self._gcs_timeout_sec = gcs_timeout_sec
    self._logging_queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
    self._stopped = False
    self._stop_event = threading.Event()
    self._abandoned = False
    self._executor: futures.ThreadPoolExecutor | None = None
    if num_workers > 1:
      self._executor = futures.ThreadPoolExecutor(
          max_workers=num_workers,
          thread_name_prefix='tunix_traj_ser',
      )
    self._staging_tempdir = (
        tempfile.TemporaryDirectory(prefix='tunix_traj_stage_')
        if _is_gcs_path(log_dir)
        else None
    )

    def _worker():
      while not self._abandoned:
        try:
          item = self._logging_queue.get(timeout=0.5)
        except queue.Empty:
          if self._stop_event.is_set():
            break
          continue

        if item is None:  # Sentinel for stopping
          self._logging_queue.task_done()
          break

        # Batching: drain the queue to log items in groups
        items = [item]
        stop_received = False
        while not self._logging_queue.empty():
          try:
            next_item = self._logging_queue.get_nowait()
            if next_item is None:
              # Acknowledge the sentinel and terminate after logging current
              # batch.
              self._logging_queue.task_done()
              stop_received = True
              break
            items.append(next_item)
          except queue.Empty:
            break

        if self._abandoned:
          for _ in range(len(items)):
            self._logging_queue.task_done()
          break

        try:
          log_item(
              self._log_dir,
              items,
              self._file_suffix,
              file_format=self._file_format,
              num_workers=self._num_workers,
              executor=self._executor,
              gcs_timeout_sec=self._gcs_timeout_sec,
              local_staging_dir=(
                  self._staging_tempdir.name
                  if self._staging_tempdir is not None
                  else None
              ),
          )
        except Exception:  # pylint: disable=broad-except
          logging.exception('Failed to log trajectories.')
        finally:
          for _ in range(len(items)):
            self._logging_queue.task_done()
        if (
            stop_received
            or self._abandoned
            or (self._stop_event.is_set() and self._logging_queue.empty())
        ):
          break

    self._logging_thread = threading.Thread(target=_worker, daemon=True)
    self._logging_thread.start()

    # Register cleanup
    atexit.register(self.stop)

    # Register signal handlers for robust termination
    if threading.current_thread() is threading.main_thread():
      try:
        signal.signal(signal.SIGINT, self._handle_signal)  # pyrefly: ignore[bad-argument-type]
        signal.signal(signal.SIGTERM, self._handle_signal)  # pyrefly: ignore[bad-argument-type]
        signal.signal(signal.SIGHUP, self._handle_signal)  # pyrefly: ignore[bad-argument-type]
      except ValueError:
        logging.warning('Failed to register signal handlers.')

    logging.info('Started trajectory logging thread.')

  def _handle_signal(self, signum: int, frame: types.FrameType):
    """Gracefully stops the logger and exits."""
    del frame  # Unused.
    logging.info('Received signal %d, flushing trajectory logger...', signum)
    self.stop()
    # Restore default handler and re-send signal to self
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

  def __del__(self):
    """Ensures stop is called when the object is destroyed."""
    self.stop()

  def stop(self):
    """Stops the background logging thread gracefully with a bounded timeout."""
    with self._stop_lock:
      if self._stopped:
        return
      self._stopped = True
      self._stop_event.set()

    logging.info('Stopping trajectory logging thread...')
    try:
      self._logging_queue.put_nowait(None)
    except queue.Full:
      logging.warning(
          'Trajectory logging queue is full during shutdown; signaling stop'
          ' event without blocking.'
      )

    self._logging_thread.join(timeout=self._stop_timeout_sec)
    if self._logging_thread.is_alive():
      self._abandoned = True
      if self._executor is not None:
        self._executor.shutdown(wait=False, cancel_futures=True)
      logging.warning(
          'Trajectory logging thread did not terminate within %.1fs; proceeding'
          ' with shutdown to avoid deadlock.',
          self._stop_timeout_sec,
      )
    else:
      if self._executor is not None:
        self._executor.shutdown(wait=True)
      if self._staging_tempdir is not None:
        try:
          self._staging_tempdir.cleanup()
        except Exception:  # pylint: disable=broad-except
          pass
      logging.info('Stopped trajectory logging thread.')

  def log_item_async(self, item: dict[str, Any] | Any):
    """Adds an item to the logging queue to be logged asynchronously."""
    if self._stopped:
      logging.warning('Trajectory logger already stopped.')
      return
    if not self._logging_thread.is_alive():
      logging.warning(
          'Trajectory logging background thread is not alive; dropping item.'
      )
      return
    try:
      self._logging_queue.put_nowait(item)
    except queue.Full:
      logging.warning(
          'Trajectory logging queue is full (maxsize=%d); dropping trajectory'
          ' item to avoid blocking caller.',
          self._logging_queue.maxsize,
      )
