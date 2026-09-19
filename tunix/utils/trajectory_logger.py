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

"""Logging utilities for trajectory data, saving as CSV."""

import atexit
from collections.abc import Callable
import dataclasses
import os
import pathlib
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
import types
from typing import Any, TypeVar

from absl import logging
from etils import epath
from google.protobuf import json_format
from google.protobuf import message
import numpy as np
import pandas as pd

_T = TypeVar('_T')

_DEFAULT_GCS_TIMEOUT_SEC = 10.0
_DEFAULT_STOP_TIMEOUT_SEC = 15.0
_DEFAULT_MAX_QUEUE_SIZE = 1000


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


def _read_gcs_csv(
    file_path: Any, gcs_timeout_sec: float | None
) -> pd.DataFrame | None:
  """Reads an existing CSV from GCS with a timeout."""

  def _do_read() -> pd.DataFrame:
    with file_path.open('r') as f:
      try:
        return pd.read_csv(f)
      except pd.errors.ParserError:
        f.seek(0)
        return pd.read_csv(f, engine='python')

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
  if isinstance(item, dict):
    return {key: _make_serializable(value) for key, value in item.items()}
  elif isinstance(item, list):
    return [_make_serializable(item) for item in item]
  elif isinstance(item, tuple):
    return tuple(_make_serializable(item) for item in item)
  elif dataclasses.is_dataclass(item):
    return _make_serializable(dataclasses.asdict(item))
  elif isinstance(item, message.Message):
    return json_format.MessageToDict(item)
  elif isinstance(item, np.ndarray):
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


def _get_item_name(item: Any) -> str | None:
  """Returns item class name if it's a dataclass, else None."""
  if dataclasses.is_dataclass(item):
    return item.__class__.__name__
  return None


def _is_gcs_path(path: Any) -> bool:
  """Returns True if `path` points to Google Cloud Storage."""
  return str(path).startswith('gs://')


def log_item(
    log_path: str,
    item: dict[str, Any] | Any,
    suffix: str | None = None,
    *,
    gcs_timeout_sec: float | None = _DEFAULT_GCS_TIMEOUT_SEC,
    local_staging_dir: str | os.PathLike[str] | None = None,
):
  """Logs a dictionary, dataclass or list to a csv file.

  The filename is determined by item type if it is a dataclass, otherwise
  it defaults to 'trajectory_log.csv'. If item is a list, the type of
  the first element is used.

  Args:
    log_path: Directory to log to.
    item: Item to log.
    suffix: Optional suffix to add to filename before `.csv`.
    gcs_timeout_sec: Timeout in seconds for GCS read/write operations.
    local_staging_dir: Optional local directory used to stage incremental CSV
      appends before uploading to GCS, avoiding quadratic re-reads from GCS.
  """

  if log_path is None:
    raise ValueError('No directory for logging provided.')

  if isinstance(item, list) and not item:
    logging.warning('Trying to log an empty list, skipping.')
    return

  if dataclasses.is_dataclass(item) or isinstance(item, (dict, list)):
    serialized_item = _make_serializable(item)
  else:
    raise ValueError(f'Item {item} is not a dataclass, dictionary or list.')

  log_path = epath.Path(log_path)  # pyrefly: ignore[bad-assignment]
  log_path.mkdir(parents=True, exist_ok=True)  # pyrefly: ignore[missing-attribute]

  # GCS has no real directories: `mkdir()` is a no-op and `is_dir()` stays
  # False until an object exists under the prefix, so only enforce the
  # directory check on filesystems that model directories.
  assert (
      _is_gcs_path(log_path) or log_path.is_dir()  # pyrefly: ignore[missing-attribute]
  ), f'log_path `{log_path}` must be a directory.'

  if isinstance(item, list):
    item_name = _get_item_name(item[0])
  else:
    item_name = _get_item_name(item)

  file_stem = item_name if item_name else 'trajectory_log'
  filename = f'{file_stem}_{suffix}.csv' if suffix else f'{file_stem}.csv'
  file_path = log_path / filename  # pyrefly: ignore[unsupported-operation]
  logging.log_first_n(logging.INFO, f'Logging item to {file_path}', 1)

  df = pd.DataFrame(
      serialized_item if isinstance(item, list) else [serialized_item]
  )
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
        if remote_exists:
          try:
            old_df = _read_gcs_csv(file_path, gcs_timeout_sec)
          except TimeoutError as e:
            logging.warning(
                'Timed out reading existing GCS file %s; skipping flush to'
                ' avoid overwriting remote state: %s',
                file_path,
                e,
            )
            return
          if old_df is not None:
            df = pd.concat([old_df, df], ignore_index=True)
        with staging_file.open('w', encoding='utf-8', newline='') as f:
          df.to_csv(f, header=True, index=False)
      else:
        existing_cols = pd.read_csv(
            staging_file, nrows=0, encoding='utf-8'
        ).columns.tolist()
        if list(df.columns) == existing_cols:
          with staging_file.open('a', encoding='utf-8', newline='') as f:
            df.to_csv(f, header=False, index=False)
        elif set(df.columns).issubset(set(existing_cols)):
          df = df.reindex(columns=existing_cols)
          with staging_file.open('a', encoding='utf-8', newline='') as f:
            df.to_csv(f, header=False, index=False)
        else:
          staged_df = pd.read_csv(staging_file, encoding='utf-8')
          combined_df = pd.concat([staged_df, df], ignore_index=True)
          with staging_file.open('w', encoding='utf-8', newline='') as f:
            combined_df.to_csv(f, header=True, index=False)

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

      if remote_exists:
        try:
          old_df = _read_gcs_csv(file_path, gcs_timeout_sec)
        except TimeoutError as e:
          logging.warning(
              'Timed out reading existing GCS file %s; skipping flush to avoid'
              ' overwriting remote state: %s',
              file_path,
              e,
          )
          return
        if old_df is not None:
          df = pd.concat([old_df, df], ignore_index=True)

      aborted = threading.Event()

      def _write_and_replace():
        with tmp_file_path.open('w') as f:
          df.to_csv(f, header=True, index=False)
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
    with file_path.open('a') as f:
      df.to_csv(f, header=write_header, index=False)


class AsyncTrajectoryLogger:
  """A logger that logs trajectories asynchronously in a background thread."""

  def __init__(
      self,
      log_dir: str,
      *,
      max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
      stop_timeout_sec: float = _DEFAULT_STOP_TIMEOUT_SEC,
      gcs_timeout_sec: float | None = _DEFAULT_GCS_TIMEOUT_SEC,
  ):
    self._log_dir = log_dir
    self._file_suffix = str(int(time.time()))
    self._stop_timeout_sec = stop_timeout_sec
    self._gcs_timeout_sec = gcs_timeout_sec
    self._logging_queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
    self._stopped = False
    self._stop_event = threading.Event()
    self._stop_lock = threading.RLock()
    self._staging_tempdir = (
        tempfile.TemporaryDirectory(prefix='tunix_traj_stage_')
        if _is_gcs_path(log_dir)
        else None
    )

    def _worker():
      while True:
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

        try:
          log_item(
              self._log_dir,
              items,
              self._file_suffix,
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
        if stop_received or (
            self._stop_event.is_set() and self._logging_queue.empty()
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
      logging.warning(
          'Trajectory logging thread did not terminate within %.1fs; proceeding'
          ' with shutdown to avoid deadlock.',
          self._stop_timeout_sec,
      )
    else:
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
