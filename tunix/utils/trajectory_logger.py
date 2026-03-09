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
import dataclasses
import queue
import threading
import time
from typing import Any

from absl import logging
from etils import epath
from google.protobuf import json_format
from google.protobuf import message
import numpy as np
import pandas as pd

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


def log_item(
    log_path: str, item: dict[str, Any] | Any, suffix: str | None = None
):
  """Logs a dictionary, dataclass or list to a csv file.

  The filename is determined by item type if it is a dataclass, otherwise
  it defaults to 'trajectory_log.csv'. If item is a list, the type of
  the first element is used.

  Args:
    log_path: Directory to log to.
    item: Item to log.
    suffix: Optional suffix to add to filename before `.csv`.
  """

  if log_path is None:
    raise ValueError('No directory for logging provided.')

  if isinstance(item, list) and not item:
    logging.warning('Trying to log an empty list, skipping.')
    return

  logging.log_first_n(logging.INFO, f'Logging item to {log_path}', 1)
  if dataclasses.is_dataclass(item) or isinstance(item, (dict, list)):
    serialized_item = _make_serializable(item)
  else:
    raise ValueError(f'Item {item} is not a dataclass, dictionary or list.')

  log_path = epath.Path(log_path)
  log_path.mkdir(parents=True, exist_ok=True)

  assert log_path.is_dir(), f'log_path `{log_path}` must be a directory.'

  if isinstance(item, list):
    item_name = _get_item_name(item[0])
  else:
    item_name = _get_item_name(item)

  file_stem = item_name if item_name else 'trajectory_log'
  filename = f'{file_stem}_{suffix}.csv' if suffix else f'{file_stem}.csv'
  file_path = log_path / filename
  write_header = not file_path.exists()

  df = pd.DataFrame(
      serialized_item if isinstance(item, list) else [serialized_item]
  )
  if str(file_path).startswith('gs://'):
    if file_path.exists():
      with file_path.open('r') as f:
        old_df = pd.read_csv(f)
      df = pd.concat([old_df, df], ignore_index=True)
    with file_path.open('w') as f:
      df.to_csv(f, header=True, index=False)
  else:
    with file_path.open('a') as f:
      df.to_csv(f, header=write_header, index=False)


class AsyncTrajectoryLogger:
  """A logger that logs trajectories asynchronously in a background thread."""

  def __init__(self, log_dir: str):
    self._log_dir = log_dir
    self._file_suffix = str(int(time.time()))
    self._logging_queue = queue.Queue()
    self._stopped = False

    def _worker():
      while True:
        item = self._logging_queue.get()
        if item is None:  # Sentinel for stopping
          self._logging_queue.task_done()
          break
        try:
          log_item(self._log_dir, item, self._file_suffix)
        except Exception:  # pylint: disable=broad-except
          logging.exception('Failed to log trajectory.')
        finally:
          self._logging_queue.task_done()

    self._logging_thread = threading.Thread(target=_worker, daemon=True)
    self._logging_thread.start()
    atexit.register(self.stop)
    logging.info('Started trajectory logging thread.')

  def __del__(self):
    """Ensures stop is called when the object is destroyed."""
    self.stop()

  def stop(self):
    """Stops the background logging thread gracefully."""
    if self._stopped:
      return
    logging.info('Stopping trajectory logging thread...')
    self._logging_queue.put(None)
    self._logging_queue.join()
    self._logging_thread.join(timeout=10)
    self._stopped = True
    logging.info('Stopped trajectory logging thread.')

  def log_item_async(self, item: dict[str, Any] | Any):
    """Adds an item to the logging queue to be logged asynchronously."""
    if self._stopped:
      logging.warning('Trajectory logger already stopped.')
      return
    self._logging_queue.put(item)
