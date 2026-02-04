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

import dataclasses
from typing import Any

from absl import logging
from etils import epath
from google.protobuf import json_format
from google.protobuf import message
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


def log_item(log_path: str, item: dict[str, Any] | Any):
  """Logs a dictionary, dataclass or list."""

  if log_path is None:
    raise ValueError('No directory for logging provided.')

  logging.log_first_n(logging.INFO, f'Logging item to {log_path}', 1)
  if dataclasses.is_dataclass(item) or isinstance(item, (dict, list)):
    serialized_item = _make_serializable(item)
  else:
    raise ValueError(f'Item {item} is not a dataclass, dictionary or list.')

  log_path = epath.Path(log_path)
  log_path.mkdir(parents=True, exist_ok=True)

  assert log_path.is_dir(), f'log_path `{log_path}` must be a directory.'
  file_path = log_path / 'trajectory_log.csv'
  write_header = not file_path.exists()

  df = pd.DataFrame(
      serialized_item if isinstance(item, list) else [serialized_item]
  )
  with file_path.open('a') as f:
    df.to_csv(f, header=write_header, index=False)
