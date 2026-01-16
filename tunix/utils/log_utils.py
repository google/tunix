"""Logging utilities for trajectory data."""

import dataclasses
import os
from typing import Any

from absl import logging
from google.protobuf import json_format
from google.protobuf import message


@dataclasses.dataclass
class TrajectoryData:
  """Rollout metadata."""

  prompt: str
  completion: str
  trajectory_id: str
  global_step: int


def make_serializable(item: Any) -> Any:
  """Makes an object serializable."""
  if isinstance(item, dict):
    return {key: make_serializable(value) for key, value in item.items()}
  elif isinstance(item, list):
    return [make_serializable(item) for item in item]
  elif isinstance(item, tuple):
    return tuple(make_serializable(item) for item in item)
  elif dataclasses.is_dataclass(item):
    return make_serializable(dataclasses.asdict(item))
  elif isinstance(item, message.Message):
    return json_format.MessageToDict(item)
  elif isinstance(item, (float, int, bool, str)):
    return item
  else:
    # Serialize other types by stringifying them.
    logging.warning(
        'Could not serialize item of type %s, turning to string', type(item)
    )
    return str(item)


Dataclass = Any  # Dataclasses have no supertype, we handle them dynamically


def _escape_for_markdown_table(text: str) -> str:
  return text.replace('|', '\\|').replace('\n', '<br>')


def _create_markdown_table_row(item: dict[str, Any], keys: list[str]) -> str:
  """Creates a markdown table row from a dictionary and a list of keys."""
  escaped_values = [
      _escape_for_markdown_table(str(item.get(key, ''))) for key in keys
  ]
  return f"| {' | '.join(escaped_values)} |\n"


def log_item(log_path: str | None, item: dict[str, Any] | Dataclass):
  """Logs a dictionary or a dataclass. Log location depends on _LOG_TYPE."""

  if log_path is None:
    logging.warning('No directory for logging provided, skipping logging.')
    return

  logging.info('Logging item to %s', log_path)
  if dataclasses.is_dataclass(item) or isinstance(item, dict):
    item = make_serializable(item)
  else:
    raise ValueError(f'Item {item} is not a dataclass or a dictionary.')

  open_fn = open
  isdir_fn = os.path.isdir
  path_exists_fn = os.path.exists
  #

  assert isdir_fn(
      log_path
  ), f'log_path `{log_path}` must be an existing directory.'
  file_path = os.path.join(log_path, 'trajectory_log.md')
  write_header = not path_exists_fn(file_path)

  # Ensure a consistent order of keys for the table.
  keys = sorted(item.keys())

  with open_fn(file_path, 'a') as f:
    if write_header:
      header = f"| {' | '.join(keys)} |\n"
      separator = f"| {' | '.join(['---'] * len(keys))} |\n"
      f.write(header)
      f.write(separator)
    f.write(_create_markdown_table_row(item, keys))
