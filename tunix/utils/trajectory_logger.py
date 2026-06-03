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

"""Logging utilities for trajectory data."""

import atexit
import dataclasses
import os
import queue
import signal
import sys
import threading
import time
import types
from typing import Any, Sequence

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
  logging.log_first_n(logging.INFO, f'Logging item to {file_path}', 1)
  write_header = not file_path.exists()

  df = pd.DataFrame(
      serialized_item if isinstance(item, list) else [serialized_item]
  )
  if str(file_path).startswith('gs://'):
    if file_path.exists():
      old_df = None
      try:
        with file_path.open('r') as f:
          old_df = pd.read_csv(f, engine='python')
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Could not read existing GCS file (possibly partial write): %s', e
        )
      if old_df is not None:
        df = pd.concat([old_df, df], ignore_index=True)

    tmp_file_path = (
        file_path.parent
        / f'{file_path.name}.{pd.Timestamp.now().nanosecond}.tmp'
    )
    try:
      with tmp_file_path.open('w') as f:
        df.to_csv(f, header=True, index=False)
      # epath.Path.replace() handles the GCS 'rename' (copy + delete)
      tmp_file_path.replace(file_path)
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Failed to finalize write to %s: %s', file_path, e)
      if tmp_file_path.exists():
        tmp_file_path.unlink()  # Cleanup
  else:
    with file_path.open('a') as f:
      df.to_csv(f, header=write_header, index=False)


def _is_main_process() -> bool:
  jax = sys.modules.get('jax')
  if jax is None:
    return True
  return jax.process_index() == 0


def _to_metadata_value(value: Any) -> Any:
  """Converts array-like values to JSON-compatible metadata."""
  if value is None:
    return None
  if isinstance(value, np.ndarray):
    if value.size == 1:
      return value.item()
    return [_to_metadata_value(item) for item in value.tolist()]
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {key: _to_metadata_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_to_metadata_value(item) for item in value]
  if hasattr(value, '__array__'):
    return _to_metadata_value(np.asarray(value))
  return value


def _sequence_item(values: Any, index: int) -> Any:
  if values is None:
    return None
  try:
    return values[index]
  except (IndexError, TypeError):
    return None


def _prompt_for_completion(
    prompts: Sequence[str | list[dict[str, str]]],
    completion_index: int,
    num_completions: int,
) -> str | list[dict[str, str]]:
  if len(prompts) == num_completions:
    return prompts[completion_index]
  if len(prompts) > 0 and num_completions % len(prompts) == 0:
    completions_per_prompt = num_completions // len(prompts)
    return prompts[completion_index // completions_per_prompt]
  return prompts[completion_index % len(prompts)]


def _messages_for_trace(
    prompt: str | list[dict[str, str]], completion: str
) -> list[dict[str, str]]:
  if isinstance(prompt, list):
    messages = [dict(message) for message in prompt]
  else:
    messages = [{'role': 'user', 'content': prompt}]
  messages.append({'role': 'assistant', 'content': completion})
  return messages


@dataclasses.dataclass(slots=True)
class CsvTrajectoryLogBackend:
  """Logs trajectory records as CSV files."""

  log_dir: str
  file_suffix: str

  def log_items(self, items: list[dict[str, Any] | Any]) -> None:
    log_item(self.log_dir, items, self.file_suffix)

  def close(self) -> None:
    pass


@dataclasses.dataclass(slots=True)
class TrackioTrajectoryLogBackend:
  """Logs rollout and trajectory records as Trackio traces."""

  project: str | None = None
  run_name: str | None = None
  trace_key: str = 'rollout/traces'
  max_traces_per_step: int = 0
  init_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  _trackio: Any = dataclasses.field(init=False, default=None)
  _run: Any = dataclasses.field(init=False, default=None)

  @classmethod
  def from_config(cls, config: Any) -> 'TrackioTrajectoryLogBackend':
    return cls(
        project=getattr(config, 'trackio_project', None),
        run_name=getattr(config, 'trackio_run_name', None),
        trace_key=getattr(config, 'trackio_trace_key', 'rollout/traces'),
        max_traces_per_step=getattr(config, 'trackio_max_traces_per_step', 0),
        init_kwargs=dict(getattr(config, 'trackio_init_kwargs', {}) or {}),
    )

  @property
  def enabled(self) -> bool:
    return bool(self.project) and self.max_traces_per_step > 0

  def _ensure_run(self) -> Any | None:
    if not self.enabled or not _is_main_process():
      return None
    if self._run is None:
      try:
        import trackio  # pylint: disable=g-import-not-at-top
      except ImportError:
        logging.warning(
            'Trackio trace logging requested, but `trackio` is not installed.'
        )
        self.max_traces_per_step = 0
        return None
      self._trackio = trackio
      self._run = trackio.init(
          project=self.project,
          name=self.run_name,
          **self.init_kwargs,
      )
    return self._run

  def log_rollouts(
      self,
      *,
      prompts: Sequence[str | list[dict[str, str]]],
      completions: Sequence[str],
      rewards: Any = None,
      advantages: Any = None,
      mode: Any = 'train',
      step: int | None = None,
      metadata: dict[str, Any] | None = None,
  ) -> None:
    """Logs rollout completions as Trackio traces."""
    if len(prompts) == 0 or len(completions) == 0:
      return
    run = self._ensure_run()
    if run is None:
      return

    traces = []
    max_traces = min(self.max_traces_per_step, len(completions))
    for sample_index in range(max_traces):
      prompt = _prompt_for_completion(prompts, sample_index, len(completions))
      trace_metadata = {
          'mode': str(mode),
          'sample_index': sample_index,
      }
      if step is not None:
        trace_metadata['step'] = int(step)
      reward = _sequence_item(rewards, sample_index)
      if reward is not None:
        trace_metadata['reward'] = _to_metadata_value(reward)
      advantage = _sequence_item(advantages, sample_index)
      if advantage is not None:
        trace_metadata['advantages'] = _to_metadata_value(advantage)
      if metadata:
        trace_metadata.update(_to_metadata_value(metadata))

      traces.append(
          self._trackio.Trace(
              messages=_messages_for_trace(prompt, completions[sample_index]),
              metadata=trace_metadata,
          )
      )

    if traces:
      run.log({self.trace_key: traces}, step=step)

  def log_messages(
      self,
      *,
      messages_list: Sequence[list[dict[str, str]]],
      mode: Any = 'train',
      step: int | None = None,
      metadata_list: Sequence[dict[str, Any]] | None = None,
      metadata: dict[str, Any] | None = None,
      trace_key: str | None = None,
  ) -> None:
    """Logs fully assembled chat message traces."""
    if len(messages_list) == 0:
      return
    run = self._ensure_run()
    if run is None:
      return

    traces = []
    max_traces = min(self.max_traces_per_step, len(messages_list))
    for sample_index in range(max_traces):
      trace_metadata = {
          'mode': str(mode),
          'sample_index': sample_index,
      }
      if step is not None:
        trace_metadata['step'] = int(step)
      if metadata_list is not None and sample_index < len(metadata_list):
        trace_metadata.update(_to_metadata_value(metadata_list[sample_index]))
      if metadata:
        trace_metadata.update(_to_metadata_value(metadata))

      traces.append(
          self._trackio.Trace(
              messages=[
                  dict(message) for message in messages_list[sample_index]
              ],
              metadata=trace_metadata,
          )
      )

    if traces:
      run.log({trace_key or self.trace_key: traces}, step=step)

  def close(self) -> None:
    if self._run is not None:
      self._run.finish()
      self._run = None


class AsyncTrajectoryLogger:
  """A logger that logs trajectories asynchronously in a background thread."""

  def __init__(
      self,
      log_dir: str | None = None,
      backends: list[Any] | None = None,
  ):
    self._file_suffix = str(int(time.time()))
    self._logging_queue = queue.Queue()
    self._stopped = False
    self._backends = list(backends or [])
    if log_dir:
      self._backends.insert(
          0, CsvTrajectoryLogBackend(log_dir, self._file_suffix)
      )
    self._logging_thread = None
    if not self._backends:
      return

    def _worker():
      while True:
        item = self._logging_queue.get()
        if item is None:  # Sentinel for stopping
          self._logging_queue.task_done()
          break

        # Batching: drain the queue to log items in groups
        items = [item]
        while not self._logging_queue.empty():
          try:
            next_item = self._logging_queue.get_nowait()
            if next_item is None:
              # Put back the sentinel so the loop terminates next time
              self._logging_queue.put(None)
              break
            items.append(next_item)
          except queue.Empty:
            break

        try:
          for backend in self._backends:
            log_items = getattr(backend, 'log_items', None)
            if log_items is not None:
              log_items(items)
        except Exception:  # pylint: disable=broad-except
          logging.exception('Failed to log trajectories.')
        finally:
          for _ in range(len(items)):
            self._logging_queue.task_done()

    self._logging_thread = threading.Thread(target=_worker, daemon=True)
    self._logging_thread.start()

    # Register cleanup
    atexit.register(self.stop)

    # Register signal handlers for robust termination
    if threading.current_thread() is threading.main_thread():
      try:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_signal)
      except ValueError:
        logging.warning('Failed to register signal handlers.')

    logging.info('Started trajectory logging thread.')

  @classmethod
  def from_config(
      cls, log_dir: str | None = None, config: Any | None = None
  ) -> 'AsyncTrajectoryLogger':
    backends = []
    if config is not None:
      trackio_backend = TrackioTrajectoryLogBackend.from_config(config)
      if trackio_backend.enabled:
        backends.append(trackio_backend)
    return cls(log_dir=log_dir, backends=backends)

  @property
  def has_backends(self) -> bool:
    return bool(self._backends)

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
    """Stops the background logging thread gracefully."""
    if self._stopped:
      return
    if self._logging_thread is None:
      self._stopped = True
      return
    logging.info('Stopping trajectory logging thread...')
    self._logging_queue.put(None)
    self._logging_queue.join()
    self._logging_thread.join(timeout=10)
    for backend in self._backends:
      close = getattr(backend, 'close', None)
      if close is not None:
        close()
    self._stopped = True
    logging.info('Stopped trajectory logging thread.')

  def log_item_async(self, item: dict[str, Any] | Any):
    """Adds an item to the logging queue to be logged asynchronously."""
    if not self._backends:
      return
    if self._stopped:
      logging.warning('Trajectory logger already stopped.')
      return
    self._logging_queue.put(item)

  def log_rollouts(
      self,
      *,
      prompts: Sequence[str | list[dict[str, str]]],
      completions: Sequence[str],
      rewards: Any = None,
      advantages: Any = None,
      mode: Any = 'train',
      step: int | None = None,
      metadata: dict[str, Any] | None = None,
  ) -> None:
    for backend in self._backends:
      log_rollouts = getattr(backend, 'log_rollouts', None)
      if log_rollouts is not None:
        log_rollouts(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            advantages=advantages,
            mode=mode,
            step=step,
            metadata=metadata,
        )

  def log_messages(
      self,
      *,
      messages_list: Sequence[list[dict[str, str]]],
      mode: Any = 'train',
      step: int | None = None,
      metadata_list: Sequence[dict[str, Any]] | None = None,
      metadata: dict[str, Any] | None = None,
      trace_key: str | None = None,
  ) -> None:
    for backend in self._backends:
      log_messages = getattr(backend, 'log_messages', None)
      if log_messages is not None:
        log_messages(
            messages_list=messages_list,
            mode=mode,
            step=step,
            metadata_list=metadata_list,
            metadata=metadata,
            trace_key=trace_key,
        )
