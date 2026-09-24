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
import concurrent.futures
import dataclasses
import json
import os
import pathlib
import queue
import re
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
  """Makes an object serializable for JSON / DataFrame."""
  if item is None:
    return None
  if isinstance(item, dict):
    return {str(key): _make_serializable(value) for key, value in item.items()}
  elif isinstance(item, (list, tuple)):
    return [_make_serializable(x) for x in item]
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
  elif hasattr(item, 'to_dict') and callable(item.to_dict):
    try:
      return _make_serializable(item.to_dict())
    except Exception:
      pass
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
    tmp_file_path = file_path.parent / f'{file_path.name}.{time.time_ns()}.tmp'
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


def _sanitize_path_segment(val: Any, default: str = 'unknown') -> str:
  """Sanitizes a string for use as a filesystem / GCS directory segment."""
  s = str(val if val is not None else '').strip()
  if not s:
    return default
  cleaned = re.sub(r'[^a-zA-Z0-9_\-]', '_', s)
  return cleaned or default


def _extract_trajectory_steps(
    traj: Any, item: dict[str, Any]
) -> list[dict[str, Any]]:
  """Extracts per-interaction steps from a trajectory or item."""
  steps_list: list[dict[str, Any]] = []

  raw_steps = None
  if isinstance(traj, dict):
    raw_steps = traj.get('steps')
  elif hasattr(traj, 'steps'):
    raw_steps = getattr(traj, 'steps')
  elif isinstance(item, dict) and 'steps' in item:
    raw_steps = item.get('steps')

  if raw_steps:
    for idx, s in enumerate(raw_steps):
      s_dict = _make_serializable(s)
      if not isinstance(s_dict, dict):
        s_dict = {'content': s_dict}
      s_dict.setdefault('step_index', idx)
      steps_list.append(s_dict)
    return steps_list

  conv_text = None
  if isinstance(traj, dict):
    conv_text = traj.get('conversation_text')
  elif hasattr(traj, 'conversation_text'):
    conv_text = getattr(traj, 'conversation_text')
  elif isinstance(item, dict):
    conv_text = item.get('conversation_text')

  if isinstance(conv_text, list) and conv_text:
    turn_idx = 0
    i = 0
    # Skip initial system/user messages for steps (they represent problem prompt)
    while (
        i < len(conv_text)
        and conv_text[i].get('role') in ('system', 'user')
        and i < 2
    ):
      i += 1

    env_time = (
        traj.get('env_time')
        if isinstance(traj, dict)
        else getattr(traj, 'env_time', None)
    )
    step_latency = (
        env_time.get('step_latency') if isinstance(env_time, dict) else None
    )
    model_time = (
        traj.get('model_time')
        if isinstance(traj, dict)
        else getattr(traj, 'model_time', None)
    )
    model_step_latency = (
        model_time.get('step_latency') if isinstance(model_time, dict) else None
    )

    while i < len(conv_text):
      msg = conv_text[i]
      role = msg.get('role', '')
      if role == 'assistant':
        step_dict = {
            'step_index': turn_idx,
            'role': 'assistant',
            'thought_and_action': msg.get('content', ''),
        }
        if i + 1 < len(conv_text) and conv_text[i + 1].get('role') in (
            'user',
            'tool',
        ):
          i += 1
          step_dict['observation'] = conv_text[i].get('content', '')

        if isinstance(step_latency, list) and turn_idx < len(step_latency):
          step_dict['latency_sec'] = step_latency[turn_idx]
        if isinstance(model_step_latency, list) and turn_idx < len(
            model_step_latency
        ):
          step_dict['model_latency_sec'] = model_step_latency[turn_idx]

        steps_list.append(step_dict)
        turn_idx += 1
      else:
        steps_list.append({
            'step_index': turn_idx,
            'role': role,
            'content': msg.get('content', ''),
        })
        turn_idx += 1
      i += 1

  if not steps_list:
    completion = item.get('completion') or (
        traj.get('completion') if isinstance(traj, dict) else ''
    )
    if completion:
      steps_list.append({
          'step_index': 0,
          'completion': str(completion),
      })
    else:
      status = item.get('status') or (
          traj.get('status') if isinstance(traj, dict) else 'UNKNOWN'
      )
      steps_list.append({
          'step_index': 0,
          'status': str(status),
      })

  return steps_list


def log_trajectory_json(
    log_path: str | Any,
    item: dict[str, Any] | Any,
    *,
    gcs_timeout_sec: float | None = _DEFAULT_GCS_TIMEOUT_SEC,
) -> str | None:
  """Logs a trajectory and its steps as separate JSON files in a directory hierarchy.

  Directory Structure:
    <log_path>/step<global_step>/<worker_id>/<traj_id>/
        ├── metadata.json
        ├── step0.json
        ├── step1.json
        └── ...

  Args:
    log_path: Root directory or GCS URI to log to.
    item: Trajectory row dictionary or object to log.
    gcs_timeout_sec: Timeout for file write operations.

  Returns:
    Path string to the created trajectory directory, or None if failed.
  """
  if log_path is None:
    raise ValueError('No directory for logging provided.')

  if isinstance(item, list):
    res = None
    for sub_item in item:
      res = log_trajectory_json(
          log_path, sub_item, gcs_timeout_sec=gcs_timeout_sec
      )
    return res

  if not isinstance(item, dict):
    if dataclasses.is_dataclass(item):
      item_dict = dataclasses.asdict(item)
    elif hasattr(item, 'to_dict') and callable(item.to_dict):
      item_dict = item.to_dict()
    else:
      item_dict = {'content': item}
  else:
    item_dict = item

  global_step = item_dict.get('global_step')
  if global_step is None:
    global_step = item_dict.get('policy_version', 0)

  metadata = dict(item_dict.get('metadata') or {})
  traj = item_dict.get('trajectory') or item_dict.get('traj') or {}
  if not isinstance(traj, dict):
    traj = dataclasses.asdict(traj) if dataclasses.is_dataclass(traj) else {}

  worker_id_raw = (
      item_dict.get('worker_id')
      or metadata.get('worker_id')
      or f"worker{item_dict.get('group_index', 0)}"
  )
  worker_id = _sanitize_path_segment(worker_id_raw, default='worker0')

  prompt_id = str(item_dict.get('prompt_id', '0'))
  group_index = item_dict.get('group_index', 0)
  traj_id_raw = (
      item_dict.get('traj_id')
      or metadata.get('traj_id')
      or f'traj_{prompt_id}_g{group_index}'
  )
  traj_id = _sanitize_path_segment(traj_id_raw, default='traj_0_g0')

  root_path = epath.Path(log_path)
  step_dir = root_path / f'step{global_step}' / worker_id / traj_id

  steps_list = _extract_trajectory_steps(traj, item_dict)

  status = item_dict.get('status') or traj.get('status', 'UNKNOWN')
  reward = item_dict.get('reward')
  if reward is None and 'trajectory_reward' in traj:
    reward = traj.get('trajectory_reward')

  meta_summary = {
      'global_step': int(global_step),
      'consumed_policy_version': item_dict.get('consumed_policy_version', 0),
      'rollout_policy_version': item_dict.get('rollout_policy_version', 0),
      'traj_id': traj_id_raw,
      'prompt_id': prompt_id,
      'group_index': int(group_index) if group_index is not None else 0,
      'worker_id': worker_id_raw,
      'status': str(status),
      'reward': float(reward) if reward is not None else None,
      'num_steps': len(steps_list),
      'question': item_dict.get('question', ''),
      'prompt': item_dict.get('prompt', ''),
      'gold_answer': item_dict.get('gold_answer', ''),
      'completion': item_dict.get('completion', ''),
      'prompt_tokens': _make_serializable(item_dict.get('prompt_tokens')),
      'completion_tokens': _make_serializable(
          item_dict.get('completion_tokens')
      ),
      'env_time': _make_serializable(traj.get('env_time')),
      'reward_time': _make_serializable(traj.get('reward_time')),
      'model_time': _make_serializable(traj.get('model_time')),
      'metadata': _make_serializable(metadata),
  }
  meta_summary = {k: v for k, v in meta_summary.items() if v is not None}

  def _write_single_file(path: Any, text: str):
    def _write():
      if hasattr(path, 'write_text'):
        path.write_text(text, encoding='utf-8')
      else:
        with path.open('w', encoding='utf-8') as f:
          f.write(text)

    _run_with_timeout(_write, gcs_timeout_sec, f'Write({path})')

  try:
    step_dir.mkdir(parents=True, exist_ok=True)

    meta_json = json.dumps(_make_serializable(meta_summary), indent=2)
    _write_single_file(step_dir / 'metadata.json', meta_json)

    for idx, s_data in enumerate(steps_list):
      step_file = step_dir / f'step{idx}.json'
      step_json = json.dumps(_make_serializable(s_data), indent=2)
      _write_single_file(step_file, step_json)

    # Generate and write inference_metrics.json and inference_metrics.jsonl
    raw_env_time = (
        traj.get('env_time')
        if isinstance(traj, dict)
        else getattr(traj, 'env_time', {})
    )
    if not isinstance(raw_env_time, dict):
      raw_env_time = {}

    raw_reward_time = (
        traj.get('reward_time')
        if isinstance(traj, dict)
        else getattr(traj, 'reward_time', {})
    )
    if not isinstance(raw_reward_time, dict):
      raw_reward_time = {}

    raw_model_time = (
        traj.get('model_time')
        if isinstance(traj, dict)
        else getattr(traj, 'model_time', {})
    )
    if not isinstance(raw_model_time, dict):
      raw_model_time = {}

    m_step_latencies = raw_model_time.get('step_latency', [])
    m_prompt_tokens = raw_model_time.get('prompt_tokens', [])
    m_comp_tokens = raw_model_time.get('completion_tokens', [])
    m_preemptions = raw_model_time.get('num_preemptions', [])
    env_step_latencies = raw_env_time.get('step_latency', [])

    inference_step_records = []
    total_comp_tokens = 0
    for idx, s_data in enumerate(steps_list):
      lat = (
          m_step_latencies[idx]
          if (
              isinstance(m_step_latencies, list) and idx < len(m_step_latencies)
          )
          else None
      )
      if lat is None and isinstance(s_data, dict):
        lat = s_data.get('model_latency_sec') or s_data.get('latency_sec')

      p_tokens = (
          m_prompt_tokens[idx]
          if (isinstance(m_prompt_tokens, list) and idx < len(m_prompt_tokens))
          else None
      )
      if p_tokens is None and isinstance(s_data, dict):
        p_tokens = s_data.get('prompt_tokens')

      c_tokens = (
          m_comp_tokens[idx]
          if (isinstance(m_comp_tokens, list) and idx < len(m_comp_tokens))
          else None
      )
      if c_tokens is None and isinstance(s_data, dict):
        c_tokens = s_data.get('completion_tokens')
        if c_tokens is None and 'assistant_tokens' in s_data:
          asst = s_data['assistant_tokens']
          c_tokens = len(asst) if isinstance(asst, (list, np.ndarray)) else None

      if c_tokens is not None:
        total_comp_tokens += int(c_tokens)

      env_lat = (
          env_step_latencies[idx]
          if (isinstance(env_step_latencies, list) and idx < len(env_step_latencies))
          else None
      )
      if env_lat is None and isinstance(s_data, dict):
        env_lat = s_data.get('env_latency_sec') or s_data.get('env_time_sec')

      preempt = (
          m_preemptions[idx]
          if (isinstance(m_preemptions, list) and idx < len(m_preemptions))
          else None
      )
      if preempt is None and isinstance(s_data, dict):
        preempt = s_data.get('num_preemptions') or s_data.get('preemptions')

      step_total_time = None
      if lat is not None or env_lat is not None:
        step_total_time = float(lat or 0.0) + float(env_lat or 0.0)

      rec = {
          'traj_id': traj_id_raw,
          'global_step': int(global_step),
          'step_index': idx,
          'total_time': float(step_total_time) if step_total_time is not None else None,
          'total_time_sec': float(step_total_time) if step_total_time is not None else None,
          'model_time_sec': float(lat) if lat is not None else None,
          'env_time_sec': float(env_lat) if env_lat is not None else None,
          'latency_sec': float(lat) if lat is not None else None,
          'prompt_tokens': int(p_tokens) if p_tokens is not None else None,
          'completion_tokens': int(c_tokens) if c_tokens is not None else None,
          'preemptions': int(preempt) if preempt is not None else 0,
          'tokens_per_second': (
              (float(c_tokens) / float(lat))
              if (lat is not None and c_tokens is not None and float(lat) > 0)
              else None
          ),
          'tpot_ms': (
              ((float(lat) / float(c_tokens)) * 1000.0)
              if (
                  lat is not None
                  and c_tokens is not None
                  and float(c_tokens) > 0
              )
              else None
          ),
      }
      inference_step_records.append(
          {k: v for k, v in rec.items() if v is not None}
      )

    total_m_time = float(sum(m_step_latencies)) if m_step_latencies else 0.0
    if total_m_time == 0.0:
      all_lats = [
          r['latency_sec'] for r in inference_step_records if 'latency_sec' in r
      ]
      total_m_time = float(sum(all_lats)) if all_lats else 0.0

    if total_comp_tokens == 0:
      all_comps = [
          r['completion_tokens']
          for r in inference_step_records
          if 'completion_tokens' in r
      ]
      total_comp_tokens = int(sum(all_comps)) if all_comps else 0

    traj_tps = (total_comp_tokens / total_m_time) if total_m_time > 0 else 0.0
    traj_tpot_ms = (
        ((total_m_time / total_comp_tokens) * 1000.0)
        if total_comp_tokens > 0
        else 0.0
    )
    mean_step_lat = (
        (total_m_time / len(inference_step_records))
        if inference_step_records
        else 0.0
    )

    total_env_time = float(
        raw_env_time.get('reset_latency', 0.0)
        + sum(env_step_latencies if isinstance(env_step_latencies, list) else [])
        + raw_env_time.get('close_latency', 0.0)
    )
    total_reward_time = float(raw_reward_time.get('reward_latency', 0.0))
    if isinstance(raw_reward_time.get('step_latency'), list):
      total_reward_time += float(sum(raw_reward_time['step_latency']))

    traj_total_time = (
        traj.get('total_time')
        if isinstance(traj, dict)
        else getattr(traj, 'total_time', None)
    )
    if traj_total_time is not None and float(traj_total_time) > 0:
      total_time = float(traj_total_time)
    else:
      total_time = float(total_m_time + total_env_time + total_reward_time)

    total_preemptions = raw_model_time.get('total_preemptions')
    if total_preemptions is None:
      all_preempts = [
          r.get('preemptions', 0) for r in inference_step_records
      ]
      total_preemptions = int(sum(all_preempts)) if all_preempts else 0
    else:
      total_preemptions = int(total_preemptions)

    inference_summary = {
        'type': 'summary',
        'traj_id': traj_id_raw,
        'global_step': int(global_step),
        'worker_id': worker_id_raw,
        'prompt_id': prompt_id,
        'group_index': int(group_index) if group_index is not None else 0,
        'status': str(status),
        'reward': float(reward) if reward is not None else None,
        'num_steps': len(steps_list),
        'total_time': total_time,
        'total_time_sec': total_time,
        'total_model_time_sec': total_m_time,
        'total_env_time_sec': total_env_time,
        'mean_step_latency_sec': mean_step_lat,
        'total_completion_tokens': total_comp_tokens,
        'total_preemptions': total_preemptions,
        'preemptions': total_preemptions,
        'tokens_per_second': traj_tps,
        'tpot_ms': traj_tpot_ms,
    }

    inf_json = json.dumps(_make_serializable(inference_summary), indent=2)
    _write_single_file(step_dir / 'inference_metrics.json', inf_json)

    jsonl_lines = [
        json.dumps(_make_serializable(r)) for r in inference_step_records
    ]
    jsonl_lines.append(json.dumps(_make_serializable(inference_summary)))
    _write_single_file(
        step_dir / 'inference_metrics.jsonl',
        '\n'.join(jsonl_lines) + '\n',
    )

    logging.log_first_n(
        logging.INFO,
        'Logged trajectory %s (%d steps) to %s',
        1,
        traj_id,
        len(steps_list),
        step_dir,
    )
    return str(step_dir)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('Failed to write trajectory JSON to %s: %s', step_dir, e)
    return None


class AsyncTrajectoryLogger:
  """A logger that logs trajectories asynchronously in a background thread."""

  def __init__(
      self,
      log_dir: str,
      *,
      log_format: str = 'json',
      max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
      stop_timeout_sec: float = _DEFAULT_STOP_TIMEOUT_SEC,
      gcs_timeout_sec: float | None = _DEFAULT_GCS_TIMEOUT_SEC,
      num_workers: int = 8,
  ):
    self._log_dir = log_dir
    self._log_format = log_format.lower()
    self._file_suffix = str(int(time.time()))
    self._stop_timeout_sec = stop_timeout_sec
    self._gcs_timeout_sec = gcs_timeout_sec
    self._num_workers = num_workers
    self._logging_queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
    self._stopped = False
    self._stop_event = threading.Event()
    self._stop_lock = threading.RLock()
    self._staging_tempdir = (
        tempfile.TemporaryDirectory(prefix='tunix_traj_stage_')
        if (self._log_format == 'csv' and _is_gcs_path(log_dir))
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
          if self._log_format == 'json':
            if len(items) == 1:
              log_trajectory_json(
                  self._log_dir,
                  items[0],
                  gcs_timeout_sec=self._gcs_timeout_sec,
              )
            else:
              with concurrent.futures.ThreadPoolExecutor(
                  max_workers=min(len(items), self._num_workers)
              ) as executor:
                futures = [
                    executor.submit(
                        log_trajectory_json,
                        self._log_dir,
                        it,
                        gcs_timeout_sec=self._gcs_timeout_sec,
                    )
                    for it in items
                ]
                concurrent.futures.wait(futures)
          else:
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
