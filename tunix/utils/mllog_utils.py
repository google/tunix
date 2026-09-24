# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLPerf RCP Logging Utilities for Post-Training (GRPO)."""

import collections
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Optional
import jax
import numpy as np


class _FallbackMLLogger:
  """Pure-Python MLPerf MLLogger fallback when mlperf_logging is not installed."""

  def __init__(self):
    self.logger = logging.getLogger("mllog_default")
    self.logger.propagate = False
    self.logger.setLevel(logging.INFO)
    if not self.logger.handlers:
      handler = logging.StreamHandler(stream=sys.stdout)
      handler.setLevel(logging.INFO)
      self.logger.addHandler(handler)
    self.default_namespace = ""
    self.default_stack_offset = 1
    self.root_dir: Optional[str] = None

  def _log_helper(
      self,
      event_type: str,
      key: str,
      value: Any = None,
      metadata: Optional[dict[str, Any]] = None,
      namespace: Optional[str] = None,
      time_ms: Optional[int] = None,
      stack_offset: Optional[int] = None,
  ) -> None:
    ns = self.default_namespace if namespace is None else namespace
    ts = int(time.time() * 1000) if time_ms is None else int(time_ms)
    offset = self.default_stack_offset if stack_offset is None else stack_offset
    frame = inspect.currentframe()
    for _ in range(2 + offset):
      if frame is not None and frame.f_back is not None:
        frame = frame.f_back
    caller_file = frame.f_code.co_filename if frame is not None else __file__
    caller_line = frame.f_lineno if frame is not None else 0
    log_meta: dict[str, Any] = {"file": caller_file, "lineno": caller_line}
    if isinstance(metadata, dict):
      log_meta.update(metadata)
    payload = collections.OrderedDict([
        ("namespace", ns),
        ("time_ms", ts),
        ("event_type", event_type),
        ("key", key),
        ("value", value),
        ("metadata", log_meta),
    ])
    self.logger.info(":::MLLOG %s", json.dumps(payload))

  def start(self, key: str, value: Any = None, metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
    self._log_helper("INTERVAL_START", key, value=value, metadata=metadata, **kwargs)

  def end(self, key: str, value: Any = None, metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
    self._log_helper("INTERVAL_END", key, value=value, metadata=metadata, **kwargs)

  def event(self, key: str, value: Any = None, metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
    self._log_helper("POINT_IN_TIME", key, value=value, metadata=metadata, **kwargs)


class _FallbackMLLogModule:
  """Pure-Python fallback for mlperf_logging.mllog."""

  def __init__(self, logger_instance: _FallbackMLLogger):
    self._logger_instance = logger_instance

  def get_mllogger(self) -> _FallbackMLLogger:
    return self._logger_instance

  def config(self, **kwargs: Any) -> None:
    filename = kwargs.get("filename")
    if filename:
      fh = logging.FileHandler(filename)
      fh.setLevel(logging.INFO)
      self._logger_instance.logger.addHandler(fh)
    if "root_dir" in kwargs:
      self._logger_instance.root_dir = kwargs["root_dir"]


try:
  from mlperf_logging import mllog
  from mlperf_logging.mllog import constants
  mllogger = mllog.get_mllogger()
except ImportError:
  constants = None
  mllogger = _FallbackMLLogger()
  mllog = _FallbackMLLogModule(mllogger)


_gcs_target_path: Optional[str] = None
_local_log_path: Optional[str] = None


def _download_from_gcs_if_exists(gcs_path: str, local_path: str) -> bool:
  """Downloads an existing GCS mllog file to local_path so events append cleanly."""
  try:
    import fsspec  # pylint: disable=g-import-not-at-top

    fs = fsspec.filesystem("gs")
    if fs.exists(gcs_path):
      os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
      fs.get(gcs_path, local_path)
      return True
    return False
  except Exception:  # pylint: disable=broad-exception-caught
    try:
      import tensorflow as tf  # pylint: disable=g-import-not-at-top

      if tf.io.gfile.exists(gcs_path):
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        tf.io.gfile.copy(gcs_path, local_path, overwrite=True)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
      os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
      res = subprocess.run(
          ["gsutil", "cp", gcs_path, local_path],
          capture_output=True,
          text=True,
          check=False,
      )
      if res.returncode == 0 and os.path.exists(local_path):
        return True
  return False


def get_mllog_file_path(
    metric_logger_dir: Optional[str] = None,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
) -> Optional[str]:
  """Returns the target MLLOG file path (GCS URI or local path)."""
  if filename is not None:
    return filename
  if metric_logger_dir is not None:
    if metric_logger_dir.endswith(".out") or metric_logger_dir.endswith(".log"):
      return metric_logger_dir
    seed_val = seed if seed is not None else 1
    return os.path.join(metric_logger_dir.rstrip("/"), f"seed_{seed_val}.out")
  return _gcs_target_path or _local_log_path


def _parse_topology_devices(
    topology: Optional[str], rollout_replicas: int = 1
) -> Optional[int]:
  """Parses TPU slice strings like 'tpuv5p:2x2x2+tpuv5p:2x2x1' into total chip count."""
  if not topology:
    return None
  parts = str(topology).split("+")
  total = 0
  for idx, part in enumerate(parts):
    dims_str = part.split(":")[-1]
    dims = [int(x) for x in dims_str.split("x") if x.isdigit()]
    if not dims:
      continue
    chips = 1
    for d in dims:
      chips *= d
    if idx > 0:
      chips *= max(1, int(rollout_replicas))
    total += chips
  return total if total > 0 else None


def _flush_to_gcs_if_needed() -> None:
  """Copies the local mllog file to GCS if metric_logger_dir was a gs:// URI."""
  if not (_is_master_process() and _gcs_target_path and _local_log_path):
    return
  if not os.path.exists(_local_log_path):
    return
  try:
    for h in getattr(mllogger.logger, "handlers", []):
      if isinstance(h, logging.FileHandler):
        h.flush()
    try:
      import fsspec  # pylint: disable=g-import-not-at-top

      fs = fsspec.filesystem("gs")
      fs.put(_local_log_path, _gcs_target_path)
    except Exception:  # pylint: disable=broad-exception-caught
      try:
        import tensorflow as tf  # pylint: disable=g-import-not-at-top

        tf.io.gfile.makedirs(os.path.dirname(_gcs_target_path))
        tf.io.gfile.copy(_local_log_path, _gcs_target_path, overwrite=True)
      except Exception:  # pylint: disable=broad-exception-caught
        subprocess.run(
            ["gsutil", "cp", _local_log_path, _gcs_target_path],
            capture_output=True,
            text=True,
            check=True,
        )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    logging.warning(
        "Failed to copy mllog file %s to %s: %s",
        _local_log_path,
        _gcs_target_path,
        exc,
    )


def configure_logger(
    metric_logger_dir: Optional[str] = None,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
):
  """Configures mllog output file if metric_logger_dir or filename is provided."""
  global _gcs_target_path, _local_log_path
  if not (_is_master_process() and mllog is not None and mllogger is not None):
    return

  seed_val = seed if seed is not None else 1
  if filename is not None and filename.startswith("gs://"):
    _gcs_target_path = filename
    filename = os.path.join(
        "/tmp/rcp_logs", f"seed_{seed_val}_{os.getpid()}.out"
    )
  elif filename is None and metric_logger_dir is not None:
    if metric_logger_dir.startswith("gs://"):
      if metric_logger_dir.endswith(".out") or metric_logger_dir.endswith(
          ".log"
      ):
        _gcs_target_path = metric_logger_dir
      else:
        _gcs_target_path = os.path.join(
            metric_logger_dir.rstrip("/"), f"seed_{seed_val}.out"
        )
      filename = os.path.join(
          "/tmp/rcp_logs", f"seed_{seed_val}_{os.getpid()}.out"
      )
    elif metric_logger_dir.endswith(".out") or metric_logger_dir.endswith(
        ".log"
    ):
      _gcs_target_path = None
      filename = metric_logger_dir
    else:
      _gcs_target_path = None
      filename = os.path.join(metric_logger_dir, f"seed_{seed_val}.out")
  elif filename is not None:
    _gcs_target_path = None

  if filename is not None:
    abs_filename = os.path.abspath(filename)
    _local_log_path = abs_filename
    os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
    if _gcs_target_path and not os.path.exists(abs_filename):
      _download_from_gcs_if_exists(_gcs_target_path, abs_filename)
    existing_files = [
        os.path.abspath(getattr(h, "baseFilename", ""))
        for h in getattr(mllogger.logger, "handlers", [])
        if isinstance(h, logging.FileHandler)
    ]
    if abs_filename not in existing_files:
      for h in list(getattr(mllogger.logger, "handlers", [])):
        if isinstance(h, logging.FileHandler):
          mllogger.logger.removeHandler(h)
          h.close()
      try:
        mllog.config(filename=filename)
      except TypeError:
        mllog.config(filename=filename, root_dir=os.getcwd())


def _is_master_process() -> bool:
  """Returns True if this is the coordinator process (host 0)."""
  try:
    return jax.process_index() == 0
  except Exception:
    return True


def _log_event(key: str, value: Any = None, metadata: Optional[dict[str, Any]] = None):
  if not _is_master_process():
    return
  if mllogger is not None:
    mllogger.event(key=key, value=value, metadata=metadata or {})


def _log_start(key: str, metadata: Optional[dict[str, Any]] = None):
  if not _is_master_process():
    return
  if mllogger is not None:
    mllogger.start(key=key, metadata=metadata or {})


def _log_end(key: str, metadata: Optional[dict[str, Any]] = None):
  if not _is_master_process():
    return
  if mllogger is not None:
    mllogger.end(key=key, metadata=metadata or {})


def init_start(
    args: Any = None,
    metric_logger_dir: Optional[str] = None,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
):
  """Logs CACHE_CLEAR and marks the beginning of the initialization phase."""
  if _is_master_process() and mllogger is not None:
    if args is not None:
      if metric_logger_dir is None:
        metric_logger_dir = getattr(args, "metric_logger_dir", None)
      if seed is None:
        seed = getattr(args, "seed", 1)
    if metric_logger_dir is not None or filename is not None:
      configure_logger(
          metric_logger_dir=metric_logger_dir, seed=seed, filename=filename
      )
    cache_clear_key = getattr(constants, "CACHE_CLEAR", "cache_clear")
    init_start_key = getattr(constants, "INIT_START", "init_start")
    mllogger.event(key=cache_clear_key, value=True)
    mllogger.start(key=init_start_key)


def init_stop():
  """Marks the end of the initialization phase."""
  if _is_master_process() and mllogger is not None:
    init_stop_key = getattr(constants, "INIT_STOP", "init_stop")
    mllogger.end(key=init_stop_key)


def run_start():
  """Marks the start of the training run."""
  if _is_master_process() and mllogger is not None:
    run_start_key = getattr(constants, "RUN_START", "run_start")
    mllogger.start(key=run_start_key)


def block_start(args=None, step: int = 0, samples_count: Optional[int] = None):
  """Marks the start of a training block."""
  if _is_master_process() and mllogger is not None:
    if samples_count is None and args is not None:
      global_batch_size = getattr(args, "batch_size", 1) * getattr(args, "num_generations", 1)
      max_steps = getattr(args, "max_steps", None)
      eval_interval = getattr(args, "eval_every_n_steps", max_steps if max_steps is not None else 1)
      if max_steps is not None:
        eval_interval = min(int(eval_interval), max(0, int(max_steps) - int(step)))
      samples_count = int(eval_interval) * global_batch_size

    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    mllogger.start(
        key=getattr(constants, "BLOCK_START", "block_start"),
        metadata=metadata,
    )


def train_start(args=None, step: int = 0, samples_count: Optional[int] = None):
  """Marks initialization end, run start, and the first training block start."""
  init_stop()
  run_start()
  block_start(args=args, step=step, samples_count=samples_count)
  _flush_to_gcs_if_needed()


def block_stop(
    step: int = 0,
    samples_count: Optional[int] = None,
    time_ms: Optional[int] = None,
):
  """Marks the end of a training block."""
  if _is_master_process() and mllogger is not None:
    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    kwargs: dict[str, Any] = {
        "key": getattr(constants, "BLOCK_STOP", "block_stop"),
        "metadata": metadata,
    }
    if time_ms is not None:
      kwargs["time_ms"] = int(time_ms)
    mllogger.end(**kwargs)


def train_stop(
    args: Any = None,
    step: Optional[int] = None,
    samples_count: Optional[int] = None,
    status: str = "success",
    time_ms: Optional[int] = None,
):
  """Marks the end of a training block (run_stop is emitted by evaluation)."""
  del status
  if args is not None:
    if step is None:
      step = getattr(args, "max_steps", 0)
    if samples_count is None:
      global_batch_size = getattr(args, "batch_size", 1) * getattr(
          args, "num_generations", 1
      )
      samples_count = int(step) * global_batch_size

  step_val = 0 if step is None else int(step)
  block_stop(step=step_val, samples_count=samples_count, time_ms=time_ms)
  _flush_to_gcs_if_needed()


def start_eval(
    step: int = 0,
    samples_count: Optional[int] = None,
    time_ms: Optional[int] = None,
):
  """Marks the start of an evaluation interval."""
  if _is_master_process() and mllogger is not None:
    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    kwargs: dict[str, Any] = {
        "key": getattr(constants, "EVAL_START", "eval_start"),
        "metadata": metadata,
    }
    if time_ms is not None:
      kwargs["time_ms"] = int(time_ms)
    mllogger.start(**kwargs)


def end_eval(
    step: int = 0,
    accuracy: float = 0.0,
    samples_count: Optional[int] = None,
    validation_time: Optional[float] = None,
    time_ms: Optional[int] = None,
):
  """Marks the end of an evaluation interval and records eval accuracy."""
  if _is_master_process() and mllogger is not None:
    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    extra_kwargs: dict[str, Any] = {}
    if time_ms is not None:
      extra_kwargs["time_ms"] = int(time_ms)

    if validation_time is not None:
      mllogger.event(
          key="tracked_stats",
          value={"validation_time": float(validation_time)},
          metadata={"step": int(step)},
          **extra_kwargs,
      )

    eval_accuracy_metadata = {}
    if samples_count is not None:
      eval_accuracy_metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    mllogger.event(
        key=getattr(constants, "EVAL_ACCURACY", "eval_accuracy"),
        value=float(accuracy),
        metadata=eval_accuracy_metadata,
        **extra_kwargs,
    )
    mllogger.end(
        key=getattr(constants, "EVAL_STOP", "eval_stop"),
        metadata=metadata,
        **extra_kwargs,
    )


def log_offline_eval_step(
    step: int,
    samples_count: int,
    eval_accuracy: float,
    target_accuracy: float = 0.69,
    checkpoint_timestamp_ms: Optional[int] = None,
    is_last_checkpoint: bool = False,
    validation_time: Optional[float] = None,
    emit_start_eval: bool = True,
) -> bool:
  """Logs offline evaluation events for a single checkpoint and emits backdated run_stop if converged or final."""
  passed = float(eval_accuracy) >= float(target_accuracy)
  if not (_is_master_process() and mllogger is not None):
    return passed

  if emit_start_eval:
    start_eval(step=int(step), samples_count=int(samples_count))
  end_eval(
      step=int(step),
      accuracy=float(eval_accuracy),
      samples_count=int(samples_count),
      validation_time=validation_time,
  )
  if passed:
    run_stop(
        status="success",
        samples_count=int(samples_count),
        time_ms=checkpoint_timestamp_ms,
    )
    _flush_to_gcs_if_needed()
    return True
  if is_last_checkpoint:
    run_stop(
        status="aborted",
        samples_count=int(samples_count),
        time_ms=checkpoint_timestamp_ms,
    )
    _flush_to_gcs_if_needed()
    return False

  _flush_to_gcs_if_needed()
  return False


def compute_val_start_step(
    global_batch_size: int,
    val_start_at_override: Optional[int] = None,
) -> int:
  """Computes MLPerf policy validation start step: CEIL(2.5 + 3840 / global_batch_size)."""
  if val_start_at_override is not None and int(val_start_at_override) > 0:
    return int(val_start_at_override)
  if int(global_batch_size) <= 0:
    raise ValueError(
        f"global_batch_size must be positive, got {global_batch_size}"
    )
  return int(math.ceil(2.5 + 3840.0 / float(global_batch_size)))


def _read_manifest_text(manifest_path: str) -> str:
  """Reads text content of manifest_path from GCS or local filesystem if it exists."""
  if manifest_path.startswith("gs://"):
    try:
      import fsspec  # pylint: disable=g-import-not-at-top

      fs = fsspec.filesystem("gs")
      if not fs.exists(manifest_path):
        return ""
      with fs.open(manifest_path, "r", encoding="utf-8") as f:
        return f.read()
    except Exception:  # pylint: disable=broad-exception-caught
      try:
        import tensorflow as tf  # pylint: disable=g-import-not-at-top

        if not tf.io.gfile.exists(manifest_path):
          return ""
        with tf.io.gfile.GFile(manifest_path, "r") as f:
          return f.read()
      except Exception:  # pylint: disable=broad-exception-caught
        import subprocess  # pylint: disable=g-import-not-at-top

        res = subprocess.run(
            ["gsutil", "cat", manifest_path],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout if res.returncode == 0 else ""
  if not os.path.exists(manifest_path):
    return ""
  with open(manifest_path, "r", encoding="utf-8") as f:
    return f.read()


def _write_manifest_text(manifest_path: str, content: str) -> None:
  """Writes text content to manifest_path on GCS or local filesystem."""
  if manifest_path.startswith("gs://"):
    try:
      import fsspec  # pylint: disable=g-import-not-at-top

      fs = fsspec.filesystem("gs")
      parent = os.path.dirname(manifest_path.rstrip("/"))
      if parent and parent != "gs:":
        fs.makedirs(parent, exist_ok=True)
      with fs.open(manifest_path, "w", encoding="utf-8") as f:
        f.write(content)
      return
    except Exception:  # pylint: disable=broad-exception-caught
      import tensorflow as tf  # pylint: disable=g-import-not-at-top

      tf.io.gfile.makedirs(os.path.dirname(manifest_path))
      with tf.io.gfile.GFile(manifest_path, "w") as f:
        f.write(content)
      return
  parent = os.path.dirname(os.path.abspath(manifest_path))
  if parent:
    os.makedirs(parent, exist_ok=True)
  with open(manifest_path, "w", encoding="utf-8") as f:
    f.write(content)


def append_checkpoint_manifest(
    manifest_path: str,
    record: Mapping[str, Any],
) -> None:
  """Appends or updates a checkpoint manifest JSONL record (local or gs://)."""
  if not manifest_path:
    return
  existing_text = _read_manifest_text(manifest_path)
  records_by_step: dict[int, dict[str, Any]] = {}
  for line in existing_text.splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    parsed = json.loads(stripped)
    records_by_step[int(parsed["step"])] = dict(parsed)

  clean_record = dict(record)
  records_by_step[int(clean_record["step"])] = clean_record
  sorted_records = [
      records_by_step[s] for s in sorted(records_by_step.keys())
  ]
  content = "\n".join(
      json.dumps(r, ensure_ascii=False) for r in sorted_records
  ) + "\n"
  _write_manifest_text(manifest_path, content)


def check_eval(
    args,
    step: int,
    eval_accuracy: float,
    start_step: int = 0,
    target_accuracy: Optional[float] = None,
    validation_time: Optional[float] = None,
) -> bool:
  """Logs an evaluation block completion, checks for early stopping, and handles next block."""
  target_acc = target_accuracy if target_accuracy is not None else getattr(args, "target_accuracy", 0.69)
  is_early_stop = (target_acc is not None) and (eval_accuracy >= target_acc)

  if not (_is_master_process() and mllogger is not None):
    return is_early_stop

  global_batch_size = getattr(args, "batch_size", 1) * getattr(args, "num_generations", 1)
  eval_interval = getattr(args, "eval_every_n_steps", 10)
  eval_frequency_samples = eval_interval * global_batch_size
  current_samples = (step - start_step) * global_batch_size

  mllogger.end(
      key=getattr(constants, "BLOCK_STOP", "block_stop"),
      metadata={
          getattr(constants, "SAMPLES_COUNT", "samples_count"): current_samples,
          "step": int(step),
      },
  )
  mllogger.start(
      key=getattr(constants, "EVAL_START", "eval_start"),
      metadata={
          getattr(constants, "SAMPLES_COUNT", "samples_count"): current_samples,
          "step": int(step),
      },
  )
  if validation_time is not None:
    mllogger.event(
        key="tracked_stats",
        value={"validation_time": float(validation_time)},
        metadata={"step": int(step)},
    )
  mllogger.event(
      key=getattr(constants, "EVAL_ACCURACY", "eval_accuracy"),
      value=float(eval_accuracy),
      metadata={
          getattr(constants, "SAMPLES_COUNT", "samples_count"): current_samples,
      },
  )
  mllogger.end(
      key=getattr(constants, "EVAL_STOP", "eval_stop"),
      metadata={
          getattr(constants, "SAMPLES_COUNT", "samples_count"): current_samples,
          "step": int(step),
      },
  )

  if is_early_stop:
    mllogger.end(
        key=getattr(constants, "RUN_STOP", "run_stop"),
        metadata={
            "status": "success",
            getattr(constants, "SAMPLES_COUNT", "samples_count"): current_samples,
        },
    )
  else:
    mllogger.start(
        key=getattr(constants, "BLOCK_START", "block_start"),
        metadata={
            getattr(constants, "SAMPLES_COUNT", "samples_count"): eval_frequency_samples,
            "step": int(step),
        },
    )

  return is_early_stop


def log_tracked_stats(
    stats: Mapping[str, Any],
    step: int = 0,
    samples_count: Optional[int] = None,
):
  """Logs tracked training/timing metrics to the MLPerf log."""
  if _is_master_process() and mllogger is not None:
    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    clean_stats = {}
    for k, v in stats.items():
      if v is not None:
        if hasattr(v, "item"):
          clean_stats[k] = v.item()
        else:
          clean_stats[k] = v

    if clean_stats:
      mllogger.event(
          key="tracked_stats",
          value=clean_stats,
          metadata=metadata,
      )


def _clean_metric_val(v: Any, op: Optional[Callable] = None) -> Optional[Any]:
  """Converts metric value into a JSON-serializable scalar or standard numeric type."""
  if v is None:
    return None

  # Handle WeightedMetric or custom objects with .compute()
  if hasattr(v, "compute") and callable(v.compute):
    try:
      v = v.compute()
    except Exception:
      pass

  # Convert list or scalar to numpy array if possible
  try:
    arr = np.asarray(v)
  except Exception:
    arr = v

  if isinstance(arr, np.ndarray):
    # Filter out pure string arrays
    if arr.dtype.kind in {"U", "S"}:
      return None
    if arr.dtype.kind == "O":
      if arr.size > 0 and isinstance(arr.ravel()[0], (str, np.str_)):
        return None

    if op is not None and arr.size > 0:
      try:
        arr = op(arr)
      except Exception:
        pass

    if hasattr(arr, "item"):
      try:
        return arr.item()
      except Exception:
        pass
    if hasattr(arr, "tolist"):
      try:
        res = arr.tolist()
        if isinstance(res, (int, float, bool, str)):
          return res
      except Exception:
        pass

  if op is not None and not isinstance(arr, np.ndarray):
    try:
      v = op(v)
    except Exception:
      pass

  if hasattr(v, "item"):
    try:
      return v.item()
    except Exception:
      pass

  if isinstance(v, (int, float, bool, str)):
    return v
  return None


def _extract_kv_from_metrics_buffer(metrics_buffer: Any) -> dict[str, Any]:
  """Extracts key-value metric pairs from a MetricsBuffer or dictionary."""
  kv_stats = {}
  if metrics_buffer is None:
    return kv_stats

  # Case 1: Plain dict
  if isinstance(metrics_buffer, dict):
    for k, v in metrics_buffer.items():
      val = _clean_metric_val(v)
      if val is not None:
        kv_stats[k] = val
    return kv_stats

  # Case 2: tunix.perf.metrics.MetricsBuffer or objects with .metrics dict
  metrics_dict = getattr(metrics_buffer, "metrics", None)
  if isinstance(metrics_dict, dict):
    for k, val_entry in metrics_dict.items():
      if isinstance(val_entry, tuple) and len(val_entry) == 2:
        values, op = val_entry
      else:
        values, op = val_entry, None

      # Handle list of values / WeightedMetrics
      if isinstance(values, list) and values:
        is_weighted = [
            hasattr(x, "compute") or hasattr(x, "numerator") for x in values
        ]
        if any(is_weighted):
          if op is not None and getattr(op, "__name__", "") in (
              "_weighted_metric_mean",
              "global_weighted_mean",
              "mean_of_means",
          ):
            try:
              values = op(values)
              op = None
            except Exception:
              values = [
                  x.compute() if hasattr(x, "compute") else x for x in values
              ]
          else:
            values = [
                x.compute() if hasattr(x, "compute") else x for x in values
            ]

      val = _clean_metric_val(values, op=op)
      if val is not None:
        kv_stats[k] = val

  # Case 3: Objects with .losses and .additional_metrics (e.g. peft_trainer.MetricsBuffer)
  if hasattr(metrics_buffer, "loss"):
    loss_val = _clean_metric_val(metrics_buffer.loss)
    if loss_val is not None:
      kv_stats["loss"] = loss_val
  elif hasattr(metrics_buffer, "losses") and getattr(metrics_buffer, "losses"):
    loss_val = _clean_metric_val(metrics_buffer.losses, op=np.mean)
    if loss_val is not None:
      kv_stats["loss"] = loss_val

  addl_metrics = getattr(metrics_buffer, "additional_metrics", None)
  if isinstance(addl_metrics, dict):
    for k, val_entry in addl_metrics.items():
      if isinstance(val_entry, tuple) and len(val_entry) == 2:
        values, op = val_entry
      else:
        values, op = val_entry, None
      if isinstance(values, list) and values:
        values = [x.compute() if hasattr(x, "compute") else x for x in values]
      val = _clean_metric_val(values, op=op)
      if val is not None:
        kv_stats[k] = val

  # Case 4: tunix.sft.metrics_logger.MetricsLogger or StandardRLProgram
  raw_metrics = getattr(metrics_buffer, "_metrics", None)
  if raw_metrics is None and hasattr(metrics_buffer, "metrics_logger"):
    raw_metrics = getattr(metrics_buffer.metrics_logger, "_metrics", None)
  if isinstance(raw_metrics, dict):
    for prefix_dict in raw_metrics.values():
      if not isinstance(prefix_dict, dict):
        continue
      for mode_key, mode_dict in prefix_dict.items():
        if str(mode_key) != "train" or not isinstance(mode_dict, dict):
          continue
        for k, vals in mode_dict.items():
          if vals and k not in kv_stats:
            val = _clean_metric_val(vals[-1])
            if val is not None:
              kv_stats[k] = val
              if k.startswith("trainer/"):
                short_k = k[len("trainer/") :]
                if short_k not in kv_stats:
                  kv_stats[short_k] = val

  return kv_stats


def log_rcp_step_stats(
    metrics_source: Any,
    args: Any = None,
    step: int = 1,
    samples_count: Optional[int] = None,
    total_devices: Optional[int] = None,
) -> None:
  """Emits the two MLPerf RCP tracked_stats events (train + timing) matching MLCommons seed_1.out."""
  if not (_is_master_process() and mllog is not None and mllogger is not None):
    return

  stats = _extract_kv_from_metrics_buffer(metrics_source)
  if not stats:
    return

  step_num = int(step)
  gbs = None
  if args is not None:
    gbs = getattr(args, "batch_size", 1) * getattr(args, "num_generations", 1)
  if samples_count is None and gbs is not None:
    samples_count = step_num * gbs

  # 1. Train stats event: reduced_train_loss, reward, grad_norm, global_valid_toks, global_valid_seqs
  loss_val = stats.get("reduced_train_loss", stats.get("loss", stats.get("trainer/loss")))
  reward_val = stats.get(
      "reward",
      stats.get("rewards/mean", stats.get("trajectory_rewards/mean", stats.get("train_reward"))),
  )
  grad_norm_val = stats.get("grad_norm", stats.get("trainer/grad_norm"))
  valid_seqs = stats.get(
      "global_valid_seqs",
      stats.get("rollout/global_valid_seqs", stats.get("orchestrator/num_rollouts", float(gbs) if gbs else None)),
  )
  valid_toks = stats.get("global_valid_toks", stats.get("rollout/global_valid_toks"))
  if valid_toks is None and valid_seqs is not None:
    mean_toks = stats.get(
        "rollout/total_tokens_mean",
        stats.get("rollout/completion_length_mean", stats.get("generation/completions/mean_raw_length")),
    )
    if mean_toks is not None:
      valid_toks = float(mean_toks) * float(valid_seqs)

  train_tracked = {
      "reduced_train_loss": loss_val,
      "reward": reward_val,
      "grad_norm": grad_norm_val,
      "global_valid_toks": float(valid_toks) if valid_toks is not None else None,
      "global_valid_seqs": float(valid_seqs) if valid_seqs is not None else None,
  }
  log_tracked_stats(train_tracked, step=step_num, samples_count=samples_count)

  # 2. Timing stats event: train_step_time, policy_training_time, exposed_generation_time, weight_sync_time, valid_tokens_per_sec_per_gpu
  step_time = stats.get(
      "train_step_time",
      stats.get("orchestrator/step_time_sec", stats.get("perf/global_step_time", stats.get("step_time"))),
  )
  policy_time = stats.get("policy_training_time", stats.get("orchestrator/policy_training_time"))
  exposed_gen_time = stats.get("exposed_generation_time", stats.get("orchestrator/exposed_generation_time"))
  weight_sync_time = stats.get("weight_sync_time", stats.get("orchestrator/weight_sync_time"))

  if total_devices is None and args is not None:
    total_devices = _parse_topology_devices(
        getattr(args, "tpu_topology", None),
        getattr(args, "rollout_replicas", 1),
    )

  toks_per_sec_per_gpu = stats.get("valid_tokens_per_sec_per_gpu")
  if (
      toks_per_sec_per_gpu is None
      and valid_toks is not None
      and step_time is not None
      and float(step_time) > 0
      and total_devices
  ):
    toks_per_sec_per_gpu = float(valid_toks) / (float(step_time) * float(total_devices))

  timing_tracked = {
      "train_step_time": step_time,
      "policy_training_time": policy_time,
      "exposed_generation_time": exposed_gen_time,
      "weight_sync_time": weight_sync_time,
      "valid_tokens_per_sec_per_gpu": toks_per_sec_per_gpu,
  }
  log_tracked_stats(timing_tracked, step=step_num, samples_count=samples_count)
  _flush_to_gcs_if_needed()


MLPERF_TRACKED_KEYS = frozenset({
    "step_time",
    "train_reward",
    "train_solve",
    "adv_abs_mean",
    "completion_length",
    "loss",
    "grad_norm",
    "reduced_pg_loss",
    "entropy",
    "kl",
    "log_ratio_abs",
    "clipfrac",
})


def log_metrics_buffer(
    metrics_buffer: Any,
    args: Any = None,
    global_batch_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_generations: Optional[int] = None,
    step: Optional[int] = None,
    samples_count: Optional[int] = None,
    allowed_keys: Optional[Iterable[str]] = MLPERF_TRACKED_KEYS,
    rl_engine: Any = None,
):
  """Extracts KV metric pairs from a MetricsBuffer and logs them to MLPerf RCP."""
  if not (_is_master_process() and mllog is not None and mllogger is not None):
    return

  stats = _extract_kv_from_metrics_buffer(metrics_buffer)

  # If rl_engine is provided, extract actor metrics from actor_trainer or _rl_metrics_logger
  if rl_engine is not None:
    actor_trainer = getattr(rl_engine, "actor_trainer", None)
    if actor_trainer is not None:
      trainer_buf = (
          getattr(actor_trainer, "_prev_buffered_train_metrics", None)
          or getattr(actor_trainer, "_buffered_train_metrics", None)
      )
      if trainer_buf is not None:
        trainer_stats = _extract_kv_from_metrics_buffer(trainer_buf)
        for k, v in trainer_stats.items():
          if k not in stats:
            stats[k] = v

    # Fallback to _rl_metrics_logger history if any keys are still missing
    metrics_logger = getattr(rl_engine, "_rl_metrics_logger", None)
    if metrics_logger is not None and hasattr(metrics_logger, "_metrics"):
      actor_m = metrics_logger._metrics.get("actor", {}).get("train", {})
      for k, vals in actor_m.items():
        if vals and k not in stats:
          val = _clean_metric_val(vals[-1])
          if val is not None:
            stats[k] = val

  if not stats:
    return

  # Map canonical Tunix metric names to MLPerf RCP tracked keys
  key_mappings = {
      "perf/global_step_time": "step_time",
      "generation/completions/mean_length": "completion_length",
      "trajectory_rewards/mean": "train_reward",
      "rewards/mean": "train_reward",
      "advantage/abs_mean": "adv_abs_mean",
      "log_ratio/abs_mean": "log_ratio_abs",
      "pg_clipfrac": "clipfrac",
  }
  for orig_key, rcp_key in key_mappings.items():
    if rcp_key not in stats and orig_key in stats:
      stats[rcp_key] = stats[orig_key]

  # In SWE-bench, tasks are strictly binary pass/fail (1.0 or 0.0),
  # so train_solve is equivalent to train_reward.
  if "train_solve" not in stats and "train_reward" in stats:
    stats["train_solve"] = stats["train_reward"]

  # Filter out internal/framework metrics not part of MLPerf RCP tracked stats
  if allowed_keys is not None:
    allowed_set = set(allowed_keys)
    stats = {k: v for k, v in stats.items() if k in allowed_set}

  if not stats:
    return

  # Determine step number
  if step is not None:
    step_num = int(step)
  else:
    raw_step = getattr(
        metrics_buffer, "global_steps", getattr(metrics_buffer, "step", 0)
    )
    step_num = int(raw_step) + 1

  # Determine samples count
  if samples_count is None:
    gbs = global_batch_size
    if gbs is None:
      if batch_size is not None and num_generations is not None:
        gbs = batch_size * num_generations
      elif args is not None:
        gbs = getattr(args, "batch_size", 1) * getattr(
            args, "num_generations", 1
        )
    if gbs is not None:
      samples_count = step_num * gbs

  log_tracked_stats(
      stats=stats,
      step=step_num,
      samples_count=samples_count,
  )


def create_rcp_metrics_logger(
    args: Any = None,
    global_batch_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_generations: Optional[int] = None,
    allowed_keys: Optional[Iterable[str]] = MLPERF_TRACKED_KEYS,
    rl_engine: Any = None,
) -> Callable[[Any], None]:
  """Creates an external metrics logger callable compatible with with_external_metrics_logger.

  Args:
    args: Optional arguments namespace containing batch_size and num_generations.
    global_batch_size: Optional total samples per global step.
    batch_size: Optional batch size per device / prompt.
    num_generations: Optional number of generations per prompt.
    allowed_keys: Optional set of metric keys to allow in tracked_stats.
    rl_engine: Optional RL engine instance (e.g. RLCluster) to extract actor
      trainer metrics from.

  Returns:
    A callable accepting a MetricsBuffer that logs tracked stats.
  """
  effective_gbs = global_batch_size
  if effective_gbs is None:
    if batch_size is not None and num_generations is not None:
      effective_gbs = batch_size * num_generations
    elif args is not None:
      effective_gbs = getattr(args, "batch_size", 1) * getattr(
          args, "num_generations", 1
      )

  def rcp_logger(metrics_buffer: Any) -> None:
    log_metrics_buffer(
        metrics_buffer=metrics_buffer,
        args=args,
        global_batch_size=effective_gbs,
        allowed_keys=allowed_keys,
        rl_engine=rl_engine,
    )

  return rcp_logger


rcp_metrics_logger = create_rcp_metrics_logger


def run_stop(
    status: str = "success",
    samples_count: Optional[int] = None,
    time_ms: Optional[int] = None,
):
  """Marks the end of the training run."""
  if _is_master_process() and mllogger is not None:
    metadata = {"status": status}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    kwargs: dict[str, Any] = {
        "key": getattr(constants, "RUN_STOP", "run_stop"),
        "metadata": metadata,
    }
    if time_ms is not None:
      kwargs["time_ms"] = int(time_ms)
    mllogger.end(**kwargs)
    _flush_to_gcs_if_needed()


def init_print(
    args,
    train_dataset: Any = None,
    val_dataset: Any = None,
    rollout_mesh: Any = None,
    train_mesh: Any = None,
    total_devices: Optional[int] = None,
):
  """Logs initial MLPerf submission metadata and hyperparameters for compliance."""
  if not (_is_master_process() and mllogger is not None):
    return

  if getattr(args, "metric_logger_dir", None) is not None:
    configure_logger(
        metric_logger_dir=args.metric_logger_dir,
        seed=getattr(args, "seed", 1),
    )

  # Extract batch & step configs
  batch_size = getattr(args, "batch_size", None) or int(os.getenv("BATCH_SIZE", "16"))
  num_generations = getattr(args, "num_generations", None) or int(os.getenv("NUM_GENERATIONS", "16"))
  global_batch_size = batch_size * num_generations
  mini_batch_size = getattr(args, "mini_batch_size", None) or batch_size
  train_micro_batch_size = getattr(args, "train_micro_batch_size", None) or 1
  max_steps = getattr(args, "max_steps", None) or 50
  max_prompt_length = getattr(args, "max_prompt_length", None) or 4096
  max_response_length = getattr(args, "max_response_length", None) or 61440
  max_seq_len = (
      getattr(args, "max_sequence_length", None)
      or getattr(args, "max_seq_token_per_tpu", None)
      or (
          max_prompt_length + max_response_length
          if (max_prompt_length + max_response_length) >= 65536
          else 65536
      )
  )

  # Train / Eval sample counts
  train_samples = None
  if train_dataset is not None:
    try:
      train_samples = len(train_dataset) * num_generations
    except (TypeError, AttributeError):
      pass
  if train_samples is None:
    train_samples = max_steps * global_batch_size

  eval_samples = None
  if val_dataset is not None:
    try:
      eval_samples = len(val_dataset)
    except (TypeError, AttributeError):
      pass
  if eval_samples is None:
    eval_samples = getattr(args, "eval_samples", None) or 251

  # Parallelism dimensions from meshes
  train_tp = 1
  train_sp = 1
  if train_mesh is not None and hasattr(train_mesh, "shape"):
    train_tp = train_mesh.shape.get("tp", train_mesh.shape.get("tensor", 1))
    train_sp = train_mesh.shape.get("sp", 1)
  elif getattr(args, "train_mesh_tp", None) is not None:
    train_tp = args.train_mesh_tp
    train_sp = getattr(args, "train_mesh_sp", 1) or 1

  rollout_tp = 1
  if rollout_mesh is not None and hasattr(rollout_mesh, "shape"):
    rollout_tp = rollout_mesh.shape.get("tp", rollout_mesh.shape.get("tensor", 1))
  elif getattr(args, "rollout_mesh_tp", None) is not None:
    rollout_tp = args.rollout_mesh_tp

  # Submission platform description
  platform = getattr(args, "tpu_topology", None)
  if not platform:
    if total_devices:
      platform = f"{total_devices}xTPU"
    else:
      platform = "TPU-Ironwood"

  # Gradient accumulation steps
  grad_accum_steps = max(
      1,
      (mini_batch_size * num_generations) // max(1, train_micro_batch_size),
  )

  scaled_lr = 1.0e-6 * ((float(global_batch_size) / 256.0) ** 0.5)
  base_lr = (
      float(getattr(args, "learning_rate"))
      if getattr(args, "learning_rate", None) is not None
      else scaled_lr
  )
  scaled_clip_norm = 0.125 * ((256.0 / float(global_batch_size)) ** 0.5)
  grad_clip_norm = (
      float(getattr(args, "max_grad_norm"))
      if getattr(args, "max_grad_norm", None) is not None
      else scaled_clip_norm
  )
  adam_b1 = (
      float(getattr(args, "b1"))
      if getattr(args, "b1", None) is not None
      else float(os.getenv("ADAM_B1", "0.9"))
  )
  adam_b2 = (
      float(getattr(args, "b2"))
      if getattr(args, "b2", None) is not None
      else float(os.getenv("ADAM_B2", "0.999"))
  )
  weight_decay = (
      float(getattr(args, "weight_decay"))
      if getattr(args, "weight_decay", None) is not None
      else float(os.getenv("WEIGHT_DECAY", "0.0"))
  )

  # 1. Submission Metadata
  mllogger.event(
      key=getattr(constants, "SUBMISSION_BENCHMARK", "submission_benchmark"),
      value=getattr(constants, "QWEN35_397B_GRPO", "qwen35_397b_grpo"),
  )
  mllogger.event(
      key=getattr(constants, "SUBMISSION_ORG", "submission_org"),
      value="Google",
  )
  mllogger.event(
      key=getattr(constants, "SUBMISSION_DIVISION", "submission_division"),
      value=getattr(constants, "CLOSED", "closed"),
  )
  mllogger.event(
      key=getattr(constants, "SUBMISSION_STATUS", "submission_status"),
      value=getattr(constants, "CLOUD", "cloud"),
  )
  mllogger.event(
      key=getattr(constants, "SUBMISSION_PLATFORM", "submission_platform"),
      value=str(platform),
  )

  # 2. Hyperparameters & Training Configuration
  logging_configs = {
      getattr(constants, "SEED", "seed"): getattr(args, "seed", 42),
      getattr(constants, "MAX_STEPS", "max_steps"): max_steps,
      getattr(constants, "GLOBAL_BATCH_SIZE", "global_batch_size"): global_batch_size,
      getattr(constants, "MICRO_BATCH_SIZE", "micro_batch_size"): train_micro_batch_size,
      getattr(constants, "MAX_SEQUENCE_LENGTH", "max_sequence_length"): max_seq_len,
      getattr(constants, "TRAIN_SAMPLES", "train_samples"): train_samples,
      getattr(constants, "EVAL_SAMPLES", "eval_samples"): eval_samples,
      getattr(constants, "INIT_CHECKPOINT_STEP", "init_checkpoint_step"): 0,
      getattr(constants, "OPT_NAME", "opt_name"): getattr(constants, "ADAMW", "adamw"),
      getattr(constants, "OPT_BASE_LR", "opt_base_learning_rate"): base_lr,
      getattr(constants, "OPT_END_LR", "opt_end_learning_rate"): base_lr,
      getattr(constants, "OPT_ADAMW_BETA_1", "opt_adamw_beta_1"): adam_b1,
      getattr(constants, "OPT_ADAMW_BETA_2", "opt_adamw_beta_2"): adam_b2,
      getattr(constants, "OPT_ADAMW_EPSILON", "opt_adamw_epsilon"): 1e-8,
      getattr(constants, "OPT_ADAMW_WEIGHT_DECAY", "opt_adamw_weight_decay"): weight_decay,
      getattr(constants, "OPT_GRADIENT_CLIP_NORM", "opt_gradient_clip_norm"): grad_clip_norm,
      getattr(constants, "OPT_LR_WARMUP_STEPS", "opt_learning_rate_warmup_steps"): getattr(args, "warmup_steps", 0),
      getattr(constants, "OPT_LR_DECAY_STEPS", "opt_learning_rate_decay_steps"): getattr(args, "lr_decay_steps", max_steps),
      getattr(constants, "OPT_LR_DECAY_SCHEDULE", "opt_learning_rate_decay_schedule"): getattr(args, "schedule_type", "constant") or "constant",
      getattr(constants, "TENSOR_PARALLELISM", "tensor_parallelism"): train_tp,
      getattr(constants, "PIPELINE_PARALLELISM", "pipeline_parallelism"): 1,
      getattr(constants, "CONTEXT_PARALLELISM", "context_parallelism"): train_sp,
      getattr(constants, "EXPERT_PARALLELISM", "expert_parallelism"): getattr(args, "train_mesh_expert", 1) or 1,
      "lowest_numerical_precision_in_linear": getattr(args, "lowest_numerical_precision_in_linear", "bfloat16"),
      "lowest_numerical_precision_in_attn": getattr(args, "lowest_numerical_precision_in_attn", "bfloat16"),
      "lowest_numerical_precision_in_comm": getattr(args, "lowest_numerical_precision_in_comm", "bfloat16"),
      "config_filename": getattr(args, "config_filename", None) or getattr(args, "model_id", None) or "qwen35_397b_grpo",
      "generation_backend": getattr(args, "rollout_engine", "vllm"),
      "generation_tensor_parallelism": rollout_tp,
      "generation_pipeline_parallelism": 1,
      "generation_expert_parallelism": getattr(args, "rollout_mesh_expert", 1) or 1,
      getattr(
          constants,
          "GENERATION_TRAINING_ROLLOUT_TEMPERATURE",
          "generation_training_rollout_temperature",
      ): getattr(args, "temperature", 1.0),
      getattr(
          constants,
          "GENERATION_TRAINING_ROLLOUT_TOP_P",
          "generation_training_rollout_top_p",
      ): (getattr(args, "top_p", None) if getattr(args, "top_p", None) is not None else 1.0),
      getattr(
          constants,
          "GENERATION_VALIDATION_ROLLOUT_TEMPERATURE",
          "generation_validation_rollout_temperature",
      ): 0.1,
      getattr(
          constants,
          "GENERATION_VALIDATION_ROLLOUT_TOP_P",
          "generation_validation_rollout_top_p",
      ): 0.95,
      getattr(constants, "NUM_PROMPTS_PER_STEP", "num_prompts_per_step"): batch_size,
      getattr(constants, "NUM_GENERATIONS_PER_PROMPT", "num_generations_per_prompt"): num_generations,
      "truncated_importance_sampling_ratio_min": 0.999,
      "truncated_importance_sampling_ratio": 1.002,
      "truncated_importance_sampling_type": "seq-mask-tis",
      "target_accuracy": getattr(args, "target_accuracy", 0.69),
      getattr(constants, "GRADIENT_ACCUMULATION_STEPS", "gradient_accumulation_steps"): grad_accum_steps,
  }

  for key, value in logging_configs.items():
    if value is not None:
      mllogger.event(key=key, value=value)
  _flush_to_gcs_if_needed()
