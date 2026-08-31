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

import logging
import os
from typing import Any, Callable, Iterable, Mapping, Optional
import jax
import numpy as np

try:
  from mlperf_logging import mllog
  from mlperf_logging.mllog import constants
  mllogger = mllog.get_mllogger()
except ImportError:
  mllog = None
  constants = None
  mllogger = None


def configure_logger(
    metric_logger_dir: Optional[str] = None,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
):
  """Configures mllog output file if metric_logger_dir or filename is provided."""
  if not (_is_master_process() and mllog is not None and mllogger is not None):
    return

  if filename is None and metric_logger_dir is not None:
    if metric_logger_dir.endswith(".out") or metric_logger_dir.endswith(".log"):
      filename = metric_logger_dir
    else:
      seed_val = seed if seed is not None else 1
      filename = os.path.join(metric_logger_dir, f"seed_{seed_val}.out")

  if filename is not None:
    abs_filename = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
    existing_files = [
        os.path.abspath(getattr(h, "baseFilename", ""))
        for h in getattr(mllogger.logger, "handlers", [])
        if isinstance(h, logging.FileHandler)
    ]
    if abs_filename not in existing_files:
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
      eval_interval = getattr(args, "eval_every_n_steps", getattr(args, "max_steps", 1))
      samples_count = eval_interval * global_batch_size

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


def block_stop(step: int = 0, samples_count: Optional[int] = None):
  """Marks the end of a training block."""
  if _is_master_process() and mllogger is not None:
    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    mllogger.end(
        key=getattr(constants, "BLOCK_STOP", "block_stop"),
        metadata=metadata,
    )


def train_stop(
    args: Any = None,
    step: Optional[int] = None,
    samples_count: Optional[int] = None,
    status: str = "success",
):
  """Marks the end of a training block and the training run."""
  if args is not None:
    if step is None:
      step = getattr(args, "max_steps", 0)
    if samples_count is None:
      global_batch_size = getattr(args, "batch_size", 1) * getattr(
          args, "num_generations", 1
      )
      samples_count = int(step) * global_batch_size

  step_val = 0 if step is None else int(step)
  block_stop(step=step_val, samples_count=samples_count)
  run_stop(status=status, samples_count=samples_count)


def start_eval(step: int = 0, samples_count: Optional[int] = None):
  """Marks the start of an evaluation interval."""
  if _is_master_process() and mllogger is not None:
    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    mllogger.start(
        key=getattr(constants, "EVAL_START", "eval_start"),
        metadata=metadata,
    )


def end_eval(
    step: int = 0,
    accuracy: float = 0.0,
    samples_count: Optional[int] = None,
    validation_time: Optional[float] = None,
):
  """Marks the end of an evaluation interval and records eval accuracy."""
  if _is_master_process() and mllogger is not None:
    metadata = {"step": int(step)}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    if validation_time is not None:
      mllogger.event(
          key="tracked_stats",
          value={"validation_time": float(validation_time)},
          metadata={"step": int(step)},
      )

    eval_accuracy_metadata = {}
    if samples_count is not None:
      eval_accuracy_metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    mllogger.event(
        key=getattr(constants, "EVAL_ACCURACY", "eval_accuracy"),
        value=float(accuracy),
        metadata=eval_accuracy_metadata,
    )
    mllogger.end(
        key=getattr(constants, "EVAL_STOP", "eval_stop"),
        metadata=metadata,
    )


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
    mllogger.event(
        key=getattr(constants, "TRAIN_SAMPLES", "train_samples"),
        value=current_samples,
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

  return kv_stats


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


def run_stop(status: str = "success", samples_count: Optional[int] = None):
  """Marks the end of the training run."""
  if _is_master_process() and mllogger is not None:
    metadata = {"status": status}
    if samples_count is not None:
      metadata[getattr(constants, "SAMPLES_COUNT", "samples_count")] = int(samples_count)

    mllogger.end(
        key=getattr(constants, "RUN_STOP", "run_stop"),
        metadata=metadata,
    )


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
  batch_size = getattr(args, "batch_size", 8)
  num_generations = getattr(args, "num_generations", 8)
  global_batch_size = batch_size * num_generations
  mini_batch_size = getattr(args, "mini_batch_size", batch_size)
  train_micro_batch_size = getattr(args, "train_micro_batch_size", 1)
  max_steps = getattr(args, "max_steps", 50)
  max_prompt_length = getattr(args, "max_prompt_length", 4096)
  max_response_length = getattr(args, "max_response_length", 8192)
  max_seq_len = max_prompt_length + max_response_length

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
    eval_samples = 256

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
  grad_accum_steps = max(1, batch_size // mini_batch_size)

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
      getattr(constants, "OPT_BASE_LR", "opt_base_learning_rate"): getattr(args, "learning_rate", 1e-6),
      getattr(constants, "OPT_END_LR", "opt_end_learning_rate"): getattr(args, "learning_rate", 1e-6),
      getattr(constants, "OPT_ADAMW_BETA_1", "opt_adamw_beta_1"): getattr(args, "b1", 0.9),
      getattr(constants, "OPT_ADAMW_BETA_2", "opt_adamw_beta_2"): getattr(args, "b2", 0.99),
      getattr(constants, "OPT_ADAMW_EPSILON", "opt_adamw_epsilon"): 1e-8,
      getattr(constants, "OPT_ADAMW_WEIGHT_DECAY", "opt_adamw_weight_decay"): getattr(args, "weight_decay", 0.01),
      getattr(constants, "OPT_GRADIENT_CLIP_NORM", "opt_gradient_clip_norm"): getattr(args, "max_grad_norm", 1.0),
      getattr(constants, "OPT_LR_WARMUP_STEPS", "opt_learning_rate_warmup_steps"): 0,
      getattr(constants, "OPT_LR_DECAY_STEPS", "opt_learning_rate_decay_steps"): max_steps,
      getattr(constants, "OPT_LR_DECAY_SCHEDULE", "opt_learning_rate_decay_schedule"): "constant",
      getattr(constants, "TENSOR_PARALLELISM", "tensor_parallelism"): train_tp,
      getattr(constants, "PIPELINE_PARALLELISM", "pipeline_parallelism"): 1,
      getattr(constants, "CONTEXT_PARALLELISM", "context_parallelism"): train_sp,
      getattr(constants, "EXPERT_PARALLELISM", "expert_parallelism"): 1,
      "generation_backend": getattr(args, "rollout_engine", "vllm"),
      "generation_tensor_parallelism": rollout_tp,
      "generation_pipeline_parallelism": 1,
      "generation_expert_parallelism": 1,
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
