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

import os
from typing import Any, Mapping, Optional
import jax

try:
  from mlperf_logging import mllog
  from mlperf_logging.mllog import constants
  mllogger = mllog.get_mllogger()
except ImportError:
  mllog = None
  constants = None
  mllogger = None


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


def init_start():
  """Logs CACHE_CLEAR and marks the beginning of the initialization phase."""
  if _is_master_process() and mllogger is not None:
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
