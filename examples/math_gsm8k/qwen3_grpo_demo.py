# Copyright 2025 Google LLC
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

"""Agentic GSM8K VTC GRPO recipe for Qwen3-1.7B.

This script contains the following components:

1. logging / runtime setup
2. argparse + recipe defaults
3. shared mesh construction
4. dataset loading
5. tokenizer / model loading
6. checkpoint + metrics + optimizer
7. rollout + RL cluster
8. GRPO trainer
9. training
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import re
import sys
import time
from typing import Any

from absl import logging as absl_logging

# Disable pathways subslice check by appending it to sys.argv before JAX/absl
# parse it.
if "--pathways_enforce_subset_devices_form_subslice=false" not in sys.argv:
  sys.argv.append("--pathways_enforce_subset_devices_form_subslice=false")

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_TPU_RPA_VERSION"] = "2"
os.environ["DISABLE_MOSAIC_ATTN"] = "1"

# ====== Logging Configuration ======
absl_logging.use_python_logging()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")
print("Logging configured at INFO level.")

import grain
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds

# Register the OSS builder before ``tfds.data_source('gsm8k')``.  Without this
# import the exact local image reports DatasetNotFoundError.
import tensorflow_datasets.text.gsm8k  # noqa: F401
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKDIR = os.getcwd()
if os.path.exists(os.path.join(WORKDIR, "tunix")):
  WORKSPACE_ROOT = WORKDIR
else:
  WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)

for root in [
    REPO_ROOT,
    WORKSPACE_ROOT,
    os.path.join(WORKSPACE_ROOT, "tunix"),
    os.path.join(WORKSPACE_ROOT, "pathways-utils"),
    os.path.join(WORKSPACE_ROOT, "r2egym"),
]:
  if root not in sys.path:
    sys.path.insert(0, root)

_DISTRIBUTED_INITIALIZED = False
try:
  import tunix  # pytype: disable=import-error  # noqa: F401
except Exception:
  pass

try:
  import r2egym  # pytype: disable=import-error  # noqa: F401
except Exception:
  pass

try:
  import pathwaysutils  # pytype: disable=import-error

  pathwaysutils.initialize()
  _DISTRIBUTED_INITIALIZED = True
except Exception:
  pass

if not _DISTRIBUTED_INITIALIZED:
  try:
    jax.distributed.initialize()
  except Exception as exc:
    print(f"jax.distributed.initialize() skipped: {exc}")

print("jax devices: ", jax.devices())

from tunix.cli.utils import model as model_utils
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.oss import utils as oss_utils
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import alignment as alignment_lib
from tunix.rl import dp_workloads
from tunix.rl import utils as rl_utils
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import utils as sft_utils

# ====== Argparse ======
arg_parser = argparse.ArgumentParser(
    description="Train Qwen3-1.7B on GSM8K with the VTC GRPO recipe."
)
arg_parser.add_argument("--batch_size", type=int, default=4)
arg_parser.add_argument("--mini_batch_size", type=int, default=2)
arg_parser.add_argument("--train_micro_batch_size", type=int, default=1)
arg_parser.add_argument(
    "--train_trajectory_micro_batch_size", type=int, default=None
)
arg_parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)
arg_parser.add_argument("--max_steps", type=int, default=200)
arg_parser.add_argument("--num_generations", type=int, default=8)
arg_parser.add_argument("--max_prompt_length", type=int, default=1024)
arg_parser.add_argument("--max_response_length", type=int, default=1024)
arg_parser.add_argument("--max_concurrency", type=int, default=None)
arg_parser.add_argument("--mesh_fsdp", type=int, default=None)
arg_parser.add_argument("--mesh_dp", type=int, default=None)
arg_parser.add_argument("--mesh_tp", type=int, default=None)
arg_parser.add_argument(
    "--rollout_vllm_hbm_utilization", type=float, default=0.6
)
arg_parser.add_argument("--rollout_vllm_max_num_seqs", type=int, default=None)
arg_parser.add_argument(
    "--rollout_vllm_max_num_batched_tokens", type=int, default=None
)
arg_parser.add_argument(
    "--wandb_project", type=str, default="zero-tim-gsm8k-v5p"
)
arg_parser.add_argument("--wandb_run_name", type=str, default="")
args, _ = arg_parser.parse_known_args()

CANON_GSM8K_L3 = os.getenv("CANON_GSM8K_L3", "") == "1"
CANON_GSM8K_TRAIN = os.getenv("CANON_GSM8K_TRAIN", "") == "1"
CANON_GSM8K_UPDATE_CANARY = (
    os.getenv("CANON_GSM8K_UPDATE_CANARY", "") == "1"
)
if CANON_GSM8K_L3 and CANON_GSM8K_TRAIN:
  raise ValueError(
      "CANON_GSM8K_L3 and CANON_GSM8K_TRAIN are mutually exclusive"
  )
if CANON_GSM8K_UPDATE_CANARY and not CANON_GSM8K_L3:
  raise ValueError("CANON_GSM8K_UPDATE_CANARY requires CANON_GSM8K_L3")
CANON_GSM8K_ACTIVE = CANON_GSM8K_L3 or CANON_GSM8K_TRAIN
_P32_WORKLOAD_NAME = os.getenv("CANON_P32_WORKLOAD", "")
if _P32_WORKLOAD_NAME and _P32_WORKLOAD_NAME != "gsm8k":
  raise ValueError(
      "GSM8K recipe cannot run a different P32 workload: "
      f"{_P32_WORKLOAD_NAME!r}"
  )
CANON_P32_WORKLOAD = _P32_WORKLOAD_NAME == "gsm8k"
P32_WORKLOAD = (
    dp_workloads.get_workload("gsm8k") if CANON_P32_WORKLOAD else None
)
if CANON_P32_WORKLOAD:
  dp_workloads.validate_environment(
      P32_WORKLOAD, require_reduction_admission=True
  )


# ====== Recipe Defaults ======
MODEL_NAME = "Qwen3-1.7B"
MODEL_ID = f"Qwen/{MODEL_NAME}"
SEED = 42

NUM_PROMPTS_PER_STEP = args.batch_size
NUM_GENERATIONS = args.num_generations
MINI_BATCH_SIZE = args.mini_batch_size
TRAIN_MICRO_BATCH_SIZE = args.train_micro_batch_size
TRAIN_TRAJECTORY_MICRO_BATCH_SIZE = args.train_trajectory_micro_batch_size
TRAJECTORY_MINI_BATCH_SIZE = (
    None
    if TRAIN_TRAJECTORY_MICRO_BATCH_SIZE is None
    else MINI_BATCH_SIZE * NUM_GENERATIONS
)
COMPUTE_LOGPS_MICRO_BATCH_SIZE = args.compute_logps_micro_batch_size

MAX_STEPS = (
    dp_workloads.requested_max_steps(P32_WORKLOAD)
    if CANON_P32_WORKLOAD
    else 1
    if CANON_GSM8K_L3
    else args.max_steps
)
if CANON_P32_WORKLOAD and args.max_steps != MAX_STEPS:
  raise ValueError(
      "P33 GSM8K --max_steps does not match CANON_P33_RUN_STAGE: "
      f"expected {MAX_STEPS}, found {args.max_steps}"
  )
NUM_EPOCHS = 1000
EVAL_EVERY_N_STEPS = 50
EVAL_BATCH_SIZE = 128
EVAL_AT_START = not CANON_GSM8K_ACTIVE
EVAL_AT_END = not CANON_GSM8K_ACTIVE

BETA = 0.0 if CANON_GSM8K_L3 else 0.04
EPSILON = 0.2
# NeMo's reference_policy_kl_type="k2" is exactly 0.5 * (logp-ref_logp)^2,
# which matches Tunix's "mse_kl" implementation.
KL_LOSS_MODE = "mse_kl"
LEARNING_RATE = 2.0e-7
WEIGHT_DECAY = 0.01
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1.0e-8
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 50
LR_DECAY_STEPS = 500

MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
MAX_TOTAL_SEQUENCE_LENGTH = 1024
KV_CACHE_SIZE = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256

TRAIN_TEMPERATURE = 1.0
TRAIN_TOP_P = 1.0
TRAIN_TOP_K = 0 if CANON_GSM8K_ACTIVE else None
EVAL_TEMPERATURE = 0.0
EVAL_TOP_P = 1.0
EVAL_TOP_K = 1
MAX_CONCURRENCY = args.max_concurrency or (
    NUM_PROMPTS_PER_STEP * NUM_GENERATIONS
)

ROLLOUT_ENGINE = os.getenv("ROLLOUT_ENGINE", "vllm")
USE_LORA = False
LORA_RANK = 64
LORA_ALPHA = 64.0
ENABLE_CHECKPOINTING = False
ENABLE_REMAT = False
ENABLE_FLASH_ATTENTION = True
MODEL_DTYPE = jnp.bfloat16

ARTIFACT_ROOT = os.getenv(
    "GSM8K_ARTIFACT_ROOT",
    os.path.join(REPO_ROOT, "artifacts", "qwen3_grpo_gsm8k_vtc"),
)
TFDS_DATA_DIR = os.path.join(ARTIFACT_ROOT, "data")
MODEL_DOWNLOAD_DIR = os.path.join(ARTIFACT_ROOT, "models")
INTERMEDIATE_CKPT_DIR = os.path.join(ARTIFACT_ROOT, "intermediate_ckpt")
CHECKPOINT_ROOT = os.path.join(
    ARTIFACT_ROOT, "checkpoints", str(int(time.time()))
)
TB_LOG_DIR = os.getenv(
    "GSM8K_LOG_DIR", os.path.join(ARTIFACT_ROOT, "logs")
)

for path in [
    TFDS_DATA_DIR,
    MODEL_DOWNLOAD_DIR,
    INTERMEDIATE_CKPT_DIR,
    TB_LOG_DIR,
]:
  os.makedirs(path, exist_ok=True)
if ENABLE_CHECKPOINTING:
  os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

show_hbm_usage = sft_utils.show_hbm_usage

VTC_PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags. Do not put anything else in the answer tags.

Problem: {}
<reasoning>
"""

_metric_call_idx = 0


# ====== Shared Mesh ======
if args.mesh_dp is not None and args.mesh_fsdp not in (None, 1):
  raise ValueError("mesh_dp and a sharding mesh_fsdp cannot be combined")
MESH_DP = args.mesh_dp or 1
MESH_FSDP = args.mesh_fsdp or 1
MESH_TP = args.mesh_tp or (
    jax.device_count() // (MESH_DP if args.mesh_dp is not None else MESH_FSDP)
)
SHARED_MESH_SHAPE = (
    (MESH_DP, MESH_TP)
    if args.mesh_dp is not None
    else (MESH_FSDP, MESH_TP)
)
SHARED_MESH_AXIS_NAMES = (
    ("dp", "tp") if args.mesh_dp is not None else ("fsdp", "tp")
)

if math.prod(SHARED_MESH_SHAPE) != jax.device_count():
  raise ValueError(
      "Shared mesh dimensions must multiply to device_count. "
      f"Got mesh={SHARED_MESH_SHAPE}, devices={jax.device_count()}."
  )

if CANON_P32_WORKLOAD:
  if args.mesh_dp != 16 or MESH_TP != 4:
    raise ValueError(
        "canonical GSM8K DP workload requires --mesh_dp=16 --mesh_tp=4"
    )
  shared_mesh = dp_workloads.create_mesh(jax.devices(), P32_WORKLOAD)
else:
  shared_device_list = jax._src.mesh_utils.create_device_mesh(
      SHARED_MESH_SHAPE, jax.devices()[: math.prod(SHARED_MESH_SHAPE)]
  )
  shared_mesh = jax.sharding.Mesh(
      shared_device_list,
      axis_names=SHARED_MESH_AXIS_NAMES,
      axis_types=(jax.sharding.AxisType.Auto,) * len(SHARED_MESH_SHAPE),
  )
print(f"shared_mesh.devices.shape={shared_mesh.devices.shape}")


# ====== Data ======
def _as_text(value: Any) -> str:
  return value if isinstance(value, str) else value.decode("utf-8")


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()


def build_prompt(question: str) -> str:
  return VTC_PROMPT_TEMPLATE.format(question)


def build_gsm8k_dataset(
    *,
    split: str,
    seed: int,
    batch_size: int,
    data_dir: str,
    shuffle: bool,
) -> grain.MapDataset:
  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )

  dataset = grain.MapDataset.source(data)
  if shuffle:
    dataset = dataset.shuffle(seed=seed)

  dataset = dataset.map(
      lambda x: {
          "prompts": build_prompt(_as_text(x["question"])),
          "question": _as_text(x["question"]),
          "answer": extract_hash_answer(_as_text(x["answer"])),
      }
  )
  return dataset.batch(batch_size)


def create_datasets() -> tuple[grain.MapDataset, grain.MapDataset]:
  train_dataset = build_gsm8k_dataset(
      split="train",
      seed=SEED,
      batch_size=NUM_PROMPTS_PER_STEP,
      data_dir=TFDS_DATA_DIR,
      shuffle=True,
  ).repeat(NUM_EPOCHS)
  eval_dataset = build_gsm8k_dataset(
      split="test",
      seed=SEED,
      batch_size=EVAL_BATCH_SIZE,
      data_dir=TFDS_DATA_DIR,
      shuffle=False,
  )
  return train_dataset, eval_dataset


def _normalize_example_value(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    flat = value.reshape(-1).tolist()
    if len(flat) == 1:
      return _normalize_example_value(flat[0])
    return [_normalize_example_value(v) for v in flat]
  if isinstance(value, np.bytes_):
    return value.tobytes().decode("utf-8")
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return value


def normalize_single_example(example: dict[str, Any]) -> dict[str, Any]:
  return {
      key: _normalize_example_value(value) for key, value in example.items()
  }


# ====== Reward + Metrics ======
def extract_boxed_answer(text: str) -> str | None:
  answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = answer_blocks[-1] if answer_blocks else text

  boxed = []
  stack = []
  for i, ch in enumerate(content):
    if ch == "{":
      stack.append(i)
    elif ch == "}":
      if not stack:
        continue
      open_idx = stack.pop()
      if content[:open_idx].endswith(r"\boxed"):
        boxed.append(content[open_idx + 1 : i].strip())
  if boxed:
    return boxed[-1]

  fallback = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
  if fallback:
    return fallback.group(1).strip()
  return None


def is_vtc_format_correct(text: str) -> bool:
  has_reasoning = text.count("</reasoning>") == 1
  has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
  reasoning_end = text.find("</reasoning>")
  answer_open = text.find("<answer>")
  answer_close = text.find("</answer>")
  return (
      has_reasoning
      and has_answer
      and reasoning_end != -1
      and answer_open != -1
      and answer_close != -1
      and reasoning_end < answer_open < answer_close
  )


def normalize_answer(text: str | None) -> str | None:
  if text is None:
    return None
  return str(text).replace(",", "").strip()


def _vtc_completion_outcome(
    completion: str, gold: Any
) -> tuple[float, bool, bool, bool]:
  format_ok = is_vtc_format_correct(completion)
  pred = normalize_answer(extract_boxed_answer(completion))
  true = normalize_answer(_normalize_example_value(gold))
  answer_ok = pred is not None and true is not None and pred == true
  extracted_ok = pred is not None

  if format_ok and answer_ok:
    score = 1.0
  elif format_ok and not answer_ok:
    score = 0.1
  elif not format_ok and answer_ok:
    score = 0.5
  else:
    score = 0.0
  return score, format_ok, answer_ok, extracted_ok


def vtc_env_reward(task, action):
  gold = task.get("answer")
  completion = action.action if hasattr(action, "action") else action
  score, _, _, _ = _vtc_completion_outcome(completion, gold)
  return score


def vtc_metric_fn(prompts, completions, rewards, advantages, answer, **kwargs):
  del prompts, completions, advantages, answer, kwargs
  global _metric_call_idx
  _metric_call_idx += 1

  rewards = np.asarray(rewards, dtype=np.float32)
  solve_all = bool(np.all(rewards > 0.1))
  solve_none = bool(np.all(np.isclose(rewards, 0.0)))
  solve_partial = (not solve_all) and (not solve_none)
  solve_ratio = float(np.mean(rewards > 0.1))
  reward_mean = float(rewards.mean())
  reward_max = float(rewards.max())

  absl_logging.info(
      "[rollout-metric] call=%d n=%d solve_ratio=%.3f reward_mean=%.3f"
      " reward_max=%.3f solve_all=%d solve_none=%d",
      _metric_call_idx,
      len(rewards),
      solve_ratio,
      reward_mean,
      reward_max,
      int(solve_all),
      int(solve_none),
  )
  return {
      "rewards/solve_all": (1 if solve_all else 0, np.mean),
      "rewards/solve_none": (1 if solve_none else 0, np.mean),
      "rewards/solve_partial": (1 if solve_partial else 0, np.mean),
      "rewards/solve_ratio": (solve_ratio, np.mean),
  }


# ====== Tokenizer / Model ======
class VTCRawTextParser:
  """Raw-text prompt parser matching NeMo's vtc_raw_text_processor style."""

  def parse(
      self,
      messages,
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    del add_generation_prompt, is_first_msg
    parts = []
    for message in messages:
      role = message.get("role")
      content = message.get("content", "")
      if role == "system" and content:
        parts.append(content)
      elif role == "user":
        parts.append(content)
      elif role == "assistant" and content:
        parts.append(content)
    return "\n".join(parts)

  def update_assistant_end_tokens(self, tokens):
    """Raw text has no extra assistant terminator; preserve tokens exactly."""
    return tokens, 0


class VTCGRPOLearner(GRPOLearner):
  """Demo-local learner that normalizes TFDS string payloads to Python str."""

  def _create_agent_env_pair(
      self, single_example, group_id: int, pair_index: int
  ):
    normalized_example = normalize_single_example(single_example)
    return super()._create_agent_env_pair(
        normalized_example, group_id=group_id, pair_index=pair_index
    )


def ensure_model_downloaded() -> None:
  if not os.path.isdir(MODEL_DOWNLOAD_DIR) or not any(
      filename.endswith(".safetensors")
      for filename in os.listdir(MODEL_DOWNLOAD_DIR)
  ):
    os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)
    if CANON_GSM8K_ACTIVE:
      # The public checkpoint must not trigger the legacy interactive
      # ``hf_pipeline`` login inside a bounded, unattended release gate.
      from huggingface_hub import snapshot_download

      snapshot_download(
          repo_id=MODEL_ID,
          local_dir=MODEL_DOWNLOAD_DIR,
          token=False,
      )
    else:
      oss_utils.hf_pipeline(MODEL_ID, MODEL_DOWNLOAD_DIR)


def maybe_apply_lora(model: nnx.Module, mesh: Mesh) -> nnx.Module:
  lora_config = {
      "module_path": (
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*down_proj|.*up_proj"
      ),
      "rank": LORA_RANK,
      "alpha": LORA_ALPHA,
  }
  return model_utils.apply_lora_to_model(
      model, mesh=mesh, lora_config=lora_config
  )


def put_model_on_device(model: nnx.Module) -> nnx.Module:
  graph_def, state = nnx.split(model)
  state = rl_utils.put_params_on_memory_kind(state, "device")
  return nnx.merge(graph_def, state)


def create_reference_and_actor(mesh: Mesh) -> tuple[nnx.Module, nnx.Module]:
  ensure_model_downloaded()

  config = qwen3_model_lib.ModelConfig.qwen3_1p7b()
  if ENABLE_REMAT:
    config.remat_config = qwen3_model_lib.RematConfig.DECODER
  else:
    config.remat_config = qwen3_model_lib.RematConfig.NONE
  if ENABLE_FLASH_ATTENTION:
    config.use_flash_attention = True
    config.flash_attention_block_size = 256
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  if CANON_P32_WORKLOAD:
    dp_workloads.configure_replicated_parameter_sharding(config)

  reference = qwen3_params_lib.create_model_from_safe_tensors(
      MODEL_DOWNLOAD_DIR, config, mesh, dtype=MODEL_DTYPE
  )
  actor_base = qwen3_params_lib.create_model_from_safe_tensors(
      MODEL_DOWNLOAD_DIR, config, mesh, dtype=jnp.float32
  )

  reference = put_model_on_device(reference)
  actor = maybe_apply_lora(actor_base, mesh) if USE_LORA else actor_base
  actor = put_model_on_device(actor)
  return reference, actor


# ====== Checkpoint + Metrics + Optimizer ======
if ENABLE_CHECKPOINTING:
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=MAX_STEPS,
      max_to_keep=1,
  )
else:
  checkpointing_options = None

wandb_config = vars(args).copy()
wandb_config.update({
    "model_id": MODEL_ID,
    "mesh_shape": SHARED_MESH_SHAPE,
    "num_steps": MAX_STEPS,
    "num_generations": NUM_GENERATIONS,
    "kl_loss_mode": KL_LOSS_MODE,
    "train_temperature": TRAIN_TEMPERATURE,
    "canonical_gsm8k_train": CANON_GSM8K_TRAIN,
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "max_response_length": MAX_RESPONSE_LENGTH,
})
metrics_logging_options = None
if not CANON_GSM8K_L3:
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=TB_LOG_DIR,
      project_name=(
          os.environ["CANON_WANDB_PROJECT"]
          if CANON_P32_WORKLOAD
          else args.wandb_project
      ),
      run_name=(
          os.environ["CANON_WANDB_RUN_NAME"]
          if CANON_P32_WORKLOAD
          else args.wandb_run_name
      ),
      flush_every_n_steps=1,
      backend_kwargs={
          "wandb": {
              "config": wandb_config,
              **(
                  {"group": os.environ["CANON_WANDB_GROUP"]}
                  if CANON_P32_WORKLOAD
                  else {}
              ),
          }
      },
  )


def create_optimizer() -> optax.GradientTransformation:
  if CANON_GSM8K_UPDATE_CANARY:
    # The production schedule intentionally starts warmup at LR=0.  A one-step
    # update canary would therefore call the optimizer without changing any
    # weight.  Use the recipe's registered peak LR as a default-off diagnostic
    # constant so the canary tests a real update rather than a no-op.
    learning_rate = optax.constant_schedule(LEARNING_RATE)
  else:
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=LR_DECAY_STEPS,
        end_value=0.0,
    )
  optimizer = optax.adamw(
      learning_rate=learning_rate,
      b1=ADAM_B1,
      b2=ADAM_B2,
      eps=ADAM_EPS,
      weight_decay=WEIGHT_DECAY,
  )
  return optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optimizer)


def _shutdown_rollout_runtime(rl_cluster) -> None:
  rollout = getattr(rl_cluster, "rollout", None)
  if rollout is not None:
    for method_name in ("close", "stop", "shutdown"):
      method = getattr(rollout, method_name, None)
      if callable(method):
        try:
          method()
        except Exception:
          absl_logging.exception(
              "Failed to %s rollout runtime during demo teardown.",
              method_name,
          )
        break
  gc.collect()
  try:
    jax.clear_caches()
  except Exception:
    absl_logging.exception("Failed to clear JAX caches during demo teardown.")


def main() -> None:
  # ====== Data ======
  train_dataset, eval_dataset = create_datasets()
  show_hbm_usage("Done with loading datasets")

  # ====== Tokenizer / Model ======
  tokenizer = AutoTokenizer.from_pretrained(
      MODEL_ID,
      token=os.getenv("HF_TOKEN"),
      trust_remote_code=True,
  )
  chat_parser = VTCRawTextParser()
  qwen_eos_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)

  reference, actor = create_reference_and_actor(shared_mesh)
  show_hbm_usage("after loading qwen_ref / qwen_actor")

  # ====== Rollout + RL cluster ======
  base_rollout_dict = {
      "max_prompt_length": MAX_PROMPT_LENGTH,
      "kv_cache_size": KV_CACHE_SIZE,
      "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
      "eos_tokens": qwen_eos_tokens,
      "return_logprobs": True,
  }
  train_rollout_dict = {
      "temperature": TRAIN_TEMPERATURE,
      "top_p": TRAIN_TOP_P,
      "top_k": TRAIN_TOP_K,
  }
  eval_rollout_dict = {
      "temperature": EVAL_TEMPERATURE,
      "top_p": EVAL_TOP_P,
      "top_k": EVAL_TOP_K,
  }

  vllm_max_num_seqs = (
      args.rollout_vllm_max_num_seqs
      if args.rollout_vllm_max_num_seqs is not None
      else NUM_PROMPTS_PER_STEP * NUM_GENERATIONS
  )
  vllm_max_batched_tokens = (
      args.rollout_vllm_max_num_batched_tokens
      if args.rollout_vllm_max_num_batched_tokens is not None
      else (vllm_max_num_seqs * KV_CACHE_SIZE) // 8
  )
  if CANON_GSM8K_ACTIVE:
    expected_gate_only = (
        "0" if CANON_GSM8K_UPDATE_CANARY or CANON_GSM8K_TRAIN else "1"
    )
    expected_update_canary = "1" if CANON_GSM8K_UPDATE_CANARY else "0"
    expected_train = "1" if CANON_GSM8K_TRAIN else "0"
    expected_grad_probe = "1" if CANON_GSM8K_L3 else "0"
    expected_min_token_bucket = "4096" if CANON_P32_WORKLOAD else "256"
    required = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_GATE_ONLY": expected_gate_only,
        "CANON_ALIGNMENT_UPDATE_CANARY": expected_update_canary,
        "CANON_ALIGNMENT_TRAIN": expected_train,
        "CANON_ENGINE_MODULE_C": "1",
        "CANON_RPA_VJP2": "1",
        "CANON_VJP2_MAX_SEQS": "1",
        "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
        "CANON_LOGPROB_M": "256",
        "MIN_TOKEN_BUCKET": expected_min_token_bucket,
        "CANON_GSM8K_GRAD_PROBE": expected_grad_probe,
    }
    wrong = {
        key: os.getenv(key)
        for key, expected in required.items()
        if os.getenv(key) != expected
    }
    if wrong:
      raise ValueError(f"canonical GSM8K environment mismatch: {wrong}")
    if CANON_GSM8K_UPDATE_CANARY and ENABLE_CHECKPOINTING:
      raise ValueError("update canary must not persist a checkpoint")
    if os.getenv("CANON_PALLAS_LOGSOFTMAX") not in ("0", "1"):
      raise ValueError("CANON_PALLAS_LOGSOFTMAX=0/1 must select the causal arm")
    CANON_P35_ENVELOPE = os.getenv("CANON_P35_ENVELOPE", "") == "1"
    if CANON_GSM8K_L3:
      if NUM_PROMPTS_PER_STEP != 1 or NUM_GENERATIONS != 2:
        raise ValueError(
            "canonical GSM8K L3 requires one prompt and two generations"
        )
      if (
          MAX_STEPS != 1
          or MAX_PROMPT_LENGTH != 256
          or MAX_RESPONSE_LENGTH != 64
      ):
        raise ValueError(
            "canonical GSM8K L3 requires one step, prompt cap 256 and "
            "response cap 64"
        )
      vllm_max_num_seqs = 2
    elif CANON_P35_ENVELOPE:
      if (
          MAX_STEPS != 1
          or MAX_PROMPT_LENGTH != 1024
          or MAX_RESPONSE_LENGTH not in (64, 256)
      ):
        raise ValueError(
            "canonical GSM8K P35 envelope requires one step, prompt cap 1024 "
            "and response cap 64 or 256"
        )
      expected_mesh = (16, 4) if CANON_P32_WORKLOAD else (1, 4)
      if SHARED_MESH_SHAPE != expected_mesh:
        raise ValueError(
            f"canonical GSM8K P35 envelope requires mesh={expected_mesh}"
        )
      if BETA != 0.04:
        raise ValueError(
            "canonical GSM8K P35 envelope requires reference beta=0.04"
        )
      if os.getenv("CANON_PALLAS_LOGSOFTMAX") != "1":
        raise ValueError(
            "canonical GSM8K P35 envelope requires canonical log-softmax"
        )
      global_num_seqs = NUM_PROMPTS_PER_STEP * NUM_GENERATIONS
      if CANON_P32_WORKLOAD:
        if global_num_seqs != P32_WORKLOAD.global_trajectories:
          raise ValueError(
              "real DP16 GSM8K training requires 256 global trajectories; "
              f"got {global_num_seqs}"
          )
        expected_num_seqs = P32_WORKLOAD.local_trajectories
      else:
        expected_num_seqs = global_num_seqs
      if vllm_max_num_seqs != expected_num_seqs:
        raise ValueError(
            "canonical GSM8K P35 envelope requires the topology-local "
            f"max_num_seqs; got {vllm_max_num_seqs} vs {expected_num_seqs}"
        )
      expected_batched_tokens = (
          P32_WORKLOAD.local_m if CANON_P32_WORKLOAD else 256
      )
      if vllm_max_batched_tokens != expected_batched_tokens:
        raise ValueError(
            "canonical GSM8K P35 envelope requires the topology-local all-M token "
            f"budget {expected_batched_tokens}"
        )
    else:
      if os.getenv("CANON_GSM8K_GRAD_PROBE", "") == "1":
        raise ValueError("real GSM8K training forbids diagnostic advantages")
      if MAX_PROMPT_LENGTH != 1024 or MAX_RESPONSE_LENGTH != 1024:
        raise ValueError(
            "real GSM8K training requires prompt/response 1024/1024"
        )
      expected_mesh = (16, 4) if CANON_P32_WORKLOAD else (1, 4)
      if SHARED_MESH_SHAPE != expected_mesh:
        raise ValueError(
            f"real GSM8K training requires mesh={expected_mesh}"
        )
      if BETA != 0.04:
        raise ValueError("real GSM8K training requires reference beta=0.04")
      if os.getenv("CANON_PALLAS_LOGSOFTMAX") != "1":
        raise ValueError("real GSM8K training requires canonical log-softmax")
      global_num_seqs = NUM_PROMPTS_PER_STEP * NUM_GENERATIONS
      if CANON_P32_WORKLOAD:
        if global_num_seqs != P32_WORKLOAD.global_trajectories:
          raise ValueError(
              "real DP16 GSM8K training requires 256 global trajectories; "
              f"got {global_num_seqs}"
          )
        expected_num_seqs = P32_WORKLOAD.local_trajectories
      else:
        expected_num_seqs = global_num_seqs
      if vllm_max_num_seqs != expected_num_seqs:
        raise ValueError(
            "real GSM8K training requires the topology-local max_num_seqs; "
            f"got {vllm_max_num_seqs} vs {expected_num_seqs}"
        )
      expected_batched_tokens = (
          P32_WORKLOAD.local_m if CANON_P32_WORKLOAD else 256
      )
      if vllm_max_batched_tokens != expected_batched_tokens:
        raise ValueError(
            "real GSM8K training requires the topology-local all-M token "
            f"budget {expected_batched_tokens}"
        )
    vllm_max_batched_tokens = (
        P32_WORKLOAD.local_m if CANON_P32_WORKLOAD else 256
    )
  vllm_rollout_dict = {
      "rollout_vllm_model_version": (
          MODEL_DOWNLOAD_DIR if CANON_GSM8K_ACTIVE else MODEL_ID
      ),
      "rollout_vllm_hbm_utilization": args.rollout_vllm_hbm_utilization,
      "rollout_vllm_server_mode": True,
      "rollout_vllm_async_scheduling": False,
      "tensor_parallel_size": SHARED_MESH_SHAPE[1],
      "data_parallel_size": SHARED_MESH_SHAPE[0],
      "rollout_vllm_max_num_seqs": vllm_max_num_seqs,
      "rollout_vllm_max_num_batched_tokens": vllm_max_batched_tokens,
      "rollout_vllm_kwargs": {
          "kv_cache_metrics": True,
          "disable_log_stats": False,
          "enable_prefix_caching": False,
          "dtype": "bfloat16",
      },
  }
  if jax.default_backend() == "tpu":
    vllm_rollout_dict["rollout_vllm_tpu_backend_type"] = "jax"

  if ROLLOUT_ENGINE == "vllm":
    train_rollout_config = base_rollout.RolloutConfig(
        **base_rollout_dict, **train_rollout_dict, **vllm_rollout_dict
    )
    eval_rollout_config = base_rollout.RolloutConfig(
        **base_rollout_dict, **eval_rollout_dict, **vllm_rollout_dict
    )
  elif ROLLOUT_ENGINE == "vanilla":
    train_rollout_config = base_rollout.RolloutConfig(
        **base_rollout_dict, **train_rollout_dict
    )
    eval_rollout_config = base_rollout.RolloutConfig(
        **base_rollout_dict, **eval_rollout_dict
    )
  else:
    raise ValueError(f"Unsupported rollout engine: {ROLLOUT_ENGINE}")

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: shared_mesh,
          rl_cluster_lib.Role.REFERENCE: shared_mesh,
          rl_cluster_lib.Role.ROLLOUT: shared_mesh,
      },
      rollout_engine=ROLLOUT_ENGINE,
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=create_optimizer(),
          eval_every_n_steps=EVAL_EVERY_N_STEPS,
          max_steps=MAX_STEPS,
          max_inflight_computations=1,
          mini_batch_size=MINI_BATCH_SIZE,
          train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
          trajectory_mini_batch_size=TRAJECTORY_MINI_BATCH_SIZE,
          train_trajectory_micro_batch_size=(
              TRAIN_TRAJECTORY_MICRO_BATCH_SIZE
          ),
          compute_logps_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE,
          data_sharding_axis=("dp",) if CANON_P32_WORKLOAD else ("fsdp",),
          optimizer_offload=CANON_P32_WORKLOAD,
          metrics_logging_options=metrics_logging_options,
          checkpoint_root_directory=(
              CHECKPOINT_ROOT if ENABLE_CHECKPOINTING else None
          ),
          checkpointing_options=checkpointing_options,
      ),
      rollout_config={
          rl_cluster_lib.Mode.TRAIN: train_rollout_config,
          rl_cluster_lib.Mode.EVAL: eval_rollout_config,
      },
  )

  canonical_grpo = (
      {
          # P32 consumes rollout logprobs directly. TIS would mask an
          # alignment failure instead of rejecting it at the four-boundary
          # gate, so it is deliberately absent from the promoted workload.
          "sampler_is": None if CANON_P32_WORKLOAD else "token",
          "sampler_is_threshold": 2.0,
          "force_compute_kl": False,
      }
      if CANON_GSM8K_ACTIVE
      else {}
  )
  grpo_config = GRPOConfig(
      num_generations=NUM_GENERATIONS,
      num_iterations=1,
      beta=BETA,
      kl_loss_mode=KL_LOSS_MODE,
      epsilon=EPSILON,
      epsilon_high=EPSILON,
      advantage_estimator="grpo",
      degenerate_group_masking=False,
      use_rollout_logps=CANON_GSM8K_ACTIVE,
      system_prompt="",
      max_response_length=MAX_RESPONSE_LENGTH,
      max_concurrency=MAX_CONCURRENCY,
      loss_agg_mode="sequence-mean-token-mean",
      **canonical_grpo,
  )
  if CANON_GSM8K_L3:
    if grpo_config.beta != 0.0 or grpo_config.force_compute_kl:
      raise RuntimeError("canonical GSM8K L3 must not execute reference KL")
    print(
        "[CANON_GSM8K_L3] reference_kl_skipped "
        "beta=0.0 force_compute_kl=0",
        flush=True,
    )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )
  show_hbm_usage("after RLCluster creation")
  if CANON_P32_WORKLOAD:
    wandb_attestation = dp_workloads.require_online_wandb_run(P32_WORKLOAD)
    print(
        f"[CANON_P33_WANDB] ONLINE_RUN_PASS {wandb_attestation}",
        flush=True,
    )
  if CANON_GSM8K_ACTIVE:
    contract = rl_cluster.rollout.canonical_engine_contract_attestation()
    marker = "CANON_GSM8K_TRAIN" if CANON_GSM8K_TRAIN else "CANON_GSM8K_L3"
    print(f"[{marker}] engine contract admitted: {contract}", flush=True)

  # ====== Trainer ======
  grpo_trainer = VTCGRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      chat_parser=chat_parser,
      metric_fns=[vtc_metric_fn],
      env_kwargs={"reward_fn": vtc_env_reward},
  )
  show_hbm_usage("after GRPOLearner creation")

  print("Shared mesh:", shared_mesh)
  print(
      "Config summary:",
      {
          "model_id": MODEL_ID,
          "mesh_shape": SHARED_MESH_SHAPE,
          "rollout_engine": ROLLOUT_ENGINE,
          "prompts_per_step": NUM_PROMPTS_PER_STEP,
          "num_generations": NUM_GENERATIONS,
          "mini_batch_size": MINI_BATCH_SIZE,
          "train_micro_batch_size": TRAIN_MICRO_BATCH_SIZE,
          "trajectory_mini_batch_size": TRAJECTORY_MINI_BATCH_SIZE,
          "train_trajectory_micro_batch_size": (
              TRAIN_TRAJECTORY_MICRO_BATCH_SIZE
          ),
          "compute_logps_micro_batch_size": COMPUTE_LOGPS_MICRO_BATCH_SIZE,
          "max_steps": MAX_STEPS,
          "max_prompt_length": MAX_PROMPT_LENGTH,
          "max_response_length": MAX_RESPONSE_LENGTH,
          "reference_beta": BETA,
          "wandb_project": args.wandb_project,
          "wandb_run_name": args.wandb_run_name,
          "max_concurrency": MAX_CONCURRENCY,
          "rollout_vllm_hbm_utilization": args.rollout_vllm_hbm_utilization,
          "rollout_vllm_max_num_seqs": vllm_max_num_seqs,
          "rollout_vllm_max_num_batched_tokens": vllm_max_batched_tokens,
      },
  )

  # ====== Training ======
  try:
    grpo_trainer.train(
        train_dataset,
        eval_dataset=None if CANON_GSM8K_ACTIVE else eval_dataset,
    )
    if CANON_GSM8K_L3:
      if CANON_GSM8K_UPDATE_CANARY:
        print(
            "[CANON_GSM8K_UPDATE] ONE_STEP_CANARY_PASS "
            "checkpoint_written=0 convergence_claim=0",
            flush=True,
        )
      else:
        print("[CANON_GSM8K_L3] FULL_GATE_PASS", flush=True)
      # Both bounded modes are ephemeral. Avoid blocking forever in vLLM
      # thread teardown after their durable report and PASS marker exist.
      os._exit(0)
    if CANON_GSM8K_TRAIN:
      print(
          f"[CANON_GSM8K_TRAIN] TRAINING_DONE max_steps={MAX_STEPS}",
          flush=True,
      )
  except alignment_lib.AlignmentGateError as exc:
    if CANON_GSM8K_L3:
      print(f"[CANON_GSM8K_L3] expected_gate_exit {exc}", flush=True)
      os._exit(3)
    if CANON_GSM8K_TRAIN:
      print(f"[CANON_GSM8K_TRAIN] alignment_gate_exit {exc}", flush=True)
    rl_cluster.close()
    raise
  except Exception:
    rl_cluster.close()
    raise
  finally:
    _shutdown_rollout_runtime(rl_cluster)


if __name__ == "__main__":
  main()
