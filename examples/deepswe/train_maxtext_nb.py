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

# %%
# [WIP] Reproduction of [DeepSWE](https://www.together.ai/blog/deepswe)
# with Multi-turn Agentic framework on MaxText models.

# %%
import argparse
import faulthandler
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

os.environ["VLLM_TPU_RPA_VERSION"] = "2"
os.environ["DISABLE_MOSAIC_ATTN"] = "1"
import signal
import sys

from absl import logging as absl_logging
import datasets as datasets_lib
from flax import nnx
import grain
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kubernetes import client, config as k8s_config
from maxtext.integration.vllm import maxtext_vllm_adapter
import numpy as np
import optax
from orbax import checkpoint as ocp
from pydantic import ValidationError
import qwix
from transformers import AutoTokenizer
from tunix.cli.utils import data as data_lib
from tunix.rl.agentic.agents import agent_types
from tunix.utils import compat
import vllm  # pytype: disable=import-error

# Register MaxText vLLM adapter
maxtext_vllm_adapter.register()
logging.info("Successfully registered MaxTextForCausalLM model with vLLM.")

# Disable MRoPE in vLLM to prevent slow 3D position ID generation for text-only sampling
try:
  from vllm.config import ModelConfig
  ModelConfig.uses_mrope = property(lambda self: False)
  logging.info("Successfully patched vLLM ModelConfig.uses_mrope to False for sampling.")
except Exception as e:
  logging.warning("Could not patch vLLM ModelConfig.uses_mrope: %s", e)

# Patch tpu_inference _reconstruct_routed_experts to handle evicted/finished requests gracefully
try:
  import tpu_inference.runner.tpu_runner as tr
  _orig_reconstruct = tr._reconstruct_routed_experts

  def _safe_reconstruct_routed_experts(
      runner,
      scheduler_output,
      expert_indices_cpu,
      req_ids,
      req_ids_dp,
      padded_num_scheduled_tokens_per_dp_rank,
  ):
    # Wrap runner.requests with a defaultdict-like fallback if any req_id is missing
    orig_requests = runner.requests
    class _SafeRequestsDict(dict):
      def __getitem__(self, key):
        if key in self:
          return super().__getitem__(key)
        # Return a dummy CachedRequestState with empty block_ids so slots map to 0 without KeyError
        from tpu_inference.runner.input_batch import CachedRequestState
        return CachedRequestState(
            req_id=key,
            prompt_token_ids=[],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(),
            num_computed_tokens=0,
            lora_request=None,
        )
    safe_requests = _SafeRequestsDict(orig_requests)
    runner.requests = safe_requests
    try:
      return _orig_reconstruct(
          runner,
          scheduler_output,
          expert_indices_cpu,
          req_ids,
          req_ids_dp,
          padded_num_scheduled_tokens_per_dp_rank,
      )
    finally:
      runner.requests = orig_requests

  tr._reconstruct_routed_experts = _safe_reconstruct_routed_experts
  logging.info("Successfully monkeypatched tpu_runner._reconstruct_routed_experts with SafeRequestsDict.")
except Exception as e:
  logging.warning("Could not patch tpu_runner._reconstruct_routed_experts: %s", e)

faulthandler.register(signal.SIGINT, all_threads=True)

Dataset = datasets_lib.Dataset


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ==========================================
# 0. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(
    description="DeepSWE Training with Multi-turn Agentic Framework on MaxText"
)
parser.add_argument("--scan_layers", type=str2bool, default=True)

# General Config
parser.add_argument(
    "--model_source",
    type=str,
    default="maxtext",
    choices=["maxtext"],
)
parser.add_argument(
    "--model_absolute_path",
    type=str,
    default="gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_version", type=str, default="Qwen3.5-35B-A3B")
parser.add_argument("--node_selector_val", type=str, default="deepswe-cpu-pool")
parser.add_argument("--dataset_path", type=str, default=None)

parser.add_argument("--tpu_topology", type=str, default=None)


# Data & Training Flow
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--mini_batch_size", type=int, default=8)
parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=50)
parser.add_argument("--eval_every_n_steps", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--enable_remat", type=bool, default=True)
parser.add_argument(
    "--remat_policy",
    type=str,
    default="decoder",
    choices=["none", "block", "decoder", "minimal", "full", "qwen_full"],
    help="Remat policy: 'none', 'block'/'minimal', 'decoder'/'full'.",
)
parser.add_argument(
    "--prefuse_moe_weights",
    type=str2bool,
    default=False,
    help="Whether to prefuse MoE weights in MaxText.",
)

# LoRA
# LoRA Config
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--alpha", type=float, default=64.0)
parser.add_argument("--train_with_lora", type=str2bool, default=False)

# GRPO Config
parser.add_argument("--num_generations", type=int, default=8)
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--beta", "--grpo_beta", dest="beta", type=float, default=0.0)
parser.add_argument("--epsilon", "--grpo_epsilon", dest="epsilon", type=float, default=0.2)
parser.add_argument("--epsilon_high", type=float, default=0.28)
parser.add_argument("--off_policy_steps", type=int, default=0)
parser.add_argument("--overlong_loss_masking", type=str2bool, default=False)
parser.add_argument("--seq_logprob_error_threshold", type=float, default=None)
parser.add_argument("--truncated_importance_sampling_type", type=str, default=None)
parser.add_argument("--truncated_importance_sampling_ratio_min", type=float, default=None)
parser.add_argument("--truncated_importance_sampling_ratio", type=float, default=None)
parser.add_argument(
    "--force_on_policy_ratio",
    type=str2bool,
    default=False,
    help=(
        "Pin the surrogate ratio to 1.0 (old_logp := stop_gradient(current_logp)). "
        "Valid for single-iteration on-policy training only."
    ),
)

# Rollout Config
parser.add_argument("--max_prompt_length", type=int, default=4096)
parser.add_argument("--max_response_length", type=int, default=8192)
parser.add_argument("--temperature", "--decode_sampling_temperature", dest="temperature", type=float, default=1.0)
parser.add_argument("--top_p", "--decode_sampling_nucleus_p", dest="top_p", type=float, default=None)
parser.add_argument("--top_k", "--decode_sampling_top_k", dest="top_k", type=int, default=None)
parser.add_argument("--rollout_engine", type=str, default="vllm")
parser.add_argument("--vllm_utilization", type=float, default=0.4)
parser.add_argument(
    "--vllm_reshard_chunk_size",
    type=int,
    default=None,
    help="Number of flat keys to reshard at a time. None for single-call.",
)
parser.add_argument(
    "--max_num_batched_tokens",
    type=int,
    default=8192,
    help="Max number of tokens to be processed in parallel by vLLM.",
)
parser.add_argument("--enable_prefix_caching", type=str2bool, default=False)

# Optimizer Config
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--b1", "--adam_b1", dest="b1", type=float, default=0.9)
parser.add_argument("--b2", "--adam_b2", dest="b2", type=float, default=0.99)
parser.add_argument("--weight_decay", "--adam_weight_decay", dest="weight_decay", type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=1)
parser.add_argument("--warmup_steps_fraction", type=float, default=0.0)
parser.add_argument("--learning_rate_final_fraction", type=float, default=1.0)
parser.add_argument("--float32_gate_logits", type=str2bool, default=False)
# Runs the [*, emb] x [emb, vocab] unembedding matmul in fp32 instead of bf16
# (decoders.py:832-846). Applied to BOTH the trainer and the rollout, because
# the quantity the TIS band gates on is the *difference* between the two, and
# an independent bf16 rounding on each side is a symmetric noise source whose
# Jensen term shows up as a negative bias in seq_geomean.
parser.add_argument("--logits_dot_in_fp32", type=str2bool, default=False)
# Recipe requires thinking OFF (grpo_qwen35_397b_swe_openhands_async.yaml:434).
# `BaseChatTemplateParser.__init__` defaults it to True and this entrypoint was
# not passing it, so every run so far has had chain-of-thought on inside all 30
# assistant turns.
parser.add_argument("--enable_thinking", type=str2bool, default=False)
# Stop STRINGS handed to vLLM. The reference sets `stop_strings: null`
# (grpo_qwen35_397b_swe_openhands_async.yaml:286-287). Ours terminate the turn
# on "</function>", strictly before the model emits <|im_end|>, so the recorded
# token_ids carry no terminator -- while the sampler's own next-turn prompt
# always does, because `_parse_assistant` appends eot unconditionally. Pass an
# empty string to match the reference and rely on stop_token_ids alone.
parser.add_argument(
    "--rollout_stop_strings",
    type=str,
    default="</function>,<|im_end|>,<|endoftext|>",
    help="Comma-separated vLLM stop strings; empty string disables them.",
)
parser.add_argument("--trainable_parameters_mask", type=str, default=None)
parser.add_argument(
    "--optimizer_offload",
    type=bool,
    default=False,
    help="Whether to offload optimizer states to CPU (pinned host memory).",
)  # not supported yet


# Checkpointing
parser.add_argument("--ckpt_dir", type=str, default="/tmp/cp/deepswe_ckpt/01")
parser.add_argument("--max_to_keep", type=int, default=4)
parser.add_argument("--save_interval_steps", type=int, default=500)
parser.add_argument("--checkpoint_storage_concurrent_gb", type=int, default=96)
parser.add_argument(
    "--checkpoint_storage_use_ocdbt", type=str2bool, default=True
)
parser.add_argument(
    "--checkpoint_storage_use_zarr3", type=str2bool, default=False
)


# Microbatch Sizes
parser.add_argument("--train_micro_batch_size", type=int, default=1)
parser.add_argument("--rollout_micro_batch_size", type=int, default=1)
parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)

# DeepSWE Agentic Specifics
parser.add_argument("--max_turns", type=int, default=50)
parser.add_argument("--per_turn_timeout_secs", type=int, default=300)
parser.add_argument("--episode_timeout_secs", type=int, default=3 * 60 * 60)
parser.add_argument("--step_timeout_secs", type=int, default=30 * 60)
parser.add_argument("--reward_timeout_secs", type=int, default=30 * 60)
parser.add_argument("--max_concurrency", type=int, default=200)
parser.add_argument(
    "--max_warmpool_replicas",
    "--max_warmpool_size",
    dest="max_warmpool_replicas",
    type=int,
    default=None,
    help="Max warmpool replicas per task/image. Defaults to num_generations.",
)
parser.add_argument(
    "--use_agent_sandbox",
    action="store_true",
    help=(
        "Whether to use Kubernetes Agent Sandbox runtime instead of local"
        " Docker socket."
    ),
)

parser.add_argument(
    "--overlong_filter",
    type=bool,
    default=True,
    help="Whether to filter out trajectories that exceed length limits",
)

# Mesh / Topology Config Override
parser.add_argument(
    "--rollout_mesh_fsdp",
    type=int,
    default=None,
    help="Optional override for rollout mesh FSDP dimension.",
)
parser.add_argument(
    "--rollout_mesh_tp",
    type=int,
    default=None,
    help="Optional override for rollout mesh TP dimension.",
)
parser.add_argument(
    "--train_mesh_fsdp",
    type=int,
    default=None,
    help="Optional override for train mesh FSDP dimension.",
)
parser.add_argument(
    "--train_mesh_tp",
    type=int,
    default=None,
    help="Optional override for train mesh TP dimension.",
)
parser.add_argument(
    "--train_mesh_sp",
    type=int,
    default=None,
    help="Optional override for train mesh SP dimension.",
)

parser.add_argument(
    "--rollout_split_fraction",
    type=float,
    default=0.5,
    help=(
        "Fraction of total devices to allocate to the rollout mesh. Default is"
        " 0.5 (1:1 ratio)."
    ),
)


VALID_STATUS_NAMES = [status.name for status in agent_types.TrajectoryStatus]

parser.add_argument(
    "--filter_statuses",
    type=str,
    nargs="+",
    default=None,  # Set default to None
    choices=VALID_STATUS_NAMES,
    help=(
        "List of trajectory statuses to filter out. Valid statuses:"
        f" {VALID_STATUS_NAMES}. Defaults to None."
    ),
)

parser.add_argument(
    "--loss_agg_mode", type=str, default="sequence-mean-token-scale"
)
parser.add_argument(
    "--sampler_is",
    type=str,
    default=None,
    help="Truncated importance sampling mode for GRPO (e.g. 'token' or None)",
)
parser.add_argument("--advantage_estimator", type=str, default="rloo")
parser.add_argument(
    "--use_rollout_logps",
    type=str2bool,
    default=False,
    help=(
        "Whether to use rollout-cached logprobs as old policy logps. "
        "Default is False to recompute old logps on the actor side. "
    ),
)
parser.add_argument(
    "--filter_repo",
    type=str,
    default=None,
    help="Filter dataset to a single repository name (e.g. 'pandas', 'numpy').",
)
parser.add_argument(
    "--max_examples",
    type=int,
    default=None,
    help="Optionally limit total examples loaded from dataset.",
)
parser.add_argument(
    "--docker_image_prefix",
    type=str,
    default=None,
    help=(
        "Optional prefix/registry to replace source image repo with (e.g."
        " us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix)."
    ),
)
parser.add_argument(
    "--filter_available_images_only",
    type=str2bool,
    default=False,
    help=(
        "Filter dataset items to only those whose docker images already exist"
        " in Artifact Registry."
    ),
)


# Profiler Config
parser.add_argument(
    "--enable_jax_profiler",
    type=str2bool,
    default=False,
    help="Whether to start JAX profiler server for live/on-demand XProf collection.",
)
parser.add_argument(
    "--jax_profiler_port",
    type=int,
    default=9999,
    help="Port to start JAX profiler server on.",
)

# Other
parser.add_argument("--do_mem_profiling", type=bool, default=False)

parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["bfloat16", "float16", "float32"],  # Restrict to valid inputs
    help="Data type for the model activations(e.g., bfloat16, float32)",
)
parser.add_argument(
    "--param_dtype",
    type=str,
    default="float32",
    choices=["bfloat16", "float16", "float32"],  # Restrict to valid inputs
    help="Data type for the model weights (e.g., bfloat16, float32)",
)


parser.add_argument("--use_flash_attention", type=str2bool, default=True)
parser.add_argument("--flash_attention_block_size", type=int, default=1024)
parser.add_argument("--metric_logger_dir", type=str, default=None)
parser.add_argument(
    "--logging_level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Logging level for the script and relevant libraries.",
)

args, _ = parser.parse_known_args()

MODEL_VERSION = args.model_version
NODE_SELECTOR_VAL = args.node_selector_val


from examples.deepswe import r2e_gym_helper

r2e_gym_helper.patch_kubernetes_runtime()


# ====== Logging Configuration ======
# 1. Force absl to use python logging
absl_logging.use_python_logging()

# 2. Configure the root logger
log_level = getattr(logging, args.logging_level.upper())
logging.basicConfig(
    stream=sys.stdout,
    level=log_level,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# 3. Explicitly set levels for relevant loggers
logging.getLogger().setLevel(log_level)
logging.getLogger("absl").setLevel(log_level)

# 4. Set absl verbosity so they actually print
absl_logging.set_verbosity(getattr(absl_logging, args.logging_level.upper()))
absl_logging.set_stderrthreshold(args.logging_level.lower())

# %%
# ==========================================
# 1. Path Setup
# ==========================================

# Use the current working directory as ROOT folder
workdir = os.getcwd()
tunix_root = os.path.join(workdir, "tunix")
pathways_root = os.path.join(workdir, "pathways-utils")
r2egym_root = os.path.join(workdir, "r2egym")

for root in [workdir, pathways_root, r2egym_root]:
  if root not in sys.path:
    sys.path.insert(0, root)

# Verification
try:
  import tunix
  import pathwaysutils
  import r2egym  # pytype: disable=import-error

  print("✅ tunix pathways-utils, r2egym are successfully mapped.")
except ImportError as e:
  print(f"❌ Still missing a module: {e}")

if pathwaysutils is not None and os.getenv("JAX_PLATFORMS", None) == "proxy":  # pyrefly: ignore[unbound-name]
  pathwaysutils.initialize()

if args.enable_jax_profiler:
  logging.info(
      "Starting JAX profiler server on port %d...", args.jax_profiler_port
  )
  try:
    _orig_thread = threading.Thread

    def _daemon_thread(*t_args, **t_kwargs):
      t_kwargs["daemon"] = True
      return _orig_thread(*t_args, **t_kwargs)

    threading.Thread = _daemon_thread
    try:
      jax.profiler.start_server(args.jax_profiler_port)
    finally:
      threading.Thread = _orig_thread

    logging.info(
        "JAX profiler server successfully started on port %d",
        args.jax_profiler_port,
    )
  except Exception as e:
    logging.warning(
        "Failed to start JAX profiler server on port %d: %s",
        args.jax_profiler_port,
        e,
    )


# %%
# ==========================================
# 2. Imports from Custom Modules
# ==========================================
from tunix.models.qwen3 import params as params_lib
from tunix.models.qwen3 import model as model_lib
from tunix.sft import utils as sft_utils
from tunix.sft import metrics_logger
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.parser.chat_template_parser import parser as template_parser
from tunix import PerfMetricsConfig
from tunix.perf.experimental.export import PerfMetricsExport
from tunix.rl.agentic.rewards.reward_types import RewardOutput
from examples.deepswe import swe_agent
from examples.deepswe import swe_env

# %%
# ==========================================
# 3. Environment Configuration
# ==========================================
DATASET_CACHE = os.getenv(
    "DATASET_CACHE", os.path.join(workdir, "dataset_cache")
)
os.makedirs(DATASET_CACHE, exist_ok=True)

os.environ["KUBECONFIG"] = "~/.kube/config"
os.environ["NODE_SELECTOR_KEY"] = "cloud.google.com/gke-nodepool"
os.environ["NODE_SELECTOR_VAL"] = (
    NODE_SELECTOR_VAL  # NB: change based on your node pool name
)
print(
    "Using Kubernetes node selector:"
    f" {os.environ['NODE_SELECTOR_KEY']}={os.environ['NODE_SELECTOR_VAL']}"
)


# Kubernetes Setup
try:
  k8s_config.load_kube_config()
  k8s_client = client.CoreV1Api()
  # k8s_client.list_namespace(timeout_seconds=5)
except Exception as e:
  print(f"Warning: Kubernetes config loading failed: {e}")


# %%
# ==========================================
# 4. Model & Training Hyperparameters
# ==========================================
MODEL_SOURCE = "maxtext"
MODEL_PATH = args.model_absolute_path
print(f"Using MaxText model from absolute path: {MODEL_PATH}")

# ====== Data ======
TRAIN_FRACTION = args.train_fraction

# ====== Reproducibility ======
SEED = args.seed

# ====== LoRA ======
RANK = args.rank
ALPHA = args.alpha
TRAIN_WITH_LORA = args.train_with_lora

# ====== Sharding ======
# MESH = [(4, 2), ("fsdp", "tp")]


# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
NUM_GENERATIONS = args.num_generations  # This corresponds to `G` in Algorithm 1

# === other GRPO configs ===
NUM_ITERATIONS = args.num_iterations
BETA = args.beta
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high
OFF_POLICY_STEPS = args.off_policy_steps
FORCE_ON_POLICY_RATIO = args.force_on_policy_ratio

# ====== Training ======
DTYPE_MAP = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "int32": jnp.int32,
}
DTYPE = DTYPE_MAP[args.dtype]
PARAM_DTYPE = DTYPE_MAP[args.param_dtype]
USE_FLASH_ATTENTION = args.use_flash_attention
FLASH_ATTENTION_BLOCK_SIZE = args.flash_attention_block_size
ENABLE_REMAT = args.enable_remat
REMAT_POLICY = args.remat_policy
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size


COMPUTE_LOGPS_MICRO_BATCH_SIZE = args.compute_logps_micro_batch_size
TRAIN_MICRO_BATCH_SIZE = args.train_micro_batch_size
ROLLOUT_MICRO_BATCH_SIZE = args.rollout_micro_batch_size

EVAL_EVERY_N_STEPS = args.eval_every_n_steps
NUM_EPOCHS = args.num_epochs

# Number of training steps.
MAX_STEPS = args.max_steps

# Max turns in mult-agent interaction (set to 1 for single-turn)
MAX_TURNS = args.max_turns
PER_TURN_TIMEOUT_SECS = args.per_turn_timeout_secs
EPISODE_TIMEOUT_SECS = args.episode_timeout_secs
STEP_TIMEOUT_SECS = args.step_timeout_secs
REWARD_TIMEOUT_SECS = args.reward_timeout_secs

MAX_CONCURRENCY = args.max_concurrency
USE_AGENT_SANDBOX = args.use_agent_sandbox
KV_CACHE_SIZE = max(
    4096, 1 << ((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH - 1).bit_length())
)
print(f"kv_cache_size (Capped): {KV_CACHE_SIZE}")
# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = args.learning_rate
B1 = args.b1
B2 = args.b2
WEIGHT_DECAY = args.weight_decay
# WARMUP_STEPS = int(args.warmup_ratio * MAX_STEPS)
MAX_GRAD_NORM = args.max_grad_norm
OPTIMIZER_OFFLOAD = args.optimizer_offload

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = args.save_interval_steps
MAX_TO_KEEP = args.max_to_keep
DO_MEM_PROFILING = args.do_mem_profiling

# ====== Rollout ======
ROLLOUT_ENGINE = args.rollout_engine
CKPT_DIR = (
    args.ckpt_dir
    if args.ckpt_dir and args.ckpt_dir.lower() not in ("none", "null")
    else None
)


# Max number of sequences to be processed in parallel by vllm.
VLLM_MAX_NUM_SEQS = ROLLOUT_MICRO_BATCH_SIZE * NUM_GENERATIONS

VLLM_UTILIZATION = args.vllm_utilization
VLLM_RESHARD_CHUNK_SIZE = args.vllm_reshard_chunk_size

# Max number of tokens to be processed in parallel by vllm.
VLLM_MAX_BATCHED_TOKENS = args.max_num_batched_tokens
print(f"vllm_max_batched_tokens: {VLLM_MAX_BATCHED_TOKENS}")

OVERLONG_FILTER = args.overlong_filter
FILTER_STATUSES = (
    {agent_types.TrajectoryStatus[name] for name in args.filter_statuses}
    if args.filter_statuses is not None
    else None
)
LOSS_AGG_MODE = args.loss_agg_mode
ADVANTAGE_ESTIMATOR = args.advantage_estimator
USE_ROLLOUT_LOGPS = args.use_rollout_logps


# %%
# ==========================================
# 5. Tokenizer & Dataset Preparation
# ==========================================
tokenizer_path = (
    MODEL_VERSION
    if MODEL_VERSION.startswith("Qwen/")
    else f"Qwen/{MODEL_VERSION}"
)
print(f"Loading tokenizer from HF Hub: {tokenizer_path}")

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, local_files_only=False, trust_remote_code=True
)

chat_parser = template_parser.QwenChatTemplateParser(
    tokenizer, enable_thinking=args.enable_thinking
)
print(
    f"chat parser: {type(chat_parser).__name__} "
    f"enable_thinking={args.enable_thinking} "
    f"generation_prompt={chat_parser.generation_prompt!r}",
    flush=True,
)

print("Loading Dataset...")

if args.dataset_path:
  dataset = datasets_lib.load_from_disk(args.dataset_path)
  if isinstance(dataset, datasets_lib.DatasetDict):
    dataset = dataset["train"]
else:
  dataset = datasets_lib.load_dataset(
      "R2E-Gym/R2E-Gym-Subset",
      split="train",
      cache_dir=DATASET_CACHE,
  )


if args.filter_repo:
  print(f"Filtering dataset to repo: {args.filter_repo}")
  dataset = dataset.filter(lambda x: x["repo_name"] == args.filter_repo)
  print(f"Filtered dataset size: {len(dataset)}")

if args.filter_available_images_only:
  import urllib.request, json

  ar_tags = set()
  # Query GCP Artifact Registry REST API or fallback to metadata token
  try:
    # 1. Get access token from metadata server or default credentials
    token = None
    try:
      req = urllib.request.Request(
          "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
          headers={"Metadata-Flavor": "Google"},
      )
      with urllib.request.urlopen(req, timeout=3) as resp:
        token = json.loads(resp.read().decode())["access_token"]
    except Exception:
      pass

    if token:
      # Project: cloud-tpu-multipod-dev, Location: us-central1, Repo: tunix, Package: pandas_final
      # List tags via Artifact Registry REST API
      url = "https://artifactregistry.googleapis.com/v1/projects/cloud-tpu-multipod-dev/locations/us-central1/repositories/tunix/packages/pandas_final/tags?pageSize=1000"
      while url:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
          data = json.loads(resp.read().decode())
          for tag_obj in data.get("tags", []):
            # tag name is e.g. "projects/.../tags/<tag>"
            ar_tags.add(tag_obj["name"].split("/")[-1])
          page_token = data.get("nextPageToken")
          if page_token:
            url = f"https://artifactregistry.googleapis.com/v1/projects/cloud-tpu-multipod-dev/locations/us-central1/repositories/tunix/packages/pandas_final/tags?pageSize=1000&pageToken={page_token}"
          else:
            url = None
  except Exception as e:
    print(f"Warning: Failed to fetch tags via Artifact Registry REST API: {e}")

  if not ar_tags and os.path.exists("/app/ar_available_tags.json"):
    with open("/app/ar_available_tags.json") as f:
      ar_tags = set(json.load(f))

  print(
      f"Found {len(ar_tags)} available image tags in Artifact Registry."
      " Filtering dataset..."
  )
  if ar_tags:
    dataset = dataset.filter(
        lambda x: x["docker_image"].split(":")[-1] in ar_tags
    )
    print(f"Dataset filtered to {len(dataset)} available instances.")
  else:
    print("Warning: No AR tags found, proceeding without filtering.")


if args.max_examples:
  num_to_take = min(len(dataset), args.max_examples)
  print(f"Limiting dataset to {num_to_take} examples")
  dataset = dataset.select(range(num_to_take))


def transform(entry):
  for k, v in entry.items():
    if isinstance(v, list):
      entry[k] = json.dumps(v)
  if (
      args.docker_image_prefix
      and "docker_image" in entry
      and entry["docker_image"]
  ):
    # e.g., 'namanjain12/pandas_final:tag' -> '<prefix>/pandas_final:tag'
    img_name = entry["docker_image"].split("/")[-1]
    entry["docker_image"] = f"{args.docker_image_prefix.rstrip('/')}/{img_name}"
  return entry


dataset = dataset.map(
    transform,
    keep_in_memory=True,  # pyrefly: ignore[unexpected-keyword]
)

dataset = dataset.shuffle(seed=SEED)
grain_dataset = grain.MapDataset.source(dataset)  # pyrefly: ignore[bad-argument-type]


def mixed_type_batch_fn(elements):
  """elements: A list of dicts."""
  batched_data = {}
  str_set = {
      "repo_name",
      "docker_image",
      "commit_hash",
      "parsed_commit_content",
      "execution_result_content",
  }
  dict_set = {"modified_files", "relevant_files", "modified_entity_summaries"}
  int_set = {
      "num_non_test_files",
      "num_non_test_func_methods",
      "num_non_test_lines",
      "prompt",
      "problem_statement",
      "expected_output_json",
  }
  keys = elements[0].keys()

  for key in keys:
    if key in str_set or key in dict_set:
      # Keep these as standard Python lists
      batched_data[key] = [item[key] for item in elements]

    elif key in int_set:
      # Convert these to NumPy arrays.
      # np.array() safely handles both single integers and lists of integers.
      batched_data[key] = np.array([item[key] for item in elements])

    else:
      # Fallback for any unexpected keys (defaulting to lists is usually safest)
      batched_data[key] = [item[key] for item in elements]

  return batched_data


train_dataset, _ = data_lib.post_init_dataset(
    grain_dataset,
    tokenizer,  # pyrefly: ignore[bad-argument-type]
    batch_size=BATCH_SIZE,
    num_batches=None,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
    prompt_key="problem_statement",
    custom_batch_fn=mixed_type_batch_fn,
)

fleet = None
if USE_AGENT_SANDBOX:
  fleet = swe_env._init_global_fleet(
      tasks=dataset,
      max_concurrency=MAX_CONCURRENCY,
      num_generations=NUM_GENERATIONS,
      batch_size=MINI_BATCH_SIZE,
      max_warmpool_replicas=args.max_warmpool_replicas,
  )
  train_dataset = swe_env.PrewarmDatasetIterator(
      train_dataset,
      fleet=fleet,
      num_generations=NUM_GENERATIONS,
      batch_size=MINI_BATCH_SIZE,
      max_warmpool_replicas=args.max_warmpool_replicas,
  )


# %%
# ==========================================
# 6. JAX Device, Config & Mesh Setup (MaxText)
# ==========================================
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig, types
from maxtext.utils import model_creation_utils, maxtext_utils

devices = jax.devices()
total_devices = len(devices)

# 1. Resolve Rollout Mesh Dimensions
rollout_fsdp = args.rollout_mesh_fsdp
rollout_tp = args.rollout_mesh_tp
if rollout_fsdp is None and rollout_tp is None:
  num_rollout_devices = int(total_devices * args.rollout_split_fraction)
  rollout_tp = 2
  rollout_fsdp = num_rollout_devices // rollout_tp
else:
  rollout_fsdp = rollout_fsdp if rollout_fsdp is not None else 1
  rollout_tp = rollout_tp if rollout_tp is not None else 1
  num_rollout_devices = rollout_fsdp * rollout_tp

# 2. Resolve Train Mesh Dimensions
train_fsdp = args.train_mesh_fsdp
train_tp = args.train_mesh_tp
train_sp = args.train_mesh_sp
if train_fsdp is None and train_tp is None:
  num_train_devices = total_devices - num_rollout_devices
  train_tp = 2
  train_fsdp = num_train_devices // train_tp
else:
  train_fsdp = train_fsdp if train_fsdp is not None else 1
  train_tp = train_tp if train_tp is not None else 1
  num_train_devices = train_fsdp * train_tp

if num_rollout_devices + num_train_devices > total_devices:
  raise ValueError(
      f"Requested {num_rollout_devices} rollout devices + {num_train_devices} "
      f"train devices, but cluster only has {total_devices} available."
  )

base_yml = os.path.join(
    os.path.dirname(pyconfig.__file__), "post_train", "rl.yml"
)
vllm_yml = os.path.join(
    os.path.dirname(pyconfig.__file__), "inference", "vllm.yml"
)

maxtext_remat_policy = "none"
if args.enable_remat:
  if args.remat_policy in ("decoder", "full"):
    maxtext_remat_policy = "full"
  elif args.remat_policy in ("block", "minimal"):
    maxtext_remat_policy = "minimal"
  else:
    maxtext_remat_policy = args.remat_policy

model_name_slug = MODEL_VERSION.lower().split("/")[-1]

trainer_config = pyconfig.initialize(
    [
        "",
        base_yml,
        "num_slices=1",
        f"model_name={model_name_slug}",
        f"load_parameters_path={MODEL_PATH}",
        f"ici_fsdp_parallelism={train_fsdp}",
        f"ici_tensor_parallelism={train_tp}",
        f"scan_layers={args.scan_layers}",
        f"max_target_length={KV_CACHE_SIZE}",
        f"max_prefill_predict_length={MAX_PROMPT_LENGTH}",
        f"remat_policy={maxtext_remat_policy}",
        f"dtype={args.dtype}",
        f"attention={'flash' if args.use_flash_attention else 'dot_product'}",
        f"prefuse_moe_weights={args.prefuse_moe_weights}",
        f"checkpoint_storage_use_ocdbt={args.checkpoint_storage_use_ocdbt}",
        f"checkpoint_storage_use_zarr3={args.checkpoint_storage_use_zarr3}",
        f"checkpoint_storage_concurrent_gb={args.checkpoint_storage_concurrent_gb}",
        f"float32_gate_logits={args.float32_gate_logits}",
        f"logits_dot_in_fp32={args.logits_dot_in_fp32}",
        *(
            [f"trainable_parameters_mask={args.trainable_parameters_mask}"]
            if args.trainable_parameters_mask
            else []
        ),
        "skip_jax_distributed_system=True",
        "load_checkpoint_only_once=True",
        "use_standalone_converter=False",
        "override_model_config=True",
        "use_mrope=False",
        "log_config=False",
        "allow_split_physical_axes=True",
    ],
    config_class=types.RLConfig,
    vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
)

# `log_config=False` above suppresses MaxText's own config dump, which is why no
# run log has ever contained a trainer/sampler diff. Print the keys that decide
# the numerics. The adapter prints the matching line for the model vLLM actually
# builds, so the two can be diffed by grepping one log for "resolved".
#
# getattr with a default throughout: this is diagnostics, and a MaxText version
# that renamed a key must not take the run down at startup.
NUMERICS_KEYS = (
    "model_name", "dtype", "weight_dtype", "attention", "use_mrope",
    "logits_dot_in_fp32", "float32_logits", "float32_qk_product",
    "float32_gate_logits", "float32_weight_sum", "matmul_precision",
    "scan_layers", "prefuse_moe_weights", "padded_base_moe_mlp_dim",
    "base_num_kv_heads", "use_qk_norm_in_gdn", "capacity_factor",
    "sparse_matmul", "megablox", "normalization_layer_epsilon",
    "max_target_length", "remat_policy", "dropout_rate", "model_call_mode",
)


def describe_maxtext_config(label, cfg):
  return f"[deepswe] resolved {label} config: " + " ".join(
      f"{k}={getattr(cfg, k, '<absent>')}" for k in NUMERICS_KEYS
  )


print(describe_maxtext_config("TRAINER", trainer_config), flush=True)

sampler_config = pyconfig.initialize(
    [
        "",
        vllm_yml,
        "num_slices=1",
        f"model_name={model_name_slug}",
        f"ici_data_parallelism={rollout_fsdp}",
        f"ici_tensor_parallelism={rollout_tp}",
        f"rollout_tensor_parallelism={rollout_tp}",
        f"rollout_data_parallelism={rollout_fsdp}",
        f"max_target_length={KV_CACHE_SIZE}",
        f"max_prefill_predict_length={MAX_PROMPT_LENGTH}",
        f"dtype={args.dtype}",
        "attention=vllm_rpa",
        f"float32_gate_logits={args.float32_gate_logits}",
        f"logits_dot_in_fp32={args.logits_dot_in_fp32}",
        # NOTE: nothing in this list reaches the model vLLM executes. This
        # object only supplies the rollout device mesh
        # (model_creation_utils.py:831-833). The model is built by
        # `maxtext_vllm_adapter.generate_maxtext_config`, which reads
        # `rollout_vllm_additional_config["maxtext_config"]` below and nothing
        # else. Anything that has to reach the sampler must be set in BOTH
        # places -- see the `use_mrope` comment there.
        "use_mrope=False",
        "override_model_config=True",
        "skip_jax_distributed_system=True",
        "remat_policy=none",
        "use_standalone_converter=False",
        "log_config=False",
        "allow_split_physical_axes=True",
    ],
    config_class=types.RLConfig,
    vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
)

# Printed for completeness, but note the label: this is the MESH config, not the
# model config. Compare the adapter's "resolved SAMPLER config" line against the
# TRAINER line above; where this one disagrees with the adapter's, the key never
# reached the sampler.
print(describe_maxtext_config("SAMPLER-MESH (not the model)", sampler_config), flush=True)

sampler_devices = devices[:num_rollout_devices]
trainer_devices = devices[
    num_rollout_devices : num_rollout_devices + num_train_devices
]

# %%
# ==========================================
# 7. Model Initialization via MaxText
# ==========================================
from etils import epath
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import type_handler_registry

# pathwaysutils registers CloudPathwaysArrayHandler on init, which reads
# checkpoint shards on the Pathways workers. It does not support OCDBT yet
# (b/365549911), so an OCDBT checkpoint has to fall back to the standard
# ArrayHandler -- but that one reads on the client and materializes whole arrays
# in the head container's host RAM.
#
# So only pay that cost when the checkpoint really is OCDBT. The client-side
# restore scales with the largest single array rather than with model size,
# which is why it goes unnoticed at 35B ([256, 10, 2048, 512] = 5.4 GiB, ~18 GB
# peak) and is fatal at 397B: the scan axis makes every MoE tensor
# [512, 15, 4096, 1024] = 64 GiB with ~12 in flight, so the head needs ~690 GB
# and is OOMKilled. Keeping the reads on the workers peaks at 22 GB instead.
# Note that checkpoint_storage_concurrent_gb does not bound this path.
#
# Outside Pathways this is a no-op either way: ArrayHandler is already the
# default handler for jax.Array.
if (epath.Path(MODEL_PATH) / "manifest.ocdbt").exists():
  print(
      "Base checkpoint is OCDBT, using the standard ArrayHandler:"
      f" {MODEL_PATH}",
      flush=True,
  )
  type_handler_registry.register_type_handler(
      jax.Array, jax_array_handlers.ArrayHandler(), override=True
  )
else:
  print(
      "Base checkpoint is not OCDBT, keeping the registered handler so reads"
      f" stay on the Pathways workers: {MODEL_PATH}",
      flush=True,
  )

(
    qwen_reference,
    reference_mesh,
    qwen_actor,
    actor_mesh,
    rollout_mesh,
) = model_creation_utils.create_models_and_meshes(
    trainer_config=trainer_config,
    sampler_config=sampler_config,
    trainer_devices=trainer_devices,
    sampler_devices=sampler_devices,
    tokenizer_pad_id=tokenizer.pad_token_id,
)

train_mesh = actor_mesh

print(f"*** Rollout Mesh *** | Shape: {rollout_mesh.shape}")
print(f"*** Train Mesh *** | Shape: {train_mesh.shape}")

if TRAIN_WITH_LORA:

  def get_lora_model(base_model, model_mesh):
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
            ".*gate_proj|.*down_proj|.*up_proj"
        ),
        rank=RANK,
        alpha=ALPHA,
    )
    raw_model = getattr(base_model, "base", base_model)
    model_input = (
        raw_model.get_model_input()
        if hasattr(raw_model, "get_model_input")
        else {}
    )
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )
    with compat.set_mesh(model_mesh):
      state = nnx.state(lora_model)
      pspecs = nnx.get_partition_spec(state)
      sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
      nnx.update(lora_model, sharded_state)
    return lora_model

  qwen_actor = get_lora_model(qwen_actor, train_mesh)

if hasattr(qwen_reference, "use_no_op_mappings"):
  qwen_reference.use_no_op_mappings = False
if hasattr(qwen_actor, "use_no_op_mappings"):
  qwen_actor.use_no_op_mappings = False

sft_utils.show_hbm_usage()

# %%
# ==========================================
# 8. Optimizer & Checkpointing
# ==========================================
if CKPT_DIR:
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
  )
else:
  checkpointing_options = None

metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=args.metric_logger_dir, flush_every_n_steps=2
)

warmup_steps = int(MAX_STEPS * args.warmup_steps_fraction)
if warmup_steps > 0 or args.learning_rate_final_fraction < 1.0:
  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0 if warmup_steps > 0 else LEARNING_RATE,
      peak_value=LEARNING_RATE,
      warmup_steps=warmup_steps,
      decay_steps=MAX_STEPS,
      end_value=LEARNING_RATE * args.learning_rate_final_fraction,
  )
else:
  lr_schedule = LEARNING_RATE

adamw_opt = optax.adamw(
    learning_rate=lr_schedule,
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
)

from maxtext.optimizers import optimizers as maxtext_optimizers

adamw_opt = maxtext_optimizers.apply_trainable_parameters_mask(
    adamw_opt, trainer_config
)

transforms = []
if MAX_GRAD_NORM is not None:
  transforms.append(optax.clip_by_global_norm(MAX_GRAD_NORM))
transforms.append(adamw_opt)

optimizer = optax.chain(*transforms)

# %%
# ==========================================
# 9. Rollout Engine Setup (vLLM) & RL Cluster Setup
# ==========================================
from tunix.perf import metrics as perf_metrics

base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "kv_cache_size": KV_CACHE_SIZE,
    "return_logprobs": USE_ROLLOUT_LOGPS,
    "return_routed_experts": True,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": tokenizer_path,
    "rollout_vllm_hbm_utilization": VLLM_UTILIZATION,
    "rollout_vllm_reshard_chunk_size": VLLM_RESHARD_CHUNK_SIZE,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": True,
    "rollout_vllm_init_with_random_weights": True,
    "tensor_parallel_size": rollout_mesh.shape.get("model", 1),
    "data_parallel_size": rollout_mesh.shape.get("data", 1),
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": args.enable_prefix_caching,
        "tokenizer": tokenizer_path,
        "dtype": "bfloat16",
        "enable_expert_parallel": False,
        "generation_config": "vllm",
        "hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
    },
    "rollout_mapping_config": {},
    # THIS dict, not `sampler_config` above, is what the vLLM-side MaxText model
    # is built from: `maxtext_vllm_adapter/adapter.py:82-186` layers exactly
    # these keys over `configs/inference/vllm.yml` and the model yml. A key that
    # is absent here takes the model yml's value regardless of what
    # `sampler_config` says.
    "rollout_vllm_additional_config": {
        "maxtext_config": {
            "model_name": model_name_slug,
            "model_call_mode": "inference",
            "attention": "vllm_rpa",
            "allow_split_physical_axes": True,
            "log_config": False,
            "weight_dtype": "bfloat16",
            "prefuse_moe_weights": True,
            "remat_policy": "none",
            "enable_dp_attention": False,
            "float32_gate_logits": args.float32_gate_logits,
            # configs/models/qwen3.5-35b-a3b.yml sets `use_mrope: true`, so
            # without this the sampler builds Qwen3OmniMoeThinkerTextRotaryEmbedding
            # (fp32 cos/sin) while the trainer builds PartialRotaryEmbedding
            # (bf16 cos/sin). Setting it on `sampler_config` does nothing --
            # that was the bug.
            "use_mrope": False,
            "logits_dot_in_fp32": args.logits_dot_in_fp32,
            "vllm_hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
        }
    },
    "rollout_vllm_sampling_kwargs": {
        # A stop STRING ends generation before the model emits <|im_end|>, so
        # the recorded token_ids have no turn terminator while the sampler's own
        # next-turn prompt always does (`_parse_assistant` appends eot
        # unconditionally). vllm_sampler.py:447-453 records that this same
        # omission once "produced 30+ nat sampler-trainer logp diffs".
        # Measured on a production CSV: 87.5% of assistant turns end without one.
        # The reference sets `stop_strings: null`; pass --rollout_stop_strings ""
        # to match it.
        **(
            {"stop": [s for s in args.rollout_stop_strings.split(",") if s]}
            if args.rollout_stop_strings
            else {}
        ),
        "stop_token_ids": [
            tokenizer.encode("<|im_end|>")[0],
            tokenizer.encode("<|endoftext|>")[0],
        ],
        "detokenize": True,
    },
}

if ROLLOUT_ENGINE == "vllm":
  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
  if TRAIN_WITH_LORA:
    vllm_rollout_dict["rollout_vllm_lora_config"] = {
        "max_lora_rank": RANK,
    }
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **vllm_rollout_dict
  )
elif ROLLOUT_ENGINE == "vanilla":
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)
else:
  raise ValueError(f"Unsupported rollout engine: {ROLLOUT_ENGINE}")

from maxtext.integration.vllm.maxtext_vllm_adapter import adapter

_orig_generate_maxtext_config = adapter.generate_maxtext_config


def _generate_maxtext_config_with_no_remat(vllm_config_param):
  if "maxtext_config" not in vllm_config_param.additional_config:
    vllm_config_param.additional_config["maxtext_config"] = {}
  vllm_config_param.additional_config["maxtext_config"]["remat_policy"] = "none"
  return _orig_generate_maxtext_config(vllm_config_param)


adapter.generate_maxtext_config = _generate_maxtext_config_with_no_remat

role_to_logical_axis_rule = {
    rl_engine_lib.Role.ACTOR: trainer_config.logical_axis_rules,
    rl_engine_lib.Role.REFERENCE: trainer_config.logical_axis_rules,
    rl_engine_lib.Role.ROLLOUT: sampler_config.logical_axis_rules,
}

import functools
from maxtext.integration.vllm.maxtext_vllm_rollout import MaxTextVllmRollout

rollout_engine_arg = functools.partial(
    MaxTextVllmRollout, maxtext_config=sampler_config
)

cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh={
        rl_engine_lib.Role.ACTOR: actor_mesh,
        rl_engine_lib.Role.REFERENCE: reference_mesh,
        rl_engine_lib.Role.ROLLOUT: rollout_mesh,
    },
    role_to_logical_axis_rule=role_to_logical_axis_rule,
    rollout_engine=rollout_engine_arg,
    offload_to_cpu=False,
    training_config=rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        compute_logps_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE,
        rollout_micro_batch_size=ROLLOUT_MICRO_BATCH_SIZE,
        metrics_logging_options=metrics_logging_options,
        perf_metrics_options=perf_metrics.PerfMetricsOptions(),
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=rollout_engine_config,
)
sft_utils.show_hbm_usage()

try:
  rl_engine = rl_engine_lib.RLEngine(
      actor=qwen_actor,
      reference=qwen_reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )
  if hasattr(rl_engine, "actor_trainer") and rl_engine.actor_trainer.train_steps > 0:
    print(
        f"Restored actor_trainer at step {rl_engine.actor_trainer.train_steps}; syncing rl_engine.global_steps.",
        flush=True,
    )
    rl_engine.global_steps = rl_engine.actor_trainer.train_steps
except ValidationError as e:
  print("Failed to initialize RLEngine due to ValidationError:", flush=True)
  import pprint

  pprint.pprint(e.errors())
  raise

# %%
# ==========================================
# 10. Learner & Agent Setup
# ==========================================

config_kwargs = {
    "num_generations": NUM_GENERATIONS,
    "num_iterations": NUM_ITERATIONS,
    "max_response_length": MAX_RESPONSE_LENGTH,
    "beta": BETA,
    "epsilon": EPSILON,
    "system_prompt": swe_agent.SWE_SYSTEM_PROMPT,
    "max_concurrency": MAX_CONCURRENCY,
    "epsilon_high": EPSILON_HIGH,
    "off_policy_steps": OFF_POLICY_STEPS,
    "episode_timeout": EPISODE_TIMEOUT_SECS,
    "overlong_filter": OVERLONG_FILTER,
    "filter_statuses": FILTER_STATUSES,
    "loss_agg_mode": LOSS_AGG_MODE,
    "advantage_estimator": ADVANTAGE_ESTIMATOR,
    "use_rollout_logps": USE_ROLLOUT_LOGPS,
    "force_on_policy_ratio": FORCE_ON_POLICY_RATIO,
    "sampler_is": args.sampler_is,
    "overlong_loss_masking": args.overlong_loss_masking,
    "seq_logprob_error_threshold": args.seq_logprob_error_threshold,
    "truncated_importance_sampling_type": args.truncated_importance_sampling_type,
    "truncated_importance_sampling_ratio_min": args.truncated_importance_sampling_ratio_min,
    "truncated_importance_sampling_ratio": args.truncated_importance_sampling_ratio,
}

grpo_config = agentic_grpo_learner.GRPOConfig(**config_kwargs)

agentic_grpo_learner = agentic_grpo_learner.GRPOLearner(
    rl_engine=rl_engine,
    reward_fns=None,
    agent_class=swe_agent.SWEAgent,
    agent_kwargs={},
    env_class=swe_env.SWEEnv,
    env_kwargs={
        "max_steps": MAX_TURNS,
        "step_timeout": STEP_TIMEOUT_SECS,
        "reward_timeout": REWARD_TIMEOUT_SECS,
        "verbose": True,
        "use_agent_sandbox": USE_AGENT_SANDBOX,
        "fleet": fleet,
    },
    algo_config=grpo_config,
    chat_parser=chat_parser,
)

try:
  import datetime
  import wandb  # pytype: disable=import-error

  settings = wandb.Settings(console="off")
  run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  wandb_config = {
      **vars(args),
      # Derived values not present in args
      "kv_cache_size": KV_CACHE_SIZE,
      "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
      "vllm_max_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
      # Stringify set so wandb can serialize it
      "filter_statuses": (
          [s.name for s in FILTER_STATUSES] if FILTER_STATUSES else None
      ),
      # Mesh topology
      "num_devices": len(devices),
      "rollout_mesh_fsdp": rollout_fsdp,
      "rollout_mesh_tp": rollout_tp,
      "train_mesh_fsdp": train_fsdp,
      "train_mesh_sp": train_sp,
      "train_mesh_tp": train_tp,
      "checkpoint_root_directory": CKPT_DIR,
      "save_interval_steps": SAVE_INTERVAL_STEPS,
      "max_to_keep": MAX_TO_KEEP,
  }
  if wandb.run is None:
    wandb.init(
        project="tunix", name=run_name, config=wandb_config, settings=settings
    )
except Exception as e:
  print(f"W&B initialization failed with error: {e}")

print("Syncing initial checkpoint weights to rollout workers...", flush=True)
_prev_steps = rl_engine.global_steps
rl_engine.sync_weights()
rl_engine.global_steps = _prev_steps

if (
    CKPT_DIR
    and "proxy" in os.getenv("JAX_PLATFORMS", "")
    and os.getenv("ENABLE_PATHWAYS_PERSISTENCE", "")
):
  import orbax.checkpoint.pathways as ocp_pathways

  print(
      "Registering Pathways persistence handlers for training checkpoints...",
      flush=True,
  )
  ocp_pathways.register_type_handlers(
      checkpointing_impl=ocp_pathways.CheckpointingImpl.PERSISTENCE
  )

print("Starting training...", flush=True)
agentic_grpo_learner.train(train_dataset=train_dataset)
