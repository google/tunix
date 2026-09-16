#!/usr/bin/env python
"""DeepSWE evaluation with deepscaler-style task-level parallelism.

This script runs SWE evaluation trajectories and uses RolloutOrchestrator to
parallelize tasks across a TPU cluster for MaxText models (e.g. Qwen3.5-35B-A3B)
with configurable JAX/vLLM sharding meshes.
"""

import argparse
import asyncio
import collections
import concurrent.futures
import gc
import json
import logging
import math
import os
import sys
import time

# Path Setup before JAX
workdir = os.getcwd()
pathways_root = os.path.join(workdir, "pathways-utils")
r2egym_root = os.path.join(workdir, "r2egym")

for root in [
    workdir,
    pathways_root,
    r2egym_root,
    "/usr/github/rllm",
    "/usr/github/pathways-utils",
    "/app",
]:
  if os.path.exists(root) and root not in sys.path:
    sys.path.insert(0, root)


# vLLM TPU backend environment configuration for MaxText
os.environ["VLLM_TPU_RPA_VERSION"] = "2"
os.environ["DISABLE_MOSAIC_ATTN"] = "1"

if "proxy" in os.getenv("JAX_PLATFORMS", ""):
  import pathwaysutils
  pathwaysutils.initialize()
  print("Pathways initialized successfully before JAX import.")

import datasets as datasets_lib
import jax
from jax.sharding import Mesh
from kubernetes import client
from kubernetes import config as k8s_config
import numpy as np
from transformers import AutoTokenizer
from maxtext.integration.vllm import maxtext_vllm_adapter
import swe_env
from guarded_swe_env import GuardedSWEEnv
from swe_agent import SWEAgent
from swe_env import _normalize_entry, SWEEnv

from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator
from tunix.rl.agentic.trajectory import trajectory_collect_engine

Counter = collections.Counter


def str2bool(v):
  """Parses common string representations into boolean values."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ========================== Argument Parsing ==========================

parser_cli = argparse.ArgumentParser(
    description="DeepSWE Evaluation with task-level parallelism"
)
parser_cli.add_argument(
    "--model_version",
    type=str,
    default=os.getenv("MODEL_VERSION", "Qwen/Qwen3-32B"),
    help="Model identifier",
)
parser_cli.add_argument(
    "--model_absolute_path",
    type=str,
    default=os.getenv("MODEL_ABSOLUTE_PATH", os.getenv("MODEL_PATH", None)),
    help="Absolute model path (GCS bucket or local directory)",
)
parser_cli.add_argument(
    "--dataset_path",
    type=str,
    default=os.getenv("DATASET_PATH", None),
    help="Dataset path",
)
parser_cli.add_argument(
    "--dataset_split",
    type=str,
    default=os.getenv("DATASET_SPLIT", "test"),
    help="Dataset split",
)
parser_cli.add_argument(
    "--dataset_cache",
    type=str,
    default=os.getenv("DATASET_CACHE", "/scratch/dataset_cache"),
    help="Dataset cache directory",
)
parser_cli.add_argument(
    "--max_steps",
    type=int,
    default=int(os.getenv("MAX_STEPS", "30")),
    help="Max agent steps per trajectory",
)
parser_cli.add_argument(
    "--max_model_len",
    type=int,
    default=int(os.getenv("MAX_MODEL_LEN", "32768")),
    help="Max context length of the LLM",
)
parser_cli.add_argument(
    "--max_response_length",
    type=int,
    default=int(
        os.getenv(
            "MAX_RESPONSE_LENGTH", os.getenv("MAX_GENERATION_STEPS", "4096")
        )
    ),
    help="Max response tokens per generation",
)
parser_cli.add_argument(
    "--max_concurrent",
    type=int,
    default=int(os.getenv("MAX_CONCURRENT", "128")),
    help="Max concurrent tasks/trajectories",
)
parser_cli.add_argument(
    "--timeout",
    type=float,
    default=float(os.getenv("TIMEOUT", "3600")),
    help="Timeout in seconds for a single trajectory",
)
parser_cli.add_argument(
    "--eval_max_examples",
    type=int,
    default=(
        int(os.environ["EVAL_MAX_EXAMPLES"])
        if os.getenv("EVAL_MAX_EXAMPLES")
        else None
    ),
    help=(
        "Optionally limit total examples loaded for evaluation. Defaults to"
        " None, i.e. evaluate the whole dataset."
    ),
)
parser_cli.add_argument(
    "--max_context_limit",
    type=int,
    default=None,
    help="Max context token limit before terminating",
)
parser_cli.add_argument(
    "--enable_guard",
    type=str2bool,
    default=os.getenv("ENABLE_GUARD", "false").lower() == "true",
    help="Enable action guard",
)
parser_cli.add_argument(
    "--rollout_engine",
    type=str,
    default=os.getenv("ROLLOUT_ENGINE", "vllm"),
    help="Rollout engine (defaults to vllm)",
)
parser_cli.add_argument(
    "--vllm_utilization",
    type=float,
    default=float(
        os.getenv("VLLM_UTILIZATION", os.getenv("VLLM_HBM_UTILIZATION", "0.85"))
    ),
    help="HBM utilization ratio for vLLM",
)
parser_cli.add_argument(
    "--vllm_max_num_seqs",
    type=int,
    default=int(os.getenv("VLLM_MAX_NUM_SEQS", "128")),
)
parser_cli.add_argument(
    "--vllm_max_batched_tokens",
    type=int,
    default=int(os.getenv("VLLM_MAX_BATCHED_TOKENS", "165888")),
)
parser_cli.add_argument(
    "--mesh_fsdp",
    type=int,
    default=int(os.getenv("MESH_FSDP", "4")),
    help="FSDP dimension size of the mesh",
)
parser_cli.add_argument(
    "--mesh_tp",
    type=int,
    default=int(os.getenv("MESH_TP", "4")),
    help="TP dimension size of the mesh",
)
parser_cli.add_argument(
    "--scan_layers",
    type=str2bool,
    default=os.getenv("SCAN_LAYERS", "true").lower() == "true",
    help="Whether to scan layers for MaxText models",
)
parser_cli.add_argument(
    "--allow_split_physical_axes",
    type=str2bool,
    default=os.getenv("ALLOW_SPLIT_PHYSICAL_AXES", "true").lower() == "true",
    help=(
        "Whether to allow splitting physical axes in MaxText (can cause"
        " performance hit)"
    ),
)
parser_cli.add_argument(
    "--checkpoint_storage_concurrent_gb",
    type=int,
    default=int(os.getenv("CHECKPOINT_STORAGE_CONCURRENT_GB", "32")),
    help="Concurrent GB limit for Orbax checkpoint storage restore",
)
parser_cli.add_argument(
    "--checkpoint_storage_use_ocdbt",
    type=str2bool,
    default=os.getenv("CHECKPOINT_STORAGE_USE_OCDBT", "true").lower() == "true",
    help="Whether the MaxText checkpoint uses OCDBT storage",
)
parser_cli.add_argument(
    "--checkpoint_storage_use_zarr3",
    type=str2bool,
    default=os.getenv("CHECKPOINT_STORAGE_USE_ZARR3", "false").lower()
    == "true",
    help="Whether the MaxText checkpoint uses Zarr3 storage",
)
parser_cli.add_argument(
    "--weight_dtype",
    type=str,
    default=os.getenv("WEIGHT_DTYPE", "bfloat16"),
    help="Weight data type for MaxText model",
)
parser_cli.add_argument(
    "--prefuse_moe_weights",
    type=str2bool,
    default=os.getenv("PREFUSE_MOE_WEIGHTS", "true").lower() == "true",
    help="Whether to prefuse MoE weights in MaxText",
)
parser_cli.add_argument(
    "--maxtext_attention",
    type=str,
    default=os.getenv("MAXTEXT_ATTENTION", "vllm_rpa"),
    help="Attention backend for MaxText in vLLM",
)
parser_cli.add_argument(
    "--enable_continue_decode",
    type=str2bool,
    default=os.getenv("ENABLE_CONTINUE_DECODE", "false").lower() == "true",
    help="Whether to enable continue decode in vLLM",
)
parser_cli.add_argument(
    "--enable_prefix_caching",
    type=str2bool,
    default=os.getenv("ENABLE_PREFIX_CACHING", "false").lower() == "true",
    help="Whether to enable vLLM prefix caching",
)
parser_cli.add_argument(
    "--temperature",
    type=float,
    default=float(os.getenv("TEMPERATURE", "0.0")),
    help="Sampling temperature (0 for greedy)",
)
parser_cli.add_argument(
    "--top_p",
    type=float,
    default=(float(os.environ["TOP_P"]) if os.getenv("TOP_P") else None),
    help="Nucleus sampling probability",
)
parser_cli.add_argument(
    "--top_k",
    type=int,
    default=(int(os.environ["TOP_K"]) if os.getenv("TOP_K") else None),
    help="Top-k sampling cutoff",
)
parser_cli.add_argument(
    "--num_rollouts_per_instance",
    type=int,
    default=int(os.getenv("NUM_ROLLOUTS_PER_INSTANCE", "1")),
    help="Trajectories to sample per instance (>1 enables pass@k)",
)
parser_cli.add_argument(
    "--scaffold",
    type=str,
    default=os.getenv("SCAFFOLD", "r2egym"),
    choices=["r2egym", "sweagent", "openhands"],
    help="Agent scaffold/sandbox toolset",
)
parser_cli.add_argument(
    "--step_timeout_secs",
    type=int,
    default=int(os.getenv("STEP_TIMEOUT_SECS", str(30 * 60))),
    help="Timeout for a single environment step",
)
parser_cli.add_argument(
    "--reward_timeout_secs",
    type=int,
    default=int(os.getenv("REWARD_TIMEOUT_SECS", str(30 * 60))),
    help="Timeout for reward computation",
)
parser_cli.add_argument(
    "--docker_image_prefix",
    type=str,
    default=os.getenv("DOCKER_IMAGE_PREFIX", None),
    help=(
        "Optional registry prefix to replace the dataset image repo with (e.g."
        " us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix)"
    ),
)
parser_cli.add_argument(
    "--node_selector_val",
    type=str,
    default=os.getenv("NODE_SELECTOR_VAL", "cpu-np"),
    help="Node selector value for GKE nodepool",
)
parser_cli.add_argument(
    "--output_dir",
    type=str,
    default=os.getenv(
        "OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "eval_results")
    ),
    help="Output directory for results",
)
parser_cli.add_argument(
    "--logging_level",
    type=str,
    default=os.getenv("LOGGING_LEVEL", "INFO"),
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
parser_cli.add_argument(
    "--use_agent_sandbox",
    type=str2bool,
    nargs="?",
    const=True,
    default=os.getenv("USE_AGENT_SANDBOX", "false").lower() == "true",
    help=(
        "Whether to use Kubernetes Agent Sandbox runtime instead of local"
        " Docker socket."
    ),
)
parser_cli.add_argument(
    "--max_warmpool_replicas",
    "--max_warmpool_size",
    dest="max_warmpool_replicas",
    type=int,
    default=(
        int(os.getenv("MAX_WARMPOOL_SIZE", os.getenv("MAX_WARMPOOL_REPLICAS")))
        if os.getenv("MAX_WARMPOOL_SIZE") or os.getenv("MAX_WARMPOOL_REPLICAS")
        else None
    ),
    help=(
        "Max warmpool replicas per task/image. Defaults to"
        " num_rollouts_per_instance."
    ),
)

args, _ = parser_cli.parse_known_args()

DATASET_PATH = args.dataset_path
DATASET_SPLIT = args.dataset_split
DATASET_CACHE = args.dataset_cache

MODEL_VERSION = args.model_version
MODEL_SOURCE = "maxtext"
MODEL_PATH = args.model_absolute_path or os.path.join(
    "/scratch/models/", MODEL_VERSION
)

MAX_STEPS = args.max_steps
MAX_MODEL_LEN = args.max_model_len
MAX_RESPONSE_LENGTH = args.max_response_length
MAX_CONCURRENT = args.max_concurrent
TIMEOUT = args.timeout
EVAL_MAX_EXAMPLES = args.eval_max_examples
MAX_CONTEXT_LIMIT = (
    args.max_context_limit
    if args.max_context_limit is not None
    else max(1, MAX_MODEL_LEN - 256)
)

ENABLE_GUARD = args.enable_guard
ROLLOUT_ENGINE = args.rollout_engine

VLLM_HBM_UTILIZATION = args.vllm_utilization
VLLM_MAX_NUM_SEQS = args.vllm_max_num_seqs
VLLM_MAX_BATCHED_TOKENS = args.vllm_max_batched_tokens

MESH_FSDP = args.mesh_fsdp
MESH_TP = args.mesh_tp
SCAN_LAYERS = args.scan_layers
ALLOW_SPLIT_PHYSICAL_AXES = args.allow_split_physical_axes
CHECKPOINT_STORAGE_CONCURRENT_GB = args.checkpoint_storage_concurrent_gb
WEIGHT_DTYPE = args.weight_dtype
PREFUSE_MOE_WEIGHTS = args.prefuse_moe_weights
MAXTEXT_ATTENTION = args.maxtext_attention
ENABLE_CONTINUE_DECODE = args.enable_continue_decode
ENABLE_PREFIX_CACHING = args.enable_prefix_caching
CHECKPOINT_STORAGE_USE_OCDBT = args.checkpoint_storage_use_ocdbt
CHECKPOINT_STORAGE_USE_ZARR3 = args.checkpoint_storage_use_zarr3

# Longest prompt MaxText must be able to prefill, i.e. the context window
# minus the tokens reserved for the response.
MAX_PREFILL_LENGTH = max(1, MAX_MODEL_LEN - MAX_RESPONSE_LENGTH)

TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
NUM_ROLLOUTS_PER_INSTANCE = args.num_rollouts_per_instance

SCAFFOLD = args.scaffold
STEP_TIMEOUT_SECS = args.step_timeout_secs
REWARD_TIMEOUT_SECS = args.reward_timeout_secs
DOCKER_IMAGE_PREFIX = args.docker_image_prefix

NODE_SELECTOR_VAL = args.node_selector_val
OUTPUT_DIR = args.output_dir
USE_AGENT_SANDBOX = args.use_agent_sandbox
MAX_WARMPOOL_REPLICAS = args.max_warmpool_replicas

ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"

# ========================== Logging ==========================

log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(
    stream=sys.stdout,
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("deepswe_eval")

# ========================== JAX / Pathways ==========================

logger.info("JAX backend initialized.")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Register MaxText vLLM adapter
maxtext_vllm_adapter.register()
logger.info("Successfully registered MaxTextForCausalLM model with vLLM.")

# ========================== Dataset ==========================

logger.info(
    "Loading dataset %s split=%s ...",
    DATASET_PATH or "R2E-Gym/R2E-Gym-Subset",
    DATASET_SPLIT,
)
if args.dataset_path:
  dataset = datasets_lib.load_from_disk(args.dataset_path)
  if isinstance(dataset, datasets_lib.DatasetDict):
    dataset = dataset[DATASET_SPLIT]
else:
  dataset = datasets_lib.load_dataset(
      "R2E-Gym/R2E-Gym-Subset",
      split=DATASET_SPLIT,
      cache_dir=DATASET_CACHE,
  )
  


entries = [
    _normalize_entry(e, docker_image_prefix=DOCKER_IMAGE_PREFIX)
    for e in dataset
    if "docker_image" in e
]
if EVAL_MAX_EXAMPLES:
  entries = entries[:EVAL_MAX_EXAMPLES]

unique_images = set(e["docker_image"] for e in entries)
logger.info(
    "Loaded %d instances (%d unique Docker images)",
    len(entries),
    len(unique_images),
)

# ========================== Kubernetes ==========================

os.environ.setdefault("KUBECONFIG", "~/.kube/config")
os.environ.setdefault("NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool")
os.environ.setdefault("NODE_SELECTOR_VAL", NODE_SELECTOR_VAL)


if os.getenv("KUBERNETES_SERVICE_HOST"):
  k8s_config.load_incluster_config()
else:
  k8s_config.load_kube_config()
k8s_client = client.CoreV1Api()
logger.info("Kubernetes connection verified.")

fleet = None
if USE_AGENT_SANDBOX:
  fleet = swe_env._init_global_fleet(
      tasks=entries,
      max_concurrency=MAX_CONCURRENT,
      num_generations=NUM_ROLLOUTS_PER_INSTANCE,
      batch_size=min(MAX_CONCURRENT, 8),
      max_warmpool_replicas=MAX_WARMPOOL_REPLICAS,
      scaffold=SCAFFOLD,
  )
  logger.info('[Main] Starting warmpools for planned tasks on K8s...')
  fleet.start_warmpools(wait=False)


# ========================== Model & Mesh ==========================

# Tokenizer Setup
tokenizer_path = MODEL_PATH
local_files_only = True
if MODEL_PATH.startswith("gs://"):
  if MODEL_VERSION.startswith("Qwen/"):
    tokenizer_path = MODEL_VERSION
  else:
    tokenizer_path = f"Qwen/{MODEL_VERSION}"
  local_files_only = False
  logger.info("Loading tokenizer from HF Hub: %s", tokenizer_path)
else:
  logger.error("Model path %s must start with gs://", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, local_files_only=local_files_only, trust_remote_code=True
)
tokenizer_for_agentic = tok_adapter.TokenizerAdapter(tokenizer)
chat_parser = parser.QwenChatTemplateParser(tokenizer, enable_thinking=True)
qwen_eos_tokens = [tokenizer.encode("<|im_end|>")[0]]

# The r2egym scaffold terminates every action with `</function>`; stopping
# there matches the training rollouts and avoids generating past the action.
STOP_STRINGS = ["</function>", "<|im_end|>", "<|endoftext|>"]
STOP_TOKEN_IDS = [
    tokenizer.encode("<|im_end|>")[0],
    tokenizer.encode("<|endoftext|>")[0],
]

# Mesh Setup
devices = jax.devices()
total_mesh_devices = MESH_FSDP * MESH_TP
if total_mesh_devices > len(devices):
  raise ValueError(
      f"Requested mesh FSDP={MESH_FSDP} * TP={MESH_TP} "
      f"({total_mesh_devices} devices) exceeds the {len(devices)} available "
      "devices. Set --mesh_fsdp/--mesh_tp to match the topology."
  )

mesh_devices = np.array(devices[:total_mesh_devices]).reshape(
    MESH_FSDP, MESH_TP
)
mesh = Mesh(mesh_devices, axis_names=("fsdp", "tp"))
logger.info(
    "Using mesh shape fsdp=%d tp=%d (total_devices=%d, used_devices=%d)",
    mesh.shape["fsdp"],
    mesh.shape["tp"],
    len(devices),
    total_mesh_devices,
)

# pathwaysutils registers CloudPathwaysArrayHandler, which reads checkpoint
# shards on the Pathways workers but does not support OCDBT yet (b/365549911).
# Fall back to the standard ArrayHandler only when the checkpoint really is
# OCDBT, since that path materializes whole arrays in the head container's RAM.
try:
  from etils import epath
  from orbax.checkpoint._src.serialization import jax_array_handlers
  from orbax.checkpoint._src.serialization import type_handler_registry

  if (epath.Path(MODEL_PATH) / "manifest.ocdbt").exists():
    type_handler_registry.register_type_handler(
        jax.Array, jax_array_handlers.ArrayHandler(), override=True
    )
    logger.info(
        "Checkpoint is OCDBT; registered the standard ArrayHandler: %s",
        MODEL_PATH,
    )
  else:
    logger.info(
        "Checkpoint is not OCDBT; keeping the registered handler so reads"
        " stay on the Pathways workers: %s",
        MODEL_PATH,
    )
except Exception as e:
  logger.warning("Could not inspect checkpoint storage format: %s", e)

# ========================== Sampler ==========================

logger.info("Creating sampler with engine=%s ...", ROLLOUT_ENGINE)

from tunix.generate import mappings
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

maxtext_cfg = {
    "model_name": MODEL_VERSION.lower().split("/")[-1],
    "model_call_mode": "inference",
    # vLLM inference requires unrolled layers for PagedAttention KV cache indexing (see maxtext vllm.yml).
    # The base checkpoint remains scanned; weight_converter automatically unrolls scanned layers into vLLM.
    "scan_layers": False,
    "enable_dp_attention": False,
    "allow_split_physical_axes": ALLOW_SPLIT_PHYSICAL_AXES,
    "log_config": False,
    "weight_dtype": WEIGHT_DTYPE,
    "prefuse_moe_weights": PREFUSE_MOE_WEIGHTS,
    "attention": MAXTEXT_ATTENTION,
    "remat_policy": "none",
    "max_target_length": MAX_MODEL_LEN,
    "max_prefill_predict_length": MAX_PREFILL_LENGTH,
    "checkpoint_storage_use_ocdbt": CHECKPOINT_STORAGE_USE_OCDBT,
    "checkpoint_storage_use_zarr3": CHECKPOINT_STORAGE_USE_ZARR3,
    "checkpoint_storage_concurrent_gb": (
        CHECKPOINT_STORAGE_CONCURRENT_GB
    ),
    "load_parameters_path": MODEL_PATH,
}

additional_config = {
    "enable_continue_decode": ENABLE_CONTINUE_DECODE,
    "maxtext_config": maxtext_cfg,
}

# The adapter regenerates the MaxText config and would otherwise reinstate
# a remat policy, which is pure overhead for inference.
try:
  from maxtext.integration.vllm.maxtext_vllm_adapter import adapter  # pytype: disable=import-error

  _orig_generate_maxtext_config = adapter.generate_maxtext_config

  def _generate_maxtext_config_with_no_remat(vllm_config_param):
    if "maxtext_config" not in vllm_config_param.additional_config:
      vllm_config_param.additional_config["maxtext_config"] = {}
    mc = vllm_config_param.additional_config["maxtext_config"]
    mc["remat_policy"] = "none"
    mc["scan_layers"] = False
    return _orig_generate_maxtext_config(vllm_config_param)

  adapter.generate_maxtext_config = _generate_maxtext_config_with_no_remat
  logger.info("Patched generate_maxtext_config to force remat_policy=none.")
except ImportError as e:
  logger.warning("Could not patch generate_maxtext_config: %s", e)

mapping_config = mappings.MappingConfig()
engine_kwargs = {
    "model": tokenizer_path,
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": VLLM_MAX_NUM_SEQS,
    "max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "enable_prefix_caching": False,
    "async_scheduling": True,
    "kv_cache_metrics": True,
    "disable_log_stats": False,
    "tokenizer": tokenizer_path,
    "hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
    "dtype": "bfloat16",
    "enable_expert_parallel": False,
}

# Must be set here rather than at the call site: `VllmSampler.__call__`
# forwards unknown kwargs via `setattr` and swallows failures. Stop strings
# additionally require `detokenize=True`, which the sampler otherwise
# hardcodes to False.
sampling_kwargs = {
    "stop": STOP_STRINGS,
    "stop_token_ids": STOP_TOKEN_IDS,
    "detokenize": True,
}

vllm_config = VllmConfig(
    mesh=mesh,
    hbm_utilization=VLLM_HBM_UTILIZATION,
    init_with_random_weights=False,
    tpu_backend_type="jax",
    server_mode=True,
    tensor_parallel_size=mesh.shape["tp"],
    data_parallel_size=mesh.shape["fsdp"],
    mapping_config=mapping_config,
    additional_config=additional_config,
    engine_kwargs=engine_kwargs,
    sampling_kwargs=sampling_kwargs,
)

logger.info(
    "Initializing VllmSampler directly with checkpoint from %s ...",
    MODEL_PATH,
)
sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)
logger.info(
    "VllmSampler successfully initialized directly with model weights."
)

# ========================== Model Call ==========================

class PromptTooLongError(ValueError):
  """Raised when a prompt exceeds the model context limit before sampling."""


def _is_prompt_overflow_error(exc: Exception) -> bool:
  message = str(exc)
  return (
      "maximum input length" in message
      or "context length is only" in message
      or "Prompt too long before sampler call" in message
      or "input_tokens" in message
      and "max_model_len" in message
  )


# `eos_tokens` is not a parameter of `VllmSampler.__call__`; it would land in
# **kwargs and be silently dropped by the `setattr(SamplingParams, ...)` path.
# For vLLM, stop tokens are configured via `VllmConfig.sampling_kwargs` instead.
SAMPLER_CALL_KWARGS = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
}

def model_call(
    chat_completions,
    env_unused=None,
    max_generation_steps=None,
    **kwargs,
):
  """Model inference via tunix sampler."""
  max_gen_steps = min(max_generation_steps or MAX_RESPONSE_LENGTH, 4096)
  pair_index = None
  instance_id = "unknown"
  if env_unused is not None:
    pair_index = getattr(env_unused, "extra_kwargs", {}).get("pair_index")
    instance_id = getattr(env_unused, "entry", {}).get("instance_id", "unknown")

  prompt = chat_parser.parse(
      chat_completions,
      add_generation_prompt=True,
      is_first_msg=True,
  )
  prompt_token_count = len(tokenizer.encode(prompt))
  logger.info(
      "[pair=%s instance=%s] model_call start prompt_chars=%d prompt_tokens=%d"
      " max_context_limit=%d",
      pair_index,
      instance_id,
      len(prompt),
      prompt_token_count,
      MAX_CONTEXT_LIMIT,
  )
  if prompt_token_count >= MAX_CONTEXT_LIMIT:
    raise PromptTooLongError(
        "Prompt too long before sampler call:"
        f" prompt_tokens={prompt_token_count},"
        f" max_context_limit={MAX_CONTEXT_LIMIT}"
    )
  t0 = time.time()
  try:
    out = sampler(
        prompt,
        max_generation_steps=max_gen_steps,
        echo=False,
        **SAMPLER_CALL_KWARGS,
    )
    gc.collect()
  except Exception as exc:
    if _is_prompt_overflow_error(exc):
      raise PromptTooLongError(str(exc)) from exc
    raise
  logger.info(
      "[pair=%s instance=%s] model_call end response_chars=%d (%.1fs)",
      pair_index,
      instance_id,
      len(out.text[0]) if out.text else 0,
      time.time() - t0,
  )
  return out


# ========================== Evaluation ==========================


class EvalTrajectoryCollectEngine(
    trajectory_collect_engine.TrajectoryCollectEngine
):
  """Trajectory engine that converts prompt overflows into per-trajectory termination and always grades working tree."""

  async def collect(self, mode: str = "Conversation"):
    try:
      return await super().collect(mode=mode)
    except Exception as exc:
      logger.exception(
          "[pair=%s instance=%s] unexpected fatal error in collect(), returning partial trajectory: %s",
          self.env.extra_kwargs.get("pair_index"),
          self.env.entry.get("instance_id", "unknown"),
          exc,
      )
      try:
        await self._close()
      except Exception:
        pass
      return self.agent.trajectory

  async def _one_step(self) -> bool:
    try:
      return await super()._one_step()
    except PromptTooLongError as exc:
      logger.warning(
          "[pair=%s instance=%s] terminating trajectory due to prompt"
          " overflow: %s",
          self.env.extra_kwargs.get("pair_index"),
          self.env.entry.get("instance_id", "unknown"),
          exc,
      )
      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
      )
      if self.agent.trajectory.steps:
        self.agent.trajectory.steps[-1].done = True
      return True
    except Exception as exc:
      logger.exception(
          "[pair=%s instance=%s] unexpected exception in _one_step, terminating trajectory gracefully: %s",
          self.env.extra_kwargs.get("pair_index"),
          self.env.entry.get("instance_id", "unknown"),
          exc,
      )
      if self.agent.trajectory.steps:
        self.agent.trajectory.steps[-1].done = True
      return True

  async def _append_final_reward(self):
    pair_index = self.env.extra_kwargs.get("pair_index")
    instance_id = self.env.entry.get("instance_id", "unknown")
    logger.info(
        "[pair=%s instance=%s] final_reward_fn start (steps=%d status=%s)",
        pair_index,
        instance_id,
        len(self.agent.trajectory.steps),
        getattr(self.agent.trajectory, "status", "UNKNOWN"),
    )
    t0 = time.time()
    await super()._append_final_reward()
    last_step = self.agent.get_current_step()
    rew = last_step.reward if last_step is not None else 0.0
    logger.info(
        "[pair=%s instance=%s] final_reward_fn end reward=%.1f (%.1fs)",
        pair_index,
        instance_id,
        rew,
        time.time() - t0,
    )


class _EvalLoggingEnvMixin:
  """Adds phase-level reset/step logs for eval debugging."""

  def reset(self):
    pair_index = self.extra_kwargs.get("pair_index")
    instance_id = self.entry.get("instance_id", "unknown")
    logger.info("[pair=%s instance=%s] reset start", pair_index, instance_id)
    t0 = time.time()
    obs, info = super().reset()
    logger.info(
        "[pair=%s instance=%s] reset end (%.1fs)",
        pair_index,
        instance_id,
        time.time() - t0,
    )
    return obs, info

  def step(self, action):
    pair_index = self.extra_kwargs.get("pair_index")
    instance_id = self.entry.get("instance_id", "unknown")
    step_idx = self.step_count + 1
    action_name = action
    if isinstance(action, str):
      action_name = action.split("\n", 1)[0][:120]
    logger.info(
        "[pair=%s instance=%s] env.step start step=%s action=%s",
        pair_index,
        instance_id,
        step_idx,
        action_name,
    )
    t0 = time.time()
    obs, reward, done, info = super().step(action)
    has_valid_fn = bool(
        action
        and isinstance(action, str)
        and "<function=" in action
        and not action.startswith("<function=>")
    )
    if not has_valid_fn and not done:
      obs = (
          "[ACTION GUARD] Your previous response did not include a valid function call. "
          "You must output exactly one tool call in the required XML format. For example:\n"
          "<function=execute_bash>\n"
          "<parameter=command>git status</parameter>\n"
          "</function>\n"
          "Do NOT call submit until you have modified code and verified the fix."
      )
    elif not obs and not done:
      obs = "(Command executed successfully with no output.)"
    if isinstance(obs, str) and len(obs) > 12000:
      obs = obs[:6000] + "\n...<response clipped>...\n" + obs[-6000:]
    logger.info(
        "[pair=%s instance=%s] env.step end step=%s reward=%.1f done=%s"
        " (%.1fs)",
        pair_index,
        instance_id,
        step_idx,
        reward,
        done,
        time.time() - t0,
    )
    return obs, reward, done, info


class LoggedSWEEnv(_EvalLoggingEnvMixin, SWEEnv):
  pass


class LoggedGuardedSWEEnv(_EvalLoggingEnvMixin, GuardedSWEEnv):
  pass


def pairs_generator():
  """Yield NUM_ROLLOUTS_PER_INSTANCE trajectory tasks per dataset entry."""
  for pair_index in range(len(entries) * NUM_ROLLOUTS_PER_INSTANCE):
    entry = entries[pair_index // NUM_ROLLOUTS_PER_INSTANCE]
    agent = SWEAgent(scaffold=SCAFFOLD)
    env_cls = LoggedGuardedSWEEnv if ENABLE_GUARD else LoggedSWEEnv
    env = env_cls(
        entry=entry,
        max_steps=MAX_STEPS,
        pair_index=pair_index,
        group_id=pair_index,
        scaffold=SCAFFOLD,
        step_timeout=STEP_TIMEOUT_SECS,
        reward_timeout=REWARD_TIMEOUT_SECS,
        use_agent_sandbox=USE_AGENT_SANDBOX,
        fleet=fleet,
    )
    yield agent, env


async def run_evaluation():
  """Run evaluation with orchestrator-managed task-level parallelism."""
  if not OUTPUT_DIR.startswith("gs://"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

  loop = asyncio.get_running_loop()
  executor = concurrent.futures.ThreadPoolExecutor(
      max_workers=max(MAX_CONCURRENT, 32),
      thread_name_prefix="model_call_worker",
  )
  loop.set_default_executor(executor)

  orchestrator = RolloutOrchestrator(
      engine_cls=EvalTrajectoryCollectEngine,
      engine_kwargs=dict(
          model_call=model_call,
          timeout=TIMEOUT,
          max_response_length=MAX_RESPONSE_LENGTH,
          tokenizer=tokenizer_for_agentic,
          chat_parser=chat_parser,
      ),
      max_concurrency=MAX_CONCURRENT,
      rollout_sync_lock=agentic_utils.RolloutSyncLock(),
  )

  results = []
  start_time = time.time()

  producer = asyncio.create_task(
      orchestrator.run_producers_from_stream(
          pairs_stream=pairs_generator(),
          group_size=1,
          group_key_fn=lambda i, env, traj: env.extra_kwargs["group_id"],
          collect_mode="Trajectory",
      )
  )

  await asyncio.sleep(0)

  async for batch in orchestrator.yield_batches(batch_size=1):
    for item in batch:
      traj = item.traj
      entry_index = item.group_index // NUM_ROLLOUTS_PER_INSTANCE
      entry = entries[entry_index]
      guard_reasons = sorted({
          (getattr(step, "info", {}) or {}).get("guard_reason", "unknown")
          for step in traj.steps
          if (getattr(step, "info", {}) or {}).get("guard_blocked")
      })
      step_actions = [
          getattr(step, "action", "").split("\n", 1)[0][:80]
          for step in traj.steps
      ]
      result = {
          "pair_index": item.group_index,
          "entry_index": entry_index,
          "instance_id": entry.get("instance_id", entry_index),
          "reward": float(traj.reward),
          "num_steps": len(traj.steps),
          "status": getattr(traj.status, "name", str(traj.status)),
          "guard_blocked_steps": sum(
              1
              for step in traj.steps
              if (getattr(step, "info", {}) or {}).get("guard_blocked")
          ),
          "guard_reasons": guard_reasons,
          "step_actions": step_actions,
      }
      results.append(result)
      elapsed = time.time() - start_time
      logger.info(
          "[%d/%d] Instance %s: reward=%.1f, steps=%d, status=%s (%.0fs"
          " elapsed)",
          len(results),
          len(entries) * NUM_ROLLOUTS_PER_INSTANCE,
          result["instance_id"],
          result["reward"],
          result["num_steps"],
          result["status"],
          elapsed,
      )
      logger.info(
          "%s[%s] FINAL TRAJECTORY REWARD=%.1f%s",
          ANSI_RED,
          result["instance_id"],
          result["reward"],
          ANSI_RESET,
      )

  try:
    await producer
    return results
  finally:
    executor.shutdown(wait=False)


# ========================== Results ==========================


def _estimate_pass_at_k(n: int, c: int, k: int):
  """Unbiased estimator for pass@k given n samples and c correct."""
  if n < k:
    return None
  if n - c < k:
    return 1.0
  return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k(results):
  """Computes and logs evaluation metrics such as Pass@k (k=1, 4, 5) and average reward."""
  total = len(results)
  if total == 0:
    logger.warning("No results to evaluate.")
    return

  correct = sum(1 for r in results if r["reward"] > 0)
  total_reward = sum(float(r["reward"]) for r in results)
  total_steps = sum(r["num_steps"] for r in results)
  status_counts = Counter(r["status"] for r in results)

  instance_groups = collections.defaultdict(list)
  for r in results:
    instance_groups[r["instance_id"]].append(r)

  pass_at_k_metrics = {}
  for k in (1, 4):
    scores = []
    for inst_results in instance_groups.values():
      n = len(inst_results)
      c = sum(1 for r in inst_results if r["reward"] > 0)
      score = _estimate_pass_at_k(n, c, k)
      if score is not None:
        scores.append(score)
    pass_at_k_metrics[k] = sum(scores) / len(scores) if scores else None

  guard_blocked_trajectories = sum(
      1 for r in results if r["guard_blocked_steps"] > 0
  )
  total_guard_blocks = sum(r["guard_blocked_steps"] for r in results)
  guard_reason_counts = Counter()
  for r in results:
    for reason in r["guard_reasons"]:
      guard_reason_counts[reason] += 1

  avg_reward = total_reward / total
  avg_steps = total_steps / total

  logger.info("=" * 50)
  logger.info("Evaluation Results")
  logger.info("=" * 50)
  logger.info("Total instances:  %d", total)
  logger.info("Resolved:         %d", correct)
  logger.info(
      "Pass@1:           %.4f",
      pass_at_k_metrics[1]
      if pass_at_k_metrics[1] is not None
      else correct / total,
  )
  if pass_at_k_metrics[4] is not None:
    logger.info("Pass@4:           %.4f", pass_at_k_metrics[4])
  else:
    logger.info("Pass@4:           N/A")
  logger.info("Avg reward:       %.4f", avg_reward)
  logger.info("Avg steps:        %.2f", avg_steps)
  logger.info("Status counts:    %s", dict(status_counts))
  logger.info(
      "Guarded trajs:    %d/%d (%.2f%%)",
      guard_blocked_trajectories,
      total,
      100.0 * guard_blocked_trajectories / total,
  )
  logger.info("Guard blocks:     %d", total_guard_blocks)
  if guard_reason_counts:
    logger.info("Guard reasons:    %s", dict(guard_reason_counts))
  logger.info("=" * 50)


def save_results(results):
  """Saves the evaluation results to a JSONL file and uploads to GCS if needed."""
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  filename = f"eval_{MODEL_VERSION.replace('/', '_')}_{timestamp}.jsonl"

  local_dir = (
      "/tmp/eval_results" if OUTPUT_DIR.startswith("gs://") else OUTPUT_DIR
  )
  os.makedirs(local_dir, exist_ok=True)
  output_file = os.path.join(local_dir, filename)

  with open(output_file, "w") as f:
    for r in results:
      entry = entries[r["entry_index"]]
      record = {
          "pair_index": r.get("pair_index", -1),
          "instance_id": entry.get("instance_id", r["instance_id"]),
          "docker_image": entry.get("docker_image", ""),
          "reward": r["reward"],
          "num_steps": r["num_steps"],
          "status": r["status"],
          "guard_blocked_steps": r["guard_blocked_steps"],
          "guard_reasons": r["guard_reasons"],
          "step_actions": r.get("step_actions", []),
      }
      f.write(json.dumps(record) + "\n")

  logger.info("Results saved to %s", output_file)

  if OUTPUT_DIR.startswith("gs://"):
    from google.cloud import storage

    gcs_path = OUTPUT_DIR[5:]
    bucket_name, *prefix_parts = gcs_path.split("/")
    blob_prefix = "/".join(prefix_parts)
    blob_name = (
        os.path.join(blob_prefix, filename) if blob_prefix else filename
    )
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(output_file)
    logger.info("Uploaded results to gs://%s/%s", bucket_name, blob_name)

  return output_file


# ========================== Main ==========================

if __name__ == "__main__":
  logger.info(
      "Starting deepscaler-style evaluation: model=%s (source=%s), %d"
      " instances, max_concurrent=%d, max_steps=%d, engine=%s, mesh=(%d, %d)",
      MODEL_VERSION,
      MODEL_SOURCE,
      len(entries),
      MAX_CONCURRENT,
      MAX_STEPS,
      ROLLOUT_ENGINE,
      MESH_FSDP,
      MESH_TP,
  )

  try:
    eval_results = asyncio.run(run_evaluation())
    compute_pass_at_k(eval_results)
    save_results(eval_results)
  finally:
    if USE_AGENT_SANDBOX and fleet is not None:
      logger.info(
          "[Main] Explicitly tearing down SandboxFleet on clean exit..."
      )
      fleet.teardown()

