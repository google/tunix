#!/usr/bin/env python
"""DeepSWE evaluation with deepscaler-style task-level parallelism.

This script runs SWE evaluation trajectories and uses RolloutOrchestrator to
parallelize tasks across a TPU cluster. It supports both HuggingFace and MaxText
models (e.g. Qwen3.5-35B-A3B) and configurable JAX/vLLM sharding meshes.
"""

import argparse
import asyncio
import base64
import collections
import concurrent.futures
import gc
import json
import logging
import math
import os
import re
import shlex
import sys
import threading
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


def _early_model_source() -> str:
  """Reads --model_source before argparse runs.

  `VLLM_TPU_RPA_VERSION` and `DISABLE_MOSAIC_ATTN` are read at import time by
  the vLLM TPU backend, so they must be set before JAX is imported -- which is
  well before the argument parser is built.
  """
  for i, arg in enumerate(sys.argv):
    if arg == "--model_source" and i + 1 < len(sys.argv):
      return sys.argv[i + 1]
    if arg.startswith("--model_source="):
      return arg.split("=", 1)[1]
  return os.getenv("MODEL_SOURCE", "huggingface")


if _early_model_source() == "maxtext":
  os.environ["VLLM_TPU_RPA_VERSION"] = "2"
  os.environ["DISABLE_MOSAIC_ATTN"] = "1"

if "proxy" in os.getenv("JAX_PLATFORMS", ""):
  try:
    import pathwaysutils

    pathwaysutils.initialize()
    print("Pathways initialized successfully before JAX import.")
  except Exception as e:
    print(f"Failed to initialize pathwaysutils: {e}")

from datasets import load_dataset
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from kubernetes import client
from kubernetes import config as k8s_config
import numpy as np
from transformers import AutoTokenizer

try:
  import swe_env
  from guarded_swe_env import GuardedSWEEnv
  from swe_agent import SWEAgent
  from swe_env import SWEEnv
except ImportError:
  from examples.deepswe import swe_env  # pytype: disable=import-error
  from examples.deepswe.guarded_swe_env import GuardedSWEEnv  # pytype: disable=import-error
  from examples.deepswe.swe_agent import SWEAgent  # pytype: disable=import-error
  from examples.deepswe.swe_env import SWEEnv  # pytype: disable=import-error

from huggingface_hub import snapshot_download
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.models.automodel import AutoModel, ModelSource
from tunix.models.qwen3 import model as model_lib
from tunix.models.qwen3 import params as params_lib
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.sft import utils as sft_utils

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
    "--model_source",
    type=str,
    default=os.getenv("MODEL_SOURCE", "huggingface"),
    choices=["huggingface", "maxtext"],
    help="Model source (huggingface or maxtext)",
)
parser_cli.add_argument(
    "--model_absolute_path",
    type=str,
    default=os.getenv("MODEL_ABSOLUTE_PATH", os.getenv("MODEL_PATH", None)),
    help="Absolute model path (GCS bucket or local directory)",
)
parser_cli.add_argument(
    "--dataset_name",
    type=str,
    default=os.getenv("DATASET_NAME", "R2E-Gym/SWE-Bench-Verified"),
    help="Dataset name or path",
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
    "--dataset_num_proc",
    type=int,
    default=int(os.getenv("DATASET_NUM_PROC", "32")),
    help="Number of processes for dataset loading (0 or 1 for single-process)",
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
    choices=["vllm", "vanilla", "sglang_jax"],
    help="Rollout engine",
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
    "--vllm_server_mode",
    type=str2bool,
    default=os.getenv("VLLM_SERVER_MODE", "true").lower() == "true",
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
    "--vllm_reshard_chunk_size",
    type=int,
    default=int(os.getenv("VLLM_RESHARD_CHUNK_SIZE", "30")),
    help="Reshard chunk size for vLLM weight sync",
)
parser_cli.add_argument(
    "--vllm_init_random_weights",
    type=str2bool,
    default=os.getenv("VLLM_INIT_RANDOM_WEIGHTS", "false").lower() == "true",
    help="Initialize vLLM with random weights and sync real weights chunked",
)
parser_cli.add_argument(
    "--sglang_mem_fraction_static",
    type=float,
    default=float(os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.4")),
)
parser_cli.add_argument(
    "--sglang_init_random_weights",
    type=str2bool,
    default=os.getenv("SGLANG_INIT_RANDOM_WEIGHTS", "true").lower() == "true",
)
parser_cli.add_argument(
    "--sglang_max_running_requests",
    type=int,
    default=int(os.getenv("SGLANG_MAX_RUNNING_REQUESTS", "1")),
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
    default=os.getenv("ENABLE_PREFIX_CACHING", "true").lower() == "true",
    help="Whether to enable vLLM prefix caching",
)
parser_cli.add_argument(
    "--patch_qwen_mrope",
    type=str2bool,
    default=os.getenv("PATCH_QWEN_MROPE", "true").lower() == "true",
    help=(
        "Whether to patch Qwen3OmniMoeThinkerTextRotaryEmbedding for 3D MRoPE"
        " position normalization. Training does not apply this patch."
    ),
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

DATASET_NAME = args.dataset_name
DATASET_SPLIT = args.dataset_split
DATASET_CACHE = args.dataset_cache
DATASET_NUM_PROC = args.dataset_num_proc

MODEL_VERSION = args.model_version
MODEL_SOURCE = args.model_source
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
VLLM_SERVER_MODE = args.vllm_server_mode
VLLM_MAX_NUM_SEQS = args.vllm_max_num_seqs
VLLM_MAX_BATCHED_TOKENS = args.vllm_max_batched_tokens
VLLM_RESHARD_CHUNK_SIZE = args.vllm_reshard_chunk_size
VLLM_INIT_RANDOM_WEIGHTS = args.vllm_init_random_weights or (
    args.vllm_reshard_chunk_size > 0
)

SGLANG_MEM_FRACTION_STATIC = args.sglang_mem_fraction_static
SGLANG_INIT_RANDOM_WEIGHTS = args.sglang_init_random_weights
SGLANG_MAX_RUNNING_REQUESTS = args.sglang_max_running_requests

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
PATCH_QWEN_MROPE = args.patch_qwen_mrope
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

# Register MaxText vLLM adapter if using a MaxText model
if MODEL_SOURCE == "maxtext":
  try:
    from maxtext.integration.vllm import maxtext_vllm_adapter  # pytype: disable=import-error

    maxtext_vllm_adapter.register()
    logger.info("Successfully registered MaxTextForCausalLM model with vLLM.")
  except ImportError as e:
    logger.warning("Could not import maxtext_vllm_adapter: %s", e)

  # Not applied by train_maxtext_nb.py. If the model being evaluated routes
  # through this rotary class, leaving this enabled makes eval diverge from the
  # behavior the model was trained with.
  if PATCH_QWEN_MROPE:
    try:
      import jax.numpy as jnp
      from maxtext.layers.embeddings import Qwen3OmniMoeThinkerTextRotaryEmbedding

      _orig_qwen_mrope_call = Qwen3OmniMoeThinkerTextRotaryEmbedding.__call__

      def _patched_qwen_mrope_call(self, inputs, position, *args, **kwargs):
        if position is not None:
          # Case 1: vLLM (3, N, 1) -> (N, 1, 3)
          if (
              position.ndim == 3
              and position.shape[0] == 3
              and position.shape[-1] == 1
          ):
            position = jnp.transpose(position.squeeze(-1), (1, 0))[:, None, :]
          # Case 2: vLLM (3, N) -> (N, 1, 3)
          elif position.ndim == 2 and position.shape[0] == 3:
            position = jnp.transpose(position, (1, 0))[:, None, :]
          # Case 3: (N,) -> (N, 1, 3)
          elif position.ndim == 1:
            position = jnp.broadcast_to(
                position[:, None, None], (position.shape[0], 1, 3)
            )
          # Case 4: (B, S) -> (B, S, 3)
          elif position.ndim == 2:
            position = jnp.broadcast_to(
                position[..., None], position.shape + (3,)
            )
          # Case 5: (B, S, 1) -> (B, S, 3)
          elif position.ndim == 3 and position.shape[-1] == 1:
            position = jnp.broadcast_to(position, position.shape[:-1] + (3,))
        return _orig_qwen_mrope_call(self, inputs, position, *args, **kwargs)

      Qwen3OmniMoeThinkerTextRotaryEmbedding.__call__ = (
          _patched_qwen_mrope_call
      )
      logger.info(
          "Successfully patched Qwen3OmniMoeThinkerTextRotaryEmbedding for 3D"
          " MRoPE position normalization."
      )
    except Exception as e:
      logger.warning(
          "Failed to patch Qwen3OmniMoeThinkerTextRotaryEmbedding: %s", e
      )

# ========================== Dataset ==========================

logger.info("Loading dataset %s split=%s ...", DATASET_NAME, DATASET_SPLIT)
if DATASET_NAME.startswith("gs://") or os.path.isdir(DATASET_NAME):
  try:
    import datasets as datasets_lib
    dataset = datasets_lib.load_from_disk(DATASET_NAME)
    if isinstance(dataset, datasets_lib.DatasetDict):
      dataset = dataset[DATASET_SPLIT]
  except Exception as e:
    logger.warning("load_from_disk failed (%s), falling back to load_dataset...", e)
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        cache_dir=DATASET_CACHE,
    )
elif DATASET_NUM_PROC > 1:
  try:
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        cache_dir=DATASET_CACHE,
        num_proc=DATASET_NUM_PROC,
    )
  except Exception as e:
    logger.warning(
        "load_dataset failed with num_proc=%d (%s), retrying without"
        " num_proc...",
        DATASET_NUM_PROC,
        e,
    )
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        cache_dir=DATASET_CACHE,
    )
else:
  dataset = load_dataset(
      DATASET_NAME,
      split=DATASET_SPLIT,
      cache_dir=DATASET_CACHE,
  )


def _normalize_entry(entry):
  """Prepares a raw dataset row for SWEEnv.

  `SWEEnv._unpack_entry` rejects list values of length != 1, and R2E-Gym rows
  carry several list-valued columns, so those are JSON-encoded the same way the
  training script does.
  """
  normalized = {
      k: json.dumps(v) if isinstance(v, list) else v for k, v in entry.items()
  }
  if "instance_id" not in normalized or not normalized["instance_id"]:
    normalized["instance_id"] = (
        entry.get("commit_hash")
        or (f"{entry.get('repo_name', 'repo')}__{entry.get('commit_hash', '')[:8]}"
            if entry.get("commit_hash") else None)
    )
  if DOCKER_IMAGE_PREFIX and normalized.get("docker_image"):
    # If image is from swebench-verified and prefix does not target swebench, keep original
    if "swebench-verified" in normalized["docker_image"] and "swebench-verified" not in DOCKER_IMAGE_PREFIX:
      pass
    else:
      # e.g. 'namanjain12/pandas_final:tag' -> '<prefix>/pandas_final:tag'
      image_name = normalized["docker_image"].split("/")[-1]
      normalized["docker_image"] = (
          f"{DOCKER_IMAGE_PREFIX.rstrip('/')}/{image_name}"
      )
  return normalized


entries = [_normalize_entry(e) for e in dataset if "docker_image" in e]
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


def patch_kubernetes_runtime():
  """Monkeypatch r2egym DockerRuntime to dynamically configure Kubernetes nodeSelector."""
  try:
    from r2egym.agenthub.runtime.docker import DockerRuntime  # pytype: disable=import-error

    original_start_kubernetes_pod = DockerRuntime._start_kubernetes_pod

    def patched_start_kubernetes_pod(
        self, docker_image, command, pod_name, **docker_kwargs
    ):
      original_create_namespaced_pod = self.client.create_namespaced_pod

      def patched_create_namespaced_pod(*a, **kw):
        body = kw.get("body")
        if body and "spec" in body:
          key = os.environ.get(
              "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
          )
          val = os.environ.get("NODE_SELECTOR_VAL", NODE_SELECTOR_VAL)
          body["spec"]["nodeSelector"] = {key: val}
          logger.info("[Monkeypatch] Overrode nodeSelector to %s=%s", key, val)
        return original_create_namespaced_pod(*a, **kw)

      self.client.create_namespaced_pod = patched_create_namespaced_pod
      try:
        return original_start_kubernetes_pod(
            self, docker_image, command, pod_name, **docker_kwargs
        )
      finally:
        self.client.create_namespaced_pod = original_create_namespaced_pod

    DockerRuntime._start_kubernetes_pod = patched_start_kubernetes_pod
    logger.info(
        "[Monkeypatch] Successfully patched DockerRuntime._start_kubernetes_pod"
    )
  except Exception as ex:
    logger.warning("[Monkeypatch] Note on patching DockerRuntime: %s", ex)


patch_kubernetes_runtime()

try:
  if os.getenv("KUBERNETES_SERVICE_HOST"):
    k8s_config.load_incluster_config()
  else:
    k8s_config.load_kube_config()
  k8s_client = client.CoreV1Api()
  logger.info("Kubernetes connection verified.")
except Exception as e:
  logger.warning("Kubernetes config loading note: %s", e)

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

  # Resilient acquire: self-heal if another job deletes warmpools in shared namespace
  orig_acquire = fleet.acquire

  def resilient_acquire(task):
    for attempt in range(10):
      try:
        return orig_acquire(task)
      except Exception as exc:
        exc_str = str(exc)
        if "SandboxWarmPool" in exc_str or "not found" in exc_str.lower():
          logger.warning(
              "[ResilientAcquire] Warmpool missing for %s (attempt %d/10): %s; recreating...",
              task.image,
              attempt + 1,
              exc,
          )
          try:
            entry = fleet.plan_.for_image(task.image) if fleet.plan_ else None
            cluster = fleet.registry.get(entry.cluster if entry else "default")
            fleet._ensure_pool(cluster, task.image, 1)
          except Exception as ce:
            logger.warning("[ResilientAcquire] Pool recreation error: %s", ce)
          time.sleep(5)
        else:
          raise
    return orig_acquire(task)

  fleet.acquire = resilient_acquire

  def safe_teardown(*args, **kwargs):
    logger.info(
        "[Main] Safely tearing down ONLY our own warmpools and claims (oh-img-*)..."
    )
    try:
      fleet.release_all()
      for c in fleet.registry:
        try:
          pools = c.resources.list_warmpools(label_selector=None)
          for p in pools:
            if p.startswith("pool-oh-img-"):
              try:
                c.resources.delete_warmpool(p)
              except Exception:
                pass
        except Exception as pe:
          logger.warning("Error cleaning up warmpools: %s", pe)
        try:
          tmpls = c.resources.list_templates(label_selector=None)
          for t in tmpls:
            if t.startswith("oh-img-"):
              try:
                c.resources.delete_template(t)
              except Exception:
                pass
        except Exception as te:
          logger.warning("Error cleaning up templates: %s", te)
    except Exception as e:
      logger.warning("Error during safe teardown: %s", e)

  fleet.teardown = safe_teardown
  fleet._teardown = safe_teardown

# ========================== Model & Mesh ==========================

# Tokenizer Setup
tokenizer_path = MODEL_PATH
local_files_only = True
if MODEL_SOURCE == "maxtext" and MODEL_PATH.startswith("gs://"):
  if MODEL_VERSION.startswith("Qwen/"):
    tokenizer_path = MODEL_VERSION
  else:
    tokenizer_path = f"Qwen/{MODEL_VERSION}"
  local_files_only = False
  logger.info("Loading tokenizer from HF Hub: %s", tokenizer_path)
else:
  if not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_VERSION,
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
    )
  logger.info("Loading tokenizer from local path: %s", tokenizer_path)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, local_files_only=local_files_only, trust_remote_code=True
)
tokenizer_for_agentic = tok_adapter.TokenizerAdapter(tokenizer)
class FixedQwenChatTemplateParser(parser.QwenChatTemplateParser):
  """Fixes Qwen3/3.5 chat template formatting to match official HuggingFace chat_template:
  1. Appends '<think>\n' to generation prompt when enable_thinking=True.
  2. Preserves '<think>...</think>' blocks in multi-step tool-use turns (after last user query).
  3. Wraps environment observations (idx >= 2) in '<tool_response>...</tool_response>'.
  4. Strips trailing '<|im_end|>' from sampler outputs to prevent double '<|im_end|>' tokens.
  """

  def _init_generation_prompt(self) -> str:
    if self.enable_thinking:
      return "<|im_start|>assistant\n<think>\n"
    return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

  def parse(
      self,
      messages,
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    if len(messages) > 1:
      norm_messages = []
      for idx, m in enumerate(messages):
        role = m.get("role", "user")
        content = str(m.get("content", "")).strip()
        if role == "assistant":
          while content.endswith("<|im_end|>") or content.endswith("<|endoftext|>"):
            if content.endswith("<|im_end|>"):
              content = content[:-len("<|im_end|>")].rstrip()
            elif content.endswith("<|endoftext|>"):
              content = content[:-len("<|endoftext|>")].rstrip()
          if "</think>" in content:
            if not content.lstrip().startswith("<think>"):
              content = "<think>\n" + content.lstrip()
          else:
            if "<function=" in content:
              f_idx = content.find("<function=")
              thought = content[:f_idx].strip()
              if thought.startswith("<think>"):
                thought = thought[len("<think>"):].strip()
              func_part = content[f_idx:].strip()
              content = f"<think>\n{thought}\n</think>\n\n{func_part}"
            else:
              thought = content
              if thought.startswith("<think>"):
                thought = thought[len("<think>"):].strip()
              content = f"<think>\n{thought}\n</think>\n\n"
        elif role == "user" and idx >= 2:
          if not content.startswith("<tool_response>"):
            content = f"<tool_response>\n{content}\n</tool_response>"
        norm_messages.append({"role": role, "content": content})
      return self.tokenizer.apply_chat_template(
          norm_messages, tokenize=False, add_generation_prompt=add_generation_prompt
      )
    # Single message case (used by tokenize_and_generate_masks)
    msg = messages[0]
    role = msg.get("role", "user")
    content = str(msg.get("content", "")).strip()
    if role == "system":
      res = f"<|im_start|>system\n{content}<|im_end|>"
    elif role == "user":
      res = f"<|im_start|>user\n{content}<|im_end|>"
    elif role == "assistant":
      while content.endswith("<|im_end|>") or content.endswith("<|endoftext|>"):
        if content.endswith("<|im_end|>"):
          content = content[:-len("<|im_end|>")].rstrip()
        elif content.endswith("<|endoftext|>"):
          content = content[:-len("<|endoftext|>")].rstrip()
      res = f"<|im_start|>assistant\n{content}<|im_end|>"
    else:
      res = f"<|im_start|>{role}\n{content}<|im_end|>"
    if add_generation_prompt:
      res += "\n" + self._init_generation_prompt()
    if not is_first_msg:
      res = "\n" + res
    return res


chat_parser = FixedQwenChatTemplateParser(tokenizer)
qwen_eos_tokens = [tokenizer.encode("<|im_end|>")[0]]

# Install resilient XML action/parameter parsing and OpenHands step patches
try:
  from r2egym.agenthub.action import Action as _R2EAction

  def _resilient_to_xml_string(self) -> str:
    if not getattr(self, "function_name", ""):
      return ""
    xml_str = f"<function={self.function_name}>\n"
    for param_key, param_value in (self.parameters or {}).items():
      xml_str += f"  <parameter={param_key}>{param_value}</parameter>\n"
    xml_str += "</function>"
    return xml_str

  def _resilient_to_bashcmd(self) -> str:
    if not getattr(self, "function_name", ""):
      return ""
    elif self.function_name in ("finish", "submit"):
      return "echo '<<<Finished>>>'"
    cmd_parts = [shlex.quote(self.function_name)]
    base_command = (self.parameters or {}).get("command")
    if base_command is not None:
      cmd_parts.append(shlex.quote(str(base_command)))
    for param_key, param_value in (self.parameters or {}).items():
      if param_key == "command":
        continue
      param_value_quoted = shlex.quote(str(param_value))
      cmd_parts.append(f"--{param_key}={param_value_quoted}")
    return " ".join(cmd_parts)

  @classmethod
  def _resilient_action_from_string(cls, action_str: str):
    if not action_str or not isinstance(action_str, str):
      return cls("", {})

    fn = ""
    default_cmd = None
    fn_match = re.search(r"<function\s*=\s*([^\s>]+)", action_str)
    if fn_match:
      fn = fn_match.group(1).strip()
    elif "<tool_call>" in action_str:
      try:
        tc_body = action_str.split("<tool_call>", 1)[1].split("</tool_call>")[0].strip()
        tc_json = json.loads(tc_body)
        fn = tc_json.get("name", "")
        params = tc_json.get("arguments", tc_json.get("parameters", {}))
        if isinstance(params, str):
          params = json.loads(params)
        if isinstance(params, dict):
          return cls(fn, {str(k): str(v) for k, v in params.items()})
      except Exception:
        pass

    fn = re.sub(r"</?function.*$", "", fn, flags=re.IGNORECASE)
    fn = re.sub(r"</?tool_call.*$", "", fn, flags=re.IGNORECASE)
    fn = fn.strip('<>\x22\x27 `/\\')
    fn_lower = fn.lower()

    if "submit" in fn_lower or "finish" in fn_lower:
      fn = "submit"
    elif any(k in fn_lower for k in ("str", "replace", "edit", "view", "create", "insert", "undo")):
      if fn_lower in ("view", "create", "insert", "str_replace", "undo_edit"):
        default_cmd = fn_lower
      fn = "str_replace_editor"
    elif any(k in fn_lower for k in ("bash", "exec", "cmd", "run", "shell")):
      fn = "execute_bash"
    else:
      fn = ""

    if not fn:
      return cls("", {})
    if fn == "submit":
      return cls("submit", {})

    params = {}
    code_keys = {"old_str", "new_str", "file_text"}

    def _clean_val(k: str, v: str) -> str:
      if k in code_keys:
        v = re.sub(r"^\r?\n", "", v)
        v = re.sub(r"\r?\n[ \t]*$", "", v)
        return v
      return v.strip()

    param_starts = list(re.finditer(r"<parameter\s*=\s*([^>]+)>", action_str))
    for idx, m in enumerate(param_starts):
      key = m.group(1).strip().strip("\"'")
      val_start = m.end()
      next_tag_start = param_starts[idx + 1].start() if idx + 1 < len(param_starts) else len(action_str)
      segment = action_str[val_start:next_tag_start]
      if "</parameter>" in segment:
        val = segment.split("</parameter>", 1)[0]
      else:
        val = re.sub(r"</function>.*$", "", segment, flags=re.DOTALL)
      if key and key not in params:
        params[key] = _clean_val(key, val)

    if not params:
      for m in re.finditer(
          r"<parameter\s+name=[\"']?([^\"'>]+)[\"']?>(.*?)(?=</parameter>|<parameter|</function>|$)",
          action_str,
          flags=re.DOTALL,
      ):
        key = m.group(1).strip()
        val = m.group(2)
        if key and key not in params:
          params[key] = _clean_val(key, val)

    for tag in ("command", "cmd", "path", "file_path", "file_text", "old_str", "new_str", "insert_line", "view_range"):
      if tag not in params:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", action_str, flags=re.DOTALL)
        if m:
          params[tag] = _clean_val(tag, m.group(1))

    if fn == "str_replace_editor":
      if "path" not in params:
        for alt in ("file_path", "filepath", "file", "target_file"):
          if alt in params:
            params["path"] = params.pop(alt)
            break
      if "path" in params:
        params["path"] = re.sub(r"<.*$", "", params["path"]).strip().strip("\"'")
      if "command" not in params:
        if default_cmd:
          params["command"] = default_cmd
        elif "old_str" in params:
          params["command"] = "str_replace"
        elif "file_text" in params:
          params["command"] = "create"
        elif "insert_line" in params:
          params["command"] = "insert"
        elif "path" in params:
          params["command"] = "view"
      if not params.get("command") or not params.get("path"):
        return cls("", {})

    elif fn == "execute_bash":
      if "command" not in params and "cmd" in params:
        params["command"] = params.pop("cmd")
      if "command" not in params:
        body = re.sub(r"^.*?<function[^>]*>", "", action_str, flags=re.DOTALL)
        body = re.sub(r"</function>.*$", "", body, flags=re.DOTALL)
        body = re.sub(r"</?parameter[^>]*>", "", body).strip()
        if body:
          params["command"] = body
      if not params.get("command"):
        return cls("", {})

    return cls(fn, params)

  _R2EAction.from_string = _resilient_action_from_string
  _R2EAction.to_xml_string = _resilient_to_xml_string
  _R2EAction.to_bashcmd = _resilient_to_bashcmd
  logger.info("Installed resilient Action methods (from_string, to_xml_string, to_bashcmd).")
except Exception as e:
  logger.warning("Could not install resilient Action methods: %s", e)

try:
  def _resilient_parse_xml_response(response_text: str):
    if not response_text or not isinstance(response_text, str):
      return "", _R2EAction("", {})
    if "<function=" in response_text:
      idx = response_text.find("<function=")
      thought = response_text[:idx].strip()
      action_block = response_text[idx:].strip()
    elif "<tool_call>" in response_text:
      idx = response_text.find("<tool_call>")
      thought = response_text[:idx].strip()
      action_block = response_text[idx:].strip()
    else:
      thought = response_text.strip()
      action_block = ""
    action_obj = _R2EAction.from_string(action_block)
    return thought, action_obj

  SWEAgent.update_from_model.__globals__["parse_xml_response"] = _resilient_parse_xml_response
  try:
    import swe_agent as _swe_agent_mod
    _swe_agent_mod.parse_xml_response = _resilient_parse_xml_response
  except Exception:
    pass
  try:
    from examples.deepswe import swe_agent as _swe_agent_mod2
    _swe_agent_mod2.parse_xml_response = _resilient_parse_xml_response
  except Exception:
    pass
  logger.info("Installed resilient parse_xml_response across SWEAgent modules.")
except Exception as e:
  logger.warning("Could not install resilient parse_xml_response: %s", e)

try:
  from tunix.rl.agentic.environments.base_environment import EnvStepResult as _EnvStepResult

  def _strip_cat_n_line_numbers(text: str) -> str:
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
      return text
    if all(re.match(r"^[ \t]*\d+[\t ]", l) for l in non_empty):
      return "\n".join(re.sub(r"^[ \t]*\d+[\t ]?", "", l) for l in lines)
    return text

  def _try_fallback_str_replace(file_content: str, old_str: str, new_str: str):
    clean_old = _strip_cat_n_line_numbers(old_str)
    clean_new = _strip_cat_n_line_numbers(new_str)
    if clean_old != old_str and file_content.count(clean_old) == 1:
      return True, file_content.replace(clean_old, clean_new, 1)

    old_lines = clean_old.splitlines()
    new_lines = clean_new.splitlines()
    while old_lines and not old_lines[0].strip():
      old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
      old_lines.pop()
    if not old_lines:
      return False, file_content

    file_lines = file_content.splitlines()
    n_old = len(old_lines)
    matches = []
    for i in range(len(file_lines) - n_old + 1):
      window = file_lines[i : i + n_old]
      if all(w.strip() == o.strip() for w, o in zip(window, old_lines)):
        matches.append(i)

    if len(matches) != 1:
      return False, file_content

    match_idx = matches[0]
    matched_window = file_lines[match_idx : match_idx + n_old]

    def _get_indent(s: str) -> int:
      return len(s) - len(s.lstrip(" "))

    file_indent_0 = _get_indent(matched_window[0])
    old_indent_0 = _get_indent(old_lines[0])
    new_indent_0 = _get_indent(new_lines[0]) if new_lines and new_lines[0].strip() else 0

    if n_old > 1:
      deltas = [
          _get_indent(w) - _get_indent(o)
          for w, o in zip(matched_window[1:], old_lines[1:])
          if w.strip() and o.strip()
      ]
      common_delta = deltas[0] if deltas else (file_indent_0 - old_indent_0)
    else:
      common_delta = file_indent_0 - old_indent_0

    adjusted_new_lines = []
    for idx, nl in enumerate(new_lines):
      if not nl.strip():
        adjusted_new_lines.append("")
        continue
      cur_indent = _get_indent(nl)
      if idx == 0 and old_indent_0 == 0 and file_indent_0 > 0 and new_indent_0 == 0:
        adjusted_new_lines.append(" " * file_indent_0 + nl.lstrip(" "))
      elif common_delta > 0 and cur_indent + common_delta >= 0:
        adjusted_new_lines.append(" " * (cur_indent + common_delta) + nl.lstrip(" "))
      else:
        adjusted_new_lines.append(nl)

    updated_lines = file_lines[:match_idx] + adjusted_new_lines + file_lines[match_idx + n_old :]
    updated_content = "\n".join(updated_lines)
    if file_content.endswith("\n") and not updated_content.endswith("\n"):
      updated_content += "\n"
    return True, updated_content

  _oh_mod = SWEEnv._step_impl.__globals__["openhands_utils"]
  _orig_step_openhands = _oh_mod.step_openhands

  def _patched_step_openhands(env, action_obj):
    max_steps = getattr(env, "max_steps", None)
    fn = getattr(action_obj, "function_name", "")
    if fn == "execute_bash" and getattr(env, "workspace", None) is not None:
      params = getattr(action_obj, "parameters", {}) or {}
      cmd = params.get("command") or params.get("cmd")
      if not cmd:
        return _EnvStepResult(
            observation="ERROR: No command specified for execute_bash.",
            reward=0,
            done=False,
            info={"max_steps": max_steps},
        )
      try:
        step_timeout = getattr(env, "step_timeout", 30.0)
        wrapped_cmd = f"(cd /testbed 2>/dev/null || cd /workspace) && {cmd}"
        result = env.workspace.execute_command(wrapped_cmd, timeout=float(step_timeout))
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stderr:
          obs = f"{stdout}\n{stderr}".strip() if stdout else stderr.strip()
        else:
          obs = stdout
        if not obs.strip() and result.exit_code == 0:
          obs = "(Command executed successfully with no output.)"
      except Exception as e:
        obs = f"Command execution failed: {e}"
      if hasattr(env, "total_steps"):
        env.total_steps += 1
      return _EnvStepResult(
          observation=obs,
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )

    res = _orig_step_openhands(env, action_obj)

    # Fallback whitespace-tolerant str_replace when exact match fails
    if (
        fn in ("str_replace_editor", "file_editor")
        and (getattr(action_obj, "parameters", {}) or {}).get("command") == "str_replace"
        and getattr(env, "workspace", None) is not None
        and ("ERROR: No occurrences of" in str(res.observation) or "ERROR: Multiple occurrences of" in str(res.observation))
    ):
      params = action_obj.parameters
      path = params.get("path")
      old_str = params.get("old_str")
      new_str = params.get("new_str", "")
      if path and old_str:
        try:
          read_res = env.workspace.execute_command(f"cat {shlex.quote(path)}", timeout=15.0)
          if read_res.exit_code == 0 and read_res.stdout:
            ok, updated_text = _try_fallback_str_replace(read_res.stdout, old_str, new_str)
            if ok:
              b64 = base64.b64encode(updated_text.encode("utf-8")).decode("ascii")
              write_cmd = (
                  f"python3 -c 'import base64, sys; "
                  f"open(sys.argv[1], \"w\").write(base64.b64decode(sys.argv[2]).decode(\"utf-8\"))' "
                  f"{shlex.quote(path)} {b64}"
              )
              if path.endswith(".py"):
                write_cmd += f" && python3 -m py_compile {shlex.quote(path)}"
              w_res = env.workspace.execute_command(write_cmd, timeout=15.0)
              if w_res.exit_code == 0:
                logger.info("Applied fallback str_replace successfully on %s", path)
                res = _EnvStepResult(
                    observation=f"The file {path} has been edited successfully.",
                    reward=0,
                    done=False,
                    info=res.info,
                )
        except Exception as e:
          logger.warning("Fallback str_replace failed on %s: %s", path, e)
    return res

  _oh_mod.step_openhands = _patched_step_openhands
  logger.info("Installed patched step_openhands with stderr preservation & fallback str_replace.")
except Exception as e:
  logger.warning("Could not install patched step_openhands: %s", e)

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
if MODEL_SOURCE == "maxtext":
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

if ROLLOUT_ENGINE == "vllm":
  from tunix.generate import mappings
  from tunix.generate.vllm_sampler import VllmConfig, VllmSampler

  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

  additional_config = None
  if MODEL_SOURCE == "maxtext":
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
    }
    if not VLLM_INIT_RANDOM_WEIGHTS:
      maxtext_cfg["load_parameters_path"] = MODEL_PATH

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
        if getattr(vllm_config_param, "load_config", None) and getattr(vllm_config_param.load_config, "load_format", None) == "dummy":
          mc.pop("load_parameters_path", None)
        return _orig_generate_maxtext_config(vllm_config_param)

      adapter.generate_maxtext_config = _generate_maxtext_config_with_no_remat
      logger.info("Patched generate_maxtext_config to force remat_policy=none.")
    except ImportError as e:
      logger.warning("Could not patch generate_maxtext_config: %s", e)

  mapping_config = mappings.MappingConfig()
  engine_kwargs = {
      "model": tokenizer_path if MODEL_SOURCE == "maxtext" else MODEL_PATH,
      "max_model_len": MAX_MODEL_LEN,
      "max_num_seqs": VLLM_MAX_NUM_SEQS,
      "max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
      "enable_prefix_caching": ENABLE_PREFIX_CACHING,
      "async_scheduling": True,
      "kv_cache_metrics": True,
      "disable_log_stats": False,
      "tokenizer": tokenizer_path,
  }
  if MODEL_SOURCE == "maxtext":
    engine_kwargs["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
    engine_kwargs["dtype"] = "bfloat16"
    engine_kwargs["enable_expert_parallel"] = False

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
      init_with_random_weights=VLLM_INIT_RANDOM_WEIGHTS,
      tpu_backend_type="jax",
      server_mode=VLLM_SERVER_MODE,
      tensor_parallel_size=mesh.shape["tp"],
      data_parallel_size=mesh.shape["fsdp"],
      mapping_config=mapping_config,
      additional_config=additional_config,
      reshard_chunk_size=VLLM_RESHARD_CHUNK_SIZE
      if VLLM_RESHARD_CHUNK_SIZE > 0
      else None,
      engine_kwargs=engine_kwargs,
      sampling_kwargs=sampling_kwargs,
  )

  if VLLM_INIT_RANDOM_WEIGHTS:
    from flax import nnx

    logger.info("Loading base model weights to transfer into VllmSampler...")
    if MODEL_SOURCE == "maxtext":
      logger.info(
          "Loading MaxText model %s from %s (scan_layers=%s)...",
          MODEL_VERSION,
          MODEL_PATH,
          SCAN_LAYERS,
      )
      from maxtext.configs import pyconfig
      from maxtext import model_creation_utils

      base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
      model_name_slug = MODEL_VERSION.lower().split("/")[-1]

      trainer_config = pyconfig.initialize(
          [
              "",
              base_yml,
              "num_slices=1",
              f"model_name={model_name_slug}",
              f"load_parameters_path={MODEL_PATH}",
              f"ici_fsdp_parallelism={MESH_FSDP}",
              f"ici_tensor_parallelism={MESH_TP}",
              f"scan_layers={SCAN_LAYERS}",
              f"max_target_length={MAX_MODEL_LEN}",
              f"max_prefill_predict_length={MAX_PREFILL_LENGTH}",
              "remat_policy=none",
              f"dtype={WEIGHT_DTYPE}",
              f"attention={'flash' if MAXTEXT_ATTENTION == 'flash' else 'dot_product'}",
              f"prefuse_moe_weights={PREFUSE_MOE_WEIGHTS}",
              f"checkpoint_storage_use_ocdbt={CHECKPOINT_STORAGE_USE_OCDBT}",
              f"checkpoint_storage_use_zarr3={CHECKPOINT_STORAGE_USE_ZARR3}",
              f"checkpoint_storage_concurrent_gb={CHECKPOINT_STORAGE_CONCURRENT_GB}",
              f"allow_split_physical_axes={ALLOW_SPLIT_PHYSICAL_AXES}",
              "skip_jax_distributed_system=True",
              "load_checkpoint_only_once=True",
              "use_standalone_converter=False",
              "log_config=False",
          ],
          vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
      )

      try:
        from etils import epath
        from orbax.checkpoint._src.serialization import jax_array_handlers
        from orbax.checkpoint._src.serialization import type_handler_registry

        if (epath.Path(MODEL_PATH) / "manifest.ocdbt").exists():
          type_handler_registry.register_type_handler(
              jax.Array, jax_array_handlers.ArrayHandler(), override=True
          )
          logger.info("Registered standard ArrayHandler for OCDBT checkpoint.")
      except Exception as e:
        logger.warning("Could not register ArrayHandler for OCDBT: %s", e)

      model, _ = model_creation_utils.from_pretrained(
          trainer_config,
          devices=devices[:total_mesh_devices],
          wrap_with_tunix_adapter=True,
          tokenizer_pad_id=tokenizer.pad_token_id,
      )
    elif MODEL_VERSION == "Qwen/Qwen3-4B-Instruct-2507":
      model_config = model_lib.ModelConfig.qwen3_4b_instruct_2507()
      logger.info("Loading model weights from %s ...", MODEL_PATH)
      model = params_lib.create_model_from_safe_tensors(
          MODEL_PATH, model_config, mesh, dtype=jnp.float32
      )
    elif MODEL_VERSION in ("Qwen/Qwen3-32B", "Qwen3-32B"):
      model_config = model_lib.ModelConfig.qwen3_32b()
      logger.info("Loading model weights from %s ...", MODEL_PATH)
      model = params_lib.create_model_from_safe_tensors(
          MODEL_PATH, model_config, mesh, dtype=jnp.float32
      )
    else:
      logger.info(
          "Loading model weights via AutoModel for %s ...", MODEL_VERSION
      )
      model, _ = AutoModel.from_pretrained(
          model_id=MODEL_VERSION,
          mesh=mesh,
          model_source=ModelSource.HUGGINGFACE,
          model_path=MODEL_PATH,
      )

    logger.info(
        "Initializing VllmSampler with random weights (chunked sync enabled)..."
    )
    sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)

    model_state = nnx.state(model)
    if "qwen" in MODEL_VERSION.lower() and SCAN_LAYERS:
      from maxtext.integration.vllm.maxtext_vllm_rollout import (
          unroll_qwen_scanned_weights,
          validate_direct_sync_layer_coverage,
      )
      logger.info("Unrolling Qwen scanned weights (heterogeneous cycle=4)...")
      model_state = unroll_qwen_scanned_weights(model_state, scan_axis=1)
      covered_params = validate_direct_sync_layer_coverage(
          model_state, sampler.transformer_state
      )
      logger.info(
          "Successfully validated direct sync coverage for all %d Qwen layer parameters!",
          covered_params,
      )

    logger.info(
        "Transferring model weights to VllmSampler with reshard_chunk_size=%s...",
        VLLM_RESHARD_CHUNK_SIZE,
    )
    sampler.update_params(model_state)
    logger.info(
        "Weight transfer complete. Freeing temporary model weights..."
    )
    del model, model_state
    gc.collect()
    sft_utils.show_hbm_usage()
  else:
    logger.info(
        "Initializing VllmSampler directly with checkpoint from %s ...",
        MODEL_PATH,
    )
    sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)
    logger.info(
        "VllmSampler successfully initialized directly with model weights."
    )

elif ROLLOUT_ENGINE in ("vanilla", "sglang_jax"):
  if MODEL_SOURCE == "maxtext":
    logger.info(
        "Loading MaxText model %s from %s (scan_layers=%s)...",
        MODEL_VERSION,
        MODEL_PATH,
        SCAN_LAYERS,
    )
    model, _ = AutoModel.from_pretrained(
        model_id=MODEL_VERSION,
        mesh=mesh,
        model_source=ModelSource.MAXTEXT,
        model_path=MODEL_PATH,
        enable_checkpointing=True,
        allow_split_physical_axes=ALLOW_SPLIT_PHYSICAL_AXES,
        scan_layers=SCAN_LAYERS,
        checkpoint_storage_concurrent_gb=CHECKPOINT_STORAGE_CONCURRENT_GB,
    )
  elif MODEL_VERSION == "Qwen/Qwen3-4B-Instruct-2507":
    model_config = model_lib.ModelConfig.qwen3_4b_instruct_2507()
    logger.info("Loading model weights from %s ...", MODEL_PATH)
    model = params_lib.create_model_from_safe_tensors(
        MODEL_PATH, model_config, mesh, dtype=jnp.float32
    )
  elif MODEL_VERSION in ("Qwen/Qwen3-32B", "Qwen3-32B"):
    model_config = model_lib.ModelConfig.qwen3_32b()
    logger.info("Loading model weights from %s ...", MODEL_PATH)
    model = params_lib.create_model_from_safe_tensors(
        MODEL_PATH, model_config, mesh, dtype=jnp.float32
    )
  else:
    logger.info("Loading model weights via AutoModel for %s ...", MODEL_VERSION)
    model, _ = AutoModel.from_pretrained(
        model_id=MODEL_VERSION,
        mesh=mesh,
        model_source=ModelSource.HUGGINGFACE,
        model_path=MODEL_PATH,
    )

  sft_utils.show_hbm_usage()

  if ROLLOUT_ENGINE == "vanilla":
    from tunix.generate import sampler as sampler_lib

    sampler = sampler_lib.Sampler(
        model,
        tokenizer,
        sampler_lib.CacheConfig(
            cache_size=16384,
            num_layers=getattr(model, "config", None)
            and getattr(model.config, "num_layers", 32)
            or 32,
            num_kv_heads=getattr(model, "config", None)
            and getattr(model.config, "num_kv_heads", 8)
            or 8,
            head_dim=getattr(model, "config", None)
            and getattr(model.config, "head_dim", 128)
            or 128,
        ),
    )

  elif ROLLOUT_ENGINE == "sglang_jax":
    from flax import nnx
    from tunix.generate import mappings
    from tunix.generate.sglang_jax_sampler import SglangJaxConfig, SglangJaxSampler

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=None,
        model=model,
        backend="sglang_jax",
    )
    sampler = SglangJaxSampler(
        tokenizer=tokenizer,
        config=SglangJaxConfig(
            mesh=mesh,
            mapping_config=mapping_config,
            model_version=MODEL_VERSION,
            context_length=MAX_MODEL_LEN,
            mem_fraction_static=SGLANG_MEM_FRACTION_STATIC,
            init_with_random_weights=SGLANG_INIT_RANDOM_WEIGHTS,
            disable_radix_cache=True,
            enable_deterministic_sampling=False,
            precompile_token_paddings=[8192, 16384],
            precompile_bs_paddings=[1],
            max_running_requests=SGLANG_MAX_RUNNING_REQUESTS,
        ),
    )
    if SGLANG_INIT_RANDOM_WEIGHTS:
      sampler.load_checkpoint(nnx.state(model))
      logger.info("Synced model weights to sglang_jax engine.")

else:
  raise ValueError(
      f"Unsupported ROLLOUT_ENGINE: {ROLLOUT_ENGINE!r}. "
      "Choose from: 'vanilla', 'vllm', 'sglang_jax'"
  )

# ========================== Model Call ==========================

sampler_lock = None
if ROLLOUT_ENGINE == "vanilla" or (
    ROLLOUT_ENGINE == "vllm" and not VLLM_SERVER_MODE
):
  sampler_lock = threading.Lock()


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
if ROLLOUT_ENGINE != "vllm":
  SAMPLER_CALL_KWARGS["eos_tokens"] = qwen_eos_tokens


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
    if sampler_lock is None:
      out = sampler(
          prompt,
          max_generation_steps=max_gen_steps,
          echo=False,
          **SAMPLER_CALL_KWARGS,
      )
    else:
      with sampler_lock:
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
    if not obs and not done:
      if has_valid_fn:
        obs = "(Command executed successfully with no output.)"
      else:
        obs = (
            "[ACTION GUARD] Your previous response did not include a valid function call. "
            "You must output exactly one tool call in the required XML format "
            "(<function=execute_bash>, <function=str_replace_editor>, or <function=submit>)."
        )
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


class Qwen35SWEAgent(SWEAgent):
  """SWEAgent subclass that formats Qwen3.5 tool responses and normalizes assistant thoughts."""

  def _observation_to_messages(
      self, observation, reward: float, done: bool, info: dict
  ) -> None:
    obs_str = str(observation)
    if len(self._trajectory.steps) > 0 and not obs_str.strip().startswith("<tool_response>"):
      obs_str = f"<tool_response>\n{obs_str.strip()}\n</tool_response>"
    self._messages.append({"role": "user", "content": obs_str})

  def update_from_model(self, response: str, **kwargs):
    clean_resp = response.strip()
    while clean_resp.endswith("<|im_end|>") or clean_resp.endswith("<|endoftext|>"):
      if clean_resp.endswith("<|im_end|>"):
        clean_resp = clean_resp[:-len("<|im_end|>")].rstrip()
      elif clean_resp.endswith("<|endoftext|>"):
        clean_resp = clean_resp[:-len("<|endoftext|>")].rstrip()
    if "</think>" in clean_resp:
      if not clean_resp.lstrip().startswith("<think>"):
        clean_resp = "<think>\n" + clean_resp.lstrip()
    else:
      if "<function=" in clean_resp:
        f_idx = clean_resp.find("<function=")
        thought = clean_resp[:f_idx].strip()
        if thought.startswith("<think>"):
          thought = thought[len("<think>"):].strip()
        func_part = clean_resp[f_idx:].strip()
        clean_resp = f"<think>\n{thought}\n</think>\n\n{func_part}"
      else:
        thought = clean_resp
        if thought.startswith("<think>"):
          thought = thought[len("<think>"):].strip()
        clean_resp = f"<think>\n{thought}\n</think>\n\n"
    action_res = super().update_from_model(clean_resp, **kwargs)
    cur_step = self._trajectory.steps[-1]
    think_block = clean_resp.split("</think>")[0] + "</think>" if "</think>" in clean_resp else clean_resp
    if (
        cur_step.action
        and cur_step.action.startswith(("<function=execute_bash>", "<function=str_replace_editor>", "<function=submit>", "<function=finish>"))
    ):
      self._messages[-1]["content"] = f"{think_block}\n\n{cur_step.action}"
    else:
      # Never leave <function=> or broken tags in assistant chat history
      self._messages[-1]["content"] = think_block
    return action_res


def pairs_generator():
  """Yield NUM_ROLLOUTS_PER_INSTANCE trajectory tasks per dataset entry."""
  for pair_index in range(len(entries) * NUM_ROLLOUTS_PER_INSTANCE):
    entry = entries[pair_index // NUM_ROLLOUTS_PER_INSTANCE]
    agent = Qwen35SWEAgent(scaffold=SCAFFOLD)
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
  for k in (1, 4, 5):
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
  if pass_at_k_metrics[5] is not None:
    logger.info("Pass@5:           %.4f", pass_at_k_metrics[5])
  else:
    logger.info("Pass@5:           N/A")
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

  if OUTPUT_DIR.startswith("gs://"):
    local_output_dir = "/tmp/eval_results"
    os.makedirs(local_output_dir, exist_ok=True)
    output_file = os.path.join(local_output_dir, filename)
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
    logger.info("Results saved locally to %s", output_file)
    try:
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
    except Exception as e:
      logger.warning("Failed to upload to GCS via storage client: %s", e)
      try:
        import subprocess

        gcs_target = os.path.join(OUTPUT_DIR, filename)
        subprocess.run(
            ["gcloud", "storage", "cp", output_file, gcs_target], check=False
        )
        logger.info("Uploaded results via gcloud storage to %s", gcs_target)
      except Exception as ex:
        logger.warning("Failed to upload to GCS via gcloud: %s", ex)
    return output_file
  else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, filename)
    with open(output_file, "w") as f:
      for r in results:
        entry = entries[r["entry_index"]]
        record = {
            "instance_id": entry.get("instance_id", r["instance_id"]),
            "docker_image": entry.get("docker_image", ""),
            "reward": r["reward"],
            "num_steps": r["num_steps"],
            "status": r["status"],
            "guard_blocked_steps": r["guard_blocked_steps"],
            "guard_reasons": r["guard_reasons"],
        }
        f.write(json.dumps(record) + "\n")
    logger.info("Results saved to %s", output_file)
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

