#!/usr/bin/env python
"""DeepSWE evaluation with deepscaler-style task-level parallelism.

This script runs SWE evaluation trajectories and uses RolloutOrchestrator to
parallelize tasks across a TPU cluster. It supports both HuggingFace and MaxText
models (e.g. Qwen3.5-35B-A3B) and configurable JAX/vLLM sharding meshes.
"""

import argparse
import asyncio
import collections
import json
import logging
import os
import sys
import threading
import time

# Path Setup before JAX
workdir = os.getcwd()
tunix_root = os.path.join(workdir, "tunix")
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
  from guarded_swe_env import GuardedSWEEnv
  from swe_agent import SWEAgent
  from swe_env import SWEEnv
except ImportError:
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
    default=float(os.getenv("TIMEOUT", "600")),
    help="Timeout in seconds for a single trajectory",
)
parser_cli.add_argument(
    "--tasks_limit",
    type=int,
    default=int(os.getenv("TASKS_LIMIT", "0")),
    help="Max tasks to evaluate (0 for all)",
)
parser_cli.add_argument(
    "--max_context_limit",
    type=int,
    default=None,
    help="Max context token limit before terminating",
)
parser_cli.add_argument(
    "--enable_guard",
    type=bool,
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
    "--vllm_init_random_weights",
    type=bool,
    default=os.getenv("VLLM_INIT_RANDOM_WEIGHTS", "true").lower() == "true",
)
parser_cli.add_argument(
    "--vllm_server_mode",
    type=bool,
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
    "--sglang_mem_fraction_static",
    type=float,
    default=float(os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.4")),
)
parser_cli.add_argument(
    "--sglang_init_random_weights",
    type=bool,
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
    type=bool,
    default=os.getenv("SCAN_LAYERS", "true").lower() == "true",
    help="Whether to scan layers for MaxText models",
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

args, _ = parser_cli.parse_known_args()

DATASET_NAME = args.dataset_name
DATASET_SPLIT = args.dataset_split
DATASET_CACHE = args.dataset_cache

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
TASKS_LIMIT = args.tasks_limit
MAX_CONTEXT_LIMIT = (
    args.max_context_limit
    if args.max_context_limit is not None
    else max(1, MAX_MODEL_LEN - 256)
)

ENABLE_GUARD = args.enable_guard
ROLLOUT_ENGINE = args.rollout_engine

VLLM_HBM_UTILIZATION = args.vllm_utilization
VLLM_INIT_RANDOM_WEIGHTS = args.vllm_init_random_weights
VLLM_SERVER_MODE = args.vllm_server_mode
VLLM_MAX_NUM_SEQS = args.vllm_max_num_seqs
VLLM_MAX_BATCHED_TOKENS = args.vllm_max_batched_tokens
VLLM_RESHARD_CHUNK_SIZE = args.vllm_reshard_chunk_size

SGLANG_MEM_FRACTION_STATIC = args.sglang_mem_fraction_static
SGLANG_INIT_RANDOM_WEIGHTS = args.sglang_init_random_weights
SGLANG_MAX_RUNNING_REQUESTS = args.sglang_max_running_requests

MESH_FSDP = args.mesh_fsdp
MESH_TP = args.mesh_tp
SCAN_LAYERS = args.scan_layers
NODE_SELECTOR_VAL = args.node_selector_val
OUTPUT_DIR = args.output_dir

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

# ========================== Dataset ==========================

logger.info("Loading dataset %s split=%s ...", DATASET_NAME, DATASET_SPLIT)
try:
  dataset = load_dataset(
      DATASET_NAME,
      split=DATASET_SPLIT,
      cache_dir=DATASET_CACHE,
      num_proc=32,
  )
except Exception as e:
  logger.warning(
      "load_dataset failed with num_proc=32 (%s), retrying without num_proc...",
      e,
  )
  dataset = load_dataset(
      DATASET_NAME,
      split=DATASET_SPLIT,
      cache_dir=DATASET_CACHE,
  )

entries = [e for e in dataset if "docker_image" in e]
if TASKS_LIMIT > 0:
  entries = entries[:TASKS_LIMIT]

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
  k8s_config.load_kube_config()
  k8s_client = client.CoreV1Api()
  logger.info("Kubernetes connection verified.")
except Exception as e:
  logger.warning("Kubernetes config loading note: %s", e)

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
chat_parser = parser.QwenChatTemplateParser(tokenizer)
qwen_eos_tokens = [tokenizer.encode("<|im_end|>")[0]]

# Mesh Setup
devices = jax.devices()
total_mesh_devices = MESH_FSDP * MESH_TP
if total_mesh_devices > len(devices):
  logger.warning(
      "Requested mesh FSDP=%d * TP=%d (%d devices) > total devices (%d)."
      " Adjusting mesh...",
      MESH_FSDP,
      MESH_TP,
      total_mesh_devices,
      len(devices),
  )
  if len(devices) >= 8:
    MESH_FSDP = len(devices) // 8
    MESH_TP = 8
  else:
    MESH_FSDP = 1
    MESH_TP = len(devices)
  total_mesh_devices = MESH_FSDP * MESH_TP

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

# Model Instantiation
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
      allow_split_physical_axes=True,
      scan_layers=SCAN_LAYERS,
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

# ========================== Sampler ==========================

logger.info("Creating sampler with engine=%s ...", ROLLOUT_ENGINE)

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

elif ROLLOUT_ENGINE == "vllm":
  from flax import nnx
  from tunix.generate import mappings
  from tunix.generate.vllm_sampler import VllmConfig, VllmSampler

  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

  additional_config = None
  if MODEL_SOURCE == "maxtext":
    additional_config = {
        "maxtext_config": {
            "model_name": MODEL_VERSION.lower().split("/")[-1],
            "model_call_mode": "inference",
            "enable_dp_attention": False,
            "allow_split_physical_axes": True,
            "log_config": False,
            "weight_dtype": "bfloat16",
            "prefuse_moe_weights": True,
            "attention": "vllm_rpa",
        }
    }
    if hasattr(model, "use_no_op_mappings"):
      model.use_no_op_mappings = True
      logger.info("Set use_no_op_mappings=True on model.")

  mapping_config = mappings.MappingConfig.build(
      mapping_obj=None,
      model=model,
      backend="vllm_jax",
  )
  engine_kwargs = {
      "model": tokenizer_path if MODEL_SOURCE == "maxtext" else MODEL_PATH,
      "max_model_len": MAX_MODEL_LEN,
      "max_num_seqs": VLLM_MAX_NUM_SEQS,
      "max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
      "enable_prefix_caching": True,
      "kv_cache_metrics": True,
      "disable_log_stats": False,
      "tokenizer": tokenizer_path,
  }
  if MODEL_SOURCE == "maxtext":
    engine_kwargs["hf_overrides"] = {
        "architectures": ["MaxTextForCausalLM"]
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
      reshard_chunk_size=VLLM_RESHARD_CHUNK_SIZE,
      engine_kwargs=engine_kwargs,
  )
  sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)
  sampler.load_checkpoint(nnx.state(model))
  del model
  import gc
  gc.collect()
  logger.info("Synced model weights to vLLM engine and freed original model.")

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


def model_call(chat_completions, env_unused):
  """Model inference via tunix sampler."""
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
      " max_model_len=%d",
      pair_index,
      instance_id,
      len(prompt),
      prompt_token_count,
      MAX_MODEL_LEN,
  )
  if prompt_token_count >= MAX_MODEL_LEN:
    raise PromptTooLongError(
        "Prompt too long before sampler call:"
        f" prompt_tokens={prompt_token_count}, max_model_len={MAX_MODEL_LEN}"
    )
  t0 = time.time()
  try:
    if sampler_lock is None:
      out = sampler(
          prompt,
          max_generation_steps=MAX_RESPONSE_LENGTH,
          echo=False,
          eos_tokens=qwen_eos_tokens,
      )
    else:
      with sampler_lock:
        out = sampler(
            prompt,
            max_generation_steps=MAX_RESPONSE_LENGTH,
            echo=False,
            eos_tokens=qwen_eos_tokens,
        )
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
  """Trajectory engine that converts prompt overflows into per-trajectory termination."""

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
      self._skip_final_reward = True
      if self.agent.trajectory.steps:
        self.agent.trajectory.steps[-1].done = True
      return True

  async def _append_final_reward(self):
    if getattr(self, "_skip_final_reward", False):
      return
    await super()._append_final_reward()

  def compute_trajectory_reward(self):
    if getattr(self, "_skip_final_reward", False):
      self.agent.trajectory.reward = 0.0
      return self.agent.trajectory
    return super().compute_trajectory_reward()


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
  """Yield one full (agent, env) trajectory task per dataset entry."""
  for pair_index, entry in enumerate(entries):
    agent = SWEAgent()
    env_cls = LoggedGuardedSWEEnv if ENABLE_GUARD else LoggedSWEEnv
    env = env_cls(
        entry=entry,
        max_steps=MAX_STEPS,
        pair_index=pair_index,
        group_id=pair_index,
    )
    yield agent, env


async def run_evaluation():
  """Run evaluation with orchestrator-managed task-level parallelism."""
  if not OUTPUT_DIR.startswith("gs://"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

  orchestrator = RolloutOrchestrator(
      engine_cls=EvalTrajectoryCollectEngine,
      engine_kwargs=dict(
          model_call=model_call,
          timeout=TIMEOUT,
          max_context_limit=MAX_CONTEXT_LIMIT,
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
      entry = entries[item.pair_index]
      guard_reasons = sorted({
          (getattr(step, "info", {}) or {}).get("guard_reason", "unknown")
          for step in traj.steps
          if (getattr(step, "info", {}) or {}).get("guard_blocked")
      })
      result = {
          "pair_index": item.pair_index,
          "instance_id": entry.get("instance_id", item.pair_index),
          "reward": float(traj.reward),
          "num_steps": len(traj.steps),
          "status": getattr(traj.status, "name", str(traj.status)),
          "guard_blocked_steps": sum(
              1
              for step in traj.steps
              if (getattr(step, "info", {}) or {}).get("guard_blocked")
          ),
          "guard_reasons": guard_reasons,
      }
      results.append(result)
      elapsed = time.time() - start_time
      logger.info(
          "[%d/%d] Instance %s: reward=%.1f, steps=%d, status=%s (%.0fs"
          " elapsed)",
          len(results),
          len(entries),
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

  await producer
  return results


# ========================== Results ==========================


def compute_pass_at_k(results):
  """Computes and logs evaluation metrics such as Pass@1 and average reward."""
  total = len(results)
  if total == 0:
    logger.warning("No results to evaluate.")
    return

  correct = sum(1 for r in results if r["reward"] > 0)
  total_reward = sum(float(r["reward"]) for r in results)
  total_steps = sum(r["num_steps"] for r in results)
  status_counts = Counter(r["status"] for r in results)

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
  logger.info("Pass@1:           %.4f", correct / total)
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
        entry = entries[r["pair_index"]]
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
    logger.info("Results saved locally to %s", output_file)
    try:
      import subprocess

      gcs_target = os.path.join(OUTPUT_DIR, filename)
      subprocess.run(
          ["gcloud", "storage", "cp", output_file, gcs_target], check=False
      )
      logger.info("Uploaded results to %s", gcs_target)
    except Exception as e:
      logger.warning("Failed to upload to GCS: %s", e)
    return output_file
  else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, filename)
    with open(output_file, "w") as f:
      for r in results:
        entry = entries[r["pair_index"]]
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

  eval_results = asyncio.run(run_evaluation())
  compute_pass_at_k(eval_results)
  save_results(eval_results)
