#!/usr/bin/env python
"""DeepSWE evaluation with deepscaler-style task-level parallelism.

This script intentionally does not modify the existing eval entrypoint.
It runs one full SWE trajectory per task and uses RolloutOrchestrator to
parallelize whole tasks, similar to how deepscaler training/eval relies on
the framework orchestrator rather than a custom outer runner.

Environment Variables:
  DATASET_NAME: The HuggingFace dataset name (default:
  "R2E-Gym/SWE-Bench-Verified").
  DATASET_SPLIT: The split of the dataset to evaluate (default: "test").
  DATASET_CACHE: Path to cache the downloaded dataset (default:
  "/scratch/dataset_cache").
  MODEL_VERSION: The model identifier on HuggingFace to evaluate (default:
  "Qwen/Qwen3-32B").
  MAX_STEPS: Maximum number of agent steps per task/trajectory (default: 30).
  MAX_MODEL_LEN: Maximum context length of the LLM (default: 32768).
  MAX_RESPONSE_LENGTH / MAX_GENERATION_STEPS: Maximum tokens the model can
  generate in a single response (default: 8192).
  MAX_CONCURRENT: Maximum number of concurrent tasks/trajectories to run in
  parallel (default: 256).
  TIMEOUT: Timeout in seconds for a single trajectory evaluation (default: 600).
  TASKS_LIMIT: Maximum number of tasks to evaluate. 0 means all tasks (default:
  0).
  MAX_CONTEXT_LIMIT: Limit on the number of context tokens before terminating
  the episode (default: MAX_MODEL_LEN - 256).
  ENABLE_GUARD: Set to "true" to enable action guard via GuardedSWEEnv (default:
  "false").
  ROLLOUT_ENGINE: The underlying LLM engine to use. Choices are 'vanilla',
  'vllm', or 'sglang_jax' (default: "vllm").
  VLLM_HBM_UTILIZATION: HBM utilization ratio for vLLM engine (default: 0.4).
  VLLM_INIT_RANDOM_WEIGHTS: Set to "true" to init vLLM with random weights and
  sync later (default: "true").
  VLLM_SERVER_MODE: Set to "true" to run vLLM in server mode (default: "true").
  VLLM_MAX_NUM_SEQS: Maximum number of concurrent sequences in vLLM (default:
  128).
  VLLM_MAX_BATCHED_TOKENS: Maximum batched tokens for vLLM (default: 165888).
  SGLANG_MEM_FRACTION_STATIC: Static memory fraction for SGLang (default: 0.4).
  SGLANG_INIT_RANDOM_WEIGHTS: Set to "true" to init SGLang with random weights
  and sync later (default: "true").
  SGLANG_MAX_RUNNING_REQUESTS: Maximum running requests for SGLang (default: 1).
  OUTPUT_DIR: Directory where the evaluation results JSONL file will be saved
  (default: "eval_results").
  JAX_PLATFORMS: If set to "proxy", initializes Pathways utils.

Usage:
  # Full evaluation with default settings:
  #   - Qwen/Qwen3-32B
  #   - vLLM sampler
  #   - MAX_CONCURRENT=256
  #   - ENABLE_GUARD=false
  #   - full evaluation split
  python3 examples/deepswe/eval_deepswe.py
"""

import asyncio
import os
import sys
import threading

from datasets import load_dataset
from examples.deepswe import deepswe_utils
from examples.deepswe import eval_utils
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from kubernetes import client
from kubernetes import config as k8s_config
import numpy as np
from swe_agent import SWEAgent
from transformers import AutoTokenizer
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.models.qwen3 import model as model_lib
from tunix.models.qwen3 import params as params_lib
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.sft import utils as sft_utils

# ========================== Configuration ==========================

sys.path.insert(0, "/usr/github/rllm")
sys.path.insert(0, "/usr/github/pathways-utils")

DATASET_NAME = os.getenv("DATASET_NAME", "R2E-Gym/SWE-Bench-Verified")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "test")
DATASET_CACHE = os.getenv("DATASET_CACHE", "/scratch/dataset_cache")

MODEL_VERSION = os.getenv("MODEL_VERSION", "Qwen/Qwen3-32B")
MODEL_PATH = os.path.join("/scratch/models/", MODEL_VERSION)

MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
MAX_RESPONSE_LENGTH = int(
    os.getenv("MAX_RESPONSE_LENGTH", os.getenv("MAX_GENERATION_STEPS", "8192"))
)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "256"))
TIMEOUT = float(os.getenv("TIMEOUT", "600"))
TASKS_LIMIT = int(os.getenv("TASKS_LIMIT", "0"))
MAX_CONTEXT_LIMIT = int(
    os.getenv("MAX_CONTEXT_LIMIT", str(max(1, MAX_MODEL_LEN - 256)))
)

ENABLE_GUARD = False
if os.getenv("ENABLE_GUARD", "false").lower() == "true":
  ENABLE_GUARD = True

ROLLOUT_ENGINE = os.getenv("ROLLOUT_ENGINE", "vllm")

VLLM_HBM_UTILIZATION = float(os.getenv("VLLM_HBM_UTILIZATION", "0.4"))
VLLM_INIT_RANDOM_WEIGHTS = (
    os.getenv("VLLM_INIT_RANDOM_WEIGHTS", "true").lower() == "true"
)
VLLM_SERVER_MODE = os.getenv("VLLM_SERVER_MODE", "true").lower() == "true"
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "128"))
VLLM_MAX_BATCHED_TOKENS = int(os.getenv("VLLM_MAX_BATCHED_TOKENS", "165888"))

SGLANG_MEM_FRACTION_STATIC = float(
    os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.4")
)
SGLANG_INIT_RANDOM_WEIGHTS = (
    os.getenv("SGLANG_INIT_RANDOM_WEIGHTS", "true").lower() == "true"
)
SGLANG_MAX_RUNNING_REQUESTS = int(os.getenv("SGLANG_MAX_RUNNING_REQUESTS", "1"))

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "eval_results")
)

# ========================== Logging ==========================

logger = deepswe_utils.setup_logging(level="INFO")


# ========================== JAX / Pathways ==========================

if os.getenv("JAX_PLATFORMS", None) == "proxy":
  import pathwaysutils

  pathwaysutils.initialize()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ========================== Dataset ==========================

logger.info("Loading dataset %s split=%s ...", DATASET_NAME, DATASET_SPLIT)
dataset = load_dataset(
    DATASET_NAME,
    split=DATASET_SPLIT,
    cache_dir=DATASET_CACHE,
    num_proc=32,
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
os.environ.setdefault("NODE_SELECTOR_VAL", "deepswe-cpu-pool")

k8s_config.load_kube_config()
k8s_client = client.CoreV1Api()
k8s_client.list_namespace(timeout_seconds=5)
logger.info("Kubernetes connection verified.")

# ========================== Model ==========================

if not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH):
  os.makedirs(MODEL_PATH, exist_ok=True)
  snapshot_download(
      repo_id=MODEL_VERSION,
      local_dir=MODEL_PATH,
      local_dir_use_symlinks=False,
  )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer_for_agentic = tok_adapter.TokenizerAdapter(tokenizer)
chat_parser = parser.QwenChatTemplateParser(tokenizer)
qwen_eos_tokens = [tokenizer.encode("<|im_end|>")[0]]

devices = jax.devices()
# Force pure tensor parallelism for eval: DP=1, TP=min(8, len(devices)).
# Qwen3-32B has tensors such as (5120, 8, 128), so TP must not exceed 8 for
# shardings that partition that dimension on the tp axis.
TP_SIZE = min(int(os.getenv("TP_SIZE", "8")), len(devices))
mesh_devices = np.array(devices[:TP_SIZE]).reshape(1, TP_SIZE)
mesh = Mesh(mesh_devices, axis_names=("fsdp", "tp"))
logger.info(
    "Using mesh shape fsdp=%d tp=%d (pure TP eval, total_devices=%d,"
    " used_devices=%d)",
    mesh.shape["fsdp"],
    mesh.shape["tp"],
    len(devices),
    TP_SIZE,
)

if MODEL_VERSION == "Qwen/Qwen3-4B-Instruct-2507":
  model_config = model_lib.ModelConfig.qwen3_4b_instruct_2507()
elif MODEL_VERSION == "Qwen/Qwen3-32B":
  model_config = model_lib.ModelConfig.qwen3_32b()
else:
  raise ValueError(f"Unsupported MODEL_VERSION: {MODEL_VERSION}")

logger.info("Loading model weights from %s ...", MODEL_PATH)
model = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, model_config, mesh, dtype=jnp.float32
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
          num_layers=model_config.num_layers,
          num_kv_heads=model_config.num_kv_heads,
          head_dim=model_config.head_dim,
      ),
  )

elif ROLLOUT_ENGINE == "vllm":
  from tunix.generate import mappings
  from tunix.generate.vllm_sampler import VllmConfig, VllmSampler

  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

  mapping_config = mappings.MappingConfig.build(
      mapping_obj=None,
      model=model,
      backend="vllm_jax",
  )
  vllm_config = VllmConfig(
      mesh=mesh,
      hbm_utilization=VLLM_HBM_UTILIZATION,
      init_with_random_weights=VLLM_INIT_RANDOM_WEIGHTS,
      tpu_backend_type="jax",
      server_mode=VLLM_SERVER_MODE,
      tensor_parallel_size=mesh.shape["tp"],
      data_parallel_size=mesh.shape["fsdp"],
      mapping_config=mapping_config,
      engine_kwargs={
          "model": MODEL_PATH,
          "max_model_len": MAX_MODEL_LEN,
          "max_num_seqs": VLLM_MAX_NUM_SEQS,
          "max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
          "enable_prefix_caching": True,
          "kv_cache_metrics": True,
          "disable_log_stats": False,
      },
  )
  sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)

  from flax import nnx

  sampler.load_checkpoint(nnx.state(model))
  logger.info("Synced model weights to vLLM engine.")

elif ROLLOUT_ENGINE == "sglang_jax":
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
    from flax import nnx

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

model_call = eval_utils.create_model_call(
    sampler=sampler,
    tokenizer=tokenizer,
    chat_parser=chat_parser,
    max_response_length=MAX_RESPONSE_LENGTH,
    max_context_limit=MAX_CONTEXT_LIMIT,
    sampler_kwargs={"eos_tokens": qwen_eos_tokens},
    sampler_lock=sampler_lock,
    logger=logger,
)


# ========================== Evaluation ==========================


class EvalTrajectoryCollectEngine(eval_utils.EvalTrajectoryCollectEngine):
  """Trajectory engine that skips final reward grading on prompt overflow."""

  skip_final_reward_on_overflow: bool = True


def pairs_generator():
  """Yield one full (agent, env) trajectory task per dataset entry."""
  for pair_index, entry in enumerate(entries):
    agent = SWEAgent()
    env_cls = (
        eval_utils.LoggedGuardedSWEEnv
        if ENABLE_GUARD
        else eval_utils.LoggedSWEEnv
    )
    env = env_cls(
        entry=entry,
        max_steps=MAX_STEPS,
        pair_index=pair_index,
        group_id=pair_index,
    )
    yield agent, env


# ========================== Main ==========================

if __name__ == "__main__":
  logger.info(
      "Starting deepscaler-style evaluation: %d instances, max_concurrent=%d, "
      "max_steps=%d, engine=%s",
      len(entries),
      MAX_CONCURRENT,
      MAX_STEPS,
      ROLLOUT_ENGINE,
  )

  eval_results = asyncio.run(
      eval_utils.run_evaluation(
          entries=entries,
          pairs_stream=pairs_generator(),
          model_call=model_call,
          tokenizer_for_agentic=tokenizer_for_agentic,
          chat_parser=chat_parser,
          timeout=TIMEOUT,
          max_concurrent=MAX_CONCURRENT,
          output_dir=OUTPUT_DIR,
          engine_cls=EvalTrajectoryCollectEngine,
          logger=logger,
          use_custom_executor=False,
      )
  )
  eval_utils.compute_pass_at_k(
      eval_results, ks=(1,), log_guard_stats=True, logger=logger
  )
  eval_utils.save_results(
      eval_results,
      entries=entries,
      output_dir=OUTPUT_DIR,
      filename_prefix="eval_deepscaler_style",
      include_pair_index=False,
      include_step_actions=False,
      include_guard_stats=True,
      logger=logger,
  )
