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
import collections
import json
import logging
import os
import sys
import threading
import time

from datasets import load_dataset
from guarded_swe_env import GuardedSWEEnv
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from kubernetes import client
from kubernetes import config as k8s_config
import numpy as np
from swe_agent import SWEAgent
from swe_env import SWEEnv
from transformers import AutoTokenizer
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.models.qwen3 import model as model_lib
from tunix.models.qwen3 import params as params_lib
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.sft import utils as sft_utils

Counter = collections.Counter

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
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"

# ========================== Logging ==========================

for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deepswe_eval")


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
# Force pure tensor parallelism for eval: DP=1, TP=8.
# Qwen3-32B has tensors such as (5120, 8, 128), so TP must not exceed 8 for
# shardings that partition that dimension on the tp axis.
TP_SIZE = 8
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
    """Resets the environment and logs the timing.

    This method calls the superclass's reset and logs the start and end of the
    reset operation, including the time taken.

    Returns:
      The observation and info returned by the superclass's reset method.
    """
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
    """Steps the environment and logs the action and timing."""
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
  """Computes and logs evaluation metrics such as Pass@1 and average reward.

  Args:
    results: A list of dictionaries, where each dictionary contains the
      evaluation results for a single instance, including 'reward', 'num_steps',
      'status', 'guard_blocked_steps', and 'guard_reasons'.
  """
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
  """Saves the evaluation results to a JSONL file.

  The results are saved in a timestamped file within the OUTPUT_DIR. Each line
  in the file is a JSON object representing the evaluation outcome for a single
  instance.

  Args:
    results: A list of dictionaries, where each dictionary contains the
      evaluation results for a single instance, including 'pair_index',
      'reward', 'num_steps', 'status', 'guard_blocked_steps', and
      'guard_reasons'.

  Returns:
    The path to the saved JSONL file.
  """
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  output_file = os.path.join(
      OUTPUT_DIR, f"eval_deepscaler_style_{timestamp}.jsonl"
  )

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
      "Starting deepscaler-style evaluation: %d instances, max_concurrent=%d, "
      "max_steps=%d, engine=%s",
      len(entries),
      MAX_CONCURRENT,
      MAX_STEPS,
      ROLLOUT_ENGINE,
  )

  eval_results = asyncio.run(run_evaluation())
  compute_pass_at_k(eval_results)
  save_results(eval_results)
