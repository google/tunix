#!/usr/bin/env python
"""DeepSWE Evaluation Script for Tunix.

Runs SWE-bench evaluation using tunix's JAX-based inference and
R2E-Gym Docker environments. Parallels rllm's run_deepswe.py but
uses tunix components (Sampler, RolloutOrchestrator).

Usage:
  # Small test run
  TASKS_LIMIT=2 MAX_CONCURRENT=1 python eval_deepswe.py

  # Full evaluation
  TASKS_LIMIT=0 MAX_CONCURRENT=16 python eval_deepswe.py
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time

# ========================== Configuration ==========================

# Paths — adjust based on your environment
sys.path.insert(0, "/usr/github/rllm")
sys.path.insert(0, "/usr/github/pathways-utils")

# Dataset
DATASET_NAME = os.getenv("DATASET_NAME", "R2E-Gym/SWE-Bench-Verified")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "test")
DATASET_CACHE = os.getenv("DATASET_CACHE", "/scratch/dataset_cache")

# Model
MODEL_VERSION = os.getenv("MODEL_VERSION", "Qwen/Qwen3-4B-Instruct-2507")
MODEL_PATH = os.path.join("/scratch/models/", MODEL_VERSION)

# Evaluation
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
MAX_GENERATION_STEPS = int(os.getenv("MAX_GENERATION_STEPS", "512"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))
TIMEOUT = float(os.getenv("TIMEOUT", "600"))
TASKS_LIMIT = int(os.getenv("TASKS_LIMIT", "10"))  # 0 = all

# Output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/scratch/eval_results")

# ========================== Logging ==========================

for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ========================== JAX / Pathways ==========================

if os.getenv("JAX_PLATFORMS", None) == "proxy":
  import pathwaysutils
  pathwaysutils.initialize()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ========================== Dataset ==========================

from datasets import load_dataset

logger.info("Loading dataset %s split=%s ...", DATASET_NAME, DATASET_SPLIT)
dataset = load_dataset(
    DATASET_NAME,
    split=DATASET_SPLIT,
    cache_dir=DATASET_CACHE,
    num_proc=32,
    trust_remote_code=True,
)

entries = [e for e in dataset if "docker_image" in e]
if TASKS_LIMIT > 0:
  entries = entries[:TASKS_LIMIT]

unique_images = set(e["docker_image"] for e in entries)
logger.info(
    "Loaded %d instances (%d unique Docker images)", len(entries), len(unique_images)
)

# ========================== Kubernetes ==========================

os.environ.setdefault("KUBECONFIG", "~/.kube/config")
os.environ.setdefault("NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool")
os.environ.setdefault("NODE_SELECTOR_VAL", "lance-cpu-pool")

from kubernetes import client, config as k8s_config

k8s_config.load_kube_config()
k8s_client = client.CoreV1Api()
k8s_client.list_namespace(timeout_seconds=5)
logger.info("Kubernetes connection verified.")

# ========================== Model ==========================

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from tunix.models.qwen3 import model as model_lib
from tunix.models.qwen3 import params as params_lib
from tunix.sft import utils as sft_utils
from tunix.rl.agentic.parser.chat_template_parser import parser

# Download model if needed
if not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH):
  os.makedirs(MODEL_PATH, exist_ok=True)
  snapshot_download(
      repo_id=MODEL_VERSION,
      local_dir=MODEL_PATH,
      local_dir_use_symlinks=False,
  )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
chat_parser = parser.QwenChatTemplateParser(tokenizer)

# Create mesh and load weights
devices = jax.devices()
mesh_devices = np.array(devices).reshape(len(devices), 1)
mesh = Mesh(mesh_devices, axis_names=("fsdp", "tp"))

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

# ========================== Model Call ==========================

sampler_lock = threading.Lock()


def model_call(chat_completions, env_unused):
  """Thread-safe model inference via tunix sampler."""
  prompt = chat_parser.parse(chat_completions)
  with sampler_lock:
    out = sampler(prompt, max_generation_steps=MAX_GENERATION_STEPS, echo=False)
  return out.text[0]


# ========================== Evaluation ==========================

from swe_agent import SWEAgent
from swe_env import SWEEnv
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator


def pairs_generator():
  """Yield (agent, env) pairs for each dataset entry."""
  for entry in entries:
    agent = SWEAgent()
    env = SWEEnv(entry=entry, max_steps=MAX_STEPS)
    yield agent, env


async def run_evaluation():
  """Run parallel evaluation using RolloutOrchestrator."""
  orchestrator = RolloutOrchestrator(
      engine_kwargs=dict(
          model_call=model_call,
          max_steps=MAX_STEPS,
          timeout=TIMEOUT,
      ),
      max_concurrency=MAX_CONCURRENT,
      rollout_sync_lock=agentic_utils.RolloutSyncLock(),
  )

  producer = asyncio.create_task(
      orchestrator.run_producers_from_stream(
          pairs_stream=pairs_generator(),
          group_size=1,
          num_episodes=1,
          collect_mode="Trajectory",
          group_key=lambda i, env, traj: i,
      )
  )

  results = []
  start_time = time.time()

  async for batch in orchestrator.yield_batches(batch_size=1):
    for item in batch:
      traj = item.traj
      results.append({
          "pair_index": item.pair_index,
          "trajectory": traj,
      })
      elapsed = time.time() - start_time
      logger.info(
          "[%d/%d] Instance %d: reward=%.1f, steps=%d (%.0fs elapsed)",
          len(results),
          len(entries),
          item.pair_index,
          traj.reward,
          len(traj.steps),
          elapsed,
      )

  await producer
  return results


# ========================== Results ==========================


def compute_pass_at_k(results):
  """Compute and print Pass@1 statistics."""
  total = len(results)
  if total == 0:
    logger.warning("No results to evaluate.")
    return

  correct = sum(1 for r in results if r["trajectory"].reward > 0)
  logger.info("=" * 50)
  logger.info("Evaluation Results")
  logger.info("=" * 50)
  logger.info("Total instances:  %d", total)
  logger.info("Resolved:         %d", correct)
  logger.info("Pass@1:           %.4f", correct / total)
  logger.info("=" * 50)


def save_results(results):
  """Save trajectory summaries to JSONL."""
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  output_file = os.path.join(OUTPUT_DIR, f"eval_{timestamp}.jsonl")

  with open(output_file, "w") as f:
    for r in results:
      traj = r["trajectory"]
      entry = entries[r["pair_index"]]
      record = {
          "instance_id": entry.get("instance_id", r["pair_index"]),
          "docker_image": entry.get("docker_image", ""),
          "reward": traj.reward,
          "num_steps": len(traj.steps),
      }
      f.write(json.dumps(record) + "\n")

  logger.info("Results saved to %s", output_file)
  return output_file


# ========================== Main ==========================

if __name__ == "__main__":
  logger.info(
      "Starting evaluation: %d instances, max_concurrent=%d, max_steps=%d",
      len(entries),
      MAX_CONCURRENT,
      MAX_STEPS,
  )

  eval_results = asyncio.run(run_evaluation())
  compute_pass_at_k(eval_results)
  save_results(eval_results)
