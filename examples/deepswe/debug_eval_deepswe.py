#!/usr/bin/env python
"""DeepSWE evaluation with trajectory logging.

Runs SWE-bench instances and writes clean trajectory logs (action, observation,
reward per step) to per-instance files under OUTPUT_DIR. All other debug noise
(JAX, vLLM, k8s) stays on stderr and is NOT written to trajectory files.

Usage:
  # Local model (single instance):
  ENABLE_GUARD=false \
  MODEL_VERSION="Qwen/Qwen3-1.7B" \
  TASK_INDEX=0 \
  MAX_STEPS=10 \
  MAX_GENERATION_STEPS=256 \
  TIMEOUT=600 \
  python3 -u examples/deepswe/debug_eval_deepswe.py

  # vLLM sampler (single instance):
  ENABLE_GUARD=false \
  ROLLOUT_ENGINE=vllm \
  MODEL_VERSION="Qwen/Qwen3-4B-Instruct-2507" \
  TASK_INDEX=0 \
  MAX_STEPS=30 \
  TIMEOUT=600 \
  python3 -u examples/deepswe/debug_eval_deepswe.py

  # SGLang-JAX sampler (single instance):
  ENABLE_GUARD=false \
  ROLLOUT_ENGINE=sglang_jax \
  MODEL_VERSION="Qwen/Qwen3-4B-Instruct-2507" \
  TASK_INDEX=0 \
  MAX_STEPS=30 \
  TIMEOUT=600 \
  python3 -u examples/deepswe/debug_eval_deepswe.py

  # Multiple instances by indices:
  ENABLE_GUARD=false \
  ROLLOUT_ENGINE=vllm \
  MODEL_VERSION="Qwen/Qwen3-4B-Instruct-2507" \
  TASK_INDICES="0,1,5,10" \
  python3 -u examples/deepswe/debug_eval_deepswe.py

  # Multiple instances by count (first N):
  ENABLE_GUARD=false \
  NUM_TASKS=5 \
  python3 -u examples/deepswe/debug_eval_deepswe.py

  # By instance ID:
  INSTANCE_ID=django__django-12345 \
  python3 -u examples/deepswe/debug_eval_deepswe.py

  # Gemini API mode:
  USE_API=gemini \
  API_KEY="your-api-key" \
  API_MODEL="gemini-2.5-flash" \
  TASK_INDEX=0 \
  MAX_STEPS=10 \
  TIMEOUT=600 \
  python3 -u examples/deepswe/debug_eval_deepswe.py

Outputs:
  - Trajectory log: OUTPUT_DIR/trajectory.log  (all instances in one file)
"""

import json
import logging
import os
import sys
import threading
import time
import traceback

# ========================== Configuration ==========================

sys.path.insert(0, "/usr/github/rllm")
sys.path.insert(0, "/usr/github/pathways-utils")

DATASET_NAME = os.getenv("DATASET_NAME", "R2E-Gym/SWE-Bench-Verified")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "test")
DATASET_CACHE = os.getenv("DATASET_CACHE", "/scratch/dataset_cache")

MODEL_VERSION = os.getenv("MODEL_VERSION", "Qwen/Qwen3-4B-Instruct-2507")
MODEL_PATH = os.path.join("/scratch/models/", MODEL_VERSION)

MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
MAX_GENERATION_STEPS = int(os.getenv("MAX_GENERATION_STEPS", "512"))
TIMEOUT = float(os.getenv("TIMEOUT", "600"))

# Task selection (priority: INSTANCE_ID > TASK_INDICES > NUM_TASKS > TASK_INDEX)
TASK_INDEX = int(os.getenv("TASK_INDEX", "0"))
INSTANCE_ID = os.getenv("INSTANCE_ID", "")
TASK_INDICES = os.getenv("TASK_INDICES", "")  # e.g. "0,1,5,10"
NUM_TASKS = int(os.getenv("NUM_TASKS", "0"))  # run first N tasks

OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "debug_eval_output"))

# Rollout engine: "vanilla", "vllm", or "sglang_jax"
ROLLOUT_ENGINE = os.getenv("ROLLOUT_ENGINE", "vanilla")

# vLLM-specific
VLLM_HBM_UTILIZATION = float(os.getenv("VLLM_HBM_UTILIZATION", "0.4"))
VLLM_INIT_RANDOM_WEIGHTS = os.getenv("VLLM_INIT_RANDOM_WEIGHTS", "true").lower() == "true"
VLLM_SERVER_MODE = os.getenv("VLLM_SERVER_MODE", "true").lower() == "true"

# SGLang-specific
SGLANG_MEM_FRACTION_STATIC = float(os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.4"))
SGLANG_INIT_RANDOM_WEIGHTS = os.getenv("SGLANG_INIT_RANDOM_WEIGHTS", "false").lower() == "true"
SGLANG_MAX_RUNNING_REQUESTS = int(os.getenv("SGLANG_MAX_RUNNING_REQUESTS", "1"))

# Guard: set ENABLE_GUARD=false to disable the action guard
ENABLE_GUARD = os.getenv("ENABLE_GUARD", "true").lower() == "true"

# API mode: set USE_API=gemini to use Gemini API instead of local model
USE_API = os.getenv("USE_API", "")  # "gemini" for Gemini API
API_KEY = os.getenv("API_KEY", "")
API_MODEL = os.getenv("API_MODEL", "gemini-2.5-flash")

# ========================== Logging ==========================
# Root logger at WARNING to suppress JAX/vLLM/k8s noise.
# Our logger at INFO for progress messages on stdout.

for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deepswe_eval")
logger.setLevel(logging.INFO)

# ========================== JAX / Pathways ==========================

if not USE_API:
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
)

all_entries = [e for e in dataset if "docker_image" in e]
logger.info("Dataset has %d instances with docker_image.", len(all_entries))


def select_entries():
  """Select entries based on env vars."""
  if INSTANCE_ID:
    for e in all_entries:
      if e.get("instance_id") == INSTANCE_ID:
        return [e]
    logger.error("Instance ID %s not found.", INSTANCE_ID)
    sys.exit(1)
  if TASK_INDICES:
    indices = [int(i.strip()) for i in TASK_INDICES.split(",")]
    return [all_entries[i] for i in indices if i < len(all_entries)]
  if NUM_TASKS > 0:
    return all_entries[:NUM_TASKS]
  return [all_entries[TASK_INDEX]]


selected_entries = select_entries()
logger.info("Selected %d instance(s) for evaluation.", len(selected_entries))

# ========================== Kubernetes ==========================

os.environ.setdefault("KUBECONFIG", "~/.kube/config")
os.environ.setdefault("NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool")
os.environ.setdefault("NODE_SELECTOR_VAL", "deepswe-cpu-pool")

from kubernetes import client, config as k8s_config

k8s_config.load_kube_config()
k8s_client = client.CoreV1Api()
k8s_client.list_namespace(timeout_seconds=5)
logger.info("Kubernetes connection verified.")

# ========================== Model ==========================

if not USE_API:
  import jax
  import jax.numpy as jnp
  from jax.sharding import Mesh
  import numpy as np
  from huggingface_hub import snapshot_download
  from transformers import AutoTokenizer
  from tunix.models.qwen3 import params as params_lib
  from tunix.sft import utils as sft_utils
  from tunix.rl.agentic.parser.chat_template_parser import parser

  if not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_VERSION,
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
    )

  tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
  chat_parser = parser.QwenChatTemplateParser(tokenizer)

  from tunix.models.automodel import call_model_config

  model_name = MODEL_VERSION.split("/")[-1]
  model_config = call_model_config(model_name)

  devices = jax.devices()
  num_devices = len(devices)

  # Compute TP/FSDP split based on num_kv_heads, matching train_deepswe_nb.py
  rollout_tp = np.gcd(num_devices, model_config.num_kv_heads)
  rollout_fsdp = num_devices // rollout_tp
  mesh_devices = np.array(devices).reshape(rollout_fsdp, rollout_tp)
  mesh = Mesh(mesh_devices, axis_names=("fsdp", "tp"))

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
            "max_model_len": 16384,
            "max_num_seqs": 1,
            "enable_prefix_caching": True,
            "kv_cache_metrics": True,
            "disable_log_stats": False,
        },
    )
    sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)

    # Sync actual model weights to vLLM engine (matching vllm_rollout.py init)
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
            context_length=16384 + MAX_GENERATION_STEPS + 100,
            mem_fraction_static=SGLANG_MEM_FRACTION_STATIC,
            init_with_random_weights=SGLANG_INIT_RANDOM_WEIGHTS,
            disable_radix_cache=True,
            enable_deterministic_sampling=False,
            precompile_token_paddings=[8192, 16384],
            precompile_bs_paddings=[1],
            max_running_requests=SGLANG_MAX_RUNNING_REQUESTS,
        ),
    )

  else:
    raise ValueError(
        f"Unsupported ROLLOUT_ENGINE: {ROLLOUT_ENGINE!r}. "
        f"Choose from: 'vanilla', 'vllm', 'sglang_jax'"
    )

  sampler_lock = threading.Lock()

else:
  # ========================== Gemini API Client ==========================

  import google.genai as genai

  if not API_KEY:
    logger.error("API_KEY is required when USE_API is set.")
    sys.exit(1)

  api_client = genai.Client(api_key=API_KEY)
  logger.info("Using Gemini API: model=%s", API_MODEL)


def _chat_to_gemini(chat_completions):
  """Convert OpenAI-style chat messages to Gemini API format."""
  system_parts = []
  contents = []
  for msg in chat_completions:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if role == "system":
      system_parts.append(content)
    elif role == "assistant":
      contents.append({"role": "model", "parts": [{"text": content}]})
    else:
      contents.append({"role": "user", "parts": [{"text": content}]})
  system_instruction = "\n".join(system_parts) if system_parts else None
  return contents, system_instruction

# ========================== Agent & Env ==========================

from action_guard import ActionGuard, GuardConfig
from swe_agent import SWEAgent, parse_xml_response
from swe_env import SWEEnv


# ========================== Trajectory Writer ==========================

def _write_traj(f, text):
  """Write a line to the trajectory file and print to stdout."""
  f.write(text + "\n")
  f.flush()
  print(text)


# ========================== Single Instance Eval ==========================

def run_single_eval(entry, traj_file):
  """Run one instance. Trajectory content -> traj_file handle. Returns reward."""
  instance_id = entry.get("instance_id", "unknown")

  agent = SWEAgent()
  env = SWEEnv(entry=entry, max_steps=MAX_STEPS, verbose=False)

  all_steps = []

  tf = traj_file
  _write_traj(tf, f"\n{'=' * 60}")
  _write_traj(tf, f"# Instance: {instance_id}")
  _write_traj(tf, f"# Repo: {entry.get('repo', 'N/A')}")
  _write_traj(tf, f"# Problem: {entry.get('problem_statement', '')[:200]}")
  _write_traj(tf, f"{'=' * 60}")
  _write_traj(tf, "")

  # --- Reset ---
  t0 = time.time()
  obs, info = env.reset()
  _write_traj(tf, f"[Reset] ({time.time() - t0:.1f}s)")
  _write_traj(tf, f"Observation: {str(obs)}")
  _write_traj(tf, "")

  agent.reset()
  agent.update_from_env(observation=obs, reward=0.0, done=False, info={})
  guard = ActionGuard(GuardConfig(enabled=ENABLE_GUARD))

  start_time = time.time()

  for step_idx in range(MAX_STEPS):
    elapsed = time.time() - start_time
    if elapsed > TIMEOUT:
      _write_traj(tf, f"\n[TIMEOUT after {elapsed:.0f}s at step {step_idx}]")
      logger.warning("[%s] TIMEOUT at step %d", instance_id, step_idx)
      break

    # --- Model call ---
    t0 = time.time()
    prompt_tokens = 0
    response_tokens = 0
    if USE_API:
      contents, system_instruction = _chat_to_gemini(agent.chat_completions)
      config = {}
      if system_instruction:
        config["system_instruction"] = system_instruction
      response = api_client.models.generate_content(
          model=API_MODEL,
          contents=contents,
          config=config,
      )
      model_response = response.text
      model_time = time.time() - t0
      usage = getattr(response, "usage_metadata", None)
      prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
      response_tokens = getattr(usage, "candidates_token_count", 0) or 0
    else:
      prompt = chat_parser.parse(agent.chat_completions)
      prompt_tokens = len(tokenizer.encode(prompt))
      with sampler_lock:
        out = sampler(prompt, max_generation_steps=MAX_GENERATION_STEPS, echo=False)
      model_response = out.text[0]
      model_time = time.time() - t0
      response_tokens = len(tokenizer.encode(model_response))

    # --- Parse action ---
    action_fn = ""
    action_params_str = ""
    thought = ""
    try:
      thought, action_obj = parse_xml_response(model_response)
      action_fn = action_obj.function_name
      action_params_str = json.dumps(action_obj.parameters, default=str)
    except Exception:
      thought = model_response[:300]

    # --- Update agent & guard ---
    action_result = agent.update_from_model(model_response)
    verdict = guard.evaluate(action_result.action)
    guard_blocked = verdict.blocked

    step_time = 0.0
    if guard_blocked:
      obs, reward, done, info = verdict.message, 0.0, False, {
          "guard_blocked": True, "guard_reason": verdict.reason,
      }
    else:
      t0 = time.time()
      try:
        obs, reward, done, info = env.step(action_result.action)
        step_time = time.time() - t0
      except Exception as e:
        obs, reward, done, info = str(e), 0.0, True, {}
        step_time = time.time() - t0
      guard.record_outcome(action_result.action, str(obs))

    obs_str = str(obs)

    # --- Write trajectory ---
    _write_traj(tf, f"{'─' * 60}")
    _write_traj(tf, f"Step {step_idx + 1}/{MAX_STEPS}  "
                     f"model={model_time:.1f}s  env={step_time:.1f}s  "
                     f"prompt_tok={prompt_tokens}  resp_tok={response_tokens}")
    if thought:
      _write_traj(tf, f"\n[Thought]\n{thought}")
    _write_traj(tf, f"\n[Action] {action_fn}")
    if action_params_str:
      _write_traj(tf, action_params_str)
    if guard_blocked:
      _write_traj(tf, f"\n[GUARD BLOCKED] {verdict.reason}")
    _write_traj(tf, f"\n[Observation]\n{obs_str}")
    _write_traj(tf, f"\n[Reward] {reward}  [Done] {done}")
    _write_traj(tf, "")

    # --- Log progress to stdout ---
    logger.info(
        "[%s] step %d/%d  action=%s  reward=%.1f  done=%s  (model=%.1fs env=%.1fs)",
        instance_id, step_idx + 1, MAX_STEPS, action_fn,
        reward, done, model_time, step_time,
    )

    agent.update_from_env(observation=obs, reward=reward, done=done, info=info)

    if done:
      break

  total_time = time.time() - start_time

  # --- Reward ---
  reward_val = 0.0
  try:
    t0 = time.time()
    result = env.env.runtime._calculate_reward(get_test_output=True)
    if isinstance(result, tuple):
      reward_val = result[0]
    else:
      reward_val = float(result)
  except Exception as e:
    logger.error("[%s] Reward computation failed: %s", instance_id, e)

  _write_traj(tf, f"{'=' * 60}")
  _write_traj(tf, f"FINAL REWARD: {reward_val}")
  _write_traj(tf, f"Total steps: {len(all_steps)}  Total time: {total_time:.1f}s")
  _write_traj(tf, f"{'=' * 60}")

  # --- Cleanup ---
  try:
    env.close()
  except Exception:
    pass

  return reward_val


# ========================== Main ==========================

if __name__ == "__main__":
  traj_log = os.path.join(OUTPUT_DIR, "trajectory.log")
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  results = []

  with open(traj_log, "w") as tf:
    for task_i, entry in enumerate(selected_entries):
      instance_id = entry.get("instance_id", f"idx_{task_i}")

      logger.info(
          "===== [%d/%d] Starting %s =====",
          task_i + 1, len(selected_entries), instance_id,
      )

      try:
        reward = run_single_eval(entry, tf)
      except Exception as e:
        logger.error("[%s] FAILED: %s", instance_id, e)
        traceback.print_exc()
        reward = 0.0

      results.append({"instance_id": instance_id, "reward": reward})
      logger.info("[%s] reward=%.1f", instance_id, reward)

  # --- Print summary to stdout ---
  num_solved = sum(1 for r in results if r["reward"] > 0)
  logger.info(
      "===== Done: %d/%d solved (%.1f%%) =====",
      num_solved, len(results),
      100.0 * num_solved / len(results) if results else 0,
  )
  for r in results:
    marker = "PASS" if r["reward"] > 0 else "FAIL"
    logger.info("  [%s] %s  reward=%.1f", marker, r["instance_id"], r["reward"])
  logger.info("Trajectory log -> %s", traj_log)
