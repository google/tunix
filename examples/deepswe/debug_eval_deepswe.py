#!/usr/bin/env python
"""Debug single-instance DeepSWE evaluation.

Runs ONE SWE-bench instance without the RolloutOrchestrator abstraction,
printing detailed logs at every stage so you can diagnose why reward=0.

Diagnostics printed:
  - Dataset entry metadata (instance_id, FAIL_TO_PASS, PASS_TO_PASS, etc.)
  - Each agent step: model response, parsed action, env observation
  - Reward computation: raw test output from /run_tests.sh, parsed results
  - Full conversation history dumped to JSON

Usage:
  # Default: first instance
  TASK_INDEX=0 python debug_eval_deepswe.py

  # Specific instance by ID
  INSTANCE_ID=django__django-12345 python debug_eval_deepswe.py
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

# Which task to debug: by index or by instance_id
TASK_INDEX = int(os.getenv("TASK_INDEX", "0"))
INSTANCE_ID = os.getenv("INSTANCE_ID", "")  # overrides TASK_INDEX if set

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/scratch/debug_eval")

# ========================== Logging ==========================

for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "=" * 72

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
)

entries = [e for e in dataset if "docker_image" in e]
logger.info("Dataset has %d instances with docker_image.", len(entries))

# Select target entry
entry = None
if INSTANCE_ID:
  for e in entries:
    if e.get("instance_id") == INSTANCE_ID:
      entry = e
      break
  if entry is None:
    logger.error("Instance ID %s not found in dataset.", INSTANCE_ID)
    sys.exit(1)
  logger.info("Selected instance by ID: %s", INSTANCE_ID)
else:
  if TASK_INDEX >= len(entries):
    logger.error("TASK_INDEX=%d but only %d entries.", TASK_INDEX, len(entries))
    sys.exit(1)
  entry = entries[TASK_INDEX]
  logger.info("Selected instance by index: %d", TASK_INDEX)

# ========================== Print Entry Metadata ==========================

print(f"\n{SEP}")
print("INSTANCE METADATA")
print(SEP)
print(f"  instance_id   : {entry.get('instance_id', 'N/A')}")
print(f"  repo          : {entry.get('repo', 'N/A')}")
print(f"  docker_image  : {entry.get('docker_image', 'N/A')}")
print(f"  FAIL_TO_PASS  : {entry.get('FAIL_TO_PASS', [])}")
print(f"  PASS_TO_PASS  : {entry.get('PASS_TO_PASS', [])}")
problem_stmt = entry.get("problem_statement", "")
print(f"  problem_statement (first 500 chars):")
print(f"    {problem_stmt[:500]}")
print(SEP)

# ========================== Kubernetes ==========================

os.environ.setdefault("KUBECONFIG", "~/.kube/config")
os.environ.setdefault("NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool")
os.environ.setdefault("NODE_SELECTOR_VAL", "haoyugao-cpu-pool")

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

if not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH):
  os.makedirs(MODEL_PATH, exist_ok=True)
  snapshot_download(
      repo_id=MODEL_VERSION,
      local_dir=MODEL_PATH,
      local_dir_use_symlinks=False,
  )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
chat_parser = parser.QwenChatTemplateParser(tokenizer)

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

sampler_lock = threading.Lock()

# ========================== Agent & Env ==========================

from swe_agent import SWEAgent, parse_xml_response
from swe_env import SWEEnv

agent = SWEAgent()
env = SWEEnv(entry=entry, max_steps=MAX_STEPS, verbose=True)


# ========================== Manual Rollout ==========================

def run_debug_eval():
  """Drive the agent-env loop manually with verbose logging."""
  all_steps = []

  # --- Reset ---
  print(f"\n{SEP}")
  print("RESETTING ENVIRONMENT")
  print(SEP)
  t0 = time.time()
  obs, info = env.reset()
  reset_time = time.time() - t0
  logger.info("Environment reset in %.1fs", reset_time)
  logger.info("Initial observation length: %d chars", len(str(obs)))
  print(f"  Initial observation (first 300 chars):\n    {str(obs)[:300]}")

  agent.reset()
  agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

  start_time = time.time()

  for step_idx in range(MAX_STEPS):
    elapsed = time.time() - start_time
    if elapsed > TIMEOUT:
      logger.warning("TIMEOUT after %.0fs at step %d", elapsed, step_idx)
      break

    print(f"\n{SEP}")
    print(f"STEP {step_idx + 1}/{MAX_STEPS}  (elapsed: {elapsed:.0f}s)")
    print(SEP)

    # --- Model call ---
    print("\n--- Model Input (last message) ---")
    last_msg = agent.chat_completions[-1] if agent.chat_completions else {}
    print(f"  role: {last_msg.get('role', 'N/A')}")
    content = last_msg.get("content", "")
    print(f"  content length: {len(content)} chars")
    print(f"  content (last 300 chars): ...{content[-300:]}")

    print("\n--- Calling model ---")
    t0 = time.time()
    prompt = chat_parser.parse(agent.chat_completions)
    prompt_tokens = len(tokenizer.encode(prompt))
    logger.info("Prompt tokens: %d", prompt_tokens)

    with sampler_lock:
      out = sampler(prompt, max_generation_steps=MAX_GENERATION_STEPS, echo=False)
    model_response = out.text[0]
    model_time = time.time() - t0
    response_tokens = len(tokenizer.encode(model_response))
    logger.info("Model responded in %.1fs (%d tokens)", model_time, response_tokens)

    # --- Parse response ---
    print("\n--- Model Response ---")
    print(f"  length: {len(model_response)} chars, {response_tokens} tokens")
    # Show full response (truncated if huge)
    if len(model_response) > 2000:
      print(f"  [first 1000 chars]:\n{model_response[:1000]}")
      print(f"  ...[truncated {len(model_response) - 2000} chars]...")
      print(f"  [last 1000 chars]:\n{model_response[-1000:]}")
    else:
      print(model_response)

    print("\n--- Parsed Action ---")
    try:
      thought, action_obj = parse_xml_response(model_response)
      action_str = action_obj.to_xml_string()
      print(f"  function_name : {action_obj.function_name}")
      print(f"  parameters    : {json.dumps(action_obj.parameters, indent=2, default=str)[:500]}")
      print(f"  thought (first 200 chars): {thought[:200]}")
    except Exception as e:
      logger.error("Failed to parse model response: %s", e)
      traceback.print_exc()
      action_str = ""
      thought = model_response

    # --- Update agent with model response ---
    action_result = agent.update_from_model(model_response)
    logger.info("Agent action: %s", action_result.action[:200] if action_result.action else "None")

    # --- Step environment ---
    print("\n--- Stepping Environment ---")
    t0 = time.time()
    try:
      obs, reward, done, info = env.step(action_result.action)
      step_time = time.time() - t0
      logger.info("Env step in %.1fs, reward=%.1f, done=%s", step_time, reward, done)
    except Exception as e:
      logger.error("Env step failed: %s", e)
      traceback.print_exc()
      obs, reward, done, info = str(e), 0.0, True, {}
      step_time = time.time() - t0

    print(f"\n--- Env Observation ---")
    obs_str = str(obs)
    print(f"  length: {len(obs_str)} chars")
    if len(obs_str) > 1000:
      print(f"  [first 500 chars]:\n{obs_str[:500]}")
      print(f"  ...[truncated]...")
      print(f"  [last 500 chars]:\n{obs_str[-500:]}")
    else:
      print(obs_str)

    print(f"  step_reward: {reward}")
    print(f"  done: {done}")
    print(f"  info keys: {list(info.keys()) if isinstance(info, dict) else info}")

    # Record step
    all_steps.append({
        "step": step_idx + 1,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "model_time_s": round(model_time, 1),
        "step_time_s": round(step_time, 1),
        "action_function": getattr(action_obj, "function_name", ""),
        "step_reward": reward,
        "done": done,
        "observation_length": len(obs_str),
        "model_response_length": len(model_response),
    })

    # --- Update agent with env feedback ---
    agent.update_from_env(observation=obs, reward=reward, done=done, info=info)

    if done:
      logger.info("Agent signaled done at step %d.", step_idx + 1)
      break

  total_time = time.time() - start_time
  total_steps = len(all_steps)

  # ========================== Reward Computation ==========================
  print(f"\n{SEP}")
  print("COMPUTING FINAL REWARD")
  print(SEP)

  # Use get_test_output=True to also capture test logs
  reward_val = 0.0
  test_output = ""
  try:
    logger.info("Running /run_tests.sh inside container...")
    t0 = time.time()
    result = env.env.runtime._calculate_reward(get_test_output=True)
    reward_time = time.time() - t0

    if isinstance(result, tuple):
      reward_val, test_output = result[0], result[1]
    else:
      reward_val = float(result)
      test_output = "(test output not captured)"

    logger.info("Reward computation took %.1fs", reward_time)
  except Exception as e:
    logger.error("Reward computation FAILED: %s", e)
    traceback.print_exc()

  print(f"\n  REWARD = {reward_val}")

  # Print test output for diagnosis
  print(f"\n--- Test Output from /run_tests.sh ---")
  test_output_str = str(test_output)
  if len(test_output_str) > 5000:
    print(f"  [first 2500 chars]:\n{test_output_str[:2500]}")
    print(f"  ...[truncated {len(test_output_str) - 5000} chars]...")
    print(f"  [last 2500 chars]:\n{test_output_str[-2500:]}")
  else:
    print(test_output_str)

  # Parse test results for detailed breakdown
  print(f"\n--- Test Result Breakdown ---")
  try:
    parsed = env.env.runtime.parse_logs(test_output_str)
    if parsed:
      fail_to_pass = entry.get("FAIL_TO_PASS", [])
      pass_to_pass = entry.get("PASS_TO_PASS", [])

      print(f"\n  FAIL_TO_PASS tests ({len(fail_to_pass)} expected):")
      for test in fail_to_pass:
        test_key = ".".join(test.split("::")[1:]) if "::" in test else test
        # Find matching key in parsed results
        status = parsed.get(test_key, None)
        if status is None:
          matching = next((k for k in parsed.keys() if test_key in k), None)
          status = parsed.get(matching, "NOT FOUND") if matching else "NOT FOUND"
          if matching:
            test_key = matching
        marker = "OK" if status == "PASSED" else "FAIL"
        print(f"    [{marker}] {test_key}: {status}")

      print(f"\n  PASS_TO_PASS tests ({len(pass_to_pass)} expected):")
      for test in pass_to_pass:
        test_key = ".".join(test.split("::")[1:]) if "::" in test else test
        status = parsed.get(test_key, None)
        if status is None:
          matching = next((k for k in parsed.keys() if test_key in k), None)
          status = parsed.get(matching, "NOT FOUND") if matching else "NOT FOUND"
          if matching:
            test_key = matching
        marker = "OK" if status == "PASSED" else "FAIL"
        print(f"    [{marker}] {test_key}: {status}")

      print(f"\n  All parsed test results ({len(parsed)} tests):")
      for k, v in sorted(parsed.items()):
        print(f"    {v:8s}  {k}")
    else:
      print("  (parse_logs returned empty — test output may be malformed)")
  except Exception as e:
    logger.error("Failed to parse test results: %s", e)
    traceback.print_exc()

  # ========================== Summary ==========================
  print(f"\n{SEP}")
  print("DEBUG SUMMARY")
  print(SEP)
  print(f"  instance_id   : {entry.get('instance_id', 'N/A')}")
  print(f"  total_steps   : {total_steps}")
  print(f"  total_time    : {total_time:.1f}s")
  print(f"  final_reward  : {reward_val}")
  print(f"  model_version : {MODEL_VERSION}")
  print(f"  max_steps     : {MAX_STEPS}")
  print()
  print("  Per-step summary:")
  for s in all_steps:
    print(
        f"    Step {s['step']:2d}: "
        f"fn={s['action_function']:20s} "
        f"model={s['model_time_s']:5.1f}s "
        f"env={s['step_time_s']:5.1f}s "
        f"prompt_tok={s['prompt_tokens']:5d} "
        f"resp_tok={s['response_tokens']:4d} "
        f"done={s['done']}"
    )
  print(SEP)

  # ========================== Save Debug Output ==========================
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  instance_id = entry.get("instance_id", f"idx_{TASK_INDEX}")
  safe_id = instance_id.replace("/", "_")
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  output_file = os.path.join(OUTPUT_DIR, f"debug_{safe_id}_{timestamp}.json")

  debug_record = {
      "instance_id": instance_id,
      "repo": entry.get("repo", ""),
      "docker_image": entry.get("docker_image", ""),
      "FAIL_TO_PASS": entry.get("FAIL_TO_PASS", []),
      "PASS_TO_PASS": entry.get("PASS_TO_PASS", []),
      "problem_statement": entry.get("problem_statement", ""),
      "model_version": MODEL_VERSION,
      "max_steps": MAX_STEPS,
      "total_steps": total_steps,
      "total_time_s": round(total_time, 1),
      "final_reward": reward_val,
      "steps": all_steps,
      "conversation": agent.chat_completions,
      "test_output": test_output_str[:50000],  # cap at 50k chars
  }

  with open(output_file, "w") as f:
    json.dump(debug_record, f, indent=2, default=str)
  logger.info("Debug output saved to %s", output_file)

  # ========================== Cleanup ==========================
  print(f"\n{SEP}")
  print("CLOSING ENVIRONMENT")
  print(SEP)
  try:
    env.close()
    logger.info("Environment closed.")
  except Exception as e:
    logger.error("Error closing environment: %s", e)

  return reward_val


# ========================== Main ==========================

if __name__ == "__main__":
  reward = run_debug_eval()
  print(f"\n{'#' * 72}")
  print(f"#  FINAL RESULT: reward = {reward}")
  print(f"{'#' * 72}")
