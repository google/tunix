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

"""Prepare, run, and analyze distributed-vs-agentic RL benchmarks.

Planning and reporting need only the Python standard library. Training uses
the repository's normal TPU dependencies. The first workload is FrozenLake,
but the recorded step, rollout, generation, and token metrics are not specific
to single-turn or multi-turn agents.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_MODULE = "tunix.experimental.examples.rl_efficiency_benchmark"
MODEL_ID = "google/gemma-4-E2B-it"
COMPUTE_DTYPE = "bfloat16"
LOAD_DTYPE = "float32"
WORKLOAD_DEFAULTS = {
    "batch": 64,
    "generations": 8,
    "steps": 150,
    "warmup": 5,
    "micro_groups": 2,
    "prompt": 2048,
    "response": 2048,
    "turns": 8,
    "seed": 42,
    "dataset_size": 10000,
    "concurrency": 512,
    "max_num_seqs": 32,
    "batched_tokens": 8192,
}


def write_json(path, value):
  Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def agentic_config(workload, model_dir, chip_layout="shared4"):
  """Build explicit overrides of base_agentic_config, in JSON (valid YAML)."""
  w = workload
  if chip_layout not in ("shared4", "split2x2"):
    raise ValueError(f"Unknown agentic chip layout: {chip_layout}")
  mesh = {
      "shape": "(2,1)" if chip_layout == "split2x2" else "(4,1)",
      "axis_names": "('fsdp','tp')",
  }
  rollout_mesh = (
      {"shape": "(1,2)", "axis_names": "('fsdp','tp')"}
      if chip_layout == "split2x2"
      else None
  )
  return {
      "training_mode": "agentic_grpo",
      "model_config": {
          "model_name": "gemma-4-e2b",
          "model_id": MODEL_ID,
          "model_source": "huggingface",
          "model_path": str(model_dir),
          "model_download_path": str(model_dir),
          "mesh": mesh,
          "remat_config": "DECODER",
          "dtype": COMPUTE_DTYPE,
          "rng_seed": w["seed"],
          "use_flash_attention": True,
          "flash_attention_block_size": 256,
          "use_sliding_window_kv_cache": False,
      },
      "actor_model_config": {"load_dtype": LOAD_DTYPE, "mesh": mesh},
      "reference_model_config": {"mesh": None, "same_mesh_as": "actor"},
      "rollout_model_config": {
          "mesh": rollout_mesh,
          "same_mesh_as": None if rollout_mesh else "actor",
      },
      "tokenizer_config": {
          "tokenizer_type": "huggingface",
          "tokenizer_path": str(model_dir),
          "add_bos": False,
          "add_eos": False,
      },
      "agent_class_path": "examples.frozenlake.agent.FrozenLakeAgent",
      "agent_kwargs": {"use_multistep_prompt": True},
      "env_class_path": "examples.frozenlake.env.FrozenLakeEnv",
      "env_kwargs": {"max_steps": w["turns"], "is_slippery": False},
      "data_module": BENCHMARK_MODULE + ".frozenlake_agentic",
      "data_config": {
          "size": w["dataset_size"],
          "seed": w["seed"],
          "limit": w["steps"] * w["batch"],
      },
      "apply_chat_template_to_dataset": False,
      "chat_parser_config": {"type": "gemma4", "enable_thinking": False},
      "batch_size": w["batch"],
      "num_batches": w["steps"],
      "num_train_epochs": 1,
      "train_fraction": 1.0,
      "rl_training_config": {
          "mini_batch_size": w["batch"],
          "train_micro_batch_size": w["micro_groups"],
          "compute_logps_micro_batch_size": w["micro_groups"],
          "compute_logps_chunk_size": 2048,
          "max_steps": w["steps"],
          "eval_every_n_steps": w["steps"] + 1,
          "checkpoint_root_directory": None,
          "checkpointing_options": None,
          "profiler_options": None,
          # Backend logging also enables trajectory logging in agentic.
          # Benchmark hooks write events.jsonl without logging backends.
          "metrics_logging_options": None,
          "actor_optimizer_config": {
              "opt_type": "adamw",
              "learning_rate": 1e-6,
              "schedule_type": None,
              "b1": 0.9,
              "b2": 0.95,
              "weight_decay": 0.0,
              "opt_chain_type": "clip_by_global_norm",
              "chain_kwargs": {"max_norm": 100.0},
          },
      },
      "rollout_engine": "vllm",
      "offload_to_cpu": False,
      "rollout_config": {
          "total_generation_steps": w["response"],
          "max_prompt_length": w["prompt"],
          "kv_cache_size": w["prompt"] + w["response"] + 256,
          "temperature": 0.7,
          "top_p": 1.0,
          "top_k": 0,
          "return_logprobs": True,
          "rollout_vllm_init_with_random_weights": True,
          "rollout_vllm_sampling_kwargs": {"skip_special_tokens": False},
      },
      "vllm_config": {
          "model_version": str(model_dir),
          "hbm_utilization": 0.2,
          "tpu_backend_type": "jax",
          "server_mode": True,
          "async_scheduling": False,
          "max_num_seqs": w["max_num_seqs"],
          "max_num_batched_tokens": w["batched_tokens"],
          "data_parallel_size": 1 if rollout_mesh else 2,
          "tensor_parallel_size": 2,
          "kwargs": {
              "kv_cache_metrics": True,
              "disable_log_stats": False,
              "enable_prefix_caching": False,
              "dtype": "bfloat16",
              "hf_overrides": {
                  "final_logit_softcapping": 30.0,
                  "text_config": {"final_logit_softcapping": 30.0},
                  "architectures": ["Gemma4ForCausalLM"],
              },
          },
      },
      "reward_functions": [],
      "agentic_grpo_config": {
          "exact_token_continuity": True,
          "num_generations": w["generations"],
          "num_iterations": 1,
          "beta": 0.0,
          "epsilon": 0.003,
          "epsilon_high": 0.005,
          "loss_algo": "gspo-token",
          "loss_agg_mode": "sequence-mean-token-mean",
          "kl_loss_mode": "low_var_kl",
          "advantage_estimator": "rloo",
          "sampler_is": "token",
          "sampler_is_threshold": 2.0,
          "max_concurrency": w["concurrency"],
          "off_policy_steps": 0,
          "episode_timeout": 600,
          "overlong_filter": False,
      },
  }


def dist_environment(w, output, model_dir, *, match_training_microbatch=False):
  microbatch_size = (
      w["micro_groups"] * w["generations"]
      if match_training_microbatch
      else 2
  )
  return {
      key: str(value)
      for key, value in {
          "MODEL_NAME": "gemma-4-e2b",
          "MODEL_ID": MODEL_ID,
          "MODEL_DIR": model_dir,
          "TOKENIZER_PATH": model_dir,
          "ARTIFACT_ROOT": output / "artifacts",
          "LOG_ROOT": output,
          "BATCH_SIZE": w["batch"],
          "MINI_BATCH_SIZE": w["batch"],
          "NUM_GENERATIONS": w["generations"],
          "NUM_BATCHES": w["steps"],
          "NUM_EPOCHS": 1,
          "NUM_ITERATIONS": 1,
          "MAX_STEPS": w["steps"],
          "DATASET_SIZE": w["dataset_size"],
          "SEED": w["seed"],
          "SHUFFLE": 1,
          "MAX_TURNS": w["turns"],
          "MAX_PROMPT_LENGTH": w["prompt"],
          "MAX_RESPONSE_LENGTH": w["response"],
          "VLLM_MAX_MODEL_LEN": w["prompt"] + w["response"] + 256,
          "TRAIN_MICRO_BATCH_SIZE": microbatch_size,
          "COMPUTE_LOGPS_MICRO_BATCH_SIZE": microbatch_size,
          "COMPUTE_LOGPS_CHUNK_SIZE": 2048,
          "ROLLOUT_MAX_CONCURRENCY": w["concurrency"],
          "VLLM_MAX_NUM_SEQS": w["max_num_seqs"],
          "VLLM_MAX_NUM_BATCHED_TOKENS": w["batched_tokens"],
          "VLLM_HBM_UTILIZATION": 0.2,
          "TRAINER_TPU_CHIPS": "0,1",
          "TRAINER_FSDP": 2,
          "TRAINER_TP": 1,
          "ROLLOUT_TPU_CHIPS": "2,3",
          "ROLLOUT_FSDP": 1,
          "ROLLOUT_TP": 2,
          "TPU_CHIPS_PER_HOST_BOUNDS": "1,2,1",
          "TPU_HOST_BOUNDS": "1,1,1",
          "WEIGHT_SYNC_MODE": "raiden",
          "OFF_POLICY_STEPS": 0,
          "SAMPLER": "inprocess_vllm",
          "CHECKPOINT_SAVE_INTERVAL_STEPS": 0,
          "CHECKPOINT_ROOT_DIRECTORY": output / "checkpoints",
          "LOG_DIR": "",
          "TRAJECTORY_LOG_DIR": "",
          "IS_SLIPPERY": 0,
          "USE_MULTISTEP_PROMPT": 1,
          "TEMPERATURE": 0.7,
          "TOP_P": 1.0,
          "TOP_K": 0,
          "BETA": 0.0,
          "EPSILON": 0.003,
          "EPSILON_HIGH": 0.005,
          "LOSS_ALGO": "gspo-token",
          "LOSS_AGG_MODE": "sequence-mean-token-mean",
          "KL_LOSS_MODE": "low_var_kl",
          "ADVANTAGE_ESTIMATOR": "rloo",
          "SAMPLER_IS": "token",
          "SAMPLER_IS_THRESHOLD": 2,
          "USE_ROLLOUT_LOGPS": 1,
          "LEARNING_RATE": 1e-6,
          "ADAM_B1": 0.9,
          "ADAM_B2": 0.95,
          "WEIGHT_DECAY": 0,
          "OPT_CHAIN_TYPE": "clip_by_global_norm",
          "MAX_GRAD_NORM": 100,
          "FLASH_ATTENTION_BLOCK_SIZE": 256,
          "MODEL_DTYPE": COMPUTE_DTYPE,
          "MODEL_LOAD_DTYPE": LOAD_DTYPE,
          "EPISODE_TIMEOUT_SECS": 600,
          "DEBUG": 0,
      }.items()
  }


def prepare(args):
  w = {key: getattr(args, key) for key in WORKLOAD_DEFAULTS}
  match_training_microbatch = getattr(args, "match_training_microbatch", False)
  positive = set(w) - {"seed", "warmup"}
  if any(w[key] <= 0 for key in positive):
    raise ValueError("Workload counts must be positive.")
  if w["warmup"] < 0:
    raise ValueError("warmup must be non-negative.")
  if args.timeout <= 0:
    raise ValueError("timeout must be positive.")
  if w["steps"] <= w["warmup"] or w["generations"] < 2:
    raise ValueError("Require steps > warmup and generations >= 2.")
  if w["response"] <= 512:
    raise ValueError(
        "response must exceed the 512-token environment reserve."
    )
  if (
      w["batch"] % w["micro_groups"]
      or w["steps"] * w["batch"] > w["dataset_size"]
  ):
    raise ValueError(
        "Require batch divisible by micro_groups and enough dataset rows."
    )
  output = args.output.resolve()
  model_dir = args.model_dir.resolve()
  if not any(model_dir.glob("*.safetensors")):
    raise ValueError(
        "Pre-download the model; --model-dir must contain safetensors."
    )
  if not (model_dir / "config.json").is_file():
    raise ValueError("--model-dir must contain the model config.json.")
  # A new directory prevents accidental checkpoint resume or mixed-run events.
  output.mkdir(parents=True, exist_ok=False)
  env = {
      "WANDB_MODE": "disabled",
      "TUNIX_BENCHMARK_DIR": str(output),
      "TUNIX_BENCHMARK_TIMING": args.timing_mode,
      "PYTHON_BIN": sys.executable,
      "PYTHONUNBUFFERED": "1",
      "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
      "JAX_LOG_COMPILES": "1",
  }
  if args.mode == "dist":
    env["BENCHMARK_PYTHON"] = sys.executable
    env["PYTHON_BIN"] = str(Path(__file__).parent / "python.sh")
    env.update(dist_environment(
        w, output, model_dir,
        match_training_microbatch=match_training_microbatch,
    ))
    training_contract = {
        "compute_dtype": env["MODEL_DTYPE"],
        "load_dtype": env["MODEL_LOAD_DTYPE"],
        "exact_token_continuity": True,
    }
    command = [
        "bash",
        str(
            ROOT
            / "tunix/experimental/examples/frozenlake_dist/run_gemma4_e2b.sh"
        ),
    ]
  else:
    config = agentic_config(w, model_dir, args.agentic_chip_layout)
    training_contract = {
        "compute_dtype": config["model_config"]["dtype"],
        "load_dtype": config["actor_model_config"]["load_dtype"],
        "exact_token_continuity": config["agentic_grpo_config"][
            "exact_token_continuity"
        ],
    }
    write_json(output / "agentic.json", config)
    env.update({
        "JAX_PLATFORMS": "tpu,cpu",
        "TPU_VISIBLE_DEVICES": "0,1,2,3",
        "TPU_VISIBLE_CHIPS": "0,1,2,3",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,4,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "LIBTPU_INIT_ARGS": (
            "--deepsea_chips_per_host_bounds=1,4,1 --deepsea_host_bounds=1,1,1"
        ),
    })
    command = [
        sys.executable,
        "-m",
        BENCHMARK_MODULE + ".frozenlake_agentic",
        "tunix/cli/base_agentic_config.yaml",
        f"override_config_file={output / 'agentic.json'}",
    ]
  cache = args.cache_root.resolve() / args.mode
  env.update({
      "VLLM_CACHE_ROOT": str(cache / "vllm"),
      "VLLM_XLA_CACHE_PATH": str(cache / "vllm/xla_cache"),
      "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
  })
  versions = {}
  for name in (
      "jax",
      "jaxlib",
      "libtpu",
      "vllm",
      "tpu-inference",
      "tpu_sync_jax",
      "flax",
      "optax",
  ):
    try:
      versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      versions[name] = None
  manifest = {
      "schema": 3,
      "timing_mode": args.timing_mode,
      "training_contract": training_contract,
      "mode": args.mode,
      "agentic_chip_layout": args.agentic_chip_layout,
      "match_training_microbatch": match_training_microbatch,
      "workload": w,
      "chips": 4,
      "hardware_label": args.hardware,
      "hostname": platform.node(),
      "python": sys.version,
      "host": platform.platform(),
      "versions": versions,
      "model_dir": str(model_dir),
      "cache_root": str(cache),
      "command": command,
      "environment": env,
      "revision": (
          subprocess.check_output(
              ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
          ).strip()
      ),
      "status": subprocess.check_output(
          ["git", "status", "--porcelain"], cwd=ROOT, text=True
      ),
  }
  manifest["diff_sha256"] = hashlib.sha256(
      subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
  ).hexdigest()
  manifest["benchmark_sha256"] = {
      p.name: hashlib.sha256(p.read_bytes()).hexdigest()
      for p in Path(__file__).parent.iterdir()
      if p.suffix in {".py", ".sh"}
  }
  manifest["model_metadata_sha256"] = {
      p.name: hashlib.sha256(p.read_bytes()).hexdigest()
      for p in model_dir.glob("*.json")
  }
  manifest["model_shards"] = {
      p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
      for p in model_dir.glob("*.safetensors")
  }
  manifest["runtime_environment"] = {
      name: os.environ.get(name)
      for name in (
          "XLA_FLAGS",
          "JAX_ENABLE_COMPILATION_CACHE",
          "JAX_DEFAULT_MATMUL_PRECISION",
          "SKIP_JAX_PRECOMPILE",
      )
  }
  write_json(output / "manifest.json", manifest)
  print(json.dumps(manifest, indent=2))
  if not args.execute:
    print(
        "Plan only. Re-run with --execute and a NEW --output directory to"
        " train."
    )
    return
  start = time.monotonic()
  with (output / "launcher.log").open("x") as log:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, **env},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
      code = process.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
      try:
        os.killpg(process.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass
      try:
        process.wait(timeout=45)
      except subprocess.TimeoutExpired:
        pass
      # Children can survive after their launcher has exited.
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      process.wait()
      code = 124
  write_json(
      output / "result.json",
      {"returncode": code, "total_run_time_seconds": time.monotonic() - start},
  )
  if code:
    raise RuntimeError(
        f"Training failed ({code}); inspect {output}/launcher.log and worker"
        " logs."
    )
  summary = summarize(output)
  write_json(output / "summary.json", summary)


def percentile(values, percent):
  """Returns a linearly interpolated percentile without a NumPy dependency."""
  ordered = sorted(values)
  if not ordered:
    raise ValueError("Cannot compute a percentile of an empty sequence.")
  position = (len(ordered) - 1) * percent / 100
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  fraction = position - lower
  return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def rollout_summary(events, steady_steps, expected):
  """Aggregates identical trajectory/model-call boundaries across both paths."""
  trajectories = [e for e in events if e["kind"] == "rollout_trajectory"]
  generations = [e for e in events if e["kind"] == "generation"]
  if any(not e.get("ok", True) for e in trajectories + generations):
    raise ValueError("Run contains failed rollout calls.")
  completed_statuses = {
      "SUCCEEDED",
      "MAX_STEPS_REACHED",
      "MAX_CONTEXT_LIMIT_REACHED",
  }
  if any(e.get("status") not in completed_statuses for e in trajectories):
    raise ValueError("Run contains incomplete or timed-out trajectories.")

  collection_times = []
  measured_trajectories = []
  measured_generations = []
  for step in steady_steps:
    step_trajectories = [e for e in trajectories if e.get("step") == step]
    step_generations = [e for e in generations if e.get("step") == step]
    if len(step_trajectories) != expected:
      raise ValueError(
          f"Rollout step {step} has {len(step_trajectories)} trajectories;"
          f" expected {expected}."
      )
    if not step_generations:
      raise ValueError(f"Rollout step {step} has no model generation calls.")
    start = min(e["start"] for e in step_trajectories)
    end = max(e["end"] for e in step_trajectories)
    seconds = end - start
    if seconds <= 0:
      raise ValueError("Invalid rollout timing interval.")
    collection_times.append(seconds)
    measured_trajectories.extend(step_trajectories)
    measured_generations.extend(step_generations)

  prompt_tokens = sum(e["prompt_tokens"] for e in measured_generations)
  generated_by_trajectory = {}

  def identity(event):
    return (
        event.get("step"),
        str(event.get("group_id")),
        event.get("pair_index"),
    )

  for event in measured_generations:
    key = identity(event)
    generated_by_trajectory[key] = (
        generated_by_trajectory.get(key, 0) + event["generated_tokens"]
    )
  trajectory_ids = {identity(event) for event in measured_trajectories}
  if (
      len(trajectory_ids) != len(measured_trajectories)
      or trajectory_ids != generated_by_trajectory.keys()
  ):
    raise ValueError(
        "Generation events do not map one-to-one to measured trajectories."
    )
  trajectory_by_id = {identity(e): e for e in measured_trajectories}
  for event in measured_generations:
    trajectory = trajectory_by_id[identity(event)]
    if (
        not trajectory["start"]
        <= event["start"]
        <= event["end"]
        <= trajectory["end"]
    ):
      raise ValueError("Generation timing lies outside its trajectory.")
  per_trajectory_tokens = list(generated_by_trajectory.values())
  trajectory_latencies = [e["seconds"] for e in measured_trajectories]
  generation_latencies = [e["seconds"] for e in measured_generations]
  total_environment_seconds = sum(
      e.get("environment_seconds", 0) for e in measured_trajectories
  )
  rewards = [
      e["reward"] for e in measured_trajectories if e.get("reward") is not None
  ]
  total_generation_call_seconds = sum(generation_latencies)
  return {
      "collection_time_seconds": timing_summary(collection_times),
      "trajectory_latency_seconds": {
          "mean": statistics.mean(trajectory_latencies),
          "p90": percentile(trajectory_latencies, 90),
      },
      "average_output_tokens_per_trajectory": statistics.mean(
          per_trajectory_tokens
      ),
      "average_prompt_tokens_per_model_call": (
          prompt_tokens / len(measured_generations)
      ),
      "average_turns_per_trajectory": statistics.mean(
          e.get("turns", 0) for e in measured_trajectories
      ),
      "average_reward_per_trajectory": (
          statistics.mean(rewards) if rewards else None
      ),
      "average_time_per_trajectory_seconds": {
          "model_generation_api": (
              total_generation_call_seconds / len(measured_trajectories)
          ),
          "environment": total_environment_seconds / len(measured_trajectories),
      },
      "trajectory_outcome_percent": {
          status: (
              100
              * sum(e.get("status") == status for e in measured_trajectories)
              / len(measured_trajectories)
          )
          for status in sorted(
              {e.get("status", "") for e in measured_trajectories}
          )
      },
  }


def timing_summary(values):
  return {
      "mean": statistics.mean(values),
      "median": statistics.median(values),
      "per_step": values,
  }


def api_call_summary(events, first, last, step_count):
  """Normalizes host-observed API usage by measured training steps."""
  durations = {}
  for event in events:
    if (
        event["kind"] == "span"
        and event["end"] > first
        and event["start"] < last
    ):
      durations.setdefault(event["name"], []).append(
          min(event["end"], last) - max(event["start"], first)
      )
  return {
      name: {
          "calls_per_step": len(values) / step_count,
          "host_time_per_step_seconds": sum(values) / step_count,
      }
      for name, values in sorted(durations.items())
  }


def input_work_summary(events, name, steady_steps):
  """Summarizes logical padded input volume without reading device contents."""
  selected = [
      event
      for event in events
      if event["kind"] == "span"
      and event["name"] == name
      and event.get("step") in steady_steps
      and event.get("ok", True)
  ]
  shapes = [shape for event in selected for shape in event.get("shapes") or []]
  if not selected or not shapes:
    return None
  trajectories = 0
  padded_token_slots = 0
  unique_shapes = set()
  for shape in shapes:
    prompt_shape = shape["prompt"]
    completion_shape = shape["completion"]
    if not prompt_shape or not completion_shape:
      continue
    trajectories += int(prompt_shape[0])
    padded_token_slots += math.prod(prompt_shape) + math.prod(completion_shape)
    unique_shapes.add(json.dumps(shape, sort_keys=True))
  return {
      "trajectories_per_call": trajectories / len(selected),
      "trajectories_per_step": trajectories / len(steady_steps),
      "padded_token_slots_per_step": padded_token_slots / len(steady_steps),
      "unique_tensor_shapes": [
          json.loads(shape) for shape in sorted(unique_shapes)
      ],
  }


def pipeline_summary(events, steady_steps):
  """Shows where rollout collection sits within each end-to-end step."""
  trajectories = [e for e in events if e["kind"] == "rollout_trajectory"]
  start_delays = []
  post_rollout_tails = []
  for step_event in steady_steps:
    step = step_event["step"]
    step_trajectories = [e for e in trajectories if e.get("step") == step]
    rollout_start = min(e["start"] for e in step_trajectories)
    rollout_end = max(e["end"] for e in step_trajectories)
    if (
        not step_event["start"]
        <= rollout_start
        <= rollout_end
        <= step_event["end"]
    ):
      raise ValueError("Trajectory timing lies outside its training step.")
    start_delays.append(rollout_start - step_event["start"])
    post_rollout_tails.append(step_event["end"] - rollout_end)
  return {
      "rollout_start_delay_seconds": timing_summary(start_delays),
      "post_rollout_tail_seconds": timing_summary(post_rollout_tails),
  }


def validate_completion(events, mode, timing_mode):
  """Requires observed completion waits, not just hook installation claims."""
  for event in events:
    if event["kind"] in {
        "span",
        "training_step",
        "generation",
        "rollout_trajectory",
    }:
      start, end, seconds = (event[k] for k in ("start", "end", "seconds"))
      if (
          not all(math.isfinite(v) for v in (start, end, seconds))
          or end < start
          or not math.isclose(seconds, end - start, rel_tol=1e-7, abs_tol=1e-6)
      ):
        raise ValueError("Invalid event timing interval.")
  spans = [e for e in events if e["kind"] == "span"]
  if mode == "agentic":
    required = {"weight_sync"}
    if timing_mode == "stages":
      required.update(("training", "actor_log_probs"))
    if any(
        e.get("device_complete") is not True
        for e in spans
        if e["name"] in required
    ):
      raise ValueError("Missing device completion evidence for agentic stage.")
    return

  # This workload launches every process on the same host; monotonic clocks
  # share an origin. Each worker completion must fall inside its caller span.
  pairs = [("weight_install", "weight_sync")]
  if timing_mode == "stages":
    pairs.extend(
        (m, "worker_rpc_" + m)
        for m in (
            "fwd_bwd",
            "update",
            "per_token_logps",
        )
    )
  for method, name in pairs:
    callers = [e for e in spans if e["name"] == name]
    completions = [
        e
        for e in events
        if e["kind"] == "device_completion" and e["method"] == method
    ]
    if not callers or not completions:
      raise ValueError(f"Missing device completion evidence: {method}.")
    for caller in callers:
      count = sum(
          caller["start"] <= e["time"] <= caller["end"] for e in completions
      )
      if count < 1 or (method != "weight_install" and count != 1):
        raise ValueError(f"Device completion does not match caller: {method}.")
    if any(
        sum(c["start"] <= e["time"] <= c["end"] for c in callers) != 1
        for e in completions
    ):
      raise ValueError(f"Orphan or ambiguous device completion: {method}.")

  if timing_mode == "stages":
    for outer_name, rpc_name in (
        ("training", "worker_rpc_fwd_bwd"),
        ("actor_log_probs", "worker_rpc_per_token_logps"),
    ):
      outers = [e for e in spans if e["name"] == outer_name]
      calls = [e for e in spans if e["name"] == rpc_name]
      for outer in outers:
        if (
            sum(
                outer["start"] <= c["start"] <= c["end"] <= outer["end"]
                for c in calls
            )
            != 1
        ):
          raise ValueError(
              f"Stage ended before its worker completed: {outer_name}."
          )
      for call in calls:
        if (
            sum(
                o["start"] <= call["start"] <= call["end"] <= o["end"]
                for o in outers
            )
            != 1
        ):
          raise ValueError(f"Worker call lies outside its stage: {rpc_name}.")
    for call in (e for e in spans if e["name"] == "worker_rpc_update"):
      if (
          sum(
              e["name"] == "training"
              and e["start"] <= call["start"] <= call["end"] <= e["end"]
              for e in spans
          )
          != 1
      ):
        raise ValueError("Optimizer completion lies outside training stage.")


def summarize(directory):
  directory = Path(directory)
  manifest = json.loads((directory / "manifest.json").read_text())
  if manifest.get("schema") != 3:
    raise ValueError(
        "Rerun with benchmark schema 3 for completion-aware timing."
    )
  timing_mode = manifest.get("timing_mode")
  if timing_mode not in ("stages", "pipeline"):
    raise ValueError("Missing or invalid timing mode.")
  result = json.loads((directory / "result.json").read_text())
  if result["returncode"] != 0:
    raise ValueError(f"Failed run: {directory}")
  event_files = sorted(directory.glob("events*.jsonl"))
  events = [
      json.loads(line)
      for path in event_files
      for line in path.read_text().splitlines()
  ]
  contracts = [e for e in events if e["kind"] == "timing_contract"]
  required = (
      {"dist", "trainer-worker", "rollout-worker"}
      if manifest["mode"] == "dist"
      else {"agentic"}
  )
  if {e["process"] for e in contracts} != required or any(
      e["timing_mode"] != timing_mode for e in contracts
  ):
    raise ValueError("Missing or mismatched timing hooks; rerun all workers.")
  training_steps = [e for e in events if e["kind"] == "training_step"]
  w = manifest["workload"]
  expected = w["batch"] * w["generations"]
  if len(training_steps) != w["steps"] or any(
      e["trained_trajectories"] != expected for e in training_steps
  ):
    raise ValueError(
        "Incomplete run or training work mismatch; no valid speed comparison."
    )
  if any(e["kind"] == "span" and not e.get("ok", True) for e in events):
    raise ValueError(
        "Run contains failed calls; inspect events before comparing."
    )
  validate_completion(events, manifest["mode"], timing_mode)
  if [e["step"] for e in training_steps] != list(range(w["steps"])) or any(
      previous["end"] != current["start"]
      for previous, current in zip(training_steps, training_steps[1:])
  ):
    raise ValueError("Training step boundaries are not contiguous and ordered.")
  steady = training_steps[w["warmup"] :]
  warmup_durations = [e["seconds"] for e in training_steps[: w["warmup"]]]
  durations = [e["seconds"] for e in steady]
  if any(d <= 0 for d in durations):
    raise ValueError("Invalid timing interval.")
  first, last = steady[0]["start"], steady[-1]["end"]
  steady_step_ids = {e["step"] for e in steady}
  trajectories = [
      e
      for e in events
      if e["kind"] == "rollout_trajectory" and e.get("step") in steady_step_ids
  ]
  if any(e.get("exact_token_continuity") is not True for e in trajectories):
    raise ValueError(
        "Benchmark requires exact token continuity in both stacks."
    )
  rollout = rollout_summary(events, sorted(steady_step_ids), expected)
  return {
      "mode": manifest["mode"],
      "timing_mode": timing_mode,
      "training_contract": manifest["training_contract"],
      "end_to_end": {
          "steady_state_step_time_seconds": timing_summary(durations),
          "warmup_step_time_seconds": (
              timing_summary(warmup_durations) if warmup_durations else None
          ),
      },
      "rollout": rollout,
      "pipeline": pipeline_summary(events, steady),
      "api_calls": api_call_summary(events, first, last, len(steady)),
      "training_input": input_work_summary(events, "training", steady_step_ids),
      "actor_log_probs_input": input_work_summary(
          events, "actor_log_probs", steady_step_ids
      ),
  }


def _path_value(value, path):
  for key in path:
    if value is None:
      return None
    value = value.get(key)
  return value


def _compare_values(agentic_values, dist_values, lower_is_better):
  if not agentic_values and not dist_values:
    return None
  agentic = statistics.median(agentic_values) if agentic_values else None
  dist = statistics.median(dist_values) if dist_values else None
  if agentic is None or dist is None:
    interpretation = "one_stack_only"
  elif math.isclose(agentic, dist, rel_tol=1e-9, abs_tol=1e-12):
    interpretation = "same"
  elif lower_is_better is None:
    interpretation = "different_not_ranked"
  elif (dist < agentic) == lower_is_better:
    interpretation = "dist_faster"
  else:
    interpretation = "dist_slower"
  return {
      "agentic": agentic,
      "dist": dist,
      "dist_minus_agentic": (
          dist - agentic if dist is not None and agentic is not None else None
      ),
      "dist_relative_to_agentic_percent": (
          100 * (dist / agentic - 1) if dist is not None and agentic else None
      ),
      "lower_is_better": lower_is_better,
      "interpretation": interpretation,
  }


def compare_summaries(summaries):
  """Builds a direct, multi-run distributed-vs-agentic gap report."""
  timing_modes = {summary.get("timing_mode") for summary in summaries}
  if len(timing_modes) != 1 or not timing_modes <= {"stages", "pipeline"}:
    raise ValueError(
        "Timing mode mismatch; compare stages and pipeline separately."
    )
  timing_mode = next(iter(timing_modes))
  by_mode = {
      mode: [summary for summary in summaries if summary["mode"] == mode]
      for mode in ("agentic", "dist")
  }
  if not all(by_mode.values()):
    raise ValueError(
        "Comparison requires at least one agentic and one dist run."
    )
  contracts = [summary.get("training_contract") for summary in summaries]
  if not contracts[0] or any(
      contract != contracts[0] for contract in contracts
  ):
    raise ValueError(
        "Training configuration mismatch; no valid speed comparison."
    )

  def compare(path, lower_is_better):
    return _compare_values(
        [
            value
            for summary in by_mode["agentic"]
            if (value := _path_value(summary, path)) is not None
        ],
        [
            value
            for summary in by_mode["dist"]
            if (value := _path_value(summary, path)) is not None
        ],
        lower_is_better,
    )

  metrics = {}
  for name, path, lower_is_better in (
      (
          "end_to_end_step_time_seconds",
          ("end_to_end", "steady_state_step_time_seconds", "mean"),
          True,
      ),
      (
          "rollout_collection_time_seconds",
          ("rollout", "collection_time_seconds", "mean"),
          True,
      ),
      (
          "trajectory_latency_mean_seconds",
          ("rollout", "trajectory_latency_seconds", "mean"),
          True,
      ),
      (
          "trajectory_latency_p90_seconds",
          ("rollout", "trajectory_latency_seconds", "p90"),
          True,
      ),
      (
          "rollout_start_delay_seconds",
          ("pipeline", "rollout_start_delay_seconds", "mean"),
          True,
      ),
      (
          "post_rollout_tail_seconds",
          ("pipeline", "post_rollout_tail_seconds", "mean"),
          True,
      ),
      (
          "average_output_tokens_per_trajectory",
          ("rollout", "average_output_tokens_per_trajectory"),
          None,
      ),
      (
          "average_prompt_tokens_per_model_call",
          ("rollout", "average_prompt_tokens_per_model_call"),
          None,
      ),
      (
          "average_turns_per_trajectory",
          ("rollout", "average_turns_per_trajectory"),
          None,
      ),
      (
          "average_reward_per_trajectory",
          ("rollout", "average_reward_per_trajectory"),
          None,
      ),
      (
          "model_generation_time_per_trajectory_seconds",
          (
              "rollout",
              "average_time_per_trajectory_seconds",
              "model_generation_api",
          ),
          True,
      ),
      (
          "environment_time_per_trajectory_seconds",
          ("rollout", "average_time_per_trajectory_seconds", "environment"),
          True,
      ),
  ):
    metrics[name] = compare(path, lower_is_better)

  stages = sorted(
      {stage for summary in summaries for stage in summary.get("api_calls", {})}
  )
  api_calls = {}
  for stage in stages:
    api_calls[stage] = {}
    for field in ("calls_per_step", "host_time_per_step_seconds"):
      api_calls[stage][field] = compare(("api_calls", stage, field), None)

  outcomes = sorted({
      outcome
      for summary in summaries
      for outcome in summary["rollout"]["trajectory_outcome_percent"]
  })

  # Compare the same logical work per training step. Use only outer API
  # spans here: nested worker RPCs remain in api_calls for diagnosis.
  pipeline_stages = {
      "rollout_collection": metrics["rollout_collection_time_seconds"],
      **{
          name: compare(("api_calls", name, "host_time_per_step_seconds"), True)
          for name in (
              "actor_log_probs",
              "training",
              "weight_sync",
          )
          if timing_mode == "stages"
      },
  }

  def sequence_shapes_match(section):
    shapes = [
        frozenset(
            (
                tuple(shape["prompt"][1:]),
                tuple(shape["completion"][1:]),
            )
            for shape in (summary.get(section) or {}).get(
                "unique_tensor_shapes", []
            )
        )
        for summary in summaries
    ]
    return bool(shapes[0]) and all(value == shapes[0] for value in shapes[1:])

  def scalar_match(section, field):
    values = [(summary.get(section) or {}).get(field) for summary in summaries]
    return all(value is not None for value in values) and all(
        math.isclose(values[0], value, rel_tol=1e-9, abs_tol=1e-12)
        for value in values[1:]
    )

  parity_checks = {
      "training_sequence_lengths_match": sequence_shapes_match("training_input"),
      "actor_log_probs_sequence_lengths_match": sequence_shapes_match(
          "actor_log_probs_input"
      ),
      "training_trajectories_per_step_match": scalar_match(
          "training_input", "trajectories_per_step"
      ),
      "training_padded_token_slots_per_step_match": scalar_match(
          "training_input", "padded_token_slots_per_step"
      ),
      "actor_log_probs_trajectories_per_step_match": scalar_match(
          "actor_log_probs_input", "trajectories_per_step"
      ),
      "actor_log_probs_padded_token_slots_per_step_match": scalar_match(
          "actor_log_probs_input", "padded_token_slots_per_step"
      ),
  }
  failed = [name for name, matched in parity_checks.items() if not matched]
  if failed:
    raise ValueError(
        "Training work mismatch; no valid speed comparison: "
        + ", ".join(failed)
    )
  return {
      "timing_mode": timing_mode,
      "run_count": {mode: len(values) for mode, values in by_mode.items()},
      "parity_checks": parity_checks,
      "step_time_per_run_seconds": {
          mode: [
              summary["end_to_end"]["steady_state_step_time_seconds"]["mean"]
              for summary in values
          ]
          for mode, values in by_mode.items()
      },
      "measurement_notes": [
          (
              "Trajectory latency ends after the episode, reward computation"
              " and environment cleanup complete. Collection spans the first"
              " trajectory start through the last trajectory completion; it"
              " excludes dispatch before that and result delivery afterward."
          ),
          (
              "Primary step time is the median across run means, including"
              " weight sync."
          ),
          (
              "Online rollouts are sampled independently; compare token counts,"
              " turns and rewards alongside speed."
          ),
          (
              "API spans are host wall times, include remote compute/waits and"
              " may overlap or nest; do not add them."
          ),
          (
              "Stage mode waits for training state, log-prob results and"
              " rollout weights to be ready. Module times include dispatch and"
              " RPC; training includes forward/backward and optimizer updates."
              " Only the four large modules are compared; preprocessing and"
              " other gaps mean their times need not sum to the full step."
          ),
          (
              "Stage-mode fences alter overlap: its full-step time describes"
              " synchronized execution. Use pipeline mode for end-to-end"
              " performance; its unfenced training API times cannot rank"
              " device-complete module performance. Do not mix the two modes."
          ),
          (
              "Repeat both modes in alternating order with warmed caches; a"
              " single run does not establish variability."
          ),
      ],
      "metrics": metrics,
      "pipeline_stages_seconds_per_step": pipeline_stages,
      "api_calls": api_calls,
      "trajectory_outcome_percent": {
          outcome: _compare_values(
              [
                  summary["rollout"]["trajectory_outcome_percent"].get(
                      outcome, 0.0
                  )
                  for summary in by_mode["agentic"]
              ],
              [
                  summary["rollout"]["trajectory_outcome_percent"].get(
                      outcome, 0.0
                  )
                  for summary in by_mode["dist"]
              ],
              None,
          )
          for outcome in outcomes
      },
  }


def report(runs):
  manifests = [json.loads((p / "manifest.json").read_text()) for p in runs]
  if any(manifest.get("schema") != 3 for manifest in manifests):
    raise ValueError(
        "Rerun with benchmark schema 3 for completion-aware timing."
    )
  for manifest in manifests[1:]:
    for key in (
        "schema",
        "timing_mode",
        "training_contract",
        "workload",
        "versions",
        "revision",
        "model_dir",
        "diff_sha256",
        "benchmark_sha256",
        "hardware_label",
        "hostname",
        "python",
        "model_metadata_sha256",
        "model_shards",
        "runtime_environment",
    ):
      if manifest[key] != manifests[0][key]:
        raise ValueError(
            f"Runs differ in {key}; report each separately or rerun matched"
            " configurations."
        )
  chip_layouts = [
      manifest.get("agentic_chip_layout", "shared4") for manifest in manifests
  ]
  if any(value != chip_layouts[0] for value in chip_layouts):
    raise ValueError("Runs differ in agentic_chip_layout.")
  datasets = [json.loads((p / "dataset.json").read_text()) for p in runs]
  if any(d != datasets[0] for d in datasets[1:]):
    raise ValueError("Dataset order/content mismatch.")
  summaries = [summarize(p) for p in runs]
  comparison = compare_summaries(summaries)
  comparison["agentic_chip_layout"] = chip_layouts[0]
  match_microbatch = [
      manifest.get("match_training_microbatch", False) for manifest in manifests
  ]
  if any(value != match_microbatch[0] for value in match_microbatch):
    raise ValueError("Runs differ in match_training_microbatch.")
  if match_microbatch[0]:
    expected_trajectories = (
        manifests[0]["workload"]["micro_groups"]
        * manifests[0]["workload"]["generations"]
    )
    expected_calls = (
        manifests[0]["workload"]["batch"]
        * manifests[0]["workload"]["generations"]
        / expected_trajectories
    )

    def matching_values(section, field):
      values = [summary[section][field] for summary in summaries]
      return all(value == expected_trajectories for value in values)

    checks = {
        f"{section}_trajectories_per_call_match": matching_values(
            section, "trajectories_per_call"
        )
        for section in ("training_input", "actor_log_probs_input")
    }
    for section in ("training_input", "actor_log_probs_input"):
      shapes = [summary[section]["unique_tensor_shapes"] for summary in summaries]
      checks[f"{section}_full_tensor_shapes_match"] = all(
          value == shapes[0] for value in shapes
      )
    for stage in ("training", "actor_log_probs"):
      calls = [
          summary["api_calls"][stage]["calls_per_step"]
          for summary in summaries
      ]
      checks[f"{stage}_calls_per_step_match"] = all(
          value == expected_calls for value in calls
      )
    failed = [name for name, matched in checks.items() if not matched]
    if failed:
      raise ValueError("Microbatch mismatch: " + ", ".join(failed))
    comparison["parity_checks"].update(checks)
  comparison["match_training_microbatch"] = match_microbatch[0]
  return comparison


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="action", required=True)
  run = commands.add_parser("run")
  run.add_argument(
      "--mode",
      choices=("dist", "agentic"),
      required=True,
  )
  run.add_argument("--output", type=Path, required=True)
  run.add_argument("--model-dir", type=Path, required=True)
  run.add_argument("--cache-root", type=Path, required=True)
  run.add_argument(
      "--hardware",
      required=True,
      help="Hardware label, e.g. v5p-4; confirm device inventory in logs.",
  )
  run.add_argument("--execute", action="store_true")
  run.add_argument(
      "--agentic-chip-layout",
      choices=("shared4", "split2x2"),
      default="shared4",
      help="Agentic actor/rollout mesh: shared four chips or separate two-chip slices.",
  )
  run.add_argument(
      "--match-training-microbatch",
      action="store_true",
      help=(
          "Set distributed train and log-prob microbatches to the agentic"
          " size: micro-groups × generations trajectories."
      ),
  )
  run.add_argument(
      "--timing-mode",
      choices=("stages", "pipeline"),
      default="stages",
      help=(
          "stages: wait for module completion (default); pipeline: preserve"
          " intra-step overlap and compare end-to-end time only."
      ),
  )
  run.add_argument("--timeout", type=int, default=172800)
  for name, default in WORKLOAD_DEFAULTS.items():
    run.add_argument("--" + name.replace("_", "-"), type=int, default=default)
  report_parser = commands.add_parser("report")
  report_parser.add_argument("runs", nargs="+", type=Path)
  args = parser.parse_args()
  if args.action == "run":
    prepare(args)
  else:
    print(json.dumps(report(args.runs), indent=2))


if __name__ == "__main__":
  main()
