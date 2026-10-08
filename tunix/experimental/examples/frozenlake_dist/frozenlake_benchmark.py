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

"""FrozenLake workload for the distributed-vs-agentic RL benchmark.

The distributed stack runs `run_gemma4_e2b.sh`; the agentic stack runs the
same Gemma4 E2B recipe through `tunix.cli.grpo_main`. The harness in
`rl_efficiency_benchmark/benchmark.py` loads this file by path, so the module
level imports only the standard library.
"""

import functools
from pathlib import Path

MODULE = "tunix.experimental.examples.frozenlake_dist.frozenlake_benchmark"
_RUNTIME = "tunix.experimental.examples.rl_efficiency_benchmark.runtime"
MODEL_ID = "google/gemma-4-E2B-it"
# Room for the final FrozenLake observation and chat-template boundary.
RESPONSE_RESERVE = 512
DEFAULTS = {
    "batch": 64,
    "generations": 8,
    "steps": 150,
    "warmup": 5,
    "micro_groups": 1,
    "prompt": 2048,
    "response": 2048,
    "turns": 8,
    "seed": 42,
    "dataset_size": 10000,
    "concurrency": 512,
    "max_num_seqs": 32,
    "batched_tokens": 8192,
}
# Settings from examples/frozenlake/train_frozenlake.py that launcher.sh does
# not default to. Both stacks use them.
_COMPUTE_DTYPE = "bfloat16"
_LOAD_DTYPE = "float32"
_HBM_UTILIZATION = 0.2


def validate(workload):
  counts = (
      "turns",
      "dataset_size",
      "concurrency",
      "max_num_seqs",
      "batched_tokens",
  )
  if any(workload[key] <= 0 for key in counts):
    raise ValueError("FrozenLake workload counts must be positive.")
  if workload["steps"] * workload["batch"] > workload["dataset_size"]:
    raise ValueError("Require at least steps * batch dataset rows.")


def dist_launch(workload, output, model_dir):
  """Returns the distributed recipe command and the variables it overrides.

  Every other setting keeps its run_gemma4_e2b.sh or launcher.sh default.

  Args:
    workload: Workload flag values.
    output: The run's output directory.
    model_dir: The pre-downloaded model directory.

  Returns:
    The command and the environment variables to set for it.
  """
  w = workload
  # Match the agentic microbatch of micro_groups prompt groups.
  microbatch_size = w["micro_groups"] * w["generations"]
  env = {
      "TRAINER_PROCESS_MAIN": _RUNTIME + ".trainer_main",
      "ROLLOUT_PROCESS_MAIN": _RUNTIME + ".rollout_main",
      "ORCHESTRATOR_PROCESS_MAIN": MODULE + ".dist_main",
      "MODEL_DIR": model_dir,
      "ARTIFACT_ROOT": output / "artifacts",
      "LOG_ROOT": output,
      "CHECKPOINT_ROOT_DIRECTORY": output / "checkpoints",
      "BATCH_SIZE": w["batch"],
      "MINI_BATCH_SIZE": w["batch"],
      "NUM_GENERATIONS": w["generations"],
      "NUM_BATCHES": w["steps"],
      "NUM_EPOCHS": 1,
      "MAX_STEPS": w["steps"],
      "DATASET_SIZE": w["dataset_size"],
      "SEED": w["seed"],
      "MAX_TURNS": w["turns"],
      "MAX_PROMPT_LENGTH": w["prompt"],
      "MAX_RESPONSE_LENGTH": w["response"],
      "ROLLOUT_MAX_CONCURRENCY": w["concurrency"],
      "VLLM_MAX_NUM_SEQS": w["max_num_seqs"],
      "VLLM_MAX_NUM_BATCHED_TOKENS": w["batched_tokens"],
      "MODEL_DTYPE": _COMPUTE_DTYPE,
      "MODEL_LOAD_DTYPE": _LOAD_DTYPE,
      "VLLM_HBM_UTILIZATION": _HBM_UTILIZATION,
      # Benchmark hooks write events.jsonl; checkpoints and logging backends
      # would add untimed I/O. run_frozenlake_dist.py logs to a default
      # directory unless LOG_DIR is in its environment, even if empty.
      "CHECKPOINT_SAVE_INTERVAL_STEPS": 0,
      "LOG_DIR": "",
      "TRAJECTORY_LOG_DIR": "",
      "TRAIN_MICRO_BATCH_SIZE": microbatch_size,
      "COMPUTE_LOGPS_MICRO_BATCH_SIZE": microbatch_size,
  }
  command = ["bash", str(Path(__file__).with_name("run_gemma4_e2b.sh"))]
  return command, {key: str(value) for key, value in env.items()}


def agentic_config(workload, model_dir):
  """Build explicit overrides of base_agentic_config, in JSON (valid YAML).

  The actor and rollout use separate two-chip meshes, matching the distributed
  chip allocation.

  Args:
    workload: Workload flag values.
    model_dir: The pre-downloaded model directory.

  Returns:
    The config overrides.
  """
  w = workload
  mesh = {"shape": "(2,1)", "axis_names": "('fsdp','tp')"}
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
          "dtype": _COMPUTE_DTYPE,
          "rng_seed": w["seed"],
          "use_flash_attention": True,
          "flash_attention_block_size": 256,
          "use_sliding_window_kv_cache": False,
      },
      "actor_model_config": {"load_dtype": _LOAD_DTYPE, "mesh": mesh},
      "reference_model_config": {"mesh": None, "same_mesh_as": "actor"},
      "rollout_model_config": {
          "mesh": {"shape": "(1,2)", "axis_names": "('fsdp','tp')"},
          "same_mesh_as": None,
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
      "data_module": MODULE,
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
          "hbm_utilization": _HBM_UTILIZATION,
          "tpu_backend_type": "jax",
          "server_mode": True,
          "async_scheduling": False,
          "max_num_seqs": w["max_num_seqs"],
          "max_num_batched_tokens": w["batched_tokens"],
          "data_parallel_size": 1,
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


def create_dataset(size, seed, limit=None, **unused_kwargs):
  """Agentic data_module: the same ordered maps as the distributed run."""
  import grain
  from tunix.experimental.examples.frozenlake_dist import frozenlake
  from tunix.experimental.examples.rl_efficiency_benchmark import runtime

  rows = frozenlake.create_dataset(
      size=size, seed=seed, shuffle_seed=seed, limit=limit
  )
  runtime.record_dataset(rows)
  return grain.MapDataset.source([{**row, "prompts": ""} for row in rows])


def dist_main(*args, **kwargs):
  """Distributed orchestrator entry point with benchmark hooks."""
  from tunix.experimental.examples.frozenlake_dist import frozenlake
  from tunix.experimental.examples.frozenlake_dist import run_frozenlake_dist
  from tunix.experimental.examples.rl_efficiency_benchmark import runtime

  create = frozenlake.create_dataset

  @functools.wraps(create)
  def recorded_dataset(*dataset_args, **dataset_kwargs):
    rows = create(*dataset_args, **dataset_kwargs)
    runtime.record_dataset(rows)
    return rows

  frozenlake.create_dataset = recorded_dataset
  runtime.install("dist")
  return run_frozenlake_dist.main(*args, **kwargs)


def agentic_main():
  """Agentic entry point with benchmark hooks."""
  from absl import app
  from tunix.cli import grpo_main
  from tunix.experimental.examples.rl_efficiency_benchmark import runtime

  runtime.install("agentic")
  app.run(grpo_main.main)


if __name__ == "__main__":
  agentic_main()
