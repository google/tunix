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

The distributed stack runs `run_gemma4_e2b.sh` with `section.key=value`
overrides; the agentic stack runs the same Gemma4 E2B recipe through
`tunix.cli.grpo_main`. The harness in `rl_efficiency_benchmark/benchmark.py`
loads this file by path, so the module level imports only the standard library.
"""

import functools
from pathlib import Path

MODULE = "tunix.dist_rl.examples.frozenlake_dist.frozenlake_benchmark"
_RUNTIME = "tunix.dist_rl.examples.rl_efficiency_benchmark.runtime"
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
# Settings from examples/frozenlake/train_frozenlake.py that frozenlake.yaml
# does not default to. Both stacks use them.
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
  """Returns the distributed recipe command and the variables it sets.

  The command passes the benchmark settings to run_gemma4_e2b.sh as
  `section.key=value` overrides. Every other setting keeps its
  run_gemma4_e2b.sh, frozenlake.yaml or launch/common/base_dist_config.yaml
  value.

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
  overrides = {
      "cluster_config.trainer_main": _RUNTIME + ".trainer_main",
      "cluster_config.rollout_main": _RUNTIME + ".rollout_main",
      "cluster_config.orchestrator_main": MODULE + ".dist_main",
      "model_config.model_download_path": model_dir,
      "tokenizer_config.tokenizer_path": model_dir,
      "cluster_config.artifact_root": output / "artifacts",
      "rl_training_config.checkpoint_root_directory": output / "checkpoints",
      "batch_size": w["batch"],
      "rl_training_config.mini_batch_size": w["batch"],
      "agentic_grpo_config.num_generations": w["generations"],
      "rl_training_config.max_steps": w["steps"],
      "data_config.size": w["dataset_size"],
      "data_config.seed": w["seed"],
      # One epoch.
      "data_config.limit": w["steps"] * w["batch"],
      "agentic_grpo_config.max_turns": w["turns"],
      "env_kwargs.max_steps": w["turns"],
      "rollout_config.max_prompt_length": w["prompt"],
      "rollout_config.total_generation_steps": w["response"],
      # The agentic kv_cache_size.
      "vllm_config.max_model_len": w["prompt"] + w["response"] + 256,
      "agentic_grpo_config.max_concurrency": w["concurrency"],
      "vllm_config.max_num_seqs": w["max_num_seqs"],
      "vllm_config.max_num_batched_tokens": w["batched_tokens"],
      "model_config.dtype": _COMPUTE_DTYPE,
      "model_config.load_dtype": _LOAD_DTYPE,
      "vllm_config.hbm_utilization": _HBM_UTILIZATION,
      # Benchmark hooks write events.jsonl; checkpoints, metrics backends, and
      # trajectory logs would add untimed I/O. Null metrics logging options
      # create no metrics backends, as in `agentic_config`, and with no
      # `trajectory_log_dir` no trajectories are logged either.
      "rl_training_config.checkpointing_options.save_interval_steps": 0,
      "rl_training_config.metrics_logging_options": "null",
      "dist_training_config.trajectory_log_dir": '""',
      "rl_training_config.train_micro_batch_size": microbatch_size,
      "rl_training_config.compute_logps_micro_batch_size": microbatch_size,
  }
  command = [
      "bash",
      str(Path(__file__).with_name("run_gemma4_e2b.sh")),
      *(f"{key}={value}" for key, value in overrides.items()),
  ]
  # launch/common/local.sh writes the node logs to LOG_ROOT.
  return command, {"LOG_ROOT": str(output)}


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
  from tunix.dist_rl.examples.frozenlake_dist import frozenlake
  from tunix.dist_rl.examples.rl_efficiency_benchmark import runtime

  rows = frozenlake.generate_entries(
      size=size, seed=seed, shuffle_seed=seed, limit=limit
  )
  runtime.record_dataset(rows)
  return grain.MapDataset.source([{**row, "prompts": ""} for row in rows])


def dist_main(*args, **kwargs):
  """Distributed orchestrator entry point with benchmark hooks."""
  from tunix.dist_rl.examples.frozenlake_dist import frozenlake
  from tunix.dist_rl.examples.rl_efficiency_benchmark import runtime
  from tunix.dist_rl.launch.nodes import orchestrator  # pylint: disable=g-import-not-at-top

  generate = frozenlake.generate_entries

  # The orchestrator's data module, `frozenlake.create_dataset`, generates the
  # maps with `frozenlake.generate_entries`.
  @functools.wraps(generate)
  def recorded_entries(*entry_args, **entry_kwargs):
    rows = generate(*entry_args, **entry_kwargs)
    runtime.record_dataset(rows)
    return rows

  frozenlake.generate_entries = recorded_entries
  runtime.install("dist")
  return orchestrator.main(*args, **kwargs)


def agentic_main():
  """Agentic entry point with benchmark hooks."""
  from absl import app
  from tunix.cli import grpo_main
  from tunix.dist_rl.examples.rl_efficiency_benchmark import runtime

  runtime.install("agentic")
  app.run(grpo_main.main)


if __name__ == "__main__":
  agentic_main()
