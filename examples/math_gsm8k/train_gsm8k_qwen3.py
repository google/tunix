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

"""Agentic GSM8K GRPO training recipe for Qwen3-1.7B on TPU.

Trains Qwen3-1.7B on the GSM8K mathematical reasoning task using the Tunix
Agentic RL framework (GRPOLearner + GSM8KEnv + GSM8KAgent).

Supports:
  - Single-host (e.g. v5p-8, v6e-4) or multi-host TPU topologies.
  - vLLM or vanilla rollout engines.
  - Full-parameter fine-tuning or LoRA PEFT.
  - Composite format & exact numerical accuracy rewards.
  - Periodic evaluation on the held-out GSM8K test split.
"""

import argparse
import contextlib
import logging
import math
import os
import sys
from typing import Any, Dict

from absl import logging as absl_logging
from flax import nnx
import grain
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
import transformers

# ====== Logging Configuration ======
absl_logging.use_python_logging()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")

from tunix.cli.utils import data as data_lib
from tunix.google.stubs import utils_stub as oss_utils
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import rl_utils
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import model_utils
from tunix.sft import utils as sft_utils

from examples.math_gsm8k.agent import GSM8KAgent
from examples.math_gsm8k.data import create_dataset
from examples.math_gsm8k.env import GSM8KEnv

# ====== Distributed Initialization ======
_DISTRIBUTED_INITIALIZED = False
try:
  import pathwaysutils

  pathwaysutils.initialize()
  _DISTRIBUTED_INITIALIZED = True
except Exception:
  pass

if not _DISTRIBUTED_INITIALIZED:
  try:
    jax.distributed.initialize()
  except Exception as exc:
    print(f"jax.distributed.initialize() skipped or already initialized: {exc}")

print("JAX devices:", jax.devices())


# ====== Command-line Arguments ======
def parse_args():
  parser = argparse.ArgumentParser(
      description="Train GSM8K on Qwen3-1.7B using Tunix Agentic RL."
  )
  # Model
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--model_dir", type=str, default="/tmp/models/Qwen3-1.7B")

  # PEFT / LoRA
  parser.add_argument("--use_lora", action="store_true", default=False)
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=int, default=32)

  # Training Hyperparameters
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--mini_batch_size", type=int, default=32)
  parser.add_argument("--train_micro_batch_size", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=1e-6)
  parser.add_argument("--b1", type=float, default=0.9)
  parser.add_argument("--b2", type=float, default=0.95)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--max_grad_norm", type=float, default=100.0)
  parser.add_argument("--num_batches", type=int, default=150)
  parser.add_argument("--num_epochs", type=int, default=1)

  # GRPO Hyperparameters
  parser.add_argument("--num_generations", type=int, default=8)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument("--epsilon_high", type=float, default=0.2)
  parser.add_argument(
      "--loss_algo",
      type=str,
      default="grpo",
      help="'grpo' (per-token PPO) or 'gspo-token' (sequence-mean IS).",
  )
  parser.add_argument(
      "--advantage_estimator",
      type=str,
      default="grpo",
      help="'grpo' (z-score) or 'rloo' (leave-one-out baseline).",
  )
  parser.add_argument(
      "--loss_agg_mode", type=str, default="sequence-mean-token-mean"
  )
  parser.add_argument("--kl_loss_mode", type=str, default="low_var_kl")

  # Rollout & Generation
  parser.add_argument("--max_prompt_length", type=int, default=1024)
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=0)
  parser.add_argument("--max_concurrency", type=int, default=128)
  parser.add_argument(
      "--rollout_engine",
      type=str,
      default=os.getenv("ROLLOUT_ENGINE", "vllm"),
      help="'vllm' or 'vanilla'.",
  )

  # Data & Evaluation
  parser.add_argument(
      "--data_source",
      type=str,
      default="huggingface",
      choices=["huggingface", "demo", "smoke_test", "tfds"],
  )
  parser.add_argument("--eval_every_n_steps", type=int, default=10)
  parser.add_argument("--num_test_batches", type=int, default=2)
  parser.add_argument("--seed", type=int, default=42)

  # Infrastructure & Output
  parser.add_argument("--tb_log_dir", type=str, default="/tmp/tunix-tb/gsm8k")
  parser.add_argument("--ckpt_dir", type=str, default=None)
  parser.add_argument("--save_interval_steps", type=int, default=50)

  return parser.parse_known_args()[0]


def main():
  args = parse_args()

  # ====== Mesh Setup ======
  # Default to pure tensor parallel (1, N) on single-host TPU
  shared_mesh_shape = (1, jax.device_count())
  shared_mesh_axis_names = ("fsdp", "tp")

  if jax.device_count() < math.prod(shared_mesh_shape):
    raise ValueError(
        f"Expected at least {math.prod(shared_mesh_shape)} devices for mesh "
        f"{shared_mesh_shape}, got {jax.device_count()}."
    )

  device_list = jax._src.mesh_utils.create_device_mesh(
      shared_mesh_shape, jax.devices()[: math.prod(shared_mesh_shape)]
  )
  shared_mesh = jax.sharding.Mesh(
      device_list,
      axis_names=shared_mesh_axis_names,
      axis_types=(jax.sharding.AxisType.Auto,) * len(shared_mesh_shape),
  )
  print(f"Shared mesh initialized with devices: {shared_mesh.devices.shape}")

  # ====== Tokenizer & Parser ======
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
  # Disable thinking tokens since explicit <reasoning>...</reasoning> tags are prompted
  chat_parser = parser.QwenChatTemplateParser(tokenizer, enable_thinking=False)

  # ====== Datasets ======
  raw_train_ds = create_dataset(
      split="train",
      data_source=args.data_source,
      seed=args.seed,
  )
  raw_test_ds = create_dataset(
      split="test" if args.data_source != "smoke_test" else "train",
      data_source=args.data_source,
      seed=args.seed,
  )

  # Wrap in grain MapDataset if not already
  if not hasattr(raw_train_ds, "map"):
    raw_train_ds = grain.MapDataset.source(list(raw_train_ds))
  if not hasattr(raw_test_ds, "map"):
    raw_test_ds = grain.MapDataset.source(list(raw_test_ds))

  train_dataset, _ = data_lib.post_init_dataset(
      raw_train_ds,
      tokenizer,
      batch_size=args.batch_size,
      num_batches=args.num_batches,
      max_prompt_length=args.max_prompt_length,
      num_epochs=args.num_epochs,
  )
  test_dataset, _ = data_lib.post_init_dataset(
      raw_test_ds,
      tokenizer,
      batch_size=args.batch_size,
      num_batches=args.num_test_batches,
      max_prompt_length=args.max_prompt_length,
  )
  sft_utils.show_hbm_usage("Done loading datasets")

  # ====== Model Weights Download & Instantiation ======
  if not os.path.isdir(args.model_dir) or not any(
      f.endswith(".safetensors") for f in os.listdir(args.model_dir)
  ):
    os.makedirs(args.model_dir, exist_ok=True)
    oss_utils.hf_pipeline(args.model_id, args.model_dir)

  # Config for Qwen3-1.7B
  config = qwen3_model_lib.ModelConfig.qwen3_1p7b()
  config.remat_config = qwen3_model_lib.RematConfig.DECODER
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32

  # Reference model (frozen, stored in bfloat16)
  qwen_ref = qwen3_params_lib.create_model_from_safe_tensors(
      args.model_dir, config, shared_mesh, dtype=jnp.bfloat16
  )
  sft_utils.show_hbm_usage("After loading qwen_ref")

  # Actor model (fp32 storage for optimizer precision)
  actor_base = qwen3_params_lib.create_model_from_safe_tensors(
      args.model_dir, config, shared_mesh, dtype=jnp.float32
  )

  # Apply LoRA if requested
  if args.use_lora:
    lora_config = {
        "module_path": (
            ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
            ".*gate_proj|.*down_proj|.*up_proj"
        ),
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
    }
    qwen_actor = model_utils.apply_lora_to_model(
        actor_base, mesh=shared_mesh, lora_config=lora_config
    )
  else:
    qwen_actor = actor_base

  # Pin parameters on device
  graph_def, state = nnx.split(qwen_actor)
  state = rl_utils.put_params_on_memory_kind(state, "device")
  qwen_actor = nnx.merge(graph_def, state)
  sft_utils.show_hbm_usage("After loading qwen_actor")

  # ====== Optimizer ======
  optimizer = optax.adamw(
      learning_rate=args.learning_rate,
      b1=args.b1,
      b2=args.b2,
      weight_decay=args.weight_decay,
  )
  if args.max_grad_norm is not None and args.max_grad_norm > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=args.max_grad_norm),
        optimizer,
    )

  # ====== Checkpoint & Logging ======
  if args.ckpt_dir:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=args.save_interval_steps,
        max_to_keep=2,
    )
  else:
    checkpointing_options = None

  max_steps = int(args.num_batches * args.num_epochs)
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=args.tb_log_dir,
      project_name="tunix-gsm8k-qwen3",
      flush_every_n_steps=1,
      backend_kwargs={"wandb": {"config": vars(args)}},
  )

  # ====== Rollout Configuration ======
  base_rollout_dict = {
      "max_prompt_length": args.max_prompt_length,
      "kv_cache_size": args.max_prompt_length + args.max_response_length + 256,
      "temperature": args.temperature,
      "top_p": args.top_p,
      "top_k": args.top_k,
      "return_logprobs": True,
      "max_tokens_to_generate": args.max_response_length,
  }

  vllm_max_num_seqs = 64
  vllm_max_batched_tokens = vllm_max_num_seqs * 4 * 1024 // 8
  vllm_rollout_dict = {
      "rollout_vllm_model_version": args.model_id,
      "rollout_vllm_hbm_utilization": 0.25,
      "rollout_vllm_tpu_backend_type": "jax",
      "rollout_vllm_server_mode": True,
      "rollout_vllm_async_scheduling": False,
      "rollout_vllm_init_with_random_weights": True,
      "tensor_parallel_size": shared_mesh_shape[1],
      "data_parallel_size": shared_mesh_shape[0],
      "rollout_vllm_max_num_seqs": vllm_max_num_seqs,
      "rollout_vllm_max_num_batched_tokens": vllm_max_batched_tokens,
      "rollout_vllm_kwargs": {
          "kv_cache_metrics": True,
          "disable_log_stats": False,
          "enable_prefix_caching": False,
          "dtype": "bfloat16",
      },
  }

  if args.rollout_engine == "vllm":
    rollout_engine_config = base_rollout.RolloutConfig(
        **base_rollout_dict, **vllm_rollout_dict
    )
  elif args.rollout_engine == "vanilla":
    rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)
  else:
    raise ValueError(f"Unsupported rollout engine: {args.rollout_engine}")

  cluster_config = rl_engine_lib.ClusterConfig(
      role_to_mesh={
          rl_engine_lib.Role.ACTOR: shared_mesh,
          rl_engine_lib.Role.REFERENCE: shared_mesh,
          rl_engine_lib.Role.ROLLOUT: shared_mesh,
      },
      rollout_engine=args.rollout_engine,
      offload_to_cpu=False,
      training_config=rl_engine_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=args.eval_every_n_steps,
          max_steps=max_steps,
          mini_batch_size=args.mini_batch_size,
          train_micro_batch_size=args.train_micro_batch_size,
          compute_logps_micro_batch_size=args.train_micro_batch_size,
          metrics_logging_options=metrics_logging_options,
          checkpoint_root_directory=args.ckpt_dir,
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=rollout_engine_config,
  )

  grpo_config = GRPOConfig(
      num_generations=args.num_generations,
      num_iterations=1,
      max_response_length=args.max_response_length,
      beta=args.beta,
      epsilon=args.epsilon,
      epsilon_high=args.epsilon_high,
      system_prompt="",
      max_concurrency=args.max_concurrency,
      off_policy_steps=0,
      loss_agg_mode=args.loss_agg_mode,
      kl_loss_mode=args.kl_loss_mode,
      loss_algo=args.loss_algo,
      advantage_estimator=args.advantage_estimator,
      sampler_is="token",
      sampler_is_threshold=2.0,
  )

  rl_engine = rl_engine_lib.RLEngine(
      actor=qwen_actor,
      reference=qwen_ref,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )
  sft_utils.show_hbm_usage("After RLEngine initialization")

  # ====== Evaluation / Diagnostic Metric Callback ======
  def gsm8k_metric_fn(prompts, completions, rewards, advantages, **kwargs):
    del prompts, completions, advantages, kwargs
    # GSM8K rewards: 1.0 (format+correct), 0.5 (unformatted correct), 0.1 (format only), 0.0 (neither)
    solve_mask = rewards >= 0.5
    format_mask = (rewards == 1.0) | (rewards == 0.1)

    solve_ratio = float(solve_mask.mean())
    format_ratio = float(format_mask.mean())
    reward_mean = float(rewards.mean())
    reward_max = float(rewards.max())

    absl_logging.info(
        "[gsm8k-metric] count=%d solve_ratio=%.3f format_ratio=%.3f"
        " reward_mean=%.3f reward_max=%.3f",
        len(rewards),
        solve_ratio,
        format_ratio,
        reward_mean,
        reward_max,
    )
    return {
        "rewards/solve_ratio": (solve_ratio, np.mean),
        "rewards/format_ratio": (format_ratio, np.mean),
        "rewards/reward_mean": (reward_mean, np.mean),
        "rewards/reward_max": (reward_max, np.max),
    }

  # ====== Instantiate GRPOLearner ======
  grpo_trainer = GRPOLearner(
      rl_engine=rl_engine,
      agent_class=GSM8KAgent,
      env_class=GSM8KEnv,
      env_kwargs={"max_steps": 1},
      algo_config=grpo_config,
      chat_parser=chat_parser,
      metric_fns=[gsm8k_metric_fn],
  )
  sft_utils.show_hbm_usage("After GRPOLearner instantiation")

  # ====== Execute Training Loop ======
  absl_logging.info(
      "Starting GSM8K training for %d steps with Qwen3-1.7B...", max_steps
  )
  grpo_trainer.train(train_dataset, eval_dataset=test_dataset)
  absl_logging.info("Training complete!")


if __name__ == "__main__":
  main()
