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

"""Agentic Household Exploration GRPO training recipe.

Trains an agent on multi-room, stateful household exploration tasks using
Tunix's AgenticGRPOLearner and multi-turn trajectory collection.
"""

import dataclasses
import logging
import sys

from absl import app as absl_app
from absl import flags
from absl import logging as absl_logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer
from tunix.cli.utils import data as cli_data_lib
from tunix.models import dummy_model_creator
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from examples.household import agent as agent_lib
from examples.household import data as data_lib
from examples.household import env as env_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

# Configure logging
absl_logging.use_python_logging()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)
absl_logging.set_verbosity(absl_logging.INFO)

flags.DEFINE_integer("batch_size", 16, "Total batch size.")
flags.DEFINE_integer("mini_batch_size", 16, "Mini batch size.")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate.")
flags.DEFINE_integer("num_batches", 100, "Number of training batches.")
flags.DEFINE_integer("num_generations", 4, "Rollout trajectories per prompt.")
flags.DEFINE_float("beta", 0.0, "KL penalty coefficient.")
flags.DEFINE_float("epsilon", 0.2, "PPO clipping parameter lower bound.")
flags.DEFINE_float("epsilon_high", 0.28, "PPO clipping parameter upper bound.")
flags.DEFINE_integer(
    "max_prompt_length", 2048, "Max prompt sequence length."
)
flags.DEFINE_integer(
    "max_response_length", 1024, "Max response sequence length."
)
flags.DEFINE_float("temperature", 0.7, "Sampling temperature.")
flags.DEFINE_integer(
    "max_steps", 25, "Maximum environment steps per trajectory."
)
flags.DEFINE_string("rollout_engine", "vanilla", "Rollout engine.")
flags.DEFINE_string(
    "data_dir", "/tmp/data/household", "Directory for household data."
)
flags.DEFINE_bool(
    "dummy_weights",
    False,
    "Use synthetic dummy weights for fast verification.",
)
flags.DEFINE_string(
    "model_id", "google/gemma-4-E2B-it", "Model ID / name."
)
flags.DEFINE_string(
    "tokenizer_path", None, "Optional path to tokenizer directory."
)


def metric_fn(prompts, completions, rewards, advantages, **kwargs):
  del prompts, completions, advantages, kwargs
  success = (rewards >= env_lib.HouseholdEnv.GOAL_REWARD).astype(float)
  success_rate = float(np.mean(success))
  reward_mean = float(np.mean(rewards))
  reward_max = float(np.max(rewards))

  absl_logging.info(
      "[household-rollout-metric] n=%d success_rate=%.3f reward_mean=%.3f"
      " reward_max=%.3f",
      len(rewards),
      success_rate,
      reward_mean,
      reward_max,
  )
  return {
      "rewards/success_rate": (success_rate, np.mean),
      "rewards/mean": (reward_mean, np.mean),
      "rewards/max": (reward_max, np.max),
  }


def main(argv=None):
  del argv
  args = flags.FLAGS

  absl_logging.info("Starting Household Task Exploration training...")
  absl_logging.info("JAX devices: %s", jax.devices())

  # 1. Load Data
  train_dataset = data_lib.create_dataset(
      split="train", data_dir=args.data_dir, train_size=500
  )
  eval_dataset = data_lib.create_dataset(
      split="eval", data_dir=args.data_dir, test_size=50
  )

  # 2. Tokenizer and Parser
  tokenizer_path = args.tokenizer_path or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  if "qwen" in args.model_id.lower():
    chat_parser = parser.QwenChatTemplateParser(tokenizer)
  else:
    chat_parser = parser.Gemma4ChatTemplateParser(
        tokenizer, enable_thinking=False
    )

  # 3. Batch Datasets with post_init_dataset
  train_dataset, _ = cli_data_lib.post_init_dataset(
      train_dataset,
      tokenizer,  # pyrefly: ignore[bad-argument-type]
      batch_size=args.batch_size,
      num_batches=args.num_batches,
      max_prompt_length=args.max_prompt_length,
  )
  eval_dataset, _ = cli_data_lib.post_init_dataset(
      eval_dataset,
      tokenizer,  # pyrefly: ignore[bad-argument-type]
      batch_size=args.batch_size,
      num_batches=2,
      max_prompt_length=args.max_prompt_length,
  )

  # 4. Model & Sharding
  num_devices = jax.device_count()
  device_mesh = np.array(jax.devices()).reshape((num_devices, 1))
  trainer_mesh = jax.sharding.Mesh(device_mesh, ("fsdp", "tp"))
  rollout_mesh = trainer_mesh

  if args.dummy_weights:
    absl_logging.info("Using dummy weights for fast verification...")
    # Minimal config for tracing and dry run
    dummy_config = model_lib.ModelConfig(
        num_layers=2,
        num_embed=getattr(tokenizer, "vocab_size", 256128),
        embed_dim=256,
        hidden_dim=512,
        num_heads=4,
        head_dim=64,
        num_kv_heads=1,
        sliding_window_size=128,
        per_layer_input_dim=64,
        frac_shared_layers=0.0,
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.GLOBAL,
        ),
    )
    dummy_config.shd_config = dataclasses.replace(
        dummy_config.shd_config,
        act_btnh=jax.sharding.PartitionSpec(None, None, None, None),
        act_btd=jax.sharding.PartitionSpec(None, None, "tp"),
        act_btf=jax.sharding.PartitionSpec(None, None, "tp"),
    )
    actor = dummy_model_creator.create_dummy_model(
        model_lib.Gemma4,
        dummy_config,
        trainer_mesh,
        dtype=jnp.float32,
    )
    reference = dummy_model_creator.create_dummy_model(
        model_lib.Gemma4,
        dummy_config,
        trainer_mesh,
        dtype=jnp.bfloat16,
    )
  else:
    # Full safetensors loading
    config = model_lib.ModelConfig.gemma4_e2b()
    # In agentic RL, rollouts generate single trajectories (batch_size=1).
    # Replicate KV cache and sequence activations along the batch dimension
    # (fsdp=None) to prevent IndivisibleError on multi-chip meshes.
    config.shd_config = dataclasses.replace(
        config.shd_config,
        act_btnh=jax.sharding.PartitionSpec(None, None, None, None),
        act_btd=jax.sharding.PartitionSpec(None, None, "tp"),
        act_btf=jax.sharding.PartitionSpec(None, None, "tp"),
    )
    reference = params_lib.create_model_from_safe_tensors(
        args.model_id, config, trainer_mesh, dtype=jnp.bfloat16
    )
    actor = params_lib.create_model_from_safe_tensors(
        args.model_id, config, trainer_mesh, dtype=jnp.float32
    )

  # 5. Rollout & Cluster Config
  base_rollout_dict = {
      "max_prompt_length": args.max_prompt_length,
      "kv_cache_size": max(
          args.max_prompt_length + args.max_response_length + 256, 2048
      ),
      "temperature": args.temperature,
      "top_p": 1.0,
      "top_k": 0,
      "return_logprobs": True,
      "max_tokens_to_generate": args.max_response_length,
  }
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)

  optimizer = optax.adamw(learning_rate=args.learning_rate)

  cluster_config = rl_engine_lib.ClusterConfig(
      role_to_mesh={
          rl_engine_lib.Role.ACTOR: trainer_mesh,
          rl_engine_lib.Role.REFERENCE: trainer_mesh,
          rl_engine_lib.Role.ROLLOUT: rollout_mesh,
      },
      rollout_engine=args.rollout_engine,
      offload_to_cpu=False,
      training_config=rl_engine_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=10,
          max_steps=args.num_batches,
          mini_batch_size=args.mini_batch_size,
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
  )

  # 6. Initialize RLEngine and GRPOLearner
  rl_engine = rl_engine_lib.RLEngine(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  grpo_trainer = GRPOLearner(
      rl_engine=rl_engine,
      agent_class=agent_lib.HouseholdAgent,
      agent_kwargs={"multi_shot": True},
      env_class=env_lib.HouseholdEnv,
      env_kwargs={"max_steps": args.max_steps},
      algo_config=grpo_config,
      chat_parser=chat_parser,
      metric_fns=[metric_fn],
  )

  absl_logging.info("Cluster and GRPO configurations initialized successfully.")
  grpo_trainer.train(train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
  absl_app.run(main)
