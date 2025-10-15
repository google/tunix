import os

import optax
import orbax.checkpoint as ocp
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger


def get_optimizer(train_config):
  opt_config = train_config["optimizer"]
  max_grad_norm = opt_config.get("max_grad_norm", None)
  train_steps = train_config["num_batches"] * train_config["num_epochs"]
  warmup_steps = opt_config.get("warmup_ratio", 0) * train_steps
  optimizer = optax.adamw(
      learning_rate=optax.schedules.warmup_constant_schedule(
          init_value=0.0,
          peak_value=opt_config["learning_rate"],
          warmup_steps=warmup_steps,
      ),
      b1=opt_config.get("b1", 0.9),
      b2=opt_config.get("b2", 0.99),
      weight_decay=opt_config.get("weight_decay", 0.0),
  )
  if max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=max_grad_norm),
        optimizer,
    )
  return optimizer


def get_rl_cluster_config(train_config, mesh, optimizer):
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=train_config["save_interval_steps"],
      max_to_keep=train_config["max_to_keep"],
  )
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=os.path.join(train_config["ckpt_dir"], "log"),
      flush_every_n_steps=1,
  )
  # Training config
  train_steps = train_config["num_batches"] * train_config["num_epochs"]
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.REFERENCE: mesh,
          rl_cluster_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=train_config["eval_every_n_steps"],
          max_steps=train_steps,
          gradient_accumulation_steps=1,
          # metrics logging
          metrics_logging_options=metrics_logging_options,
          # checkpoint saving
          checkpoint_root_directory=train_config["ckpt_dir"],
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=train_config["max_response_length"],
          max_prompt_length=train_config["max_prompt_length"],
          kv_cache_size=train_config["max_prompt_length"]
          + train_config["max_response_length"]
          + 128,
          temperature=train_config["rollout"]["temperature"],
          top_p=train_config["rollout"]["top_p"],
          top_k=train_config["rollout"]["top_k"],
      ),
  )
  return cluster_config


def get_trainer(train_config, rl_cluster, reward_fns):
  if train_config["trainer"] == "grpo":
    algo_config = train_config["grpo"]
    grpo_config = grpo_learner.GrpoConfig(
        num_generations=algo_config.get("num_generations", 2),
        num_iterations=algo_config.get("num_iterations", 1),
        beta=algo_config.get("beta", 0.08),
        epsilon=algo_config.get("epsilon", 0.2),
    )
    grpo_trainer = grpo_learner.GrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        grpo_config=grpo_config,
    )
    return grpo_trainer
  else:
    raise ValueError(f"Trainer {train_config['trainer']} not supported")
