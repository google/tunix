# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO learner."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Sequence, Type, TypeVar

from absl import logging
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core as ppo_helpers
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner

from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import task_environment
from tunix.perf.experimental import constants as perf_constants
from tunix.utils import trajectory_logger


TrainingInputT = agentic_rl_learner.TrainingInputT
RewardFn = agentic_rl_learner.RewardFn
MetricFn = agentic_rl_learner.MetricFn
registry = function_registry.default_registry

@flax.struct.dataclass(frozen=True)
class TrainExample(agentic_rl_learner.TrainExample):
  returns: jax.Array | None = None
  old_values: jax.Array | None = None

@dataclasses.dataclass(slots=True, kw_only=True)
class PPOConfig(agentic_rl_learner.AgenticRLConfig):
  """Configuration for PPO learner.

  Attributes:
    algo_variant: The algorithm variant to use. Default: `ppo`.
    advantage_estimator: The advantage estimator to use. Default: `gae`.
    policy_loss_fn: The policy loss function to use. Default: `ppo`.
    num_iterations: The number of optimization epochs per batch of rollouts.
      This corresponds to the number of times the policy updates its weights for
      a given batch of rollouts.
    mini_batch_size: The batch size on which the actual model updates happen.
      The rollout phase (`generate_and_compute_advantages`) happen on a larger
      batch, which is then split into "mini-batches".
    gamma: The discount factor for future rewards in GAE.
    gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
    beta: The coefficient for the KL divergence penalty.
    epsilon: Epsilon value for clipping the ratio for the policy objective.
    epsilon_low: Lower bound for clipping the ratio for the policy objective.
      Set to `epsilon` if not provided.
    epsilon_high: Upper bound for clipping the ratio for the policy objective.
      Set to `epsilon` if not provided.
    epsilon_c: Lower bound for clipping for dual-clip PPO. If not provided, we
      don't do dual-clip PPO.
      Reference: https://arxiv.org/abs/1912.09729.
    entropy_coef: Entropy coefficient for the policy loss. Set to `None` or
      `0.0` to disable entropy regularization.
    clip_range_value: The range for clipping the value function loss.
    kl_method: The method for computing KL divergence. Must be one of
      `["low_var_kl", "kl", "mse_kl"]`.
  """

  algo_variant: str = "agentic_ppo"
  advantage_estimator: str = "gae"
  policy_loss_fn: str = "ppo"
  value_loss_fn: str = "ppo"
  num_iterations: int = 1

  # PPO loss and advantage computation configs.
  gamma: float = 1.0
  gae_lambda: float = 0.95
  beta: float = 0.04
  epsilon: float = 0.2
  epsilon_low: float | None = None
  epsilon_high: float | None = None
  epsilon_c: float | None = None
  entropy_coef: float | None = None
  clip_range_value: float = 0.2
  kl_method: str = "low_var_kl"
  kl_clamp_value: float | None = None
  use_rollout_logps: bool = True
  sampler_is: str | None = None  # None | "token"
  sampler_is_threshold: float = 2.0

  def __post_init__(self):
    self.epsilon_low = self.epsilon_low if self.epsilon_low else self.epsilon
    self.epsilon_high = self.epsilon_high if self.epsilon_high else self.epsilon
    self.epsilon = self.epsilon

    if self.epsilon_c is not None and self.epsilon_c <= 1.0:
      raise ValueError(
          f"`epsilon_c` must be greater than 1. Received: {self.epsilon_c}."
      )

    if self.kl_method not in ["kl", "mse_kl", "low_var_kl"]:
      raise ValueError(
          f"Invalid KL method: {self.kl_method}. Must be one of"
          " ['low_var_kl', 'kl', 'mse_kl']."
      )


TPPOConfig = TypeVar("TPPOConfig", bound=PPOConfig)
class PPOLearner(agentic_rl_learner.AgenticRLLearner[TPPOConfig]):
  """PPO (Proximal Policy Optimization) learner for the agentic setting.

  PPO is a reinforcement learning algorithm that fine-tunes models using an
  actor-critic architecture. It optimizes a clipped surrogate objective function
  to ensure stable policy updates, preventing large, destructive changes. The
  actor (policy model) learns what actions to take, while the critic (value
  model) estimates the value of states to help calculate advantages. This
  approach balances exploration and exploitation, making it a robust choice for
  a wide range of RL tasks.

  References:
  - https://arxiv.org/abs/1707.06347
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TPPOConfig,
      reward_fns: RewardFn | List[RewardFn] | None = None,
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      agent_class: Type[
          base_agent.ConversationAgentBase
        ] = model_agent.ModelAgent,
      agent_kwargs: Dict[str, Any] | None = None,
      env_class: Type[
          base_environment.BaseTaskEnv
      ] = task_environment.TaskEnvironment,
      env_kwargs: Dict[str, Any] | None = None,

  ):
    """Initializes the `PPOLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: An instance of `PPOConfig` containing all training-specific
        configuration options.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept `prompts`, `completions`,
        `rewards`, `advantages` and optional keyword arguments, and return a
        dictionary of metric names to tuples of (metric_value, aggregation_fn):
        >>> def metric_fn(prompts, completions, rewards, advantages, **kargs):
        ...    return { ...        "prompt_min_len": (min(len(p) for p in
        prompts), np.min), ...        ... ...    }
    """
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        chat_parser=chat_parser,
        metric_fns=metric_fns,
        agent_class=agent_class,
        agent_kwargs=agent_kwargs,
        env_class=env_class,
        env_kwargs=env_kwargs,
     )
    
    self._trajectory_logger = None
    metrics_logger_options = (
        self.rl_cluster.cluster_config.training_config.metrics_logging_options
    )
    metrics_log_dir = (
        metrics_logger_options.log_dir if metrics_logger_options else None
    )

    if metrics_log_dir:
      self._trajectory_logger = trajectory_logger.AsyncTrajectoryLogger(
          metrics_log_dir
      )
    else:
      logging.warning("Metrics log dir is None, skipping trajectory logging.")

    # ===== RlCluster should have `reward` and `critic` models =====
    if bool(reward_fns) == bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    ):
      raise ValueError(
          "PPO requires one of `reward_fns` or `rl_cluster.reward` to be set. "
          f"Received: reward_fn={reward_fns}, "
          "rl_cluster.reward="
          f"{self.rl_cluster.inference_worker._models['reward']}"
      )
    if not self.rl_cluster.inference_worker._models["critic"]:
      raise ValueError(
          "PPO requires a critic model. Please pass the correct `critic` to "
          "`RlCluster`."
      )
    self._use_reward_model = bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    )

    # ===== Configure the actor (policy) trainer =====
    # policy_loss_fn is retrieved from the registry.
    policy_loss_fn = registry.get(
        "policy_loss_fn", self.algo_config.policy_loss_fn
    )
    loss_fn = lambda model, train_example, algo_config: policy_loss_fn(
        model,
        train_example,
        algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
        compute_logps_chunk_size=self.rl_cluster.cluster_config.training_config.compute_logps_chunk_size,
    )
    self.rl_cluster.actor_trainer.with_loss_fn(loss_fn, has_aux=True)
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {  # pyrefly: ignore[bad-argument-type]
            "train_example": x,
            "algo_config": self.algo_config,
        }
    )

    # ===== Configure the critic (value) trainer =====
    value_loss_fn = registry.get(
        "value_loss_fn", self.algo_config.value_loss_fn
    )
    self.rl_cluster.critic_trainer.with_loss_fn(value_loss_fn, has_aux=True)
    self.rl_cluster.critic_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "clip_range_value": self.algo_config.clip_range_value,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )

    # ===== Configure the metrics logger =====
    # We just log the metrics returned in `aux`. All other metrics are logged
    # by `RLCluster` itself.
    actor_rl_metrics_to_log = {"pg_clipfrac": np.mean}
    if self.algo_config.epsilon_c is not None:
      actor_rl_metrics_to_log["pg_clipfrac_lower"] = np.mean
    if (
        self.algo_config.entropy_coef is not None
        and self.algo_config.entropy_coef > 0.0
    ):
      actor_rl_metrics_to_log["loss/entropy"] = np.mean
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log(
        actor_rl_metrics_to_log  # pyrefly: ignore[bad-argument-type]
    )

    self.rl_cluster.critic_trainer.with_rl_metrics_to_log({
        "vpred_mean": np.mean,
        "vf_clipfrac": np.mean,
    })

  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      expected_step: int | None = None
  ) -> List[TrainExample]:
    """Generates completions and computes advantages for PPO training.

    Args:
      trajectories: A list of trajectory results for a single PPO prompt.
      mode: The mode to use for logging metrics.
      expected_step: The expected training step.

    Returns:
      A list of `TrainExample` instances containing the processed input data for PPO.
    """

    logging.debug(
        "Processing results to compute advantage for %d items.",
        len(trajectories),
    )

    # Extract completions and tokens from the trajectories.
    completion_texts: List[str] = []
    prompt_tokens_list: List[np.ndarray] = []
    completion_tokens_list: List[np.ndarray] = []
    completion_masks_list: List[np.ndarray] = []
    old_logprobs_list: List[np.ndarray] = []
    policy_versions_list: List[int] = []
    trajectory_rewards_list: List[float] = []
    trajectories_to_log = []

    for item in trajectories:
      trajectories_to_log.append(item.traj)
      conversation = item.traj.get("conversation_text") or []
      assistant_text = next(
          (
              message["content"]
              for message in conversation
              if message["role"] == "assistant"
          ),
          "",
      )
    
      completion_texts.append(assistant_text)
      prompt_tokens_list.append(item.traj.get("prompt_tokens"))
      completion_tokens_list.append(item.traj.get("conversation_tokens"))
      completion_masks_list.append(item.traj.get("conversation_masks"))
      old_logprobs_list.append(item.traj.get("old_logprobs"))
      policy_version = item.traj.get("policy_version")

      if policy_version is None:
        raise ValueError("policy_version is missing from trajectory task.")
      policy_versions_list.append(policy_version)
      trajectory_rewards_list.append(item.traj.get("trajectory_reward"))
    
    # Log trajectory.
    if self._trajectory_logger and trajectories_to_log:
      for traj in trajectories_to_log:
        self._trajectory_logger.log_item_async(traj)

    # Pad all prompts and completions to consistent lengths.
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      rollout_config = rollout_config[mode]
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    padded_prompt_ids = []
    padded_completion_ids = []
    padded_completion_masks = []
    padded_old_logprobs = []

    max_response_length = self.algo_config.max_response_length
    clipped_completion_count = 0
    for prompt_tokens, completion_tokens, completion_mask, old_logprobs in zip(
        prompt_tokens_list,
        completion_tokens_list,
        completion_masks_list,
        old_logprobs_list,
    ):
      if (
          len(completion_tokens) >= max_response_length
          and completion_mask[-1] != eos_value
      ):
        clipped_completion_count += 1
      padded_prompt, padded_completion, _ = (
          agentic_utils.pad_prompt_and_completion(
              prompt_tokens,  # pyrefly: ignore[bad-argument-type]
              completion_tokens,  # pyrefly: ignore[bad-argument-type]
              rollout_config.max_prompt_length,
              max_response_length,
              pad_value,
          )
      )
      padded_prompt_ids.append(padded_prompt)
      padded_completion_ids.append(padded_completion[:max_response_length])
      padded_completion_masks.append(
          agentic_utils.right_pad(completion_mask, max_response_length, 0)[
              :max_response_length
          ]
      )
      if self.algo_config.use_rollout_logps:
        if old_logprobs is not None:
          padded_old_logprobs.append(
              agentic_utils.right_pad(
                  old_logprobs,
                  length=max_response_length,
                  pad=0.0,
                  dtype=old_logprobs.dtype,
              )[:max_response_length]
          )
        else:
          padded_old_logprobs.append(
              np.zeros(max_response_length, dtype=np.float32)
          )

    prompt_ids = jnp.asarray(padded_prompt_ids)
    prompt_mask = prompt_ids != pad_value
    completion_ids = jnp.asarray(padded_completion_ids)
    completion_mask = jnp.asarray(padded_completion_masks)
    batch_size = completion_ids.shape[0]
    logits_to_keep = completion_ids.shape[1]
    
    # Sampler-trainer log-probability mismatch diagnostic. When rollout
    # logprobs are present we recompute the trainer's logprobs so the per-batch
    # diff, max, and Pearson correlation metrics can be logged below. Training
    # itself still uses whichever logp source is configured via
    # ``use_rollout_logps``. The diagnostic forward pass is skipped when the
    # actor is attached to an empty mesh (e.g. unit-test environments without a
    # device topology) because the actor sharding path requires a real mesh;
    # the metrics are still emitted when running on real accelerators. Cost
    # when active: one extra trainer forward pass per training step.
    actor_mesh = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR]
    have_actor_mesh = actor_mesh is not None and not actor_mesh.empty
    rollout_per_token_logps = None
    trainer_per_token_logps = None
    if self.algo_config.use_rollout_logps and padded_old_logprobs:
      rollout_per_token_logps = jnp.asarray(padded_old_logprobs)
      old_per_token_logps = rollout_per_token_logps
      # The diagnostic pass (and the sampler-IS ``token`` path, which needs the
      # trainer's recomputed logp as ``old_per_token_logps``) requires a real
      # actor mesh; skip when not available.
      need_trainer_logps = (
          have_actor_mesh or self.algo_config.sampler_is == "token"
      )
      if need_trainer_logps:
        trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
        )
      # When sampler-IS correction is enabled, use the trainer's recomputed
      # logp as ``old_per_token_logps`` so the PPO ratio is
      # ``exp(current_logp - trainer_logp)`` rather than against the rollout
      # sampler's logp directly. The IS weight computed below corrects for
      # the trainer-vs-sampler divergence.
      if (
          self.algo_config.sampler_is == "token"
          and trainer_per_token_logps is not None
      ):
        old_per_token_logps = trainer_per_token_logps
    elif self.algo_config.use_rollout_logps:
      old_per_token_logps = None
    else:
      trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
      )
      old_per_token_logps = trainer_per_token_logps

    if self.algo_config.num_iterations > 1 and old_per_token_logps is None:
      raise RuntimeError(
          "old_per_token_logps is not available for off-policy RL. Enable "
          " `return_logprobs` in RolloutConfig."
      )

    # Collect perf tags
    traj = trajectories[0].traj
    group_id = traj.get("group_id")
    if group_id is None:
      original_input = traj.get("original_input", {})
      group_id = original_input.get("group_id")

    perf_tags = {
        perf_constants.STEP: expected_step,
    }
    if group_id is not None:
      perf_tags[perf_constants.GROUP_ID] = group_id

    # Collect original training inputs embedded in trajectories
    original_inputs_list = [ 
        item.traj.get("original_input", {}) for item in trajectories
    ]
    original_inputs = rl_utils.merge_micro_batches(original_inputs_list)    

    logging.debug(
        "Token shapes: prompt_ids=%s, completion_ids=%s",
        prompt_ids.shape,
        completion_ids.shape,
    )

    if self.algo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_value,
      eos_id=eos_value,
     )
    else:
      ref_per_token_logps = None

    if self.algo_config.use_rollout_logps and padded_old_logprobs:
      old_per_token_logps = jnp.asarray(padded_old_logprobs)
    else:
      old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
      )

    # ===== Value computation ======
    # Get values from the value model before model weights are updated.
    values = self.rl_cluster.get_values(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_value,
        eos_id=eos_value,
    )
    # `values` start from the last *prompt* token. Shape: `[B, T]`.
    values = values[:, -logits_to_keep - 1 : -1]
    values = values * completion_mask

    # ===== Reward computation ======
    # Reward computation is in accordance with other RL libraries
    # batch reward manager (token-level rewards).
    # 1. Set all rewards (i.e., for every token) to 0s.
    # 2. A positive reward is given only at the final timestep, so we add that
    # to the tensor of zeros.
    # 3. Subtract KL divergence from the reward tensor.

    # Get rewards from the reward model. Eventual shape: `[B, T]`.
    eos_idx = jnp.max(
        common.build_positions_from_mask(completion_mask),
        axis=-1,
    )
    if self._use_reward_model:
      scores = self.rl_cluster.get_rewards(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )[:, -logits_to_keep:]
      # We use the score corresponding to the last non-padding token.
      jax_last_token_scores = scores[jnp.arange(batch_size), eos_idx]
      last_token_scores = jax.device_get(jax_last_token_scores)
    else:
      reward_kwargs = {
        key: value for key, value in original_inputs.items() if key != "prompts"
      }
      reward_kwargs["trajectory_rewards"] = trajectory_rewards_list
      prompts_texts = [
          item.traj.get("prompt_text", "") for item in trajectories
      ]
      
      with self.rl_cluster.perf_v2.span(
        perf_constants.ADVANTAGE_COMPUTATION,
        tags=perf_tags,
      ):
        last_token_scores = self._compute_rewards(
            prompts=prompts_texts,  # pyrefly: ignore[bad-argument-type]
            completions=completion_texts,
            mode=mode,
            **reward_kwargs,
            expected_step=expected_step,
        )
        jax_last_token_scores = jax.device_put(last_token_scores)

    rewards = jnp.zeros_like(completion_ids)
    rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(
        jax_last_token_scores
    )

    if self.algo_config.beta != 0.0:
      # TODO(abheesht): Add a toggle - KL can either be added directly to
      # rewards or computed in the loss function.
      kl = common.compute_kl_divergence(
          old_per_token_logps,
          ref_per_token_logps,  # pyrefly: ignore[bad-argument-type]
          method=self.algo_config.kl_method,
          clamp_value=self.algo_config.kl_clamp_value,
      )
      kl = kl * completion_mask
      rewards = rewards - self.algo_config.beta * kl

    # ===== Compute advantages using Generalised Advantage Estimation ======
    advantage_estimator = registry.get(
        "advantage_estimator", self.algo_config.advantage_estimator
    )
    advantages, returns = advantage_estimator(
        rewards=rewards,
        values=values,
        completion_mask=completion_mask,
        gamma=self.algo_config.gamma,
        gae_lambda=self.algo_config.gae_lambda,
    )

    logging.debug("Advantages computed: %s", advantages)
    policy_versions = np.array(policy_versions_list, dtype=np.int32)

    # ===== Metric logging ======
    agg_completion_mask = completion_mask.sum(axis=-1)
    
    # Log rewards.
    metrics_to_log = {
        "generation/completions/mean_length": (
            np.mean(agg_completion_mask),
            np.mean,
        ),
        "generation/completions/max_length": (
            np.max(agg_completion_mask),
            np.max,
        ),
        "generation/completions/min_length": (
            np.min(agg_completion_mask),
            np.min,
        ),  
        "generation/completions/clip_ratio": (
            clipped_completion_count / len(trajectories),
            np.mean,
        ),
        "rewards/advantage/mean": (np.mean(advantages), np.mean),
        "rewards/advantage/max": (np.max(advantages), np.max),
        "rewards/advantage/min": (np.min(advantages), np.min),
        "rewards/advantage/std": (np.std(advantages), np.mean),
    }
    
    # Log returns.
    valid_returns = np.ma.masked_array(
        returns, mask=np.logical_not(completion_mask)
    )   
    metrics_to_log.update({
        "advantages/returns/mean": (valid_returns.mean(), np.mean),
        "advantages/returns/max": (valid_returns.max(), np.max),
        "advantages/returns/min": (valid_returns.min(), np.min),
    })

    # Log values.
    valid_values = np.ma.masked_array(
        values, mask=np.logical_not(completion_mask)
    )
    metrics_to_log.update({
        "advantages/old_values/mean": (valid_values.mean(), np.mean),
        "advantages/old_values/max": (valid_values.max(), np.max),
        "advantages/old_values/min": (valid_values.min(), np.min),
    })

    if self.algo_config.beta != 0.0:
      # Average of the per-sequence mean KL
      per_sequence_mean_kl = ppo_helpers.masked_mean(
          kl, completion_mask, axis=-1  # pylint: disable=undefined-variable  # pyrefly: ignore[unbound-name]
      )
      self.rl_cluster.buffer_metrics(
          {
              "rewards/reward_kl_penalty": (
                  jax.device_get(per_sequence_mean_kl.mean()),
                  np.mean,
              ),
          },
          mode=mode,
      )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "generation/completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "generation/completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "generation/completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )

    # Log advantages.
    valid_advantages = np.ma.masked_array(
        advantages, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "advantages/mean": (valid_advantages.mean(), np.mean),
            "advantages/max": (valid_advantages.max(), np.max),
            "advantages/min": (valid_advantages.min(), np.min),
        },
        mode=mode,
    )

    valid_returns = np.ma.masked_array(
        returns, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "advantages/returns/mean": (valid_returns.mean(), np.mean),
            "advantages/returns/max": (valid_returns.max(), np.max),
            "advantages/returns/min": (valid_returns.min(), np.min),
        },
        mode=mode,
    )

    for time_key in ["env_time", "reward_time"]:
      prefix = f"trajectory/{time_key}"
      time_dicts = [item.traj.get(time_key, {}) for item in trajectories]
        
      # Safely gather all unique sub-keys (e.g., 'reset_latency') across all trajectories      
      for sub_key in {k for d in time_dicts for k in d.keys()}:
        vals = [d.get(sub_key, 0.0) for d in time_dicts]
        metrics_to_log.update({
            f"{prefix}/{sub_key}/mean": (np.mean(vals), np.mean),
            f"{prefix}/{sub_key}/max": (np.max(vals), np.max),
            f"{prefix}/{sub_key}/min": (np.min(vals), np.min),
        })
        self.rl_cluster.buffer_metrics_async(
            metrics_to_log,  # pyrefly: ignore[bad-argument-type]
            mode=mode,
            step=expected_step,  # pyrefly: ignore[bad-argument-type]
        )

    # log user defined metrics
    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=prompt_ids,
          completions=completion_texts,
          advantages=advantages,
          rewards=last_token_scores,
          **{
              key: value
              for key, value in original_inputs.items()
              if key != "prompts"
          },
      )
      self.rl_cluster.buffer_metrics_async(
          user_defined_metric, mode=mode, step=expected_step
      )
    
    sampler_is_weights = None
    if (
        self.algo_config.sampler_is == "token"
        and rollout_per_token_logps is not None
        and trainer_per_token_logps is not None
    ):
      asst_mask_f = completion_mask.astype(jnp.float32)
      log_ratio = trainer_per_token_logps - rollout_per_token_logps
      log_ratio = jnp.clip(log_ratio, min=-20.0, max=20.0)
      sampler_is_weights = jax.lax.stop_gradient(
          jnp.minimum(
              jnp.exp(log_ratio),
              self.algo_config.sampler_is_threshold,
          )
          * asst_mask_f
      )

    return [TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        returns=returns,
        old_per_token_logps=old_per_token_logps,
        old_values=values,
        policy_version=policy_versions,
        sampler_is_weights=sampler_is_weights,
    )]

  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is same as the offset of the example in the data source.

    Args:
      example: The training input data.
      steps: The number of steps taken so far.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
    batch_size = len(example["prompts"]) // self._num_generations()  # pyrefly: ignore[bad-argument-type]
    row_offset = steps * batch_size
    row_offsets = np.arange(row_offset, row_offset + batch_size)
    return row_offsets.astype(str).tolist()

  def _num_iterations(self) -> int:
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    return 1

  def train(  # pylint: disable=useless-parent-delegation
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """PPO training loop."""
    super().train(train_ds, eval_ds, skip_jit)


PpoConfig = PPOConfig
PpoLearner = PPOLearner
