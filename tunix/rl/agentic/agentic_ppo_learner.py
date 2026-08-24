# Copyright 2026 Google LLC
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

"""PPO learner for the agentic setting."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Sequence, Type, TypeVar

from absl import logging
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import algo_core as ppo_helpers
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import task_environment
from tunix.utils import trajectory_logger


TrainingInputT = agentic_rl_learner.TrainingInputT
RewardFn = agentic_rl_learner.RewardFn
MetricFn = agentic_rl_learner.MetricFn
registry = function_registry.default_registry


@flax.struct.dataclass(frozen=True)
class TrainExample(agentic_rl_learner.TrainExample):
  returns: jax.Array | None = None
  old_values: jax.Array | None = None


@dataclasses.dataclass
class ExtractedTrajectories:
  """Holds raw extracted trajectory components."""

  completion_texts: List[str]
  prompt_tokens_list: List[np.ndarray]
  completion_tokens_list: List[np.ndarray]
  completion_masks_list: List[np.ndarray]
  old_logprobs_list: List[np.ndarray]
  policy_versions_list: List[int]
  trajectory_rewards_list: List[float]
  trajectories_to_log: List[Dict[str, Any]]
  original_inputs_list: List[Dict[str, Any]]


@dataclasses.dataclass
class ProcessedTrajectoryData:
  """Holds data extracted and processed from trajectories for PPO training."""

  prompt_ids: jnp.ndarray
  prompt_mask: jnp.ndarray
  completion_ids: jnp.ndarray
  completion_mask: jnp.ndarray
  completion_texts: List[str]
  trajectory_rewards: List[float]
  padded_old_logprobs: np.ndarray
  policy_versions: np.ndarray
  original_inputs: Dict[str, Any]


@dataclasses.dataclass(slots=True, kw_only=True)
class PPOConfig(agentic_rl_learner.AgenticRLConfig):
  """Configuration for PPO learner.

  Attributes:
    algo_variant: The algorithm variant to use. Default: `agentic_ppo`.
    advantage_estimator: The advantage estimator to use. Default: `gae`.
    policy_loss_fn: The policy loss function to use. Default: `ppo`.
    value_loss_fn: The value loss function to use. Default: `ppo`.
    num_iterations: The number of optimization epochs per batch of rollouts.
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
      data_shuffle_seed: int | None = None,
      env_class: Type[
          base_environment.BaseTaskEnv
      ] = task_environment.TaskEnvironment,
      env_kwargs: Dict[str, Any] | None = None,
  ):
    """Initializes the `PPOLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference, critic and reward models.
      algo_config: An instance of `PPOConfig` containing all training-specific
        configuration options.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions.
      agent_class: User defined agent class.
      agent_kwargs: Keyword arguments for the agent class.
      data_shuffle_seed: The seed for shuffling the data.
      env_class: User defined environment class.
      env_kwargs: Keyword arguments for the environment class.
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

    self.data_shuffle_seed = data_shuffle_seed

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
          f"{self.rl_cluster.inference_worker._models.get('reward')}"
      )
    if not self.rl_cluster.inference_worker._models.get("critic", None):
      raise ValueError(
          "PPO requires a critic model. Please pass the correct `critic` to "
          "`RLCluster`."
      )
    self._use_reward_model = bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    )

    # ===== Configure the actor (policy) trainer =====
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
        lambda x: {
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
    actor_rl_metrics_to_log = {"pg_clipfrac": np.mean}
    if self.algo_config.epsilon_c is not None:
      actor_rl_metrics_to_log["pg_clipfrac_lower"] = np.mean
    if (
        self.algo_config.entropy_coef is not None
        and self.algo_config.entropy_coef > 0.0
    ):
      actor_rl_metrics_to_log["loss/entropy"] = np.mean
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log(
        actor_rl_metrics_to_log
    )

    self.rl_cluster.critic_trainer.with_rl_metrics_to_log({
        "vpred_mean": np.mean,
        "vf_clipfrac": np.mean,
    })

  def _extract_trajectory_data(
      self, trajectories: List[Any]
  ) -> ExtractedTrajectories:
    """Extracts tokens, texts, rewards, and metadata from a list of trajectory results."""
    completion_texts: List[str] = []
    prompt_tokens_list: List[np.ndarray] = []
    completion_tokens_list: List[np.ndarray] = []
    completion_masks_list: List[np.ndarray] = []
    old_logprobs_list: List[np.ndarray] = []
    policy_versions_list: List[int] = []
    trajectory_rewards_list: List[float] = []
    trajectories_to_log: List[Dict[str, Any]] = []
    original_inputs_list: List[Dict[str, Any]] = []

    for item in trajectories:
      traj = item.traj
      trajectories_to_log.append(traj)

      conversation = traj.get("conversation_text") or []
      assistant_text = next(
          (
              message["content"]
              for message in conversation
              if message.get("role") == "assistant"
          ),
          "",
      )
      completion_texts.append(assistant_text)

      prompt_tokens_list.append(traj.get("prompt_tokens"))
      completion_tokens_list.append(traj.get("conversation_tokens"))
      completion_masks_list.append(traj.get("conversation_masks"))
      old_logprobs_list.append(traj.get("old_logprobs"))
      original_inputs_list.append(traj.get("original_input", {}))

      policy_version = traj.get("policy_version")
      if policy_version is None:
        raise ValueError("policy_version is missing from trajectory task.")
      policy_versions_list.append(policy_version)

      trajectory_rewards_list.append(traj.get("trajectory_reward"))

    return ExtractedTrajectories(
        completion_texts=completion_texts,
        prompt_tokens_list=prompt_tokens_list,
        completion_tokens_list=completion_tokens_list,
        completion_masks_list=completion_masks_list,
        old_logprobs_list=old_logprobs_list,
        policy_versions_list=policy_versions_list,
        trajectory_rewards_list=trajectory_rewards_list,
        trajectories_to_log=trajectories_to_log,
        original_inputs_list=original_inputs_list,
    )

  def _pad_prompts_and_completions(
      self, extracted: ExtractedTrajectories, mode: rl_cluster_lib.Mode
  ):
    """Pads prompt and completion tokens, masks, and rollout logprobs to fixed lengths."""
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      rollout_config = rollout_config[mode]

    padded_prompt_ids = []
    padded_completion_ids = []
    padded_completion_masks = []
    padded_old_logprobs = []

    max_response_length = self.algo_config.max_response_length
    clipped_completion_count = 0
    for (
        prompt_tokens,
        completion_tokens,
        completion_mask,
        old_logprobs,
    ) in zip(
        extracted.prompt_tokens_list,
        extracted.completion_tokens_list,
        extracted.completion_masks_list,
        extracted.old_logprobs_list,
    ):
      if (
          len(completion_tokens) >= max_response_length
          and completion_mask[-1] != eos_value
      ):
        clipped_completion_count += 1

      padded_prompt, padded_completion, _ = (
          agentic_utils.pad_prompt_and_completion(
              prompt_tokens,
              completion_tokens,
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

    return (
        padded_prompt_ids,
        padded_completion_ids,
        padded_completion_masks,
        padded_old_logprobs,
        clipped_completion_count,
    )

  def _collect_perf_tags(
      self, trajectories: List[Any], expected_step: int | None
  ) -> Dict[str, Any]:
    """Collects performance tracking tags from trajectories."""
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

    return perf_tags

  def _compute_values(self, traj_data: ProcessedTrajectoryData) -> jnp.ndarray:
    """Computes value function estimates for completion tokens."""
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    logits_to_keep = traj_data.completion_ids.shape[1]

    values = self.rl_cluster.get_values(
        prompt_tokens=traj_data.prompt_ids,
        completion_tokens=traj_data.completion_ids,
        pad_id=pad_value,
        eos_id=eos_value,
    )
    # `values` start from the last *prompt* token. Shape: `[B, T]`.
    values = values[:, -logits_to_keep - 1 : -1]
    values = values * traj_data.completion_mask
    return values

  def _compute_rollout_trainer_logps(
      self, traj_data: ProcessedTrajectoryData
  ):
    """Computes rollout logprobs, trainer logprobs, and old_per_token_logps."""
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    actor_mesh = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR]
    have_actor_mesh = actor_mesh is not None and not actor_mesh.empty
    rollout_per_token_logps = None
    trainer_per_token_logps = None
    if self.algo_config.use_rollout_logps and traj_data.padded_old_logprobs:
      rollout_per_token_logps = jnp.asarray(traj_data.padded_old_logprobs)
      old_per_token_logps = rollout_per_token_logps
      need_trainer_logps = (
          have_actor_mesh or self.algo_config.sampler_is == "token"
      )
      if need_trainer_logps:
        trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
            prompt_tokens=traj_data.prompt_ids,
            completion_tokens=traj_data.completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
        )
      if (
          self.algo_config.sampler_is == "token"
          and trainer_per_token_logps is not None
      ):
        old_per_token_logps = trainer_per_token_logps
    elif self.algo_config.use_rollout_logps:
      old_per_token_logps = None
    else:
      trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
          prompt_tokens=traj_data.prompt_ids,
          completion_tokens=traj_data.completion_ids,
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

    return rollout_per_token_logps, trainer_per_token_logps, old_per_token_logps

  def _sampler_v_trainer_logp_diag(
      self,
      completion_mask: jnp.ndarray,
      rollout_per_token_logps: jnp.ndarray | None,
      trainer_per_token_logps: jnp.ndarray | None,
      metrics_to_log: Dict[str, Any],
  ) -> None:
    """Per-token sampler-vs-trainer log-probability agreement diagnostic."""
    if rollout_per_token_logps is None or trainer_per_token_logps is None:
      return

    mask = completion_mask.astype(jnp.bool_)
    mask_f = mask.astype(jnp.float32)
    mask_sum = jnp.maximum(mask_f.sum(), 1.0)
    diff = jnp.abs(rollout_per_token_logps - trainer_per_token_logps)
    diff_mean = float((diff * mask_f).sum() / mask_sum)
    diff_max = float(jnp.where(mask, diff, 0.0).max())

    rp = jnp.exp(rollout_per_token_logps)
    tp = jnp.exp(trainer_per_token_logps)
    prob_diff = jnp.abs(rp - tp)
    prob_diff_mean = float((prob_diff * mask_f).sum() / mask_sum)
    prob_diff_max = float(jnp.where(mask, prob_diff, 0.0).max())

    rp_flat = rp.reshape(-1)
    tp_flat = tp.reshape(-1)
    mf = mask_f.reshape(-1)
    rp_mean = (rp_flat * mf).sum() / mask_sum
    tp_mean = (tp_flat * mf).sum() / mask_sum
    rp_d = (rp_flat - rp_mean) * mf
    tp_d = (tp_flat - tp_mean) * mf
    cov = (rp_d * tp_d).sum() / mask_sum
    rp_var = (rp_d * rp_d).sum() / mask_sum
    tp_var = (tp_d * tp_d).sum() / mask_sum
    pearson = float(cov / jnp.sqrt(jnp.maximum(rp_var * tp_var, 1e-12)))

    metrics_to_log.update({
        "sampler_trainer/logp_diff_mean": (diff_mean, np.mean),
        "sampler_trainer/logp_diff_max": (diff_max, np.max),
        "sampler_trainer/prob_diff_mean": (prob_diff_mean, np.mean),
        "sampler_trainer/prob_diff_max": (prob_diff_max, np.max),
        "sampler_trainer/probs_pearson_corr": (pearson, np.mean),
    })
    logging.info(
        "sampler-trainer: logp_diff=(%.5f,%.5f) prob_diff=(%.5f,%.5f)"
        " pearson=%.5f",
        diff_mean,
        diff_max,
        prob_diff_mean,
        prob_diff_max,
        pearson,
    )

  def _compute_sampler_is_weights(
      self,
      completion_mask: jnp.ndarray,
      rollout_per_token_logps: jnp.ndarray | None,
      trainer_per_token_logps: jnp.ndarray | None,
      metrics_to_log: Dict[str, Any],
  ) -> jnp.ndarray | None:
    """Computes truncated importance-sampling (TIS) correction weights."""
    if (
        self.algo_config.sampler_is != "token"
        or rollout_per_token_logps is None
        or trainer_per_token_logps is None
    ):
      return None

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
    mask_sum = jnp.maximum(asst_mask_f.sum(), 1.0)
    is_mean = float((sampler_is_weights * asst_mask_f).sum() / mask_sum)
    is_max = float(jnp.where(asst_mask_f > 0, sampler_is_weights, 0.0).max())
    frac_clipped = float(
        (
            (
                (jnp.exp(log_ratio) > self.algo_config.sampler_is_threshold)
                & (asst_mask_f > 0)
            ).astype(jnp.float32)
        ).sum()
        / mask_sum
    )
    metrics_to_log.update({
        "sampler_is/weight_mean": (is_mean, np.mean),
        "sampler_is/weight_max": (is_max, np.max),
        "sampler_is/frac_clipped_at_threshold": (frac_clipped, np.mean),
    })
    logging.info(
        "sampler_is: weight_mean=%.4f weight_max=%.4f frac_clipped=%.4f"
        " (threshold=%.2f)",
        is_mean,
        is_max,
        frac_clipped,
        self.algo_config.sampler_is_threshold,
    )

    return sampler_is_weights

  def _compute_ref_logps(
      self, perf_tags: Dict[str, Any], traj_data: ProcessedTrajectoryData
  ) -> jnp.ndarray | None:
    """Computes reference policy logprobs if beta != 0."""
    if self.algo_config.beta != 0.0:
      pad_value = self.rl_cluster.rollout.pad_id()
      eos_value = self.rl_cluster.rollout.eos_id()

      with self.rl_cluster.perf_v2.span(
          perf_constants.REFERENCE_INFERENCE,
          devices=self.rl_cluster.r2m[rl_cluster_lib.Role.REFERENCE].devices,
          tags=perf_tags,
      ) as interval_v2:
        ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
            prompt_tokens=traj_data.prompt_ids,
            completion_tokens=traj_data.completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
        )
        interval_v2.async_end([ref_per_token_logps])
    else:
      ref_per_token_logps = None

    return ref_per_token_logps

  def _get_rewards(
      self,
      trajectories: List[Any],
      traj_data: ProcessedTrajectoryData,
      perf_tags: Dict[str, Any],
      mode: rl_cluster_lib.Mode,
      expected_step: int | None,
  ):
    """Computes token-level reward matrix and returns (rewards, last_token_scores)."""
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    batch_size = traj_data.completion_ids.shape[0]
    logits_to_keep = traj_data.completion_ids.shape[1]

    eos_idx = jnp.max(
        common.build_positions_from_mask(traj_data.completion_mask),
        axis=-1,
    )

    if self._use_reward_model:
      scores = self.rl_cluster.get_rewards(
          prompt_tokens=traj_data.prompt_ids,
          completion_tokens=traj_data.completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )[:, -logits_to_keep:]
      jax_last_token_scores = scores[jnp.arange(batch_size), eos_idx]
      last_token_scores = jax.device_get(jax_last_token_scores)
    else:
      reward_kwargs = {
          key: value
          for key, value in traj_data.original_inputs.items()
          if key != "prompts"
      }
      reward_kwargs["trajectory_rewards"] = traj_data.trajectory_rewards
      prompts_texts = traj_data.original_inputs.get(
          "prompts",
          [item.traj.get("prompt_text", "") for item in trajectories],
      )

      with self.rl_cluster.perf_v2.span(
          perf_constants.ADVANTAGE_COMPUTATION,
          tags=perf_tags,
      ):
        last_token_scores = self._compute_rewards(
            prompts=prompts_texts,
            completions=traj_data.completion_texts,
            mode=mode,
            expected_step=expected_step,
            **reward_kwargs,
        )
      jax_last_token_scores = jax.device_put(last_token_scores)

    rewards = jnp.zeros_like(traj_data.completion_ids, dtype=jnp.float32)
    rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(
        jax_last_token_scores
    )

    return rewards, last_token_scores

  def _get_KL_divergence(
      self,
      old_per_token_logps: jnp.ndarray,
      ref_per_token_logps: jnp.ndarray,
      completion_mask: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes masked token-level KL divergence between old policy and reference policy."""
    kl = common.compute_kl_divergence(
        old_per_token_logps,
        ref_per_token_logps,
        method=self.algo_config.kl_method,
        clamp_value=self.algo_config.kl_clamp_value,
    )
    kl = kl * completion_mask
    return kl

  def _compute_advantages_and_returns(
      self,
      rewards: jnp.ndarray,
      values: jnp.ndarray,
      completion_mask: jnp.ndarray,
  ):
    """Computes advantages and returns using GAE."""
    advantage_estimator = function_registry.get_advantage_estimator(
        self.algo_config.advantage_estimator
    )
    advantages, returns = advantage_estimator(
        rewards=rewards,
        values=values,
        completion_mask=completion_mask,
        gamma=self.algo_config.gamma,
        gae_lambda=self.algo_config.gae_lambda,
    )

    logging.debug("Advantages computed: %s", advantages)
    return advantages, returns

  def _extract_time_metrics(
      self, trajectories: List[Any], metrics_to_log: Dict[str, Any]
  ) -> None:
    """Extracts env_time and reward_time metrics across trajectories."""
    for time_key in ["env_time", "reward_time"]:
      prefix = f"trajectory/{time_key}"
      time_dicts = [item.traj.get(time_key, {}) for item in trajectories]

      for sub_key in {k for d in time_dicts for k in d.keys()}:
        vals = [d.get(sub_key, 0.0) for d in time_dicts]
        metrics_to_log.update({
            f"{prefix}/{sub_key}/mean": (np.mean(vals), np.mean),
            f"{prefix}/{sub_key}/max": (np.max(vals), np.max),
            f"{prefix}/{sub_key}/min": (np.min(vals), np.min),
        })

  def _eval_user_metrics(
      self,
      traj_data: ProcessedTrajectoryData,
      advantages: jnp.ndarray,
      rewards: Any,
      mode: rl_cluster_lib.Mode,
      expected_step: int | None,
  ) -> None:
    """Evaluates custom user-defined metric functions and buffers them."""
    for metric_fn in self.metric_fns:
      prompts = traj_data.original_inputs.get("prompts", [])
      user_defined_metric = metric_fn(
          prompts=prompts,
          completions=traj_data.completion_texts,
          advantages=advantages,
          rewards=rewards,
          **{
              key: value
              for key, value in traj_data.original_inputs.items()
              if key != "prompts"
          },
      )
      self.rl_cluster.buffer_metrics_async(
          user_defined_metric, mode=mode, step=expected_step
      )

  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      expected_step: int | None = None,
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

    pad_value = self.rl_cluster.rollout.pad_id()

    # Extract trajectory data
    extracted = self._extract_trajectory_data(trajectories)

    # Log trajectory.
    if self._trajectory_logger and extracted.trajectories_to_log:
      for traj in extracted.trajectories_to_log:
        self._trajectory_logger.log_item_async(traj)

    # Pad prompts and completions
    (
        padded_prompt_ids,
        padded_completion_ids,
        padded_completion_masks,
        padded_old_logprobs,
        clipped_completion_count,
    ) = self._pad_prompts_and_completions(extracted, mode)

    prompt_ids = jnp.asarray(padded_prompt_ids)
    prompt_mask = prompt_ids != pad_value
    completion_ids = jnp.asarray(padded_completion_ids)
    completion_mask = jnp.asarray(padded_completion_masks)
    original_inputs = rl_utils.merge_micro_batches(
        extracted.original_inputs_list
    )
    policy_versions = np.array(extracted.policy_versions_list, dtype=np.int32)

    traj_data = ProcessedTrajectoryData(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_texts=extracted.completion_texts,
        trajectory_rewards=extracted.trajectory_rewards_list,
        padded_old_logprobs=padded_old_logprobs,
        policy_versions=policy_versions,
        original_inputs=original_inputs,
    )

    logging.debug(
        "Token shapes: prompt_ids=%s, completion_ids=%s",
        traj_data.prompt_ids.shape,
        traj_data.completion_ids.shape,
    )

    # Log prompt length metrics
    prompt_token_len = len(extracted.prompt_tokens_list[0])
    self.rl_cluster.buffer_metrics_async(
        {
            "generation/prompts/mean_length": (prompt_token_len, np.mean),
            "generation/prompts/max_length": (prompt_token_len, np.max),
            "generation/prompts/min_length": (prompt_token_len, np.min),
        },
        mode=mode,
        step=expected_step,
    )

    (
        rollout_per_token_logps,
        trainer_per_token_logps,
        old_per_token_logps,
    ) = self._compute_rollout_trainer_logps(traj_data)

    perf_tags = self._collect_perf_tags(trajectories, expected_step)
    ref_per_token_logps = self._compute_ref_logps(perf_tags, traj_data)
    values = self._compute_values(traj_data)
    rewards, last_token_scores = self._get_rewards(
        trajectories, traj_data, perf_tags, mode, expected_step
    )

    kl = None
    if self.algo_config.beta != 0.0:
      kl = self._get_KL_divergence(
          old_per_token_logps, ref_per_token_logps, traj_data.completion_mask
      )
      rewards = rewards - self.algo_config.beta * kl

    advantages, returns = self._compute_advantages_and_returns(
        rewards, values, traj_data.completion_mask
    )

    # Metric logging dictionary
    agg_completion_mask = completion_mask.sum(axis=-1)
    metrics_to_log = {
        "rewards/score/mean": (np.mean(last_token_scores), np.mean),
        "rewards/score/max": (np.max(last_token_scores), np.max),
        "rewards/score/min": (np.min(last_token_scores), np.min),
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

    sequence_rewards = jax.device_get(rewards.sum(-1))
    metrics_to_log.update({
        "rewards/reward/mean": (np.mean(sequence_rewards), np.mean),
        "rewards/reward/max": (np.max(sequence_rewards), np.mean),
        "rewards/reward/min": (np.min(sequence_rewards), np.mean),
    })

    if kl is not None:
      per_sequence_mean_kl = ppo_helpers.masked_mean(
          kl, completion_mask, axis=-1
      )
      metrics_to_log["rewards/reward_kl_penalty"] = (
          jax.device_get(per_sequence_mean_kl.mean()),
          np.mean,
      )

    valid_returns = np.ma.masked_array(
        returns, mask=np.logical_not(completion_mask)
    )
    metrics_to_log.update({
        "advantages/returns/mean": (valid_returns.mean(), np.mean),
        "advantages/returns/max": (valid_returns.max(), np.max),
        "advantages/returns/min": (valid_returns.min(), np.min),
    })

    valid_values = np.ma.masked_array(
        values, mask=np.logical_not(completion_mask)
    )
    metrics_to_log.update({
        "advantages/old_values/mean": (valid_values.mean(), np.mean),
        "advantages/old_values/max": (valid_values.max(), np.max),
        "advantages/old_values/min": (valid_values.min(), np.min),
    })

    self._sampler_v_trainer_logp_diag(
        traj_data.completion_mask,
        rollout_per_token_logps,
        trainer_per_token_logps,
        metrics_to_log,
    )

    sampler_is_weights = self._compute_sampler_is_weights(
        traj_data.completion_mask,
        rollout_per_token_logps,
        trainer_per_token_logps,
        metrics_to_log,
    )

    self._extract_time_metrics(trajectories, metrics_to_log)

    self.rl_cluster.buffer_metrics_async(
        metrics_to_log,
        mode=mode,
        step=expected_step,
    )

    self._eval_user_metrics(
        traj_data, advantages, last_token_scores, mode, expected_step
    )

    return [
        TrainExample(
            prompt_ids=traj_data.prompt_ids,
            prompt_mask=traj_data.prompt_mask,
            completion_ids=traj_data.completion_ids,
            completion_mask=traj_data.completion_mask,
            ref_per_token_logps=ref_per_token_logps,
            advantages=advantages,
            returns=returns,
            old_per_token_logps=old_per_token_logps,
            old_values=values,
            policy_version=policy_versions,
            sampler_is_weights=sampler_is_weights,
        )
    ]

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
    batch_size = len(example["prompts"]) // self._num_generations()
    row_offset = steps * batch_size
    row_offsets = np.arange(row_offset, row_offset + batch_size)
    return row_offsets.astype(str).tolist()

  def _num_iterations(self) -> int:
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    return self.algo_config.num_generations

  def train(
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """PPO training loop."""
    super().train(train_ds, eval_ds, skip_jit)


PpoConfig = PPOConfig
PpoLearner = PPOLearner
