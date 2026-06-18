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

"""Implements an RLLearner for the Agentic GRPO algorithm.

This learner orchestrates the process of generating multiple text completions
for each prompt from a dataset, computing rewards and advantages according to
the GRPO (Group-wise Reward Policy Optimization) algorithm, and then training
the actor model.

The data flow is designed around an asynchronous producer-consumer pattern:
1. A producer generates rollouts (text generations) in parallel for each prompt.
2. These rollouts are grouped by the original prompt.
3. For each group, rewards and advantages are computed.
4. The resulting training examples are put into a queue.
5. The main training loop consumes these examples to update the model weights.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Sequence, Type, TypeVar

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core  # pylint: disable=unused-import
from tunix.perf.experimental import constants as perf_constants
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

TrainExample = agentic_rl_learner.TrainExample


@dataclasses.dataclass(kw_only=True)
class GRPOConfig(agentic_rl_learner.AgenticRLConfig):
  """Configuration for GRPO algorithm.

  Attributes:
    algo_variant: Algorithm variant name.
    advantage_estimator: Name of the advantage estimator function.
    policy_loss_fn: Name of the policy loss function.
    loss_agg_mode: Method for aggregating the loss. Supported values:
      "token-mean", "sequence-mean-token-mean", "sequence-mean-token-scale",
      "seq-mean-token-sum", "sequence-mean-token-sum-norm".
    num_generations: Number of samples per prompt (G in the paper). Must be > 1.
    num_iterations: Number of GRPO iterations per batch (μ in the paper).
    beta: KL penalty coefficient.
    kl_loss_mode: Method for computing the KL loss.
    force_compute_kl: Whether to force compute KL divergence for logging even
      when it would normally be skipped (e.g., when beta is 0.0).
    epsilon: PPO-style clipping epsilon.
    epsilon_high: PPO-style clipping epsilon upper bound.
    loss_algo: "grpo" or "gspo-token".
    system_prompt: System prompt for the agent.
    max_concurrency: Maximum number of concurrent rollout engines.
    off_policy_steps: Number of off-policy steps can be accepted before a policy
      update.
    degenerate_group_masking: Whether to mask out degenerate groups with all-0
      advantages.
  """

  algo_variant: str = "agentic_grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  loss_algo: (
      str
  ) = (  # grpo or gspo-token # TODO(sizhi): Remove this option once gspo is
      # refactored to a separate loss fn.
      "grpo"
  )
  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  kl_loss_mode: str = "kl"
  force_compute_kl: bool = False
  epsilon: float = 0.2
  system_prompt: str = ""
  max_concurrency: int = 16
  epsilon_high: float | None = None  # 0.28 from DAPO.
  off_policy_steps: int = 0
  degenerate_group_masking: bool = (
      False  # Whether to mask out degenerate groups with all-0 advantages.
  )
  use_rollout_logps: bool = True
  # Truncated importance-sampling (TIS) correction for the residual mismatch
  # between the rollout sampler and the trainer's recomputed log-probabilities.
  # Set to ``"token"`` to enable per-token TIS weights. When enabled, the loss
  # path uses the trainer's start-of-step recomputed logp as
  # ``old_per_token_logps`` (so the PPO ratio is taken against the trainer's
  # own policy at step start, rather than directly against the sampler's logp)
  # and multiplies each per-token pg-loss term by a detached weight
  #   w_t = clip(exp(clip(trainer_logp_t - sampler_logp_t, ±20)), max=threshold)
  # dampening positions where the trainer's recomputed probability disagrees
  # significantly with the rollout sampler. Without this correction, importance
  # ratios computed directly against the sampler's logp can spike on outlier
  # tokens, producing large-variance gradient updates.
  sampler_is: str | None = None  # None | "token"
  sampler_is_threshold: float = 2.0

  def __post_init__(self):
    if self.num_generations <= 1:
      raise ValueError(
          "num_generations must be greater than 1. Received: "
          f"{self.num_generations}"
      )
    if self.epsilon_high is None:
      self.epsilon_high = self.epsilon
    if self.loss_algo not in ["grpo", "gspo-token"]:
      raise ValueError(
          "loss_algo should be either grpo or gspo-token. Received: "
          f"{self.loss_algo}"
      )


TGrpoConfig = TypeVar("TGrpoConfig", bound=GRPOConfig)


class GRPOLearner(agentic_rl_learner.AgenticRLLearner[TGrpoConfig]):
  """An RLLearner that implements the GRPO algorithm in an agentic setting.

  GRPO is a reinforcement learning algorithm designed to enhance the reasoning
  abilities of large language models, like mathematical problem-solving. It is
  a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
  eliminating the need for a separate value function model. GRPO works by
  generating multiple responses for a given prompt, evaluating these responses
  using a reward model, and then calculating a relative advantage based on the
  group's performance to update the policy.

  References:
    - https://arxiv.org/abs/2402.03300
  """

  def _logps_sequence_micro_batch_size(self) -> int:
    """Returns the flattened sequence micro-batch size for log-prob passes."""
    return (
        self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size
        * self.algo_config.num_generations
    )

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TGrpoConfig,
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
    """Initializes the `GRPOTrainer`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      reward_fns: A single callable or a list of callables that compute a
        scalar reward for given prompts and completions. Each function should
        accept `prompts`, `completions` and optional keyword arguments, and
        return a list of float rewards.
      algo_config: An instance of `GRPOConfig` containing all GRPO specific
        parameters.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return
        a dictionary of metric names to tuples of
        ``(metric_value, aggregation_fn)``:

           >>> def metric_fn(
           ...     prompts, completions, rewards, advantages, **kargs
           ... ):
           ...     return {
           ...       # ...
           ...       "prompt_min_len": (min(len(p) for p in prompts), np.min),
           ...       # ... }
      agent_class: The class of the agent to be used.
      agent_kwargs: Keyword arguments to pass to the agent class.
      env_class: The class of the environment to be used.
      env_kwargs: Keyword arguments to pass to the environment class.
    """  # fmt: skip
    super().__init__(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        algo_config=algo_config,
        chat_parser=chat_parser,
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

    self.algo_config.temperature = self.rl_cluster.get_rollout_config(
        mode=rl_cluster_lib.Mode.TRAIN
    ).temperature

    # Workaround to pass loss fn with algorithm flag
    policy_loss_fn = function_registry.get_policy_loss_fn(
        self.algo_config.policy_loss_fn
    )
    loss_fn = lambda model, train_example, algo_config: policy_loss_fn(
        model,
        train_example,
        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
    )

    self.rl_cluster.actor_trainer.with_loss_fn(
        loss_fn,
        has_aux=True,
    )
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "algo_config": self.algo_config,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({
        "kl": np.mean,
        "entropy": np.mean,
        "pg_loss": np.mean,
        "pg_clipfrac": np.mean,
        "ppo_kl": np.mean,
        "kl_loss": np.mean,
        "is_ratio/mean": np.mean,
        "is_ratio/max": np.max,
        "is_ratio/min": np.min,
        "log_ratio/abs_mean": np.mean,
        "pg_loss/unclipped_mean": np.mean,
        "pg_loss/clipped_mean": np.mean,
        "advantage/abs_mean": np.mean,
        "advantage/max": np.max,
        "advantage/min": np.min,
        "advantage/nonzero_frac": np.mean,
        "sampler_is/weight_mean": np.mean,
        "sampler_is/weight_min": np.min,
    })
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        lambda: "kl"
        if self.algo_config.force_compute_kl or self.algo_config.beta != 0.0
        else None,
    ])

  def _decode_token_debug(self, token_ids: Sequence[int]) -> str:
    """Best-effort token decode that preserves special tokens when possible."""
    ids = [int(x) for x in token_ids]
    try:
      return self.tokenizer.decode(ids, skip_special_tokens=False)
    except TypeError:
      return self.tokenizer.decode(ids)
    except Exception:  # pylint: disable=broad-exception-caught
      return repr(ids)

  def _maybe_log_sampler_trainer_diff_debug(
      self,
      *,
      group_id: Any,
      completion_ids: jax.Array,
      completion_mask: jax.Array,
      rollout_per_token_logps: jax.Array,
      trainer_per_token_logps: jax.Array,
      completion_texts: Sequence[str],
  ) -> None:
    """Logs top sampler-vs-trainer logprob mismatches when requested."""
    if not os.getenv("TUNIX_DEBUG_LOGP_DIFF"):
      return
    try:
      limit = int(os.getenv("TUNIX_DEBUG_LOGP_DIFF_TOPK", "8"))
    except ValueError:
      limit = 8
    if limit <= 0:
      return
    try:
      completion_ids_np = np.asarray(jax.device_get(completion_ids))
      mask_np = np.asarray(jax.device_get(completion_mask.astype(jnp.bool_)))
      rollout_np = np.asarray(jax.device_get(rollout_per_token_logps))
      trainer_np = np.asarray(jax.device_get(trainer_per_token_logps))
      diff_np = np.abs(rollout_np - trainer_np)
      masked_flat = np.where(mask_np, diff_np, -np.inf).reshape(-1)
      if not np.isfinite(masked_flat).any():
        logging.warning(
            "sampler-trainer debug group=%s has no masked completion tokens",
            group_id,
        )
        return
      top_flat_idxs = np.argsort(masked_flat)[::-1][:limit]
      for rank, flat_idx in enumerate(top_flat_idxs, start=1):
        if not np.isfinite(masked_flat[flat_idx]):
          continue
        seq_idx, tok_idx = np.unravel_index(flat_idx, diff_np.shape)
        token_id = int(completion_ids_np[seq_idx, tok_idx])
        start = max(0, tok_idx - 8)
        end = min(completion_ids_np.shape[1], tok_idx + 9)
        window_ids = completion_ids_np[seq_idx, start:end]
        text_snippet = (
            completion_texts[seq_idx][:240]
            if seq_idx < len(completion_texts)
            else ""
        )
        logging.warning(
            "sampler-trainer top_diff group=%s rank=%d seq=%d tok=%d"
            " token_id=%d token=%r rollout_logp=%.6f trainer_logp=%.6f"
            " diff=%.6f rollout_prob=%.6g trainer_prob=%.6g"
            " window_ids=%r window_text=%r completion_prefix=%r",
            group_id,
            rank,
            seq_idx,
            tok_idx,
            token_id,
            self._decode_token_debug([token_id]),
            float(rollout_np[seq_idx, tok_idx]),
            float(trainer_np[seq_idx, tok_idx]),
            float(diff_np[seq_idx, tok_idx]),
            float(np.exp(rollout_np[seq_idx, tok_idx])),
            float(np.exp(trainer_np[seq_idx, tok_idx])),
            [int(x) for x in window_ids],
            self._decode_token_debug(window_ids),
            text_snippet,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Failed sampler-trainer logp diff debug: %s", e)

  def _trajectory_log_item(
      self, item: Any, expected_step: int | None
  ) -> dict[str, Any]:
    """Builds a lightweight trajectory record for CSV logging."""
    traj = item.traj
    return {
        "global_step": expected_step,
        "group_id": item.group_id,
        "pair_index": item.pair_index,
        "trajectory_reward": traj.get("trajectory_reward"),
        "conversation_text": traj.get("conversation_text"),
        "original_input": traj.get("original_input"),
    }

  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ) -> List[TrainExample]:
    """Processes generation results, computes rewards and advantages.

    This is a core method that performs several steps:
    1. Extracts completions from the raw trajectory results.
    2. Pads prompt and completion tokens to a consistent length.
    3. Computes masks for prompts and completions.
    4. Gets reference and old model log probabilities if required.
    5. Computes rewards for each completion using the provided reward functions.
    6. Computes GRPO-specific advantages from the rewards.
    7. Buffers metrics for logging.
    8. Constructs and returns a list of `TrainExample` objects.

    Args:
      trajectories: A list of trajectory results for a single GRPO group.
      mode: The current mode (TRAIN or EVAL).
      expected_step: The expected training step.

    Returns:
      A list of `TrainExample` instances containing all data needed for the
      loss function.

    Raises:
      ValueError: If `policy_version` is missing from any trajectory task.
      RuntimeError: If `old_per_token_logps` is not available for off-policy RL.
    """
    logging.debug(
        "Processing results to compute advantage for %d items.",
        len(trajectories),
    )
    # With a full group, sorting by pair_index is not necessary as they all
    # originate from the same initial prompt.
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    # Extract completions and tokens from the group of G results.
    completion_texts: List[str] = []
    prompt_tokens_list: List[np.ndarray] = []
    completion_tokens_list: List[np.ndarray] = []
    completion_masks_list: List[np.ndarray] = []
    old_logprobs_list: List[np.ndarray] = []
    policy_versions_list: List[int] = []
    trajectory_rewards_list: List[float] = []
    raw_completion_lengths: List[int] = []
    trajectories_to_log = []

    for item in trajectories:
      trajectories_to_log.append(self._trajectory_log_item(item, expected_step))
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
      raw_completion_lengths.append(
          min(len(completion_tokens), max_response_length)
      )
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

    prompt_ids = jnp.asarray(padded_prompt_ids)
    prompt_mask = prompt_ids != pad_value
    completion_ids = jnp.asarray(padded_completion_ids)
    completion_mask = jnp.asarray(padded_completion_masks)
    logging.debug(
        "Token shapes: prompt_ids=%s, completion_ids=%s",
        prompt_ids.shape,
        completion_ids.shape,
    )

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
            micro_batch_size=self._logps_sequence_micro_batch_size(),
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
          micro_batch_size=self._logps_sequence_micro_batch_size(),
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

    if self.algo_config.force_compute_kl or self.algo_config.beta != 0.0:
      with self.rl_cluster.perf_v2.span(
          perf_constants.REFERENCE_INFERENCE,
          devices=self.rl_cluster.r2m[rl_cluster_lib.Role.REFERENCE].devices,
          tags=perf_tags,
      ) as interval_v2:
        ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=self._logps_sequence_micro_batch_size(),
        )
        interval_v2.async_end([ref_per_token_logps])
    else:
      ref_per_token_logps = None

    # Rewards & advantages
    # Prepare arguments for reward computation by forwarding all training inputs
    # except for prompts, which is passed explicitly.
    original_inputs_list = [
        item.traj["original_input"] for item in trajectories
    ]
    original_inputs = rl_utils.merge_micro_batches(original_inputs_list)

    prompt_token_len = len(prompt_tokens_list[0])
    self.rl_cluster.buffer_metrics_async(
        {
            "generation/prompts/mean_length": (prompt_token_len, np.mean),
            "generation/prompts/max_length": (prompt_token_len, np.max),
            "generation/prompts/min_length": (prompt_token_len, np.min),
        },
        mode=mode,
        step=expected_step,
    )

    reward_kwargs = {
        key: value for key, value in original_inputs.items() if key != "prompts"
    }
    reward_kwargs["trajectory_rewards"] = trajectory_rewards_list
    with self.rl_cluster.perf_v2.span(
        perf_constants.ADVANTAGE_COMPUTATION,
        tags=perf_tags,
    ):
      rewards = self._compute_rewards(
          prompts=original_inputs["prompts"],
          completions=completion_texts,
          mode=mode,
          **reward_kwargs,
          expected_step=expected_step,
      )

      advantage_estimator = function_registry.get_advantage_estimator(
          self.algo_config.advantage_estimator
      )
      advantages = advantage_estimator(
          rewards=rewards, num_generations=self.algo_config.num_generations
      )

    logging.debug("Advantages computed: %s", advantages)

    if self.algo_config.degenerate_group_masking:
      if jnp.all(jnp.isclose(advantages, 0.0)):
        logging.info(
            "Filtering degenerate group %s with all-0 advantages.", group_id
        )
        completion_mask = jnp.zeros_like(completion_mask)

    policy_versions = np.array(policy_versions_list, dtype=np.int32)

    # Log completion lengths, rewards and env time.
    agg_completion_mask = completion_mask.sum(axis=-1)
    raw_completion_lengths_np = np.asarray(
        raw_completion_lengths, dtype=np.int32
    )
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
        # Raw length mirrors rLLM/VERL response_length: all trajectory response
        # tokens after the initial prompt, including env/user tokens, clamped to
        # max_response_length. The existing *_length metrics remain loss-mask
        # lengths over assistant-generated tokens only.
        "generation/completions/mean_raw_length": (
            np.mean(raw_completion_lengths_np),
            np.mean,
        ),
        "generation/completions/max_raw_length": (
            np.max(raw_completion_lengths_np),
            np.max,
        ),
        "generation/completions/min_raw_length": (
            np.min(raw_completion_lengths_np),
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

    # Per-token sampler-vs-trainer log-probability agreement diagnostic. When
    # this diverges from zero, importance ratios used in the policy update
    # are biased and gradient quality degrades. A mean per-token diff well
    # under 0.01 nat indicates the trainer and rollout sampler are computing
    # log-probabilities consistently.
    if (
        rollout_per_token_logps is not None
        and trainer_per_token_logps is not None
    ):
      # ``completion_mask`` is the assistant-vs-env mask built upstream
      # (1 for assistant-generated tokens, 0 for env-injected tokens), and
      # already correctly scopes the comparison to model-emitted positions.
      # We deliberately do NOT additionally drop positions where the rollout
      # logprob equals exactly 0.0 — that value can legitimately occur for
      # near-certain tokens (e.g. format chars after a structured response)
      # and excluding them removes the most consistent positions from the
      # statistic, inflating the per-position mean.
      mask = completion_mask.astype(jnp.bool_)
      mask_f = mask.astype(jnp.float32)
      mask_sum = jnp.maximum(mask_f.sum(), 1.0)
      diff = jnp.abs(rollout_per_token_logps - trainer_per_token_logps)
      diff_mean = float((diff * mask_f).sum() / mask_sum)
      diff_max = float(jnp.where(mask, diff, 0.0).max())
      # Per-position probability-space diff |exp(rollout) - exp(trainer)|.
      # More representative than logp_diff for confidence agreement: logp can
      # diverge arbitrarily for very low-probability tokens while their
      # contribution to the importance ratio is negligible. prob_diff weights
      # each position by its actual probability mass.
      rp = jnp.exp(rollout_per_token_logps)
      tp = jnp.exp(trainer_per_token_logps)
      prob_diff = jnp.abs(rp - tp)
      prob_diff_mean = float((prob_diff * mask_f).sum() / mask_sum)
      prob_diff_max = float(jnp.where(mask, prob_diff, 0.0).max())
      # Pearson correlation between exp(logp) at masked positions.
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
      self._maybe_log_sampler_trainer_diff_debug(
          group_id=group_id,
          completion_ids=completion_ids,
          completion_mask=completion_mask,
          rollout_per_token_logps=rollout_per_token_logps,
          trainer_per_token_logps=trainer_per_token_logps,
          completion_texts=completion_texts,
      )
    # Truncated importance-sampling (TIS) correction weights.
    # Compute per-token TIS weights from the trainer-vs-sampler log-ratio,
    # mask to assistant tokens only (we dampen offending model-emitted
    # positions, not env tokens), clamp at the configured threshold, and
    # detach. The policy loss picks these up via
    # ``train_example.sampler_is_weights``.
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

    # Extract time metrics (env_time and reward_time)
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
            metrics_to_log,
            mode=mode,
            step=expected_step,
        )

    for metric_fn in self.metric_fns:
      user_defined_metric = metric_fn(
          prompts=original_inputs["prompts"],
          completions=completion_texts,
          advantages=advantages,
          rewards=rewards,
          **{
              key: value
              for key, value in original_inputs.items()
              if key != "prompts"
          },
      )
      self.rl_cluster.buffer_metrics_async(
          user_defined_metric, mode=mode, step=expected_step
      )

    combined_batch = TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
        policy_version=policy_versions,
        sampler_is_weights=sampler_is_weights,
    )
    return [combined_batch]


GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
