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
from typing import Any, Dict, List, Mapping, Sequence, Type, TypeVar

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core  # pylint: disable=unused-import
from tunix.rl import alignment
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import common
from tunix.rl import deepswe_contract
from tunix.rl import deepswe_debug
from tunix.rl import envelope_probe
from tunix.rl import function_registry
from tunix.rl import perf_log
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


def _canonical_alignment_sampler_is_valid(
    sampler_is: str | None,
    workload_name: str,
    *,
    p57_tim_study: bool = False,
    p34_deepswe: bool = False,
    p34_disable_sampler_is: bool = False,
    p34_disable_tis: bool = False,
) -> bool:
  """Return whether sampler IS preserves the workload contract."""
  if sampler_is == "token":
    return True
  if sampler_is is not None:
    return False
  if workload_name == "gsm8k" or p57_tim_study:
    return True
  return p34_deepswe and p34_disable_sampler_is and p34_disable_tis


def _p57_tim_purity_enabled(env: Mapping[str, str]) -> bool:
  """Return whether the signed P57 training purity contract applies."""
  return (
      env.get("CANON_P57_RUN_KIND") == "train"
      and env.get("CANON_P57_TIM_ARM") in ("mismatch", "zero")
      and env.get("CANON_PROFILE_FILE")
      == "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
      and env.get("CANON_P32_WORKLOAD") == "frozenlake-dp8-tp8"
  )


def _validate_p57_tim_purity(
    *,
    sampler_is: str | None,
    use_rollout_logps: bool,
    rollout_logps_present: bool,
    old_logps_are_rollout: bool,
    sampler_is_weights_present: bool,
) -> None:
  """Fail closed if a TIM-aware mitigation can affect a P57 update."""
  failures = []
  if sampler_is is not None:
    failures.append(f"sampler_is={sampler_is!r}")
  if not use_rollout_logps:
    failures.append("use_rollout_logps=0")
  if not rollout_logps_present:
    failures.append("rollout_logps=absent")
  if not old_logps_are_rollout:
    failures.append("old_logps!=rollout")
  if sampler_is_weights_present:
    failures.append("tis_weights=present")
  if failures:
    raise alignment.AlignmentGateError(
        "P57 TIM purity contract failed: " + ", ".join(failures)
    )


def _canonical_gsm8k_gate_advantages(advantages: Any) -> np.ndarray:
  """Return the registered nonzero cotangent for the two-rollout gate."""
  original = np.asarray(advantages, dtype=np.float32)
  if original.shape != (2,) or not np.array_equal(
      original, np.zeros(2, dtype=np.float32)
  ):
    raise alignment.AlignmentGateError(
        "GSM8K gradient probe expected deterministic degenerate advantages, "
        f"got {original.tolist()}"
    )
  return np.asarray([-1.0, 1.0], dtype=np.float32)


def _canonical_frozenlake_c0_advantages(advantages: Any) -> np.ndarray:
  """Return the registered nonzero cotangent for the C0 GRPO pair."""
  original = np.asarray(advantages, dtype=np.float32)
  if original.shape != (2,) or not np.isfinite(original).all():
    raise alignment.AlignmentGateError(
        "FrozenLake C0 gradient probe requires two finite advantages, "
        f"got shape={original.shape} values={original.tolist()}"
    )
  return np.asarray([-1.0, 1.0], dtype=np.float32)


def _canonical_frozenlake_release_advantages(advantages: Any) -> np.ndarray:
  """Return four fixed zero-mean cotangents for the P27 4x2 batch."""
  original = np.asarray(advantages, dtype=np.float32)
  if original.shape != (8,) or not np.isfinite(original).all():
    raise alignment.AlignmentGateError(
        "FrozenLake P27 gradient probe requires the frozen eight finite "
        "advantages before 4x2 trajectory splitting, "
        f"got shape={original.shape} values={original.tolist()}"
    )
  return np.tile(np.asarray([-1.0, 1.0], dtype=np.float32), 4)


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
    loss_scale_factor: Optional explicit fixed denominator for
      `sequence-mean-token-scale`. When set, it must equal the compiled
      response width.
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
      advantages. Deprecated. Will remove in the next release.
  """

  algo_variant: str = "agentic_grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  loss_scale_factor: int | None = None
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
  # Deprecated. Will remove in the next release.
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
    if self.loss_scale_factor is not None:
      if self.loss_agg_mode != "sequence-mean-token-scale":
        raise ValueError(
            "loss_scale_factor requires sequence-mean-token-scale"
        )
      if self.loss_scale_factor <= 0:
        raise ValueError("loss_scale_factor must be positive")
      if self.loss_scale_factor != self.max_response_length:
        raise ValueError(
            "loss_scale_factor must equal max_response_length: "
            f"{self.loss_scale_factor} != {self.max_response_length}"
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
    self._p57_tim_purity_announced = False
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

    self.algo_config.temperature = self.rl_cluster.get_rollout_config(  # pyrefly: ignore[missing-attribute]
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
        compute_logps_chunk_size=self.rl_cluster.cluster_config.training_config.compute_logps_chunk_size,
    )

    self.rl_cluster.actor_trainer.with_loss_fn(
        loss_fn,
        has_aux=True,
    )
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {  # pyrefly: ignore[bad-argument-type]
            "train_example": x,
            "algo_config": self.algo_config,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({
        "kl": common.mean_of_means,
        "entropy": common.mean_of_means,
        "reduced_pg_loss": common.mean_of_means,
        "unreduced_pg_loss": common.global_weighted_mean,
        "pg_clipfrac": common.mean_of_means,
        "ppo_kl": common.mean_of_means,
        "kl_loss": common.mean_of_means,
        "is_ratio/mean": common.mean_of_means,
        "is_ratio/max": np.max,
        "is_ratio/min": np.min,
        "log_ratio/abs_mean": common.mean_of_means,
        "pg_loss/unclipped_mean": common.mean_of_means,
        "pg_loss/clipped_mean": common.mean_of_means,
        "advantage/abs_mean": common.mean_of_means,
        "advantage/max": np.max,
        "advantage/min": np.min,
        "advantage/nonzero_frac": common.mean_of_means,
        "sampler_is/weight_mean": common.mean_of_means,
        "sampler_is/weight_min": np.min,
    })
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([  # pyrefly: ignore[bad-argument-type]
        lambda: "kl"
        if self.algo_config.force_compute_kl or self.algo_config.beta != 0.0
        else None,
    ])

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
    prompt_lengths_list: List[int | None] = []
    completion_tokens_list: List[np.ndarray] = []
    completion_masks_list: List[np.ndarray] = []
    old_logprobs_list: List[np.ndarray] = []
    policy_versions_list: List[int] = []
    trajectory_rewards_list: List[float] = []
    raw_completion_lengths: List[int] = []
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
      prompt_lengths_list.append(item.traj.get("prompt_length"))
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
    padded_prompt_masks = []
    padded_completion_ids = []
    padded_completion_masks = []
    padded_completion_valid_masks = []
    padded_old_logprobs = []

    max_response_length = self.algo_config.max_response_length
    clipped_completion_count = 0
    for (
        prompt_tokens,
        prompt_length,
        completion_tokens,
        completion_mask,
        old_logprobs,
    ) in zip(
        prompt_tokens_list,
        prompt_lengths_list,
        completion_tokens_list,
        completion_masks_list,
        old_logprobs_list,
    ):
      raw_completion_length = min(len(completion_tokens), max_response_length)
      raw_completion_lengths.append(raw_completion_length)
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
      if prompt_length is None:
        # Legacy rollout engines do not expose exact pre-padding length.
        prompt_valid = np.asarray(prompt_tokens) != pad_value
      else:
        prompt_length = int(prompt_length)
        if prompt_length < 0 or prompt_length > len(prompt_tokens):
          raise ValueError(
              f"invalid prompt length {prompt_length} for width {len(prompt_tokens)}"
          )
        prompt_valid = agentic_utils.left_pad(
            np.ones(prompt_length, dtype=np.int32),
            len(prompt_tokens),
            0,
        )
      padded_prompt_masks.append(
          agentic_utils.left_pad(
              prompt_valid,
              rollout_config.max_prompt_length,
              0,
          )
      )
      padded_completion_ids.append(padded_completion[:max_response_length])
      padded_completion_masks.append(
          agentic_utils.right_pad(completion_mask, max_response_length, 0)[
              :max_response_length
          ]
      )
      # The assistant mask above is a loss mask, not a sequence-validity mask.
      # Multi-turn environment tokens and parser-appended delimiters have loss
      # mask 0 but remain causal context for every later assistant action.
      padded_completion_valid_masks.append(
          agentic_utils.right_pad(
              np.ones(raw_completion_length, dtype=np.bool_),
              max_response_length,
              False,
          )[:max_response_length]
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
    prompt_mask = jnp.asarray(padded_prompt_masks, dtype=jnp.bool_)
    completion_ids = jnp.asarray(padded_completion_ids)
    completion_mask = jnp.asarray(padded_completion_masks)
    completion_valid_mask = jnp.asarray(
        padded_completion_valid_masks, dtype=jnp.bool_
    )
    if bool(jnp.any(completion_mask.astype(jnp.bool_) & ~completion_valid_mask)):
      raise ValueError("assistant completion mask is not a subset of valid tokens")
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
    configured_compute_logps = (
        self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size
    )
    compute_logps_micro_batch_size = (
        configured_compute_logps * self.algo_config.num_generations
        if configured_compute_logps
        else len(trajectories)
    )
    if deepswe_debug.enabled():
      marker = deepswe_debug.marker_prefix()
      print(
          f"[{marker}.LOGPS_BATCH] "
          f"configured_prompts={configured_compute_logps} "
          f"generations={self.algo_config.num_generations} "
          f"execution_trajectories={compute_logps_micro_batch_size} "
          f"observed_trajectories={len(trajectories)}",
          flush=True,
      )
    rollout_per_token_logps = None
    trainer_per_token_logps = None
    if self.algo_config.use_rollout_logps and padded_old_logprobs:
      rollout_per_token_logps = jnp.asarray(padded_old_logprobs)
      old_per_token_logps = rollout_per_token_logps
      # The diagnostic pass (and the sampler-IS ``token`` path, which needs the
      # trainer's recomputed logp as ``old_per_token_logps``) requires a real
      # actor mesh; skip when not available.
      need_trainer_logps = (
          (have_actor_mesh and not deepswe_debug.rollout_only())
          or self.algo_config.sampler_is == "token"
      )
      if need_trainer_logps:
        trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=compute_logps_micro_batch_size,
            prompt_mask=prompt_mask,
            completion_mask=completion_valid_mask,
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
          micro_batch_size=compute_logps_micro_batch_size,
          prompt_mask=prompt_mask,
          completion_mask=completion_valid_mask,
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
            micro_batch_size=compute_logps_micro_batch_size,
            prompt_mask=prompt_mask,
            completion_mask=completion_valid_mask,
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
        step=expected_step,  # pyrefly: ignore[bad-argument-type]
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

    # The bounded GSM8K numerical gate must exercise a real backward program.
    # Its deterministic two rollouts currently both receive reward zero, so
    # ordinary GRPO produces [0, 0] advantages and XLA can DCE the entire
    # gradient path.  In gate-only mode, inject a fixed zero-mean cotangent to
    # test the exact same rollout tokens/logprobs without mutating the model.
    # The separately authorized update canary reuses the same cotangent solely
    # to prove optimizer/update plumbing; it is not a GSM8K learning claim.
    gsm8k_grad_probe = os.environ.get("CANON_GSM8K_GRAD_PROBE", "") == "1"
    frozenlake_c0_grad_probe = (
        os.environ.get("CANON_FROZENLAKE_GRAD_PROBE", "") == "1"
    )
    frozenlake_release_grad_probe = (
        os.environ.get("CANON_FROZENLAKE_RELEASE_GRAD_PROBE", "") == "1"
    )
    frozenlake_c0 = os.environ.get("CANON_FROZENLAKE_C0", "") == "1"
    if frozenlake_c0_grad_probe != frozenlake_c0:
      raise alignment.AlignmentGateError(
          "CANON_FROZENLAKE_C0 and CANON_FROZENLAKE_GRAD_PROBE must be "
          "enabled or disabled together"
      )
    if sum((
        gsm8k_grad_probe,
        frozenlake_c0_grad_probe,
        frozenlake_release_grad_probe,
    )) > 1:
      raise alignment.AlignmentGateError(
          "diagnostic cotangent modes are mutually exclusive"
      )
    if gsm8k_grad_probe:
      if not alignment.enabled():
        raise alignment.AlignmentGateError(
            "CANON_GSM8K_GRAD_PROBE requires the alignment gate"
        )
      mode = alignment.execution_mode()
      original_advantages = np.asarray(advantages, dtype=np.float32)
      advantages = _canonical_gsm8k_gate_advantages(original_advantages)
      if mode == "gate-only":
        # Keep the frozen release-gate marker stable.
        logging.info(
            "[CANON_GSM8K_L3] diagnostic_advantages original=%s "
            "injected=%s gate_only=1",
            original_advantages.tolist(),
            advantages.tolist(),
        )
      else:
        logging.info(
            "[CANON_GSM8K_UPDATE] diagnostic_advantages original=%s "
            "injected=%s mode=update-canary",
            original_advantages.tolist(),
            advantages.tolist(),
        )
    elif frozenlake_c0_grad_probe:
      if not alignment.enabled() or alignment.execution_mode() != "gate-only":
        raise alignment.AlignmentGateError(
            "CANON_FROZENLAKE_GRAD_PROBE requires alignment gate-only mode"
        )
      original_advantages = np.asarray(advantages, dtype=np.float32)
      advantages = _canonical_frozenlake_c0_advantages(original_advantages)
      logging.info(
          "[CANON_FROZENLAKE_C0] diagnostic_advantages original=%s "
          "injected=%s gate_only=1",
          original_advantages.tolist(),
          advantages.tolist(),
      )
    elif frozenlake_release_grad_probe:
      if (
          os.environ.get("CANON_FROZENLAKE_P27", "") != "1"
          or not alignment.enabled()
          or alignment.execution_mode() != "gate-only"
      ):
        raise alignment.AlignmentGateError(
            "CANON_FROZENLAKE_RELEASE_GRAD_PROBE requires P27 gate-only mode"
        )
      original_advantages = np.asarray(advantages, dtype=np.float32)
      advantages = _canonical_frozenlake_release_advantages(
          original_advantages
      )
      logging.info(
          "[CANON_FROZENLAKE_P27] diagnostic_advantages original=%s "
          "injected=%s gate_only=1",
          original_advantages.tolist(),
          advantages.tolist(),
      )

    logging.debug("Advantages computed: %s", advantages)

    if deepswe_debug.enabled() and mode == rl_cluster_lib.Mode.TRAIN:
      if expected_step is None:
        raise ValueError("DeepSWE debug artifacts require an expected step")
      p58_artifacts = False
      if deepswe_debug.onehost():
        artifact_model_id = "Qwen/Qwen3-4B-Instruct-2507"
      else:
        workload = deepswe_contract.active_workload(os.environ)
        if workload.contract_name not in (
            "p34-production",
            "p43-64chip-debug",
            "p44-qwen4b-parity-64",
            "p44-qwen4b-parity-128",
            "p46-qwen32b-train-64",
            "p46-qwen32b-train-256",
            "p58-qwen4b-tim-128",
        ):
          raise ValueError(
              "DeepSWE artifacts require P34 production, P43, or P44"
          )
        artifact_model_id = workload.model_id
        p58_artifacts = workload.contract_name == "p58-qwen4b-tim-128"
      artifact_step = int(expected_step)
      optimizer_step = None
      if p58_artifacts:
        if not hasattr(self, "_p58_debug_batch_index"):
          self._p58_debug_batch_index = deepswe_debug.next_batch_index(
              deepswe_debug.artifact_directory()
          )
        artifact_step = self._p58_debug_batch_index
        optimizer_step = int(expected_step)
      debug_metrics = deepswe_debug.persist_batch(
          trajectories,
          rewards,
          advantages,
          expected_step=artifact_step,
          optimizer_step=optimizer_step,
          output_dir=deepswe_debug.artifact_directory(),
          model_id=artifact_model_id,
      )
      if p58_artifacts:
        self._p58_debug_batch_index += 1
      trajectory_count = debug_metrics["trajectories"]
      prompt_group_count = debug_metrics["prompt_groups"]
      self.rl_cluster.buffer_metrics_async(
          {
              "deepswe/trajectory_solve_ratio": (
                  debug_metrics["trajectory_solve_ratio"], np.mean
              ),
              "deepswe/solved_trajectories": (
                  debug_metrics["solved_trajectories"], np.mean
              ),
              "deepswe/complete_trajectory_ratio": (
                  debug_metrics["complete_trajectories"] / trajectory_count,
                  np.mean,
              ),
              "deepswe/all_solved_prompt_ratio": (
                  debug_metrics["all_solved_prompt_groups"]
                  / prompt_group_count,
                  np.mean,
              ),
              "deepswe/all_solved_prompt_groups": (
                  debug_metrics["all_solved_prompt_groups"], np.mean
              ),
              "deepswe/all_failed_prompt_ratio": (
                  debug_metrics["all_failed_prompt_groups"]
                  / prompt_group_count,
                  np.mean,
              ),
              "deepswe/all_failed_prompt_groups": (
                  debug_metrics["all_failed_prompt_groups"], np.mean
              ),
              "deepswe/mixed_prompt_ratio": (
                  debug_metrics["mixed_prompt_groups"] / prompt_group_count,
                  np.mean,
              ),
              "deepswe/mixed_prompt_groups": (
                  debug_metrics["mixed_prompt_groups"], np.mean
              ),
              "deepswe/incomplete_prompt_ratio": (
                  debug_metrics["incomplete_prompt_groups"]
                  / prompt_group_count,
                  np.mean,
              ),
              "deepswe/incomplete_prompt_groups": (
                  debug_metrics["incomplete_prompt_groups"], np.mean
              ),
              "deepswe/effective_prompt_ratio": (
                  debug_metrics["effective_prompt_groups"]
                  / prompt_group_count,
                  np.mean,
              ),
              "deepswe/effective_prompt_groups": (
                  debug_metrics["effective_prompt_groups"], np.mean
              ),
              "deepswe/nonzero_advantage_ratio": (
                  debug_metrics["nonzero_advantage_ratio"], np.mean
              ),
              "deepswe/raw_nonzero_advantage_ratio": (
                  debug_metrics["raw_nonzero_advantage_ratio"], np.mean
              ),
              "deepswe/compact_filtered_trajectory_ratio": (
                  debug_metrics["compact_filtered_trajectory_ratio"],
                  np.mean,
              ),
              "deepswe/compact_filtered_prompt_ratio": (
                  debug_metrics["compact_filtered_prompt_groups"]
                  / prompt_group_count,
                  np.mean,
              ),
              "deepswe/compact_filtered_trajectories": (
                  debug_metrics["compact_filtered_trajectories"], np.mean
              ),
              "deepswe/compact_filtered_prompt_groups": (
                  debug_metrics["compact_filtered_prompt_groups"], np.mean
              ),
              **{
                  key: (value, np.mean)
                  for key, value in deepswe_debug.timeout_wandb_metrics(
                      debug_metrics
                  ).items()
              },
          },
          mode=mode,
          step=expected_step,
      )

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
            metrics_to_log,  # pyrefly: ignore[bad-argument-type]
            mode=mode,
            step=expected_step,  # pyrefly: ignore[bad-argument-type]
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
          user_defined_metric, mode=mode, step=expected_step  # pyrefly: ignore[bad-argument-type]
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
        completion_valid_mask=completion_valid_mask,
    )
    if _p57_tim_purity_enabled(os.environ):
      _validate_p57_tim_purity(
          sampler_is=self.algo_config.sampler_is,
          use_rollout_logps=self.algo_config.use_rollout_logps,
          rollout_logps_present=rollout_per_token_logps is not None,
          old_logps_are_rollout=(
              old_per_token_logps is rollout_per_token_logps
          ),
          sampler_is_weights_present=sampler_is_weights is not None,
      )
      if not self._p57_tim_purity_announced:
        print(
            "[P57.TIM_PURITY] PASS sampler_is=none old_logps=rollout "
            "tis_weights=absent trainer_rescore=observer-only",
            flush=True,
        )
        self._p57_tim_purity_announced = True
    if (
        deepswe_debug.enabled()
        and deepswe_debug.rollout_only()
    ):
      return [combined_batch]
    if alignment.enabled():
      if not self.algo_config.use_rollout_logps:
        raise alignment.AlignmentGateError(
            "FrozenLake alignment requires use_rollout_logps=True"
        )
      if not _canonical_alignment_sampler_is_valid(
          self.algo_config.sampler_is,
          os.environ.get("CANON_P32_WORKLOAD", ""),
          p57_tim_study=_p57_tim_purity_enabled(os.environ),
          p34_deepswe=(
              os.environ.get("CANON_P34_DEEPSWE", "") == "1"
          ),
          p34_disable_sampler_is=(
              os.environ.get("CANON_P34_DISABLE_SAMPLER_IS", "") == "1"
          ),
          p34_disable_tis=(
              os.environ.get("CANON_P34_DISABLE_TIS", "") == "1"
          ),
      ):
        raise alignment.AlignmentGateError(
            "canonical alignment requires sampler_is='token'; sampler_is=None "
            "is admitted only by the signed GSM8K, P34 DeepSWE, or P57 "
            "causal-study contract"
        )
      if rollout_per_token_logps is None or trainer_per_token_logps is None:
        raise alignment.AlignmentGateError(
            "alignment batch is missing S_decode or trainer T_old"
        )
      if os.environ.get("CANON_P34_DEEPSWE", "") == "1":
        try:
          deepswe_contract.persist_weight_attestation(
              self.rl_cluster.attest_actor_anchor_matches_engine(),
              step=int(expected_step),
              report_path=os.environ.get("CANON_P34_WEIGHT_REPORT", ""),
          )
        except (RuntimeError, ValueError) as exc:
          raise alignment.AlignmentGateError(
              "P34 requires exact rollout/trainer weights before A/B/C"
          ) from exc
      rescore_source = self.rl_cluster.rollout.get_prefill_rescore_logps

      def _perf_sink(stage, seconds):
        self.rl_cluster.buffer_metrics_async(
            {f"perf/{stage}_seconds": (seconds, np.mean)},
            mode=rl_cluster_lib.Mode.TRAIN,
            step=int(self.rl_cluster.global_steps),
        )

      with perf_log.phase(
          "rescore_b",
          step=int(self.rl_cluster.global_steps),
          sink=_perf_sink,
      ) as perf_info:
        perf_info["rows"] = int(completion_ids.shape[0])
        if envelope_probe.enabled():
          s_prefill = self.rl_cluster.get_prefill_rescore_logps(
              prompt_ids,
              completion_ids,
              completion_lengths=np.asarray(
                  raw_completion_lengths, dtype=np.int32
              ),
              diagnostic_arm="A",
          )
        else:
          s_prefill = self.rl_cluster.get_prefill_rescore_logps(
              prompt_ids,
              completion_ids,
              completion_lengths=np.asarray(
                  raw_completion_lengths, dtype=np.int32
              ),
          )
      rollout_config = self.rl_cluster.get_rollout_config(
          mode=rl_cluster_lib.Mode.TRAIN
      )
      combined_batch = alignment.wrap_train_example(
          combined_batch,
          s_decode=rollout_per_token_logps,
          s_prefill=s_prefill,
          t_old=trainer_per_token_logps,
          action_mask=completion_mask,
          completion_valid_mask=completion_valid_mask,
          prompt_mask=prompt_mask,
          tokens=completion_ids,
          policy_version=policy_versions,
          temperature=rollout_config.temperature,
          top_k=rollout_config.top_k,
          top_p=rollout_config.top_p,
          s_prefill_source=rescore_source,
      )
      logging.info(
          "[CANON_ALIGN] attached host sidecar rows=%d completion_width=%d",
          completion_ids.shape[0],
          completion_ids.shape[1],
      )
      if envelope_probe.enabled():
        try:
          data_size = int(os.environ.get(envelope_probe.DATA_SIZE_ENV, "0"))
          local_m = int(os.environ.get(envelope_probe.LOCAL_M_ENV, "0"))
        except ValueError as exc:
          raise envelope_probe.EnvelopeProbeError(
              "P35 DP size and local M must be integers"
          ) from exc
        report_path = os.environ.get(envelope_probe.REPORT_ENV, "")
        metadata_dir = os.environ.get(envelope_probe.METADATA_DIR_ENV, "")
        if not report_path or not metadata_dir:
          raise envelope_probe.EnvelopeProbeError(
              "P35 requires explicit report and metadata paths"
          )

        a_full = np.asarray(s_prefill)
        c_full = np.asarray(trainer_per_token_logps)
        action_full = np.asarray(completion_mask, dtype=np.bool_)
        rows, first_ac = envelope_probe.select_reproducing_group(
            a_full,
            c_full,
            action_full,
            data_size=data_size,
        )
        selected_prompts = np.asarray(prompt_ids)[rows]
        selected_completions = np.asarray(completion_ids)[rows]
        selected_prompt_mask = np.asarray(prompt_mask, dtype=np.bool_)[rows]
        selected_valid_mask = np.asarray(
            completion_valid_mask, dtype=np.bool_
        )[rows]
        selected_action_mask = action_full[rows]
        selected_lengths = np.asarray(
            raw_completion_lengths, dtype=np.int32
        )[rows]
        sequences = envelope_probe.compact_sequences(
            selected_prompts,
            selected_completions,
            selected_prompt_mask,
            selected_valid_mask,
        )

        b_selected = self.rl_cluster.get_grouped_prefill_rescore_logps(
            selected_prompts,
            selected_completions,
            completion_lengths=selected_lengths,
            group_size=data_size,
            source_row_indices=rows,
            diagnostic_arm="B",
        )
        b_contract = (
            self.rl_cluster.canonical_p35_grouped_prefill_contract()
        )
        weight_attestation = (
            self.rl_cluster.attest_actor_anchor_matches_engine()
        )
        adapter_contract = self.rl_cluster.canonical_p35_adapter_contract()

        metadata_error = None
        try:
          metadata_attestations, metadata_summary = (
              envelope_probe.attest_metadata(
                  directory=metadata_dir,
                  expected_b_sequences=sequences,
                  expected_a_rows=int(a_full.shape[0]),
                  data_size=data_size,
                  local_m=local_m,
              )
          )
        except envelope_probe.EnvelopeProbeError as exc:
          metadata_error = str(exc)
          metadata_attestations = {
              "native_A_observed": False,
              "grouped_B_observed": False,
              "mesh_shape_expected": False,
              "device_order_expected": False,
              "local_m256_B": False,
              "positions_equal": False,
              "block_tables_B_observed": False,
              "request_distribution_B_one_per_rank": False,
              "metadata_B_matches_C": False,
              "cache_fresh_B": False,
          }
          metadata_summary = {"error": metadata_error}

        captured_mesh_ids = tuple(
            metadata_summary.get("mesh", {}).get("device_ids", ())
            if isinstance(metadata_summary.get("mesh"), dict)
            else ()
        )
        adapter_mesh_ids = tuple(adapter_contract.get("mesh_device_ids", ()))
        weight_mesh_ids = tuple(weight_attestation.get("mesh_device_ids", ()))
        metadata_attestations["device_order_expected"] = bool(
            captured_mesh_ids
            and captured_mesh_ids == adapter_mesh_ids == weight_mesh_ids
        )
        selected_policy = np.asarray(policy_versions)[rows]
        b_groups = tuple(b_contract.get("group_provenance", ()))
        attestations = {
            **metadata_attestations,
            "weights_equal": bool(weight_attestation.get("equal")),
            "policy_version_equal": bool(
                selected_policy.size
                and np.all(selected_policy == selected_policy.reshape(-1)[0])
            ),
            "selected_token_ids_equal": bool(
                len(sequences) == data_size
                and metadata_attestations.get("metadata_B_matches_C") is True
            ),
            "action_masks_equal": bool(
                selected_action_mask.shape == np.asarray(b_selected).shape
                and np.all(~selected_action_mask | selected_valid_mask)
            ),
            "validity_masks_equal": bool(
                np.array_equal(
                    selected_lengths,
                    selected_valid_mask.sum(axis=1, dtype=np.int32),
                )
            ),
            "rank_strided_group": bool(
                np.array_equal(
                    rows,
                    envelope_probe.rank_strided_row_groups(
                        a_full.shape[0], data_size
                    )[first_ac[0] % (a_full.shape[0] // data_size)],
                )
                and adapter_contract.get("rank_strided_groups") is True
            ),
            "local_m256_C": bool(
                adapter_contract.get("local_m") == 256 and local_m == 256
            ),
            "block_tables_C_canonical": bool(
                adapter_contract.get("block_tables_rank_local_contiguous")
                is True
            ),
            "prefix_cache_reset_B": bool(
                len(b_groups) == 1
                and b_groups[0].get("reset_prefix_cache") is True
            ),
            "cache_fresh_C": bool(
                adapter_contract.get("fresh_cache_per_group") is True
            ),
        }
        exact_replay_enabled = (
            os.environ.get(envelope_probe.EXACT_REPLAY_ENV, "") == "1"
        )
        exact_replay_path = None
        if exact_replay_enabled:
          exact_replay_path = os.environ.get(
              envelope_probe.EXACT_REPLAY_REPORT_ENV, ""
          )
          if not exact_replay_path:
            raise envelope_probe.EnvelopeProbeError(
                "P35.3 requires CANON_P35_EXACT_REPLAY_REPORT"
            )

        def build_base_report():
          return envelope_probe.build_report(
              a=a_full[rows],
              b=np.asarray(b_selected),
              c=c_full[rows],
              action_mask=selected_action_mask,
              selected_row_indices=rows,
              first_full_ac_mismatch=first_ac,
              attestations=attestations,
              metadata={
                  "serving": metadata_summary,
                  "adapter": adapter_contract,
                  "grouped_serving": b_contract,
                  "weights": weight_attestation,
                  "metadata_error": metadata_error,
                  "exact_replay_report": exact_replay_path,
              },
          )

        if exact_replay_enabled:
          preliminary_path = os.environ.get(
              envelope_probe.PRE_REPLAY_REPORT_ENV, ""
          ) or envelope_probe.pre_replay_report_path(report_path)
          evidence_paths = {
              os.path.abspath(os.fspath(path))
              for path in (
                  report_path,
                  preliminary_path,
                  exact_replay_path,
              )
          }
          if len(evidence_paths) != 3:
            raise envelope_probe.EnvelopeProbeError(
                "P35.3 base, preliminary, and replay reports must use "
                "three distinct paths"
            )
          preliminary_output = envelope_probe.write_report(
              build_base_report(),
              preliminary_path,
          )
          print(
              "[CANON_P35] BASE_REPORT_COMPLETE "
              f"path={preliminary_output} rows={rows.tolist()} "
              "REPLAY_PENDING",
              flush=True,
          )
          b_records = envelope_probe.load_arm_metadata_records(
              metadata_dir, "B"
          )
          replay = self.rl_cluster.p35_exact_input_replay(
              b_records,
              full_prompt_tokens=prompt_ids,
              full_completion_tokens=completion_ids,
              full_prompt_mask=prompt_mask,
              full_completion_mask=completion_valid_mask,
              selected_row_indices=rows,
          )

          def comparisons_exact(comparisons):
            return bool(
                comparisons
                and all(
                    summary.get("valid") is True
                    and summary.get("exact") is True
                    for group in comparisons.values()
                    for summary in group.values()
                )
            )

          exact_attestations = {
              "weights_equal": bool(weight_attestation.get("equal")),
              "captured_B_metadata_admitted": bool(
                  metadata_attestations.get("metadata_B_matches_C") is True
              ),
              "selected_token_ids_equal": bool(
                  attestations.get("selected_token_ids_equal") is True
              ),
              "action_masks_equal": bool(
                  attestations.get("action_masks_equal") is True
              ),
              "cache_fresh_B": bool(
                  attestations.get("cache_fresh_B") is True
              ),
              "cache_fresh_replay": True,
              "local_m256": bool(
                  attestations.get("local_m256_B") is True
                  and attestations.get("local_m256_C") is True
              ),
              "device_order_expected": bool(
                  attestations.get("device_order_expected") is True
              ),
              "repeat_exact": bool(
                  comparisons_exact(replay["repeat_comparisons"])
              ),
          }
          exact_report = envelope_probe.build_exact_replay_report(
              b=np.asarray(b_selected),
              c=c_full[rows],
              r0_live=np.asarray(replay["r0_live_logps"]),
              r1_mapped=np.asarray(replay["r1_mapped_logps"]),
              r2_adapter_direct=np.asarray(
                  replay["r2_adapter_direct_logps"]
              ),
              r3_adapter_envelope=np.asarray(
                  replay["r3_adapter_envelope_logps"]
              ),
              action_mask=selected_action_mask,
              stage_comparisons=replay["stage_comparisons"],
              repeat_comparisons=replay["repeat_comparisons"],
              attestations=exact_attestations,
              metadata={
                  "replay": replay["metadata"],
                  "selected_rows": rows.tolist(),
                  "captured_B_records": len(b_records),
              },
          )
          replay_output = envelope_probe.write_report(
              exact_report, exact_replay_path
          )
          print(
              "[CANON_P35.3] REPLAY_COMPLETE "
              f"path={replay_output}",
              flush=True,
          )
        report = build_base_report()
        output = envelope_probe.write_report(report, report_path)
        print(
            "[CANON_P35] REPORT_COMPLETE "
            f"path={output} rows={rows.tolist()} "
            "STOP_BEFORE_BACKWARD",
            flush=True,
        )
        raise envelope_probe.EnvelopeProbeComplete(
            f"P35 diagnostic complete before backward: {output}"
        )
      if alignment.precheck_enabled():
        diagnostic_only = alignment.precheck_only_enabled()
        precheck_record = alignment.check_pre_backward(
            combined_batch,
            step=int(expected_step),
            fail_closed=not diagnostic_only,
        )
        if diagnostic_only:
          alignment.stop_after_diagnostic_precheck(precheck_record)
    return [combined_batch]


GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
