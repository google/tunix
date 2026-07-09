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

"""GRPO learner."""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Sequence, TypeVar

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core  # pylint: disable=unused-import
from tunix.generate import utils
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  pass


@dataclasses.dataclass(kw_only=True)
class GRPOConfig(algo_config_lib.AlgorithmConfig):
  """Configuration for GRPO algorithms.

  Attributes:
    algo_variant: The algorithm variant to use. Default: `grpo`.
    advantage_estimator: The advantage estimator to use. Default: `grpo`.
    policy_loss_fn: The policy loss function to use. Default: `grpo`.
    loss_agg_mode: The aggregation mode for the loss function. Supported values
      include `token-mean`, `sequence-mean-token-mean`,
      `sequence-mean-token-scale`, `seq-mean-token-sum`, and
      `sequence-mean-token-sum-norm`. Default: `sequence-mean-token-mean`.
    reward_manager: The reward manager to use. Default: `sequence-level`.
    loss_algo: The loss algorithm to use. To be deprecated.
    num_generations: The number of times the policy generates multiple responses
      for a given prompt within a single training step. This corresponds to 'G'
      in Algorithm 1 in the `paper <https://arxiv.org/abs/2402.03300>`_. A
      higher value means more samples are used to compute relative advantages.
    num_iterations: The number of iterations per batch (𝜇 in GRPO algo 1).
    beta: The coefficient for the KL divergence penalty (𝛽) in the GRPO loss
      function. This term prevents policy updates from deviating too far from
      the reference model. A value of 0.0 means no KL penalty is applied.
    kl_loss_mode: The divergence mode used for KL penalty estimation. Default:
      `kl`.
    epsilon: Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to
      PPO, it ensures stable updates.
    epsilon_high: Epsilon value for upper bound clipping.
    loss_algo: use GRPO or GSPO for loss computation. GRPO loss is per-batch
      normalized instead of per-response normalized as mentioned in the paper.
      For GSPO, we use gspo-token loss which is more flexible.
    use_rollout_logps: Whether to use rollout-side log probabilities as
      old-policy log probabilities. If False, recompute old-policy log
      probabilities on the trainer actor.
    sampler_is: Optional truncated importance-sampling correction between the
      rollout sampler and trainer actor. Set to "token" to use trainer
      recomputed logps as old-policy logps and multiply the policy loss by
      detached per-token sampler/trainer correction weights.
    sampler_is_threshold: Maximum per-token TIS correction weight.

  References:
    - GRPO: https://arxiv.org/abs/2402.03300
    - GSPO: https://arxiv.org/abs/2507.18071
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  reward_manager: str = "sequence-level"
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
  epsilon: float = 0.2
  use_rollout_logps: bool = True
  sampler_is: str | None = None
  sampler_is_threshold: float = 2.0

  def __post_init__(self):
    if self.num_generations <= 1:
      raise ValueError(
          "num_generations must be greater than 1. Received: "
          f"{self.num_generations}"
      )

    if self.loss_algo not in ["grpo", "gspo-token"]:
      raise ValueError(
          "loss_algo should be either grpo or gspo-token. Received: "
          f"{self.loss_algo}"
      )
    if self.sampler_is not in (None, "token"):
      raise ValueError(
          "sampler_is should be either None or 'token'. Received: "
          f"{self.sampler_is}"
      )


TGrpoConfig = TypeVar("TGrpoConfig", bound=GRPOConfig)


class GRPOLearner(rl_learner.RLLearner[TGrpoConfig]):
  """GRPO (Group Relative Policy Optimization) learner.

  GRPO is a reinforcement learning algorithm designed to enhance the reasoning
  abilities of large language models, like mathematical problem-solving. It is
  a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
  eliminating the need for a separate value function model. GRPO works by
  generating multiple responses for a given prompt, evaluating these responses
  using a reward model, and then calculating a relative advantage based on the
  group's performance to update the policy.
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TGrpoConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `GRPOTrainer`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: An instance of `AlgorithmConfig` containing all
        training-specific configuration options.
      reward_fns: A single callable or a list of callables that compute a
        scalar reward for given prompts and completions. Each function should
        accept `prompts`, `completions` and optional keyword arguments, and
        return a list of float rewards.
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
      data_shuffle_seed: The seed used to shuffle the training data.
    """  # fmt: skip
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

    self.algo_config.temperature = self.rl_cluster.get_rollout_config(  # pyrefly: ignore[missing-attribute]
        mode=rl_cluster_lib.Mode.TRAIN
    ).temperature

    policy_loss_fn = function_registry.get_policy_loss_fn(
        self.algo_config.policy_loss_fn
    )

    # Workaround for passing in importance_sampling_algo as jax transforms
    # doesn't like partial functions with kwargs.
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
        "kl": np.mean,
        "pg_clipfrac": np.mean,
        "sampler_is/weight_mean": np.mean,
        "sampler_is/weight_min": np.min,
    })
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([  # pyrefly: ignore[bad-argument-type]
        lambda: "kl" if self.algo_config.beta != 0.0 else None,
    ])

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Generate text completions and compute the advantages for GRPO training.

    Args:
      training_input: A dictionary containing the training input data,
        containing the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data, including
      prompt IDs, completion IDs, masks, advantages, and per-token log
      probabilities from the reference and policy models.
    """
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      rollout_config = rollout_config[mode]

    training_input["prompts"] = list(training_input["prompts"])  # pyrefly: ignore[bad-argument-type]
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    # TODO (noghabi): Add mini batch and micro batch tags
    perf_tags = {
        perf_constants.STEP: self.rl_cluster.global_steps,
    }

    rollout_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
        mode=mode,
        micro_batch_size=(
            self._rollout_micro_batch_size * self.algo_config.num_generations  # pyrefly: ignore[unsupported-operation]
        ),
        trace_tags=perf_tags,
    )
    padded_completion_ids = np.array([
        utils.pad_to_length(
            completion_ids,
            target_length=rollout_config.max_tokens_to_generate,
            pad_value=pad_value,
            left=False,
        )
        for completion_ids in rollout_output.tokens
    ])
    prompt_ids = jnp.array(rollout_output.left_padded_prompt_tokens)

    # Assemble masks
    prompt_mask = prompt_ids != pad_value
    completion_mask = np.not_equal(padded_completion_ids, pad_value)

    # Convert completion_ids and completion_mask to jax arrays
    jax_completion_ids = jnp.array(padded_completion_ids)
    jax_completion_mask = jnp.array(completion_mask)
    compute_logps_micro_batch_size = (
        self._compute_logps_micro_batch_size * self.algo_config.num_generations
        if self._compute_logps_micro_batch_size
        else len(training_input["prompts"])
    )

    if self.algo_config.beta != 0.0:
      devices = self.rl_cluster.r2m[rl_cluster_lib.Role.REFERENCE].devices
      # TODO(yangmu): use function decorator to trace this part, same below.
      with self.rl_cluster.perf.span(
          "refer_inference", devices
      ) as interval, self.rl_cluster.perf_v2.span(
          perf_constants.REFERENCE_INFERENCE, devices, tags=perf_tags
      ) as interval_v2:
        ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=compute_logps_micro_batch_size,
        )
        interval.device_end([ref_per_token_logps])
        interval_v2.async_end([ref_per_token_logps])
    else:
      ref_per_token_logps = None
    rollout_per_token_logps = None
    trainer_per_token_logps = None
    old_per_token_logps = None
    sampler_is_weights = None

    if self.algo_config.use_rollout_logps and rollout_output.logprobs is not None:
      rollout_per_token_logps = jnp.asarray([
          utils.pad_to_length(
              np.asarray(logprobs),
              target_length=rollout_config.max_tokens_to_generate,
              pad_value=0.0,
              left=False,
          )[: rollout_config.max_tokens_to_generate]
          for logprobs in rollout_output.logprobs
      ])
      old_per_token_logps = rollout_per_token_logps

    needs_rollout_logps = (
        self.algo_config.use_rollout_logps
        and rollout_per_token_logps is None
        and (
            self.algo_config.num_iterations > 1
            or self.algo_config.sampler_is == "token"
        )
    )
    if needs_rollout_logps:
      devices = self.rl_cluster.r2m[rl_cluster_lib.Role.ROLLOUT].devices
      with self.rl_cluster.perf.span(
          "old_actor_inference", devices
      ) as interval, self.rl_cluster.perf_v2.span(
          perf_constants.OLD_ACTOR_INFERENCE, devices, tags=perf_tags
      ) as interval_v2:
        rollout_per_token_logps = self.rl_cluster.get_old_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            micro_batch_size=compute_logps_micro_batch_size,
        )
        interval.device_end([rollout_per_token_logps])
        interval_v2.async_end([rollout_per_token_logps])
      old_per_token_logps = rollout_per_token_logps

    actor_mesh = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR]
    have_actor_mesh = actor_mesh is not None and not actor_mesh.empty
    needs_trainer_logps = (
        not self.algo_config.use_rollout_logps
        or self.algo_config.sampler_is == "token"
        or (rollout_per_token_logps is not None and have_actor_mesh)
    )
    if needs_trainer_logps:
      devices = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR].devices
      with self.rl_cluster.perf.span(
          "old_actor_inference", devices
      ) as interval, self.rl_cluster.perf_v2.span(
          perf_constants.OLD_ACTOR_INFERENCE, devices, tags=perf_tags
      ) as interval_v2:
        trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=compute_logps_micro_batch_size,
        )
        interval.device_end([trainer_per_token_logps])
        interval_v2.async_end([trainer_per_token_logps])

    if not self.algo_config.use_rollout_logps:
      old_per_token_logps = trainer_per_token_logps
    elif (
        self.algo_config.sampler_is == "token"
        and trainer_per_token_logps is not None
    ):
      old_per_token_logps = trainer_per_token_logps

    if self.algo_config.num_iterations > 1 and old_per_token_logps is None:
      raise RuntimeError(
          "old_per_token_logps is not available for off-policy RL. Enable "
          "`return_logprobs` in RolloutConfig, keep rollout logp computation "
          "available, or set use_rollout_logps=False to recompute on the "
          "trainer actor."
      )

    with self.rl_cluster.perf.span(
        "advantage_computation"
    ), self.rl_cluster.perf_v2.span(
        perf_constants.ADVANTAGE_COMPUTATION, tags=perf_tags
    ):
      # Compute rewards and advantages
      rewards = self._compute_rewards(
          prompts=training_input["prompts"],
          completions=rollout_output.text,
          mode=mode,
          **{k: v for k, v in training_input.items() if k != "prompts"},  # pyrefly: ignore[bad-argument-type]
      )
      advantage_estimator = function_registry.get_advantage_estimator(
          self.algo_config.advantage_estimator
      )
      advantages = advantage_estimator(
          rewards=rewards, num_generations=self.algo_config.num_generations
      )

    # Log raw scores from the reward model fn
    self.rl_cluster.buffer_metrics(
        {
            "rewards/score/mean": (np.mean(rewards), np.mean),
            "rewards/score/max": (np.max(rewards), np.max),
            "rewards/score/min": (np.min(rewards), np.min),
        },
        mode=mode,
    )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )

    if (
        rollout_per_token_logps is not None
        and trainer_per_token_logps is not None
    ):
      mask = jax_completion_mask.astype(jnp.bool_)
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
      self.rl_cluster.buffer_metrics(
          {
              "sampler_trainer/logp_diff_mean": (diff_mean, np.mean),
              "sampler_trainer/logp_diff_max": (diff_max, np.max),
              "sampler_trainer/prob_diff_mean": (prob_diff_mean, np.mean),
              "sampler_trainer/prob_diff_max": (prob_diff_max, np.max),
              "sampler_trainer/probs_pearson_corr": (pearson, np.mean),
          },
          mode=mode,
      )

    if (
        self.algo_config.sampler_is == "token"
        and rollout_per_token_logps is not None
        and trainer_per_token_logps is not None
    ):
      asst_mask_f = jax_completion_mask.astype(jnp.float32)
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
      self.rl_cluster.buffer_metrics(
          {
              "sampler_is/weight_mean": (is_mean, np.mean),
              "sampler_is/weight_max": (is_max, np.max),
              "sampler_is/frac_clipped_at_threshold": (
                  frac_clipped,
                  np.mean,
              ),
          },
          mode=mode,
      )

    # log user defined metrics
    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=training_input["prompts"],
          completions=rollout_output.text,
          advantages=advantages,
          rewards=rewards,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=jax_completion_ids,
        completion_mask=jax_completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=jax.device_put(advantages),
        old_per_token_logps=old_per_token_logps,
        sampler_is_weights=sampler_is_weights,
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is a string of format {row_offset}_{group_offset} where
    row_offset is the row index of the example data source and
    group_offset is the group index of the example in the generation group.

    Args:
      example: The training input data.
      steps: The number of steps taken so far.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
    batch_size = len(example["prompts"]) // self.algo_config.num_generations  # pyrefly: ignore[bad-argument-type]
    row_offset = steps * batch_size
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.algo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.algo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _num_iterations(self) -> int:
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    return self.algo_config.num_generations

  def train(  # pylint: disable=useless-parent-delegation
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """GRPO training loop.

    Algorithm as below: extract from https://arxiv.org/abs/2402.03300 ::

        Input:
            initial policy model πθinit;
            reward models rφ;
            task prompts D;
            hyperparameters ε, β, μ

        policy model πθ ← πθinit
        for iteration = 1, ..., I do
          reference model πref ← πθ
          for step = 1, ..., M do
            Sample a batch D♭ from D
            Update the old policy model πθold ← πθ
            Sample G outputs {oi}G_i=1 ~ πθold(· | q) for each question q ∈ D♭
            Compute rewards {ri}G_i=1 for each sampled output oi by running rφ
            Compute Âi,t for the t-th token of oi through group relative
            advantage estimation.
            for GRPO iteration = 1, ..., μ do
              Update the policy model πθ by maximizing the GRPO objective
              (Equation 21)
          Update rφ through continuous training using a replay mechanism.
        Output πθ

    .. note::

        1. The outer loop (I) is ignored for now because we never update the
           reference model for now.

        2. Currently sample and train hold the same referece to the model. So
           we also omit the step to update the sampler model.

    Args:
      train_ds: An iterable of training input data, where each element is a
        dictionary containing the key 'prompts'.
      eval_ds: An iterable of evaluation input data, where each element is a
        dictionary containing the key 'prompts'.
      skip_jit: Whether to skip JIT compilation of the training loop.
    """
    super().train(train_ds, eval_ds, skip_jit)


GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
