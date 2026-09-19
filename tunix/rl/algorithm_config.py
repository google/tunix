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

from collections.abc import Sequence
import dataclasses
from absl import logging
from tunix.rl import function_registry


@dataclasses.dataclass(slots=True, kw_only=True)
class AlgorithmConfig:
  """Configuration for RL algorithms.

  Parameters:
    algo_variant: The core algorithm variant to use.
    advantage_estimator: The advantage estimator to use.
    policy_loss_fn: The policy loss function to use.
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  reward_manager: str = "sequence-level"
  # Optional symmetric clamp applied to per-token KL inside
  # `common.compute_kl_divergence`. `None` (default) disables the clamp and
  # preserves prior behavior bit-for-bit. Set to a positive float (e.g.
  # `10000.0`) to bound rare outliers — useful when the trained policy
  # briefly drifts far from the reference and the `low_var_kl` estimator's
  # `exp(diff)` term saturates bf16 / overflows fp32 and poisons the loss
  # for the rest of the step.
  kl_clamp_value: float | None = None
  # Number of worker processes `SequenceRewardManager` uses to evaluate
  # reward functions over the batch. `0` (default) keeps the serial
  # implementation and preserves prior behavior bit-for-bit. Values > 1
  # chunk the (prompts, completions) batch across a process pool — reward
  # functions are called with contiguous slices, so per-sequence reward
  # functions produce identical results, while functions that depend on the
  # batch as a whole (e.g. on a sequence's position in it) see the chunk
  # instead of the full batch, which is why the pool is opt-in. Functions
  # that cannot run in a worker (unpicklable, or spawning subprocesses of
  # their own) are detected at runtime and evaluated in the parent process
  # instead; any other failure falls back to the serial path. `-1` uses one
  # worker per CPU.
  reward_num_workers: int = 0
  # How long, in seconds, to wait for one reward-fn chunk in a worker before
  # treating the pool as unusable for that fn and evaluating it in the parent
  # process instead.
  reward_worker_timeout_seconds: float = 180.0
  # Sampling temperature used during rollout generation to compute log
  # probabilities and entropy in the loss function.
  # NB: This should not be configured manually, instead it will be set by the RL
  # engine based on the rollout config.
  temperature: float | None = None
  # Whether to use rollout-side log probabilities as old-policy log
  # probabilities. If False, recompute old-policy log probabilities on the
  # trainer actor.
  use_rollout_logps: bool = True

  def __post_init__(self):
    valid_algo_variants = [
        "grpo",
        "drgrpo",
        "gspo-token",
        "ppo",
        "dapo",
    ]
    valid_advantage_estimators = ["grpo", "gae", "drgrpo", "rloo", "grpo-loo"]
    valid_policy_loss_fns = ["grpo", "ppo"]
    if self.algo_variant not in valid_algo_variants:
      raise ValueError(
          f"algo_variant must be one of {valid_algo_variants}. "
          f"Received: {self.algo_variant!r}"
      )
    if (
        self.advantage_estimator not in valid_advantage_estimators
        and self.advantage_estimator
        not in function_registry.list_advantage_estimators()
    ):
      raise ValueError(
          f"advantage_estimator must be one of {valid_advantage_estimators} or"
          " registered in function_registry. Received:"
          f" {self.advantage_estimator}"
      )
    if (
        self.policy_loss_fn not in valid_policy_loss_fns
        and self.policy_loss_fn not in function_registry.list_policy_loss_fns()
    ):
      raise ValueError(
          f"policy_loss_fn must be one of {valid_policy_loss_fns} or registered"
          f" in function_registry. Received: {self.policy_loss_fn}"
      )
    if self.reward_num_workers < -1:
      raise ValueError(
          "reward_num_workers must be >= 0, or -1 for one worker per CPU."
          f" Received: {self.reward_num_workers}"
      )
    if self.reward_worker_timeout_seconds <= 0:
      raise ValueError(
          "reward_worker_timeout_seconds must be > 0."
          f" Received: {self.reward_worker_timeout_seconds}"
      )

    # Automatically prints configuration upon initialization.
    self.print_config()

  def print_config(self):
    """Prints all configuration fields, working dynamically for child classes."""
    logging.info(f"Initializing {self.__class__.__name__}:")
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      logging.info(f"  {field.name}: {value}")


@dataclasses.dataclass(kw_only=True)
class GRPOConfig(AlgorithmConfig):
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
    loss_algo: use GRPO or GSPO for loss computation. GRPO loss is per-batch
      normalized instead of per-response normalized as mentioned in the paper.
      For GSPO, we use gspo-token loss which is more flexible.
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
    epsilon_c: Dual-clip PPO/GRPO lower bound for clipping when advantages are
      negative.
    sampler_is: Optional truncated importance-sampling correction between the
      rollout sampler and trainer actor. Set to "token" to use trainer
      recomputed logps as old-policy logps and multiply the policy loss by
      detached per-token sampler/trainer correction weights.
    sampler_is_threshold: Maximum per-token TIS correction weight.
    overlong_loss_masking: Whether to drop sequences the rollout truncated at
      the response-length budget. They leave the loss and its denominator
      both, so the surviving sequences keep their gradient magnitude and only
      the effective batch shrinks. Requires the rollout to report a
      per-trajectory truncation status.
    seq_logprob_error_threshold: Drop a sequence when the sampler and the
      trainer disagree about its tokens by more than this, measured as
      `mean_t exp|log p_trainer - log q_sampler|`. `None` disables the gate.
      Requires rollout log-probabilities.
    truncated_importance_sampling_type: Sequence-level variant of the sampler
      importance-sampling correction. `"seq-mask-tis"` weights every token by
      its raw sampler/trainer ratio and zeroes the weights of sequences whose
      geometric-mean ratio leaves the keep-band. `None` disables it. Requires
      rollout log-probabilities.
    truncated_importance_sampling_ratio_min: Lower edge of that keep-band.
    truncated_importance_sampling_ratio: Upper edge of that keep-band.
    sampler_is_length_buckets: Completion-length bucket edges, in tokens,
      strictly increasing. Reports the sampler/trainer offset per length
      bucket, which separates offsets that shrink as sequences get longer from
      offsets that do not. `None` disables the diagnostic.

  References:
    - GRPO: https://arxiv.org/abs/2402.03300
    - GSPO: https://arxiv.org/abs/2507.18071
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  reward_manager: str = "sequence-level"
  loss_algo: str = "grpo"
  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  kl_loss_mode: str = "kl"
  epsilon: float = 0.2
  epsilon_high: float | None = None
  epsilon_c: float | None = None
  sampler_is: str | None = None
  sampler_is_threshold: float = 2.0
  overlong_loss_masking: bool = False
  seq_logprob_error_threshold: float | None = None
  truncated_importance_sampling_type: str | None = None
  truncated_importance_sampling_ratio_min: float | None = None
  truncated_importance_sampling_ratio: float | None = None
  sampler_is_length_buckets: Sequence[int] | None = None

  def __post_init__(self):
    if self.epsilon_high is None:
      self.epsilon_high = self.epsilon

    super().__post_init__()

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
    self._validate_sampler_is_sequence_options()

  def _validate_sampler_is_sequence_options(self):
    """Checks the sequence-level sampler-vs-trainer options for consistency.

    A half-specified or reversed keep-band is not a crash, it is a run that
    rejects every sequence and reports zero gradients, so these are rejected
    at construction rather than at the first training step.

    Raises:
      ValueError: If an option is unsupported, incomplete, or out of order.
    """
    lo = self.truncated_importance_sampling_ratio_min
    hi = self.truncated_importance_sampling_ratio
    if (lo is None) != (hi is None):
      raise ValueError(
          "truncated_importance_sampling_ratio_min and"
          " truncated_importance_sampling_ratio must be set together, since a"
          f" keep-band needs both ends. Received: min={lo}, max={hi}"
      )
    if lo is not None and hi is not None and lo > hi:
      raise ValueError(
          "truncated_importance_sampling_ratio_min must not exceed"
          f" truncated_importance_sampling_ratio. Received: min={lo}, max={hi}"
      )
    if self.truncated_importance_sampling_type is not None:
      if self.truncated_importance_sampling_type != "seq-mask-tis":
        raise ValueError(
            "truncated_importance_sampling_type should be either None or"
            " 'seq-mask-tis'. Received:"
            f" {self.truncated_importance_sampling_type}"
        )
      if lo is None:
        raise ValueError(
            "truncated_importance_sampling_type requires a keep-band. Set"
            " truncated_importance_sampling_ratio_min and"
            " truncated_importance_sampling_ratio."
        )
      if self.sampler_is is not None:
        raise ValueError(
            "sampler_is and truncated_importance_sampling_type both set the"
            " per-token importance weights, so only one may be used at a"
            f" time. Received: sampler_is={self.sampler_is},"
            " truncated_importance_sampling_type="
            f"{self.truncated_importance_sampling_type}"
        )
      if self.use_rollout_logps:
        # The correction would be applied twice. `use_rollout_logps=True` makes
        # the sampler's log-probabilities the denominator of the surrogate
        # ratio, so that ratio already carries the whole sampler-vs-trainer
        # correction; seq-mask-tis then multiplies the same quantity in again
        # and the per-token loss picks up its square. There is no correct
        # reading of the combination, so it is refused rather than reconciled.
        raise ValueError(
            "truncated_importance_sampling_type requires"
            " use_rollout_logps=False. With use_rollout_logps=True the"
            " sampler's log-probabilities are already the surrogate ratio's"
            " denominator, so applying the sampler correction again squares"
            " it. Set use_rollout_logps=False, which recomputes the"
            " denominator on the trainer and pins the ratio to 1."
        )
    if self.sampler_is_length_buckets is not None:
      edges = tuple(self.sampler_is_length_buckets)
      if not edges:
        raise ValueError(
            "sampler_is_length_buckets must be non-empty when set; use None"
            " to disable the diagnostic."
        )
      if any(e <= 0 for e in edges) or any(
          b <= a for a, b in zip(edges, edges[1:])
      ):
        raise ValueError(
            "sampler_is_length_buckets must be strictly increasing positive"
            f" token counts. Received: {edges}"
        )
      self.sampler_is_length_buckets = edges
