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

import dataclasses
from absl import logging

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

  def __post_init__(self):
    valid_algo_variants = [
        "grpo",
        "drgrpo",
        "gspo-token",
        "ppo",
        "dapo",
    ]
    valid_advantage_estimators = ["grpo", "gae", "drgrpo", "rloo"]
    valid_policy_loss_fns = ["grpo", "ppo"]
    if self.algo_variant not in valid_algo_variants:
      raise ValueError(
          f"algo_variant must be one of {valid_algo_variants}. "
          f"Received: {self.algo_variant!r}"
      )
    if self.advantage_estimator not in valid_advantage_estimators:
      raise ValueError(
          f"advantage_estimator must be one of {valid_advantage_estimators}."
          f" Received: {self.advantage_estimator}"
      )
    if self.policy_loss_fn not in valid_policy_loss_fns:
      raise ValueError(
          f"policy_loss_fn must be one of {valid_policy_loss_fns}."
          f" Received: {self.policy_loss_fn}"
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
    use_rollout_logps: Whether to use rollout-side log probabilities as
      old-policy log probabilities. If False, recompute old-policy log
      probabilities on the trainer actor.
    sampler_is: Optional truncated importance-sampling correction between the
      rollout sampler and trainer actor. Set to "token" to use trainer
      recomputed logps as old-policy logps and multiply the policy loss by
      detached per-token sampler/trainer correction weights.
    sampler_is_threshold: Maximum per-token TIS correction weight.
    temperature: Sampling temperature used during rollout generation to compute
      log probabilities and entropy in the loss function.

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
  temperature: float | None = None
  use_rollout_logps: bool = True
  sampler_is: str | None = None
  sampler_is_threshold: float = 2.0

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
