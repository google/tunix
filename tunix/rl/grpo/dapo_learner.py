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
"""Helper functions for GRPO Trainer."""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@dataclasses.dataclass(slots=True, kw_only=True)
class DAPOConfig(grpo_learner_lib.GRPOConfig):
  """Configuration for DAPO.

  Attributes:
   algo_variant: The algorithm variant to use.
   advantage_estimator: The advantage estimator to use.
   policy_loss_fn: The policy loss function to use.
   loss_agg_mode: The aggregation mode for the loss function.
   loss_algo: The loss algorithm to use. To be deprecated.
   num_generations: The number of times the policy generates multiple responses
     for a given prompt within a single training step. This corresponds to 'G'
     in Algorithm 1 in the paper. A higher value means more samples are used to
     compute relative advantages.
   num_iterations: The number of iterations per batch (𝜇 in GRPO algo 1).
   beta: The coefficient for the KL divergence penalty (𝛽) in the GRPO loss
     function. This term prevents policy updates from deviating too far from the
     reference model. A value of 0.0 means no KL penalty is applied. Always None
     for DAPO.
   epsilon: Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to
     PPO, it ensures stable updates.
   epsilon_high: Epsilon value for upper bound clipping.
   dynamic_sampling: Whether to use dynamic sampling.
   overlong_buffer: The overlong buffer to use for overlong reward shaping.
   References: - DAPO:
     https://arxiv.org/pdf/2503.14476
  """

  algo_variant: str = "dapo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  loss_agg_mode: str = "token-mean"
  reward_manager: str = "sequence-level"
  num_generations: int = 2
  num_iterations: int = 1
  beta: None = None  # No KL term.
  epsilon: float = 0.2
  epsilon_high: float = 0.28  # Clip higher
  dynamic_sampling: bool = True  # TODO(sizhi): Add dynamic sampling.
  overlong_buffer: Optional[Dict[str, Any]] = dataclasses.field(
      default_factory=lambda: {
          "enable": True,
          "overlong_buffer_length": 4096,  # Threshold before penalties apply.
          "overlong_buffer_penalty": 1.0,
          "max_response_length": 20480,  # Hard maximum generation length.
      }
  )

  def __post_init__(self):
    if self.epsilon_high < self.epsilon:
      raise ValueError("epsilon_high must be greater than or equal to epsilon.")

    if self.overlong_buffer is not None and self.overlong_buffer.get("enable"):
      buffer = self.overlong_buffer
      required = [
          "overlong_buffer_length",
          "overlong_buffer_penalty",
          "max_response_length",
      ]

      missing = [k for k in required if buffer.get(k) is None]
      if missing:
        raise ValueError(f"overlong_buffer is enabled but missing: {missing}")

      if buffer["overlong_buffer_penalty"] < 0:
        raise ValueError("overlong_buffer_penalty must be non-negative.")

      if buffer["overlong_buffer_length"] <= 0:
        raise ValueError("overlong_buffer_length must be positive.")

      if buffer["max_response_length"] <= 0:
        raise ValueError("max_response_length must be positive.")


def reward_shaping(
    prompts: List[str],
    completions: List[str],
    mode: rl_cluster_lib.Mode,
    overlong_buffer: Dict[str, Any] | None = None,
    **kwargs,
) -> List[float]:
  """Reward shaping function for DAPO."""
  del prompts, mode, kwargs
  if overlong_buffer is None:
    raise ValueError("reward_shaping is called but with empty overlong_buffer.")

  overlong_buffer_length = overlong_buffer["overlong_buffer_length"]
  overlong_buffer_penalty = overlong_buffer["overlong_buffer_penalty"]
  max_response_length = overlong_buffer["max_response_length"]

  expected_response_length = max_response_length - overlong_buffer_length
  scores = []
  for completion in completions:
    output_length = len(completion)
    exceed_length = output_length - expected_response_length
    overlong_reward = min(
        -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
    )
    scores.append(overlong_reward)
  return scores


class DAPOLearner(grpo_learner_lib.GrpoLearner[DAPOConfig]):
  """DAPO learner."""

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: DAPOConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `DAPOLearner`."""
    reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )
    if algo_config.overlong_buffer and algo_config.overlong_buffer["enable"]:
      reward_fns.append(reward_shaping)
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
