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
   algo_variant: The core algorithm variant to use.
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
  num_generations: int = 2
  num_iterations: int = 1
  beta: None = None  # No KL term.
  epsilon: float = 0.2
  epsilon_high: float = 0.28  # Clip higher
  dynamic_sampling: bool = True  # TODO(sizhi): Add dynamic sampling.
  overlong_buffer: Optional[Dict[str, Any]] = dataclasses.field(
      default_factory=lambda: {
          "buffer_len": 1024,
          "float": 1.0,
      }
  )  # TODO(sizhi): Add overlong buffer.


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
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
