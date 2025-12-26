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
import jax
from tunix.rl import function_registry
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@function_registry.register_advantage_estimator("drgrpo")
def compute_advantages(rewards: jax.Array, num_generations: int) -> jax.Array:
  """Group relative advantages -- done right.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=1)
  return rewards - mean_grouped_rewards.repeat(num_generations)


@dataclasses.dataclass(slots=True, kw_only=True)
class DrGRPOConfig(grpo_learner_lib.GRPOConfig):
  """Configuration for DrGRPO."""

  # https://arxiv.org/abs/2503.20783
  algo_variant: str = dataclasses.field(default="drgrpo", init=False)
  advantage_estimator: str = dataclasses.field(default="drgrpo", init=False)
  loss_agg_mode: str = dataclasses.field(default="sequence-mean-token-scale")


class DrGRPOLearner(grpo_learner_lib.GrpoLearner[DrGRPOConfig]):
  """Dr. GRPO Learner."""
