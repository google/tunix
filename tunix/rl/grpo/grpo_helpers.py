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

import jax


def compute_advantages(
    rewards: jax.Array, num_generations: int, std_scale: bool = True
) -> jax.Array:
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.
    std_scale: Whether to divide advantages by the standard deviation.

  Returns:
    Group relative advantages.
  """
  reshaped_rewards = rewards.reshape(-1, num_generations)

  mean_grouped_rewards = reshaped_rewards.mean(axis=1).repeat(num_generations)
  advantages = rewards - mean_grouped_rewards

  if not std_scale:
    return advantages

  std_grouped_rewards = reshaped_rewards.std(axis=1, ddof=1)
  return advantages / (std_grouped_rewards + 1e-4).repeat(num_generations)
