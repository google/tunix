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

# Magistral: https://arxiv.org/pdf/2506.10910
# DR-GRPO: https://arxiv.org/pdf/2503.20783

import jax
import jax.numpy as jnp


def compute_advantages(
    rewards: jax.Array,
    num_generations: int,
    dr_grpo: bool = False,
    magistral_adv_norm: bool = False,
    eliminate_non_diverse_groups: bool = False,
    debug: bool = False,
) -> jax.Array:
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.
    dr_grpo: If True, use DR-GRPO variant (mean subtraction only).
    magistral_adv_norm: If True, normalize by mini-batch std dev.
    eliminate_non_diverse_groups: If True, zero out advantages for non-diverse
      groups.
    debug: If True, print debug information.

  Returns:
    Group relative advantages.
  """
  reshaped_rewards = rewards.reshape(-1, num_generations)
  mean_grouped_rewards = reshaped_rewards.mean(axis=1, keepdims=True)
  advantages = reshaped_rewards - mean_grouped_rewards

  if dr_grpo:
    if debug:
      print(
          "[GRPO Helper] Using DR-GRPO advantage computation (mean subtraction"
          " only)."
      )
    return advantages.flatten()

  if magistral_adv_norm:
    # Normalize by mini-batch std dev instead of group std dev
    batch_std = advantages.std()
    advantages = advantages / (batch_std + 1e-4)
    if debug:
      print(
          "[GRPO Helper] Magistral Adv Norm: Normalizing with mini-batch std"
          f" dev: {batch_std}"
      )
  else:
    # Default: normalize by group std dev
    std_grouped_rewards = reshaped_rewards.std(axis=1, ddof=1, keepdims=True)
    advantages = advantages / (std_grouped_rewards + 1e-4)

  if eliminate_non_diverse_groups:
    # Introduced in Magistral paper (https://arxiv.org/pdf/2506.10910)
    # A group is non-diverse if all its rewards are equal.
    # Create a mask where groups with all equal rewards are marked.
    all_equal_mask = jnp.all(
        reshaped_rewards == reshaped_rewards[:, 0:1], axis=1, keepdims=True
    ).astype(advantages.dtype)
    advantages = advantages * (1.0 - all_equal_mask)
    if debug:
      num_eliminated = jnp.sum(all_equal_mask)
      num_total_groups = reshaped_rewards.shape[0]
      print(
          f"[GRPO Helper] Eliminating {num_eliminated}/{num_total_groups}"
          " non-diverse groups."
      )

  return advantages.flatten()
