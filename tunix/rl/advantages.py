# Copyright 2026 Google LLC
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

"""Framework-neutral NumPy group-relative advantage estimators."""

from collections.abc import Mapping
import numpy as np
from numpy import typing as npt
from tunix.rl import function_registry

MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE: int = 2
# `tied_rewards` treats two group rewards as equal within float32 tolerance.
_REWARD_TIE_ATOL: float = 1e-6


def _grouped_valid_stats(
    rewards: npt.ArrayLike,
    valid_mask: npt.ArrayLike,
    num_generations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Reshapes rewards into groups and computes valid-only group statistics.

  Args:
    rewards: Flat `[num_groups * num_generations]` rewards.
    valid_mask: Flat `[num_groups * num_generations]` boolean mask; `True` marks
      a trajectory that should contribute to (and receive) an advantage.
    num_generations: Group size `G`.

  Returns:
    Tuple `(grouped_rewards, grouped_mask, valid_counts, masked_sum)` where the
    grouped arrays are `[num_groups, num_generations]` and `valid_counts` /
    `masked_sum` are `[num_groups, 1]`. Rewards of invalid trajectories are
    zeroed in `grouped_rewards` and excluded from `masked_sum`.
  """
  grouped_rewards = np.asarray(rewards, dtype=np.float32).reshape(
      -1, num_generations
  )
  grouped_mask = (
      np.asarray(valid_mask).astype(bool).reshape(-1, num_generations)
  )
  grouped_mask = grouped_mask & np.isfinite(grouped_rewards)
  grouped_rewards = np.where(grouped_mask, grouped_rewards, 0.0)
  valid_counts = grouped_mask.sum(axis=-1, keepdims=True)
  masked_sum = grouped_rewards.sum(axis=-1, keepdims=True)
  return grouped_rewards, grouped_mask, valid_counts, masked_sum


@function_registry.register_advantage_estimator("grpo")
def compute_advantages(
    rewards: npt.ArrayLike,
    num_generations: int,
    valid_mask: npt.ArrayLike | None = None,
) -> np.ndarray:
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.
    valid_mask: Optional boolean mask, same shape as `rewards`, marking the
      trajectories that are healthy enough to be trained on. When provided, the
      group mean and (sample, `ddof=1`) std are computed over valid trajectories
      only, invalid trajectories get a `0.0` advantage, and groups with fewer
      than `MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE` valid trajectories are zeroed
      out entirely (a sample std is undefined there).

  Returns:
    Group relative advantages (`float32` ndarray).
  """
  if valid_mask is None:
    valid_mask = np.ones(np.shape(rewards), dtype=bool)

  grouped_rewards, grouped_mask, valid_counts, masked_sum = (
      _grouped_valid_stats(rewards, valid_mask, num_generations)
  )
  mean_grouped_rewards = masked_sum / np.maximum(valid_counts, 1)
  squared_deviations = np.where(
      grouped_mask, (grouped_rewards - mean_grouped_rewards) ** 2, 0.0
  ).sum(axis=-1, keepdims=True)
  std_grouped_rewards = np.sqrt(
      squared_deviations / np.maximum(valid_counts - 1, 1)
  )
  advantages = (grouped_rewards - mean_grouped_rewards) / (
      std_grouped_rewards + 1e-6
  )
  advantages = np.where(
      grouped_mask & (valid_counts >= MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE),
      advantages,
      0.0,
  )
  return advantages.reshape(np.shape(rewards)).astype(np.float32)


@function_registry.register_advantage_estimator("grpo-loo")
def compute_grpo_loo_advantages(
    rewards: npt.ArrayLike,
    num_generations: int,
    valid_mask: npt.ArrayLike | None = None,
) -> np.ndarray:
  """GRPO advantages with leave-one-out centering and std normalization.

  Combines RLOO's unbiased baseline with GRPO's variance normalization. In
  plain GRPO the sample `r_i` appears in its own baseline `mean(r)` with weight
  `1/k`, which shrinks every residual by `(k - 1) / k`; divided by the sample
  std (`ddof=1`), `A_i = ((k - 1) / k) * (r_i - mean_{-i}) / std(r)`. At small
  valid group sizes (`k = 2..4`, or larger groups after invalid trajectories are
  masked out) that `(k - 1) / k` factor attenuates the advantage by 25%--50% as
  a pure function of how many peers survived. Using the leave-one-out mean
  `mean_{-i}` removes the self-inclusion bias while keeping the group-std
  scaling that makes advantages comparable across prompts of different
  difficulty:

    `A_i = (r_i - mean_{j != i, valid} r_j) / (std_{valid}(r) + 1e-6)`

  Equivalently, `A_i = (k / (k - 1)) * A_i^{grpo}` for a group with `k` valid
  trajectories.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations `G`.
    valid_mask: Optional boolean mask, same shape as `rewards`, marking the
      trajectories that are healthy enough to be trained on. Invalid
      trajectories are excluded from every peer's leave-one-out mean and from
      the group std, receive a `0.0` advantage themselves, and groups with fewer
      than `MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE` valid trajectories are zeroed
      out entirely.

  Returns:
    Leave-one-out, std-normalized group advantages (`float32` ndarray).
  """
  if num_generations < MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE:
    return np.zeros(np.shape(rewards), dtype=np.float32)

  if valid_mask is None:
    valid_mask = np.ones(np.shape(rewards), dtype=bool)

  grouped_rewards, grouped_mask, valid_counts, masked_sum = (
      _grouped_valid_stats(rewards, valid_mask, num_generations)
  )
  loo_mean = (masked_sum - grouped_rewards) / np.maximum(valid_counts - 1, 1)
  group_mean = masked_sum / np.maximum(valid_counts, 1)
  squared_deviations = np.where(
      grouped_mask, (grouped_rewards - group_mean) ** 2, 0.0
  ).sum(axis=-1, keepdims=True)
  std_grouped_rewards = np.sqrt(
      squared_deviations / np.maximum(valid_counts - 1, 1)
  )
  advantages = (grouped_rewards - loo_mean) / (std_grouped_rewards + 1e-6)
  advantages = np.where(
      grouped_mask & (valid_counts >= MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE),
      advantages,
      0.0,
  )
  return advantages.reshape(np.shape(rewards)).astype(np.float32)


@function_registry.register_advantage_estimator("rloo")
def compute_rloo_advantages(
    rewards: npt.ArrayLike,
    num_generations: int,
    valid_mask: npt.ArrayLike | None = None,
) -> np.ndarray:
  """Compute RLOO (REINFORCE Leave-One-Out) advantages.

  RLOO computes a baseline for each completion by averaging the rewards of all
  other completions to the same prompt.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.
    valid_mask: Optional boolean mask, same shape as `rewards`, marking the
      trajectories that are healthy enough to be trained on. When provided, the
      leave-one-out baseline averages valid peers only, invalid trajectories get
      a `0.0` advantage, and groups with fewer than
      `MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE` valid trajectories are zeroed out
      entirely (no peer is left to form a baseline).

  Returns:
    RLOO advantages (`float32` ndarray).
  """
  if num_generations < MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE:
    # RLOO requires at least 2 samples to calculate a baseline.
    return np.zeros(np.shape(rewards), dtype=np.float32)

  if valid_mask is None:
    valid_mask = np.ones(np.shape(rewards), dtype=bool)

  grouped_rewards, grouped_mask, valid_counts, masked_sum = (
      _grouped_valid_stats(rewards, valid_mask, num_generations)
  )
  loo_mean = (masked_sum - grouped_rewards) / np.maximum(valid_counts - 1, 1)
  rloo_advantages = grouped_rewards - loo_mean
  rloo_advantages = np.where(
      grouped_mask & (valid_counts >= MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE),
      rloo_advantages,
      0.0,
  )
  return rloo_advantages.reshape(np.shape(rewards)).astype(np.float32)


@function_registry.register_advantage_estimator("drgrpo")
def compute_drgrpo_advantages(
    rewards: npt.ArrayLike,
    num_generations: int,
    valid_mask: npt.ArrayLike | None = None,
) -> np.ndarray:
  """Group relative advantages -- done right.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.
    valid_mask: Optional boolean mask, same shape as `rewards`, marking the
      trajectories that are healthy enough to be trained on. When provided, the
      group mean is computed over valid trajectories only, invalid trajectories
      get a `0.0` advantage, and groups with fewer than
      `MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE` valid trajectories are zeroed out
      entirely (a single valid trajectory has zero deviation from its own mean).

  Returns:
    Group relative advantages (`float32` ndarray).
  """
  if valid_mask is None:
    valid_mask = np.ones(np.shape(rewards), dtype=bool)

  grouped_rewards, grouped_mask, valid_counts, masked_sum = (
      _grouped_valid_stats(rewards, valid_mask, num_generations)
  )
  mean_grouped_rewards = masked_sum / np.maximum(valid_counts, 1)
  advantages = grouped_rewards - mean_grouped_rewards
  advantages = np.where(
      grouped_mask & (valid_counts >= MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE),
      advantages,
      0.0,
  )
  return advantages.reshape(np.shape(rewards)).astype(np.float32)


@function_registry.register_group_filter("tied_rewards")
def tied_rewards(
    rewards: npt.ArrayLike,
    valid_mask: npt.ArrayLike | None = None,
    *,
    reward_components: Mapping[str, npt.ArrayLike] | None = None,
) -> bool:
  """Returns whether a prompt group should be dropped due to tied rewards (DAPO).

  Checks finite rewards among valid trajectories. When `reward_components` is
  provided (mapping `"env"` and/or `<reward_fn.__name__>` to per-trajectory
  scores before summing), the tie check inspects the primary task reward
  (`"env"` when present, otherwise the first reward component) so continuous
  shaping terms such as length or format penalties do not mask an underlying
  task-reward tie.

  Groups with zero trainable rollouts (`mask & isfinite(total)` all `False`,
  e.g. worker or sandbox outages) return `False` so they reach step telemetry
  (`rollout/status/*`, `rollout/invalid_trajectory_frac`) instead of masking
  failures or spinning indefinitely without `max_consecutive_filtered_groups`.
  A group with fewer than 2 finite primary rewards, or whose primary or total
  rewards tie across usable rollouts, yields zero task signal and returns
  `True`.

  Args:
    rewards: 1D per-trajectory total rewards for one prompt group.
    valid_mask: Optional boolean mask aligned with `rewards` (`True` for healthy
      trajectories). Defaults to all `True`.
    reward_components: Optional per-source unsummed reward arrays aligned with
      `rewards`.

  Returns:
    `True` to drop the group and dispatch a replacement prompt; `False` to keep.
  """
  total = np.asarray(rewards, dtype=np.float64).reshape(-1)
  mask = (
      np.ones(total.shape, dtype=bool)
      if valid_mask is None
      else np.asarray(valid_mask, dtype=bool).reshape(-1)
  )
  if mask.shape != total.shape:
    raise ValueError(
        f"valid_mask shape {mask.shape} must match rewards shape {total.shape}."
    )
  primary = total
  if reward_components:
    key = "env" if "env" in reward_components else next(iter(reward_components))
    candidate = np.asarray(reward_components[key], dtype=np.float64).reshape(-1)
    if candidate.shape != total.shape:
      raise ValueError(
          f"reward_components[{key!r}] shape {candidate.shape} must match"
          f" rewards shape {total.shape}."
      )
    primary = candidate
  trainable = mask & np.isfinite(total)
  if not np.any(trainable):
    return False
  usable = trainable & np.isfinite(primary)
  if int(np.sum(usable)) < MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE:
    return True
  if float(np.ptp(primary[usable])) <= _REWARD_TIE_ATOL:
    return True
  return float(np.ptp(total[usable])) <= _REWARD_TIE_ATOL
