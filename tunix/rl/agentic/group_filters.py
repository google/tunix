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

"""Dynamic-sampling group filters (DAPO) for the agentic RL learner.

A filter looks at one scored group (its trajectories and their advantages) and
decides whether the group should be dropped. Dropped groups are not trained on;
the learner dispatches a replacement prompt instead.
"""

from collections.abc import Sequence
from typing import Any

from jax import typing as jax_typing
import numpy as np
from tunix.rl.agentic.agents import agent_types

# Trajectory statuses that mean the episode broke, not that the policy failed
# the task (the collector stores `status.name`). Deliberately narrower than
# `agent_types.INVALID_TRAJECTORY_STATUSES`: an episode `TIMEOUT` is the
# policy taking too long, which is a legitimate (zero-reward) outcome.
SYSTEM_ERROR_STATUSES = frozenset({
    agent_types.TrajectoryStatus.FAILED.name,
    agent_types.TrajectoryStatus.ENV_TIMEOUT.name,
    agent_types.TrajectoryStatus.CANCELLED.name,
})

DROP_FAILED = "drop_failed"
ZERO_VARIANCE = "zero_variance"
GROUP_FILTERS = (DROP_FAILED, ZERO_VARIANCE)

# GRPO advantages are (r - mean) / (std + 1e-6); for tied non-binary rewards
# the float error of the mean is amplified to ~1e-2, so "zero" needs slack.
_ZERO_ADVANTAGE_ATOL = 1e-3


def validate_group_filters(filters: Sequence[str]) -> None:
  """Checks that `filters` is a duplicate-free sequence of known names.

  Args:
    filters: Names from `GROUP_FILTERS`.

  Raises:
    ValueError: If `filters` is a bare string, has an unknown name or has
      duplicates.
  """
  if isinstance(filters, str):
    raise ValueError(
        f"group_filters must be a sequence of names, got the string {filters!r}"
        f" (use e.g. ['{filters}'])."
    )
  unknown = [f for f in filters if f not in GROUP_FILTERS]
  if unknown:
    raise ValueError(
        f"Unknown group_filters {unknown}; supported: {GROUP_FILTERS}."
    )
  if len(set(filters)) != len(filters):
    raise ValueError(f"group_filters has duplicates: {list(filters)}.")


def has_system_error(group: Sequence[Any]) -> bool:
  """Returns whether any trajectory of the group ended in a system error."""
  return any(item.traj["status"] in SYSTEM_ERROR_STATUSES for item in group)


def has_zero_advantage(advantages: jax_typing.ArrayLike) -> bool:
  """Returns whether every advantage is (numerically) zero: all rewards tie."""
  return bool(
      np.allclose(np.asarray(advantages), 0.0, atol=_ZERO_ADVANTAGE_ATOL)
  )


def drop_reason(
    filters: Sequence[str],
    group: Sequence[Any],
    advantages: jax_typing.ArrayLike,
) -> str | None:
  """Returns the first filter that drops the group, or None to keep it.

  Args:
    filters: Names from `GROUP_FILTERS`, checked in order.
    group: The group's trajectory items (`item.traj["status"]` is read).
    advantages: The group's advantages.

  Returns:
    The name of the first filter that rejects the group, or None.
  """
  for name in filters:
    if name == DROP_FAILED and has_system_error(group):
      return name
    if name == ZERO_VARIANCE and has_zero_advantage(advantages):
      return name
  return None
