# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Statistics of Tunix RL rollouts, the Tunix extension of `TrajectoryStats`.

A statistic belongs in `TrajectoryStats` if base ATIF fields can compute it,
and here if it needs the Tunix schema (`TunixTrajectoryMetadata`). The Explorer
keeps the Tunix figures in one slot, `TrajectoryStats.tunix`, set only for
`TunixTrajectory` instances and flattened into `tunix_*` columns of
`RunStats.df`; `RunStats.tunix` aggregates those columns per run. This is the
only Explorer module that reads the rollout engine's types (`agent_types`).
"""

from __future__ import annotations

import collections
from collections.abc import Mapping
import dataclasses
import math
import types
from typing import NamedTuple, TypedDict, cast

import pandas as pd
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types

# `TunixTrajectoryMetadata.status` stores a member's name.
TrajectoryStatus = agent_types.TrajectoryStatus

# Prefix of the `RunStats.df` columns holding `TunixStats` fields.
COLUMN_PREFIX = "tunix"

# A `tunix_*` cell of a `RunStats.df` row: a JSON scalar (see `to_row`).
Cell = str | float | None


def column(field: str) -> str:
  """Returns the `RunStats.df` column of a `TunixStats` field."""
  return f"{COLUMN_PREFIX}_{field}"


def parse_status(status: str | None) -> TrajectoryStatus | None:
  """Returns the status a rollout states, or None if it states none.

  Args:
    status: `TunixTrajectoryMetadata.status`, a `TrajectoryStatus` name in any
      letter case.

  Raises:
    ValueError: If `status` is not a `TrajectoryStatus` name.
  """
  if status is None:
    return None
  try:
    return TrajectoryStatus[status.upper()]
  except KeyError:
    raise ValueError(
        f"Unrecognized trajectory status {status!r} expected one of"
        f" {[s.name for s in TrajectoryStatus]}."
    ) from None


class TunixStatsJson(TypedDict):
  """JSON form of `TunixStats`."""

  status: str | None
  total_reward: float | None


class TunixStats(NamedTuple):
  """Figures a Tunix RL rollout reports about itself.

  Attributes:
    status: Status the rollout is in, or None if unstated.
    total_reward: Episode return, or None if unreported or non-finite.
  """

  status: TrajectoryStatus | None
  total_reward: float | None

  @classmethod
  def from_trajectory(
      cls, trajectory: trajectory_lib.TunixTrajectory
  ) -> TunixStats:
    """Derives the rollout's figures from its Tunix metadata.

    Raises:
      ValueError: If the rollout states a status that is not a
        `TrajectoryStatus` name.
    """
    total_reward = trajectory.total_reward
    if total_reward is not None and not math.isfinite(total_reward):
      total_reward = None
    return cls(
        status=parse_status(trajectory.status),
        total_reward=total_reward,
    )

  def to_json_dict(self) -> TunixStatsJson:
    """Returns the figures as a plain JSON-serializable dict."""
    return TunixStatsJson(
        status=self.status.name if self.status is not None else None,
        total_reward=self.total_reward,
    )


def to_row(
    stats: TunixStats | None,
) -> dict[str, Cell]:
  """Returns the `tunix_*` cells of a `RunStats.df` row.

  The cells are the figures' JSON form (`to_json_dict`, whose keys are the
  `TunixStats` fields), so a row and a JSON record of the same rollout agree
  cell for cell: enums are their names, scalars are kept as they are.

  Args:
    stats: The figures of a Tunix rollout, or None for a trajectory of another
      schema, whose cells are then all None.
  """
  if stats is None:
    return {column(field): None for field in TunixStats._fields}
  # To a type checker a `TypedDict` is a `Mapping[str, object]`; the JSON form
  # only ever holds `Cell` values.
  cells = cast(Mapping[str, Cell], stats.to_json_dict())
  return {column(field): value for field, value in cells.items()}


# The pandas dtype of each field's `RunStats.df` column (see `column`), all
# nullable: the columns are null for trajectories that are not Tunix rollouts.
FIELD_DTYPES: Mapping[str, str] = types.MappingProxyType({
    "status": "string",
    "total_reward": "float64",
})


class TunixRunStatsJson(TypedDict):
  """JSON form of `TunixRunStats`."""

  status_counts: dict[str, int]
  num_with_status: int
  mean_reward: float | None
  num_with_reward: int


@dataclasses.dataclass(frozen=True)
class TunixRunStats:
  """Aggregates of the Tunix figures across a run.

  Every quantity is a reduction over the `tunix_*` columns of `df` covering the
  rollouts that report it; a run without Tunix rollouts has empty counts and
  no means.

  Attributes:
    df: The run's `RunStats.df`.
  """

  df: pd.DataFrame

  @property
  def status_counts(self) -> Mapping[TrajectoryStatus, int]:
    """Returns a read-only mapping from status to its number of rollouts.

    Rollouts that report no status are not counted.
    """
    counts = self.df[column("status")].value_counts(sort=False)
    return types.MappingProxyType(
        {TrajectoryStatus[name]: int(count) for name, count in counts.items()}
    )

  @property
  def num_with_status(self) -> int:
    """Returns the number of rollouts that report a status."""
    return int(self.df[column("status")].notna().sum())

  @property
  def num_with_reward(self) -> int:
    """Returns the number of rollouts that report a reward."""
    return int(self.df[column("total_reward")].notna().sum())

  @property
  def mean_reward(self) -> float | None:
    """Returns the mean `total_reward`, or None if no rollout reports one."""
    if self.num_with_reward == 0:
      return None
    return float(self.df[column("total_reward")].mean())

  def to_json_dict(self) -> TunixRunStatsJson:
    """Returns the aggregates as a plain JSON-serializable dict."""
    ranked = collections.Counter(self.status_counts).most_common()
    return TunixRunStatsJson(
        status_counts={status.name: count for status, count in ranked},
        num_with_status=self.num_with_status,
        mean_reward=self.mean_reward,
        num_with_reward=self.num_with_reward,
    )
