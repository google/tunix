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

"""Derived statistics over ATIF trajectories.

`TrajectoryStats.from_trajectory` summarizes a single trajectory (steps, tokens,
tool calls, status, reward); `RunStats.from_trajectories` aggregates a
collection of them (a "run") into a per-trajectory `pandas.DataFrame`. A
quantity a trajectory does not report (status, reward, token counts) is `None`
rather than 0 (a null in the DataFrame). Token counts are reported as a unit:
once a trajectory reports any token field, the fields it omits are 0. See
`TokenCounts`.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import math
import types
from typing import NamedTuple, Protocol, TypedDict, runtime_checkable

import pandas as pd
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types

_Trajectory = trajectory_lib.Trajectory[trajectory_lib.StepT]
TrajectoryStatus = agent_types.TrajectoryStatus


class StepCountsJson(TypedDict):
  """JSON form of `StepCounts`, plus the derived `total`."""

  agent: int
  user: int
  system: int
  other: int
  total: int


class TokenCountsJson(TypedDict):
  """JSON form of `TokenCounts`, plus the derived `total`."""

  prompt: int
  completion: int
  cached: int
  total: int


class TrajectoryStatsJson(TypedDict):
  """JSON form of `TrajectoryStats`."""

  trajectory_id: str | None
  status: str | None
  total_reward: float | None
  steps: StepCountsJson
  tokens: TokenCountsJson | None
  tool_call_names: list[str]


class RunStatsJson(TypedDict):
  """JSON form of `RunStats`."""

  num_trajectories: int
  steps: StepCountsJson
  tokens: TokenCountsJson | None
  num_with_tokens: int
  status_counts: dict[str, int]
  num_with_status: int
  mean_reward: float | None
  num_with_reward: int
  tool_counts: dict[str, int]


@runtime_checkable
class _ReportsOutcome(Protocol):
  """A trajectory whose schema declares a status and a total reward."""

  status: str | TrajectoryStatus | None
  total_reward: float | None


class StepCounts(NamedTuple):
  """Step counts, split by who produced each step (`Source`).

  `AGENT` steps are produced by the model, while `USER` and `SYSTEM` steps come
  from the environment (user prompts, tool outputs, and system instructions).

  Attributes:
    agent: Number of steps whose source is `Source.AGENT`.
    user: Number of steps whose source is `Source.USER`.
    system: Number of steps whose source is `Source.SYSTEM`.
    other: Number of steps whose source is any other `Source` member.
  """

  agent: int = 0
  user: int = 0
  system: int = 0
  other: int = 0

  @classmethod
  def from_steps(cls, steps: Sequence[trajectory_lib.Step]) -> StepCounts:
    """Returns the counts of `steps` by source."""
    by_source = collections.Counter(step.source for step in steps)
    agent = by_source[trajectory_lib.Source.AGENT]
    user = by_source[trajectory_lib.Source.USER]
    system = by_source[trajectory_lib.Source.SYSTEM]
    return cls(
        agent=agent,
        user=user,
        system=system,
        other=len(steps) - agent - user - system,
    )

  @property
  def total(self) -> int:
    """Returns the number of steps regardless of source."""
    return sum(self)

  @property
  def env_steps(self) -> int:
    """Returns the number of environment steps, `user + system`."""
    return self.user + self.system

  def to_json_dict(self) -> StepCountsJson:
    """Returns the counts as a plain JSON-serializable dict."""
    return StepCountsJson(
        agent=self.agent,
        user=self.user,
        system=self.system,
        other=self.other,
        total=self.total,
    )


class TokenCounts(NamedTuple):
  """Token counts reported by a step, a trajectory, or a run.

  A source either reports token counts or it does not (`TokenCounts | None`).
  Once a source reports any token field, the fields it leaves unset count as 0.

  Attributes:
    prompt: Input tokens, cached and non-cached.
    completion: Generated output tokens.
    cached: A subset of prompt tokens that hit the cache.
  """

  prompt: int = 0
  completion: int = 0
  cached: int = 0

  @classmethod
  def from_fields(
      cls, prompt: int | None, completion: int | None, cached: int | None
  ) -> TokenCounts | None:
    """Returns counts from optional fields, or None if all are unset."""
    if prompt is None and completion is None and cached is None:
      return None
    return cls(
        prompt=prompt if prompt is not None else 0,
        completion=completion if completion is not None else 0,
        cached=cached if cached is not None else 0,
    )

  @classmethod
  def from_metrics(
      cls, metrics: trajectory_lib.Metrics | None
  ) -> TokenCounts | None:
    """Returns the token counts a single step's metrics report, or None.

    Args:
      metrics: A step's metrics, or None if the step recorded none.
    """
    if metrics is None:
      return None
    return cls.from_fields(
        metrics.prompt_tokens, metrics.completion_tokens, metrics.cached_tokens
    )

  @classmethod
  def from_trajectory(cls, trajectory: _Trajectory) -> TokenCounts | None:
    """Returns the token counts a trajectory reports, or None if unreported.

    `final_metrics` is the producer's own sum over steps, so when it reports
    any token field it is used as a whole. Otherwise the per-step `metrics`
    are summed.

    Args:
      trajectory: The trajectory to inspect.
    """
    final = trajectory.final_metrics
    if final is not None:
      from_final = cls.from_fields(
          final.total_prompt_tokens,
          final.total_completion_tokens,
          final.total_cached_tokens,
      )
      if from_final is not None:
        return from_final
    return cls.sum_reported(
        cls.from_metrics(step.metrics) for step in trajectory.steps
    )

  @classmethod
  def sum_reported(
      cls, counts: Iterable[TokenCounts | None]
  ) -> TokenCounts | None:
    """Returns the sum of the non-None `counts`, or None if there are none."""
    reported = [c for c in counts if c is not None]
    if not reported:
      return None
    return cls(
        prompt=sum(c.prompt for c in reported),
        completion=sum(c.completion for c in reported),
        cached=sum(c.cached for c in reported),
    )

  @property
  def total(self) -> int:
    """Returns `prompt + completion`."""
    return self.prompt + self.completion

  def to_json_dict(self) -> TokenCountsJson:
    """Returns the counts as a plain JSON-serializable dict."""
    return TokenCountsJson(
        prompt=self.prompt,
        completion=self.completion,
        cached=self.cached,
        total=self.total,
    )


@dataclasses.dataclass(frozen=True)
class TrajectoryStats:
  """Derived summary statistics of a single trajectory.

  Only the root trajectory is summarized; embedded `subagent_trajectories` are
  not traversed (call `from_trajectory` on each subagent for its figures).

  `status` and `total_reward` are read from trajectories whose schema declares
  them and are None for trajectories whose schema does not (values packed in
  `extra` are not read).

  Attributes:
    trajectory_id: The trajectory's ID, or None if unset.
    status: Run status, or None if unreported.
    total_reward: Overall score of the trajectory (its episode return in RL), or
      None if unreported or non-finite (NaN, inf).
    steps: Step counts split by source.
    tokens: Token counts, or None if the trajectory reports none.
    tool_call_names: Function name of every tool call, in invocation order, with
      repeats kept.
  """

  trajectory_id: str | None
  status: TrajectoryStatus | None
  total_reward: float | None
  steps: StepCounts
  tokens: TokenCounts | None
  tool_call_names: tuple[str, ...]

  @classmethod
  def from_trajectory(cls, trajectory: _Trajectory) -> TrajectoryStats:
    """Derives the statistics of a single trajectory.

    Args:
      trajectory: The trajectory to summarize.

    Raises:
      ValueError: If the trajectory states a status string that is not a
        `TrajectoryStatus` name.
    """
    status: TrajectoryStatus | None = None
    total_reward: float | None = None

    if isinstance(trajectory, _ReportsOutcome):
      raw_status = trajectory.status
      if isinstance(raw_status, str):
        try:
          status = TrajectoryStatus[raw_status.upper()]
        except KeyError:
          raise ValueError(
              f"Unrecognized trajectory status {raw_status!r} expected one of"
              f" {[s.name for s in TrajectoryStatus]}."
          ) from None
      else:
        status = raw_status
      total_reward = trajectory.total_reward
    if total_reward is not None and not math.isfinite(total_reward):
      total_reward = None
    tool_call_names = []
    for step in trajectory.steps:
      for call in step.tool_calls or ():
        tool_call_names.append(call.function_name)
    return cls(
        trajectory_id=trajectory.trajectory_id,
        status=status,
        total_reward=total_reward,
        steps=StepCounts.from_steps(trajectory.steps),
        tokens=TokenCounts.from_trajectory(trajectory),
        tool_call_names=tuple(tool_call_names),
    )

  def to_json_dict(self) -> TrajectoryStatsJson:
    """Returns the stats as a plain JSON-serializable dict."""
    return TrajectoryStatsJson(
        trajectory_id=self.trajectory_id,
        status=self.status.name if self.status else None,
        total_reward=self.total_reward,
        steps=self.steps.to_json_dict(),
        tokens=self.tokens.to_json_dict() if self.tokens is not None else None,
        tool_call_names=list(self.tool_call_names),
    )

  def to_row(self) -> dict[str, str | float | int | None]:
    """Returns the stats as one flat `RunStats.df` row.

    Unreported quantities are None, which become nulls in the DataFrame. The
    per-call `tool_call_names` sequence is reduced to `num_tool_calls`; the
    run keeps the per-tool tally in `RunStats.tool_counts`.
    """
    return {
        "trajectory_id": self.trajectory_id,
        "status": self.status.name if self.status else None,
        "total_reward": self.total_reward,
        **_prefixed("steps", self.steps, StepCounts._fields),
        **_prefixed("tokens", self.tokens, TokenCounts._fields),
        "num_tool_calls": len(self.tool_call_names),
    }


def _prefixed(
    prefix: str,
    counts: StepCounts | TokenCounts | None,
    fields: tuple[str, ...],
) -> dict[str, int | None]:
  """Returns `{f"{prefix}_{field}": value}`; all None if `counts` is None."""
  values = counts._asdict() if counts is not None else dict.fromkeys(fields)
  return {f"{prefix}_{field}": values[field] for field in fields}


COLUMN_DTYPES: Mapping[str, str] = types.MappingProxyType({
    "trajectory_id": "string",
    "status": "string",
    "total_reward": "float64",
    **{f"steps_{field}": "int64" for field in StepCounts._fields},
    # Capitalized "Int64" is pandas' nullable integer dtype: `tokens` may be
    # unreported.
    **{f"tokens_{field}": "Int64" for field in TokenCounts._fields},
    "num_tool_calls": "int64",
})


@dataclasses.dataclass(frozen=True)
class RunStats:
  """Aggregated statistics across a collection of trajectories (a run).

  The per-trajectory figures are the rows of `df`; every run-level quantity is a
  reduction over its columns.

  Attributes:
    df: One row per trajectory, in the order the trajectories were given, with
      the columns and dtypes of `COLUMN_DTYPES` (see `TrajectoryStats.to_row`).
      Unreported quantities are null.
    tool_counts: Read-only mapping from tool name to its number of calls across
      the run, keyed in first-seen order.
  """

  df: pd.DataFrame
  tool_counts: Mapping[str, int]

  @classmethod
  def from_trajectories(cls, trajectories: Iterable[_Trajectory]) -> RunStats:
    """Derives per-trajectory and aggregate statistics for `trajectories`."""
    rows = []
    tool_counts = collections.Counter[str]()
    for trajectory in trajectories:
      stats = TrajectoryStats.from_trajectory(trajectory)
      tool_counts.update(stats.tool_call_names)
      rows.append(stats.to_row())
    df = pd.DataFrame(rows, columns=list(COLUMN_DTYPES)).astype(
        dict(COLUMN_DTYPES)
    )
    return cls(df=df, tool_counts=types.MappingProxyType(tool_counts))

  @property
  def num_trajectories(self) -> int:
    """Returns the number of trajectories in the run."""
    return len(self.df)

  @property
  def steps(self) -> StepCounts:
    """Returns the step counts summed across the run, split by source."""
    return StepCounts(
        agent=int(self.df["steps_agent"].sum()),
        user=int(self.df["steps_user"].sum()),
        system=int(self.df["steps_system"].sum()),
        other=int(self.df["steps_other"].sum()),
    )

  @property
  def num_with_tokens(self) -> int:
    """Returns the number of trajectories that report token counts."""
    return int(self.df["tokens_prompt"].notna().sum())

  @property
  def tokens(self) -> TokenCounts | None:
    """Returns the token counts summed across the run, or None if none reported."""
    if self.num_with_tokens == 0:
      return None
    return TokenCounts(
        prompt=int(self.df["tokens_prompt"].sum()),
        completion=int(self.df["tokens_completion"].sum()),
        cached=int(self.df["tokens_cached"].sum()),
    )

  @property
  def status_counts(self) -> Mapping[TrajectoryStatus, int]:
    """Returns a read-only mapping from status to its number of trajectories.

    Trajectories that report no status are not counted.
    """
    counts = self.df["status"].value_counts(sort=False)
    return types.MappingProxyType(
        {TrajectoryStatus[name]: int(count) for name, count in counts.items()}
    )

  @property
  def num_with_status(self) -> int:
    """Returns the number of trajectories that report a status."""
    return int(self.df["status"].notna().sum())

  @property
  def num_with_reward(self) -> int:
    """Returns the number of trajectories that report a reward."""
    return int(self.df["total_reward"].notna().sum())

  @property
  def mean_reward(self) -> float | None:
    """Returns the mean `total_reward`, or None if no trajectory reports one."""
    if self.num_with_reward == 0:
      return None
    return float(self.df["total_reward"].mean())

  def to_json_dict(self) -> RunStatsJson:
    """Returns the totals as a plain JSON-serializable dict."""
    status_counts = collections.Counter(self.status_counts).most_common()
    tool_counts = collections.Counter(self.tool_counts).most_common()
    return RunStatsJson(
        num_trajectories=self.num_trajectories,
        steps=self.steps.to_json_dict(),
        tokens=self.tokens.to_json_dict() if self.tokens is not None else None,
        num_with_tokens=self.num_with_tokens,
        status_counts={status.name: count for status, count in status_counts},
        num_with_status=self.num_with_status,
        mean_reward=self.mean_reward,
        num_with_reward=self.num_with_reward,
        tool_counts=dict(tool_counts),
    )
