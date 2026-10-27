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
tool calls); `RunStats.from_trajectories` aggregates a collection of them (a
"run") into a per-trajectory `pandas.DataFrame`. A quantity a trajectory does
not report (each token count) is `None` rather than 0 (a null in the
DataFrame). See `TokenCounts`.

Every statistic here is computed from base ATIF fields. Figures that only the
Tunix schema reports (status, reward) live in `tunix_stats` and reach the
generic classes through one slot each: `TrajectoryStats.tunix` (None for any
other schema) and `RunStats.tunix`.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import types
from typing import NamedTuple, TypedDict

import pandas as pd
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import tunix_stats

_Trajectory = trajectory_lib.Trajectory[trajectory_lib.StepT]


class StepCountsJson(TypedDict):
  """JSON form of `StepCounts`, plus the derived `total`."""

  agent: int
  user: int
  system: int
  other: int
  total: int


class TokenCountsJson(TypedDict):
  """JSON form of `TokenCounts`, plus the derived `total`.

  A count the source does not report is null, and so is `total` unless both
  `prompt` and `completion` are reported.
  """

  prompt: int | None
  completion: int | None
  cached: int | None
  total: int | None


class TrajectoryStatsJson(TypedDict):
  """JSON form of `TrajectoryStats`."""

  trajectory_id: str | None
  steps: StepCountsJson
  tokens: TokenCountsJson | None
  tool_call_names: list[str]
  tunix: tunix_stats.TunixStatsJson | None


class RunStatsJson(TypedDict):
  """JSON form of `RunStats`."""

  num_trajectories: int
  steps: StepCountsJson
  tokens: TokenCountsJson | None
  num_with_tokens: int
  tool_counts: dict[str, int]
  tunix: tunix_stats.TunixRunStatsJson


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

  def __str__(self) -> str:
    """Formats the counts as `total (agent a, user u, system s, other o)`."""
    return (
        f"{self.total} (agent {self.agent}, user {self.user}, system"
        f" {self.system}, other {self.other})"
    )

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

  Each count is None when the source does not report it, never a fabricated
  0: a source may report some counts and not others (e.g. an RL rollout
  reports only the completion tokens it generated). At least one count is
  reported: build instances with `from_fields`, `from_metrics` or
  `from_trajectory`, which return None instead of an all-None `TokenCounts`.

  Attributes:
    prompt: Input tokens, cached and non-cached.
    completion: Generated output tokens.
    cached: A subset of prompt tokens that hit the cache.
  """

  prompt: int | None = None
  completion: int | None = None
  cached: int | None = None

  @classmethod
  def from_fields(
      cls, prompt: int | None, completion: int | None, cached: int | None
  ) -> TokenCounts | None:
    """Returns counts from optional fields, or None if all are unset.

    Fields may also be nullable pandas values, e.g. `RunStats.df` cells:
    `pd.NA` counts as unset, and numpy integers are converted to `int`.
    """
    fields = [
        None if v is None or pd.isna(v) else int(v)
        for v in (prompt, completion, cached)
    ]
    return None if all(v is None for v in fields) else cls(*fields)

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
    any token field it is used as a whole (the fields it omits stay None).
    Otherwise each of `prompt`, `completion` and `cached` is summed over the
    steps whose `metrics` report it, and stays None if no step does. When
    steps report different fields, the sums (and hence `total`) cover
    different steps.

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
    step_counts = [cls.from_metrics(step.metrics) for step in trajectory.steps]
    reported = [c for c in step_counts if c is not None]
    return cls.from_fields(
        prompt=_sum_reported(c.prompt for c in reported),
        completion=_sum_reported(c.completion for c in reported),
        cached=_sum_reported(c.cached for c in reported),
    )

  @property
  def total(self) -> int | None:
    """Returns `prompt + completion`, or None unless both are reported."""
    if self.prompt is None or self.completion is None:
      return None
    return self.prompt + self.completion

  def __str__(self) -> str:
    """Returns the counts formatted as `total (prompt+completion)`.

    Without both `prompt` and `completion` there is no total, so the reported
    counts are named instead, e.g. `50 completion`.
    """
    if self.total is not None:
      return f"{self.total} ({self.prompt}+{self.completion})"
    return ", ".join(
        f"{count} {name}"
        for name, count in self._asdict().items()
        if count is not None
    )

  def to_json_dict(self) -> TokenCountsJson:
    """Returns the counts as a plain JSON-serializable dict."""
    return TokenCountsJson(
        prompt=self.prompt,
        completion=self.completion,
        cached=self.cached,
        total=self.total,
    )


def _sum_reported(values: Iterable[int | None]) -> int | None:
  """Returns the sum of the non-None `values`, or None if none is reported."""
  reported = [v for v in values if v is not None]
  return sum(reported) if reported else None


@dataclasses.dataclass(frozen=True)
class TrajectoryStats:
  """Derived summary statistics of a single trajectory.

  Only the root trajectory is summarized; embedded `subagent_trajectories` are
  not traversed (call `from_trajectory` on each subagent for its figures).

  Attributes:
    trajectory_id: The trajectory's ID, or None if unset.
    steps: Step counts split by source.
    tokens: Token counts, or None if the trajectory reports none.
    tool_call_names: Function name of every tool call, in invocation order, with
      repeats kept.
    tunix: Figures only a Tunix RL rollout reports, or None for any other
      schema.
  """

  trajectory_id: str | None
  steps: StepCounts
  tokens: TokenCounts | None
  tool_call_names: tuple[str, ...]
  tunix: tunix_stats.TunixStats | None

  @classmethod
  def from_trajectory(cls, trajectory: _Trajectory) -> TrajectoryStats:
    """Derives the statistics of a single trajectory.

    Args:
      trajectory: The trajectory to summarize.

    Raises:
      ValueError: If a Tunix rollout states a status that is not a
        `TrajectoryStatus` name.
    """
    tool_call_names = []
    for step in trajectory.steps:
      for call in step.tool_calls or ():
        tool_call_names.append(call.function_name)
    return cls(
        trajectory_id=trajectory.trajectory_id,
        steps=StepCounts.from_steps(trajectory.steps),
        tokens=TokenCounts.from_trajectory(trajectory),
        tool_call_names=tuple(tool_call_names),
        tunix=(
            tunix_stats.TunixStats.from_trajectory(trajectory)
            if isinstance(trajectory, trajectory_lib.TunixTrajectory)
            else None
        ),
    )

  def to_json_dict(self) -> TrajectoryStatsJson:
    """Returns the stats as a plain JSON-serializable dict."""
    return TrajectoryStatsJson(
        trajectory_id=self.trajectory_id,
        steps=self.steps.to_json_dict(),
        tokens=self.tokens.to_json_dict() if self.tokens is not None else None,
        tool_call_names=list(self.tool_call_names),
        tunix=self.tunix.to_json_dict() if self.tunix is not None else None,
    )

  def to_row(self) -> dict[str, str | float | int | None]:
    """Returns the stats as one flat `RunStats.df` row.

    Unreported quantities are None, which become nulls in the DataFrame. The
    per-call `tool_call_names` sequence is reduced to `num_tool_calls`; the
    run keeps the per-tool tally in `RunStats.tool_counts`. The Tunix figures
    are the `tunix_*` cells (see `tunix_stats.to_row`).
    """
    return {
        "trajectory_id": self.trajectory_id,
        **_prefixed("steps", self.steps, StepCounts._fields),
        **_prefixed("tokens", self.tokens, TokenCounts._fields),
        "num_tool_calls": len(self.tool_call_names),
        **tunix_stats.to_row(self.tunix),
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
    **{f"steps_{field}": "int64" for field in StepCounts._fields},
    # Capitalized "Int64" is pandas' nullable integer dtype: each token count
    # may be unreported.
    **{f"tokens_{field}": "Int64" for field in TokenCounts._fields},
    "num_tool_calls": "int64",
    **{
        tunix_stats.column(field): dtype
        for field, dtype in tunix_stats.FIELD_DTYPES.items()
    },
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
    """Returns the number of trajectories that report any token count."""
    token_columns = [f"tokens_{field}" for field in TokenCounts._fields]
    return int(self.df[token_columns].notna().any(axis=1).sum())

  @property
  def tokens(self) -> TokenCounts | None:
    """Returns the token counts summed across the run, or None if none reported.

    Each of `prompt`, `completion` and `cached` is summed over the trajectories
    that report it, and stays None if none of them does. A run normally comes
    from one producer; if its trajectories report different fields, the sums
    (and hence `total`) cover different trajectories.
    """
    # `min_count=1` keeps a field with no reported value null instead of 0.
    return TokenCounts.from_fields(
        *(self.df[f"tokens_{f}"].sum(min_count=1) for f in TokenCounts._fields)
    )

  @property
  def tunix(self) -> tunix_stats.TunixRunStats:
    """Returns the Tunix figures of `df` aggregated over the run.

    Each aggregate covers the rollouts that report its figure, so a run without
    Tunix rollouts has empty counts and no means.
    """
    return tunix_stats.TunixRunStats(df=self.df)

  def to_json_dict(self) -> RunStatsJson:
    """Returns the totals as a plain JSON-serializable dict."""
    tool_counts = collections.Counter(self.tool_counts).most_common()
    return RunStatsJson(
        num_trajectories=self.num_trajectories,
        steps=self.steps.to_json_dict(),
        tokens=self.tokens.to_json_dict() if self.tokens is not None else None,
        num_with_tokens=self.num_with_tokens,
        tool_counts=dict(tool_counts),
        tunix=self.tunix.to_json_dict(),
    )
