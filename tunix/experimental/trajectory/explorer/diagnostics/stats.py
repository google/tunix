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

TrajectoryStats centralizes common metrics (steps, tokens, tools, reward) so
every consumer (the Explorer CLI, Colab notebooks, evaluation scripts) derives
them identically without traversing steps. Because metrics are optional,
missing fields default to None instead of 0 to differentiate absence of data
from zero measurements. This module only computes; rendering for the terminal
lives in `explorer.commands.formatting`.

Base ATIF has no reward or status field: a TunixTrajectory carries them as
first-class fields (packed into `extra` once persisted), while other producers
pass them through the open `extra` maps. Everything else is base ATIF and is
read the same way for every trajectory.
"""

import collections
from collections.abc import Iterable, Sequence
import dataclasses
from typing import Any

from tunix.experimental.trajectory import trajectory as trajectory_lib

# Base ATIF has no reward or status field, so producers that are not Tunix pass
# them through the open `extra` maps under these keys.
_REWARD_KEY = "reward"
_STATUS_KEY = "status"


@dataclasses.dataclass(frozen=True)
class StepCounts:
  """Step totals for a trajectory, split by originator."""

  total: int = 0
  agent: int = 0
  user: int = 0
  system: int = 0

  def to_json_dict(self) -> dict[str, int]:
    """Returns the counts as a plain JSON-serializable dict."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TokenCounts:
  """Token totals for a trajectory.

  `reported` distinguishes a trajectory that genuinely consumed zero tokens
  from the much more common one that simply never recorded any, which would
  otherwise both render as `0/0/0`.
  """

  prompt: int = 0
  completion: int = 0
  cached: int = 0
  reported: bool = False

  @classmethod
  def from_metrics(
      cls, metrics: trajectory_lib.Metrics | None
  ) -> "TokenCounts":
    """Returns the token counts a single step's metrics record.

    Args:
      metrics: A step's metrics, or None if the step recorded none.

    Returns:
      The counts, with `reported` False unless at least one token field is set,
      so metrics that only carry e.g. `cost_usd` are not reported as zero.
    """
    if metrics is None or all(
        value is None
        for value in (
            metrics.prompt_tokens,
            metrics.completion_tokens,
            metrics.cached_tokens,
        )
    ):
      return cls()
    return cls(
        prompt=metrics.prompt_tokens or 0,
        completion=metrics.completion_tokens or 0,
        cached=metrics.cached_tokens or 0,
        reported=True,
    )

  @classmethod
  def sum_reported(cls, counts: Iterable["TokenCounts"]) -> "TokenCounts":
    """Returns the sum of the counts that are reported.

    Args:
      counts: Token counts to add up.

    Returns:
      The summed counts, with `reported` False if none of `counts` is reported.
    """
    reported = [c for c in counts if c.reported]
    if not reported:
      return cls()
    return cls(
        prompt=sum(c.prompt for c in reported),
        completion=sum(c.completion for c in reported),
        cached=sum(c.cached for c in reported),
        reported=True,
    )

  @property
  def total(self) -> int:
    """Returns prompt plus completion tokens."""
    return self.prompt + self.completion

  def to_json_dict(self) -> dict[str, Any]:
    """Returns the counts as a plain JSON-serializable dict, or nulls."""
    if not self.reported:
      return {"prompt": None, "completion": None, "cached": None, "total": None}
    return {
        "prompt": self.prompt,
        "completion": self.completion,
        "cached": self.cached,
        "total": self.total,
    }


@dataclasses.dataclass(frozen=True)
class ToolUsage:
  """Tool calls issued by a trajectory, in invocation order."""

  sequence: tuple[str, ...] = ()

  @property
  def total_calls(self) -> int:
    """Returns the number of tool calls issued."""
    return len(self.sequence)

  @property
  def counts(self) -> dict[str, int]:
    """Returns call counts per tool, ordered by first invocation."""
    return dict(collections.Counter(self.sequence))

  def to_json_dict(self) -> dict[str, Any]:
    """Returns the usage as a plain JSON-serializable dict."""
    return {
        "sequence": list(self.sequence),
        "counts": self.counts,
        "total_calls": self.total_calls,
    }


@dataclasses.dataclass(frozen=True)
class TrajectoryStats:
  """The derived quantities of a single trajectory."""

  trajectory_id: str | None
  session_id: str | None
  agent_name: str
  agent_version: str
  model_name: str | None
  status: str | None
  final_reward: float | None
  steps: StepCounts
  tokens: TokenCounts
  tools: ToolUsage

  def to_json_dict(self) -> dict[str, Any]:
    """Returns the stats as a plain JSON-serializable dict."""
    return {
        "trajectory_id": self.trajectory_id,
        "session_id": self.session_id,
        "agent_name": self.agent_name,
        "agent_version": self.agent_version,
        "model_name": self.model_name,
        "status": self.status,
        "final_reward": self.final_reward,
        "steps": self.steps.to_json_dict(),
        "tokens": self.tokens.to_json_dict(),
        "tools": self.tools.to_json_dict(),
    }


@dataclasses.dataclass(frozen=True)
class RunStats:
  """Totals across every trajectory the reader exposes."""

  trajectories: tuple[TrajectoryStats, ...]

  @property
  def total_steps(self) -> int:
    """Returns the step count summed over all trajectories."""
    return sum(stats.steps.total for stats in self.trajectories)

  @property
  def tokens(self) -> TokenCounts:
    """Returns token counts summed over trajectories that report them."""
    return TokenCounts.sum_reported(s.tokens for s in self.trajectories)

  @property
  def rewards(self) -> tuple[float, ...]:
    """Returns the final reward of every trajectory that reports one."""
    return tuple(
        s.final_reward for s in self.trajectories if s.final_reward is not None
    )

  @property
  def mean_reward(self) -> float | None:
    """Returns the mean final reward, or None if no trajectory reports one."""
    rewards = self.rewards
    if not rewards:
      return None
    return sum(rewards) / len(rewards)

  @property
  def tool_counts(self) -> dict[str, int]:
    """Returns call counts per tool across the run, most frequent first."""
    counter: collections.Counter[str] = collections.Counter()
    for stats in self.trajectories:
      counter.update(stats.tools.sequence)
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

  def to_json_dict(self) -> dict[str, Any]:
    """Returns the totals as a plain JSON-serializable dict."""
    return {
        "trajectories_count": len(self.trajectories),
        "total_steps": self.total_steps,
        "tokens": self.tokens.to_json_dict(),
        "mean_reward": self.mean_reward,
        "rewards_reported": len(self.rewards),
        "tool_counts": self.tool_counts,
    }


def _coerce_float(value: Any) -> float | None:
  """Returns `value` as a float, or None if it is not a real number.

  Args:
    value: An arbitrary value pulled out of an open `extra` map.

  Returns:
    The value as a float, or None if it is absent or not numeric. Booleans are
    rejected because `bool` is an `int` subclass and a flag is not a reward.
  """
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  return float(value)


def _extra_value(extra: dict[str, Any] | None, key: str) -> Any:
  """Returns `extra[key]`, or None if `extra` is absent or lacks the key.

  Args:
    extra: An optional open metadata map.
    key: Key to look up.

  Returns:
    The mapped value, or None.
  """
  if not extra:
    return None
  return extra.get(key)


def _rehydrate_tunix(
    trajectory: trajectory_lib.Trajectory[Any],
) -> trajectory_lib.Trajectory[Any]:
  """Returns `trajectory` as a TunixTrajectory if it is one, else unchanged.

  Stores persist base ATIF, packing Tunix fields into
  `extra[TUNIX_EXTENSIONS_KEY]`, so a Tunix trajectory read back from a store is
  a base Trajectory. `TrajectoryMetadata.resolve_subclass` recognizes those
  packed fields, so such a trajectory is rehydrated here and its `total_reward`
  and `status` are read like those of a TunixTrajectory built in memory.

  Args:
    trajectory: The trajectory to normalize.

  Returns:
    A TunixTrajectory if the trajectory's metadata resolves to
    TunixTrajectoryMetadata (or a subclass), otherwise `trajectory` itself.
  """
  metadata_cls = trajectory_lib.TrajectoryMetadata.resolve_subclass(trajectory)
  if not issubclass(metadata_cls, trajectory_lib.TunixTrajectoryMetadata):
    return trajectory
  return trajectory_lib.TunixTrajectory.from_atif_trajectory(trajectory)


def _extract_reward(trajectory: trajectory_lib.Trajectory[Any]) -> float | None:
  """Returns the final reward a trajectory reports.

  The reward is the Tunix `total_reward` when set; otherwise the root `extra`
  map's reward, then `final_metrics.extra`'s; otherwise, for a TunixTrajectory,
  the sum of per-step environment rewards, which is the episode return.

  Args:
    trajectory: The trajectory to inspect, already passed through
      `_rehydrate_tunix`.

  Returns:
    The final reward, or None if the trajectory does not report one.
  """
  is_tunix = isinstance(trajectory, trajectory_lib.TunixTrajectory)
  if is_tunix and trajectory.total_reward is not None:
    return trajectory.total_reward
  for extra in (
      trajectory.extra,
      trajectory.final_metrics.extra if trajectory.final_metrics else None,
  ):
    reward = _coerce_float(_extra_value(extra, _REWARD_KEY))
    if reward is not None:
      return reward
  if not is_tunix:
    return None
  step_rewards = [
      step.reward
      for step in trajectory.steps
      if isinstance(step, trajectory_lib.TunixEnvStep)
      and step.reward is not None
  ]
  return sum(step_rewards) if step_rewards else None


def _extract_status(trajectory: trajectory_lib.Trajectory[Any]) -> str | None:
  """Returns the run status a trajectory reports.

  The status is the Tunix `status` when set, otherwise the root `extra` map's.

  Args:
    trajectory: The trajectory to inspect, already passed through
      `_rehydrate_tunix`.

  Returns:
    The status, or None if the trajectory does not report one.
  """
  if (
      isinstance(trajectory, trajectory_lib.TunixTrajectory)
      and trajectory.status is not None
  ):
    return trajectory.status
  status = _extra_value(trajectory.extra, _STATUS_KEY)
  return status if isinstance(status, str) else None


def _extract_step_counts(
    steps: Sequence[trajectory_lib.Step],
) -> StepCounts:
  """Returns step totals split by originator.

  Args:
    steps: The trajectory's steps.

  Returns:
    The step counts.
  """
  by_source = collections.Counter(step.source for step in steps)
  return StepCounts(
      total=len(steps),
      agent=by_source[trajectory_lib.Source.AGENT],
      user=by_source[trajectory_lib.Source.USER],
      system=by_source[trajectory_lib.Source.SYSTEM],
  )


def _extract_token_counts(
    trajectory: trajectory_lib.Trajectory[Any],
) -> TokenCounts:
  """Returns token totals for a trajectory.

  `final_metrics` is defined as the sum over steps, so it is authoritative when
  the producer filled it in; per-step metrics are summed otherwise.

  Args:
    trajectory: The trajectory to inspect.

  Returns:
    The token counts, with `reported` False if nothing recorded any tokens.
  """
  final = trajectory.final_metrics
  if final is not None and any(
      value is not None
      for value in (
          final.total_prompt_tokens,
          final.total_completion_tokens,
          final.total_cached_tokens,
      )
  ):
    return TokenCounts(
        prompt=final.total_prompt_tokens or 0,
        completion=final.total_completion_tokens or 0,
        cached=final.total_cached_tokens or 0,
        reported=True,
    )

  return TokenCounts.sum_reported(
      TokenCounts.from_metrics(step.metrics) for step in trajectory.steps
  )


def _extract_tool_usage(
    steps: Sequence[trajectory_lib.Step],
) -> ToolUsage:
  """Returns the tools a trajectory called, in invocation order.

  Args:
    steps: The trajectory's steps.

  Returns:
    The tool usage.
  """
  sequence = []
  for step in steps:
    for call in step.tool_calls or ():
      sequence.append(call.function_name)
  return ToolUsage(sequence=tuple(sequence))


def from_trajectory(
    trajectory: trajectory_lib.Trajectory[Any],
) -> TrajectoryStats:
  """Derives the reported statistics of a single trajectory.

  Args:
    trajectory: The trajectory to summarize, either a TunixTrajectory (possibly
      persisted as base ATIF with packed Tunix fields) or a base trajectory.

  Returns:
    The derived statistics.
  """
  outcome_trajectory = _rehydrate_tunix(trajectory)
  return TrajectoryStats(
      trajectory_id=trajectory.trajectory_id,
      session_id=trajectory.session_id,
      agent_name=trajectory.agent.name,
      agent_version=trajectory.agent.version,
      model_name=trajectory.agent.model_name,
      status=_extract_status(outcome_trajectory),
      final_reward=_extract_reward(outcome_trajectory),
      steps=_extract_step_counts(trajectory.steps),
      tokens=_extract_token_counts(trajectory),
      tools=_extract_tool_usage(trajectory.steps),
  )


def from_trajectories(
    trajectories: Sequence[trajectory_lib.Trajectory[Any]],
) -> RunStats:
  """Derives per-trajectory statistics and the run totals over them.

  Args:
    trajectories: The trajectories to summarize.

  Returns:
    The run statistics.
  """
  return RunStats(trajectories=tuple(from_trajectory(t) for t in trajectories))
