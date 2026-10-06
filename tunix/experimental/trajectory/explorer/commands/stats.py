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
`summary` and `show` views render identically without traversing steps. Because
metrics are optional, missing fields default to None and render as - instead of
0 to differentiate absence of data from zero measurements.

Where a trajectory records its outcome (final reward and run status) depends on
its TrajectoryMetadata subclass, so outcomes are read by an `OutcomeExtractor`
registered per `METADATA_TYPE`; everything else is base ATIF and is read the
same way for every metadata type.
"""

import abc
import collections
from collections.abc import Sequence
import dataclasses
from typing import Any, ClassVar

from absl import logging
from tunix.experimental.trajectory import trajectory as trajectory_lib

# Base ATIF has no reward or status field, so producers that are not Tunix pass
# them through the open `extra` maps under these keys.
_REWARD_KEY = "reward"
_STATUS_KEY = "status"

# Column placeholder for a quantity the trajectory does not report.
MISSING = "-"


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
  """The quantities `summary` tabulates and `show` puts in its header."""

  trajectory_id: str
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
    reported_tokens = [s.tokens for s in self.trajectories if s.tokens.reported]
    if not reported_tokens:
      return TokenCounts()
    return TokenCounts(
        prompt=sum(t.prompt for t in reported_tokens),
        completion=sum(t.completion for t in reported_tokens),
        cached=sum(t.cached for t in reported_tokens),
        reported=True,
    )

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


@dataclasses.dataclass(frozen=True)
class Outcome:
  """How a trajectory ended, as far as the trajectory reports it."""

  final_reward: float | None = None
  status: str | None = None


class OutcomeExtractor(abc.ABC):
  """Reads the outcome a TrajectoryMetadata subclass records.

  Base ATIF has no reward or status field, so where a trajectory records them
  depends on its metadata type: TunixTrajectoryMetadata carries first-class
  `total_reward` and `status`, while base TrajectoryMetadata can only pass them
  through its open `extra` maps.

  Subclasses register themselves for the metadata type they read by declaring
  `METADATA_TYPE`, a name registered in `TrajectoryMetadata._REGISTRY`.
  `for_trajectory` resolves the extractor for the nearest metadata type in the
  trajectory's class hierarchy, so a TrajectoryMetadata subclass without an
  extractor of its own is read like its closest registered ancestor. Base
  TrajectoryMetadata is always registered, so every trajectory resolves to one.
  """

  METADATA_TYPE: ClassVar[str]
  _REGISTRY: ClassVar[dict[str, type["OutcomeExtractor"]]] = {}

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Registers `cls` for the metadata type it declares.

    Args:
      **kwargs: Forwarded to `super().__init_subclass__`.

    Raises:
      TypeError: If `cls` does not declare its own `METADATA_TYPE`.
      ValueError: If `METADATA_TYPE` names no registered TrajectoryMetadata
        subclass, or another extractor is already registered for it.
    """
    super().__init_subclass__(**kwargs)
    if "METADATA_TYPE" not in vars(cls):
      raise TypeError(f"{cls.__qualname__} must declare METADATA_TYPE.")
    meta_type = cls.METADATA_TYPE
    metadata_registry = (
        trajectory_lib.TrajectoryMetadata._REGISTRY  # pylint: disable=protected-access
    )
    if meta_type not in metadata_registry:
      raise ValueError(
          f"{cls.__qualname__} declares METADATA_TYPE {meta_type!r}, which is"
          f" not a registered metadata type; expected one of"
          f" {sorted(metadata_registry)}."
      )
    if meta_type in cls._REGISTRY:
      raise ValueError(
          f"An OutcomeExtractor for METADATA_TYPE {meta_type!r} is already"
          f" registered: {cls._REGISTRY[meta_type].__qualname__}; cannot"
          f" register {cls.__qualname__}."
      )
    cls._REGISTRY[meta_type] = cls

  @classmethod
  def for_trajectory(
      cls, trajectory: trajectory_lib.Trajectory[Any]
  ) -> "OutcomeExtractor":
    """Returns the extractor for the trajectory's nearest metadata type.

    Args:
      trajectory: The trajectory whose outcome is to be read.

    Returns:
      A new instance of the extractor registered for the first class in the
      trajectory's MRO that declares a `METADATA_TYPE` with an extractor.

    Raises:
      ValueError: If no class in the trajectory's MRO has an extractor, which
        only happens if the base extractor failed to register.
    """
    for klass in type(trajectory).__mro__:
      if "METADATA_TYPE" not in vars(klass):
        continue
      meta_type = vars(klass)["METADATA_TYPE"]
      if meta_type in cls._REGISTRY:
        return cls._REGISTRY[meta_type]()
    raise ValueError(
        f"No OutcomeExtractor is registered for {type(trajectory).__name__};"
        f" registered metadata types: {sorted(cls._REGISTRY)}."
    )

  @abc.abstractmethod
  def extract(self, trajectory: trajectory_lib.Trajectory[Any]) -> Outcome:
    """Returns the outcome `trajectory` records.

    Args:
      trajectory: A trajectory of this extractor's metadata type, or of a
        subclass of it.

    Returns:
      The final reward and run status, each None if unreported.
    """


class BaseOutcomeExtractor(OutcomeExtractor):
  """Reads the outcome of a base ATIF trajectory from its open `extra` maps."""

  METADATA_TYPE: ClassVar[str] = trajectory_lib.TrajectoryMetadata.METADATA_TYPE

  def extract(self, trajectory: trajectory_lib.Trajectory[Any]) -> Outcome:
    """Returns the reward and status passed through the `extra` maps.

    The reward is read from the root `extra` map, then from
    `final_metrics.extra`; the status only from the root `extra` map.

    Args:
      trajectory: The trajectory to inspect.

    Returns:
      The outcome, with each field None if no `extra` map reports it.
    """
    final_reward = None
    for extra in (
        trajectory.extra,
        trajectory.final_metrics.extra if trajectory.final_metrics else None,
    ):
      final_reward = _coerce_float(_extra_value(extra, _REWARD_KEY))
      if final_reward is not None:
        break
    status = _extra_value(trajectory.extra, _STATUS_KEY)
    return Outcome(
        final_reward=final_reward,
        status=status if isinstance(status, str) else None,
    )


class TunixOutcomeExtractor(BaseOutcomeExtractor):
  """Reads the outcome of a TunixTrajectory from its first-class fields."""

  METADATA_TYPE: ClassVar[str] = (
      trajectory_lib.TunixTrajectoryMetadata.METADATA_TYPE
  )

  def extract(self, trajectory: trajectory_lib.Trajectory[Any]) -> Outcome:
    """Returns the Tunix reward and status, falling back to the `extra` maps.

    The reward is `total_reward` when set; otherwise whatever the base `extra`
    maps report; otherwise the sum of per-step environment rewards, which is
    the episode return. The status is `status` when set, otherwise the base
    `extra` status.

    Args:
      trajectory: A TunixTrajectory, or a 1-indexed ATIF trajectory to
        rehydrate into one.

    Returns:
      The outcome, with each field None if the trajectory does not report it.
    """
    tunix_trajectory = trajectory_lib.TunixTrajectory.from_atif_trajectory(
        trajectory
    )
    from_extra = super().extract(tunix_trajectory)

    final_reward = tunix_trajectory.total_reward
    if final_reward is None:
      final_reward = from_extra.final_reward
    if final_reward is None:
      step_rewards = [
          step.reward
          for step in tunix_trajectory.steps
          if isinstance(step, trajectory_lib.TunixEnvStep)
          and step.reward is not None
      ]
      if step_rewards:
        final_reward = sum(step_rewards)

    status = tunix_trajectory.status
    if status is None:
      status = from_extra.status
    return Outcome(final_reward=final_reward, status=status)


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

  per_step = [step.metrics for step in trajectory.steps if step.metrics]
  if not per_step:
    return TokenCounts()
  return TokenCounts(
      prompt=sum(m.prompt_tokens or 0 for m in per_step),
      completion=sum(m.completion_tokens or 0 for m in per_step),
      cached=sum(m.cached_tokens or 0 for m in per_step),
      reported=True,
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


def _unread_extension_fields(
    trajectory: trajectory_lib.Trajectory[Any],
) -> frozenset[str]:
  """Returns subclass fields a base-typed trajectory leaves packed in `extra`.

  A store reading metadata back as base TrajectoryMetadata keeps the fields of
  whatever subclass wrote it (e.g. Tunix `total_reward` and `status`) packed
  under `extra[TUNIX_EXTENSIONS_KEY]`, where no outcome extractor reads them.

  Args:
    trajectory: The trajectory to inspect.

  Returns:
    The packed field names, or an empty set if `trajectory` was read as a
    metadata subclass or carries no packed fields.
  """
  base_type = trajectory_lib.TrajectoryMetadata.METADATA_TYPE
  if trajectory.METADATA_TYPE != base_type:
    return frozenset()
  return frozenset(trajectory.get_extensions())


def _metadata_types_with_fields(fields: frozenset[str]) -> list[str]:
  """Returns the registered metadata subclasses declaring all of `fields`.

  Args:
    fields: Field names packed by the subclass that wrote the trajectories.

  Returns:
    Sorted METADATA_TYPE names of every non-base registered subclass whose
    model declares each of `fields`.
  """
  metadata_registry = (
      trajectory_lib.TrajectoryMetadata._REGISTRY  # pylint: disable=protected-access
  )
  return sorted(
      meta_type
      for meta_type, metadata_cls in metadata_registry.items()
      if meta_type != trajectory_lib.TrajectoryMetadata.METADATA_TYPE
      and fields <= metadata_cls.model_fields.keys()
  )


def _warn_on_unread_extensions(
    trajectories: Sequence[trajectory_lib.Trajectory[Any]],
) -> None:
  """Logs one warning if any trajectory was read as base over subclass data.

  Without it, a Tunix run read with the default `--metadata_type base` shows
  `-` for its reward and status, which reads as "not reported" even though the
  values are on disk.

  Args:
    trajectories: The trajectories about to be summarized.
  """
  affected = 0
  fields: set[str] = set()
  for trajectory in trajectories:
    unread = _unread_extension_fields(trajectory)
    if unread:
      affected += 1
      fields |= unread
  if not affected:
    return
  candidates = _metadata_types_with_fields(frozenset(fields))
  suggestion = (
      " or ".join(f"--metadata_type {c}" for c in candidates)
      if candidates
      else "--metadata_type set to the metadata type that wrote them"
  )
  logging.warning(
      "%d of %d trajectories were read as metadata type %r but carry fields of"
      " a metadata subclass packed in extra[%r] (%s), so their reward and"
      " status are not read and show as %r. Re-run with %s.",
      affected,
      len(trajectories),
      trajectory_lib.TrajectoryMetadata.METADATA_TYPE,
      trajectory_lib.TUNIX_EXTENSIONS_KEY,
      ", ".join(sorted(fields)),
      MISSING,
      suggestion,
  )


def _derive_stats(
    trajectory: trajectory_lib.Trajectory[Any],
) -> TrajectoryStats:
  """Derives the reported statistics of a single trajectory, without warning.

  Args:
    trajectory: The trajectory to summarize.

  Returns:
    The derived statistics.
  """
  outcome = OutcomeExtractor.for_trajectory(trajectory).extract(trajectory)
  return TrajectoryStats(
      trajectory_id=trajectory.trajectory_id or MISSING,
      session_id=trajectory.session_id,
      agent_name=trajectory.agent.name,
      agent_version=trajectory.agent.version,
      model_name=trajectory.agent.model_name,
      status=outcome.status,
      final_reward=outcome.final_reward,
      steps=_extract_step_counts(trajectory.steps),
      tokens=_extract_token_counts(trajectory),
      tools=_extract_tool_usage(trajectory.steps),
  )


def from_trajectory(
    trajectory: trajectory_lib.Trajectory[Any],
) -> TrajectoryStats:
  """Derives the reported statistics of a single trajectory.

  Logs a warning if the trajectory was read as base metadata but carries the
  packed fields of a metadata subclass, whose outcome then goes unreported.

  Args:
    trajectory: The trajectory to summarize, of any TrajectoryMetadata subclass;
      its outcome is read by the `OutcomeExtractor` for its metadata type.

  Returns:
    The derived statistics.
  """
  _warn_on_unread_extensions([trajectory])
  return _derive_stats(trajectory)


def from_trajectories(
    trajectories: Sequence[trajectory_lib.Trajectory[Any]],
) -> RunStats:
  """Derives per-trajectory statistics and the run totals over them.

  Logs at most one warning for the whole run if any trajectory was read as
  base metadata but carries the packed fields of a metadata subclass.

  Args:
    trajectories: The trajectories to summarize.

  Returns:
    The run statistics.
  """
  _warn_on_unread_extensions(trajectories)
  return RunStats(trajectories=tuple(_derive_stats(t) for t in trajectories))


def format_reward(reward: float | None) -> str:
  """Formats a final reward for a terminal column.

  Args:
    reward: The reward, or None if unreported.

  Returns:
    The reward to two decimal places, or the missing-value placeholder.
  """
  return MISSING if reward is None else f"{reward:.2f}"


def format_tokens(tokens: TokenCounts) -> str:
  """Formats token counts as `prompt/completion/total`.

  Args:
    tokens: The counts to format.

  Returns:
    The formatted counts, or the missing-value placeholder.
  """
  if not tokens.reported:
    return MISSING
  return f"{tokens.prompt}/{tokens.completion}/{tokens.total}"


def format_steps(steps: StepCounts) -> str:
  """Formats step counts as `total (agent/user/system)`.

  Args:
    steps: The counts to format.

  Returns:
    The formatted counts.
  """
  return f"{steps.total} ({steps.agent}/{steps.user}/{steps.system})"


def format_tool_sequence(tools: ToolUsage, max_shown: int = 0) -> str:
  """Formats a tool call sequence as an arrow-joined chain.

  Args:
    tools: The usage to format.
    max_shown: Truncate to this many calls, appending a count of the remainder.
      Zero, the default, shows the whole sequence.

  Returns:
    The formatted sequence, or the missing-value placeholder.
  """
  sequence = tools.sequence
  if not sequence:
    return MISSING
  if 0 < max_shown < len(sequence):
    shown = " -> ".join(sequence[:max_shown])
    return f"{shown} -> ... (+{len(sequence) - max_shown} more)"
  return " -> ".join(sequence)
