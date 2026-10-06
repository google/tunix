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

"""Displays a single trajectory step by step under a header of its stats."""

import dataclasses
import json
import sys
from typing import Any

import simple_parsing
import termcolor
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.explorer.commands import base
from tunix.experimental.trajectory.explorer.commands import stats as stats_lib

_SOURCE_COLORS: dict[trajectory_lib.Source, str] = {
    trajectory_lib.Source.AGENT: "cyan",
    trajectory_lib.Source.USER: "green",
    trajectory_lib.Source.SYSTEM: "yellow",
}


@dataclasses.dataclass(frozen=True)
class StepView:
  """What `show` reports about a single step."""

  step_id: int
  source: trajectory_lib.Source
  message: str
  tool_calls: tuple[str, ...]
  tokens: stats_lib.TokenCounts
  reward: float | None

  def to_json_dict(self) -> dict[str, Any]:
    """Returns the step as a plain JSON-serializable dict."""
    return {
        "step_id": self.step_id,
        "source": self.source.value,
        "message": self.message,
        "tool_calls": list(self.tool_calls),
        "tokens": self.tokens.to_json_dict(),
        "reward": self.reward,
    }


def step_view(step: trajectory_lib.Step) -> StepView:
  """Derives what `show` reports about `step`.

  Args:
    step: A step of the trajectory being shown.

  Returns:
    The step view. The reward is only reported by Tunix environment steps.
  """
  metrics = step.metrics
  if metrics is None or all(
      value is None
      for value in (
          metrics.prompt_tokens,
          metrics.completion_tokens,
          metrics.cached_tokens,
      )
  ):
    tokens = stats_lib.TokenCounts()
  else:
    tokens = stats_lib.TokenCounts(
        prompt=metrics.prompt_tokens or 0,
        completion=metrics.completion_tokens or 0,
        cached=metrics.cached_tokens or 0,
        reported=True,
    )
  reward = (
      step.reward if isinstance(step, trajectory_lib.TunixEnvStep) else None
  )
  return StepView(
      step_id=step.step_id,
      source=step.source,
      message=step.message,
      tool_calls=tuple(call.function_name for call in step.tool_calls or ()),
      tokens=tokens,
      reward=reward,
  )


def truncate(text: str, max_chars: int) -> str:
  """Collapses `text` onto one line and truncates it to `max_chars`.

  Args:
    text: The text to shorten.
    max_chars: Maximum characters kept; 0 keeps the whole text.

  Returns:
    The single-line text, ending in `...` if it was truncated.
  """
  line = " ".join(text.split())
  if 0 < max_chars < len(line):
    return line[:max_chars] + "..."
  return line


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShowCommand(base.BaseCommand):
  """Shows one trajectory's stats and its steps in order."""

  trajectory_id: str = simple_parsing.field(
      help="Unique identifier of the trajectory to inspect."
  )
  max_message_chars: int = simple_parsing.field(
      default=120,
      help=(
          "Truncate each step message to this many characters in the"
          " human-readable view; 0 shows full messages. Ignored with --json."
      ),
  )

  def __post_init__(self) -> None:
    if self.max_message_chars < 0:
      raise ValueError(
          "max_message_chars must be non-negative, got"
          f" {self.max_message_chars}."
      )

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    """Prints the trajectory's stats header followed by its steps.

    Args:
      reader: The trajectory store reader backend.
      output_json: Whether to format output as machine-readable JSON.

    Returns:
      Process exit code: 0 on success, 2 if the trajectory does not exist.
    """
    try:
      (trajectory,) = reader.get_trajectories([self.trajectory_id])
    except store.TrajectoryNotFoundError as error:
      # A mistyped ID is a usage error, reported like one on stderr.
      print(f"error: {error.args[0]}", file=sys.stderr)
      return 2

    stats = stats_lib.from_trajectory(trajectory)
    steps = [step_view(step) for step in trajectory.steps]

    if output_json:
      print(
          json.dumps({
              "trajectory": stats.to_json_dict(),
              "steps": [view.to_json_dict() for view in steps],
          })
      )
      return 0

    self._print_header(stats)
    for view in steps:
      print()
      self._print_step(view)
    return 0

  def _print_header(self, stats: stats_lib.TrajectoryStats) -> None:
    """Prints the trajectory's identity and derived statistics.

    Args:
      stats: The trajectory's statistics.
    """
    print(
        termcolor.colored(f"Trajectory {stats.trajectory_id}", attrs=["bold"])
    )
    print(f"  Session:        {stats.session_id or stats_lib.MISSING}")
    print(
        f"  Agent:          {stats.agent_name} {stats.agent_version}"
        f" (model: {stats.model_name or stats_lib.MISSING})"
    )
    print(f"  Status:         {stats.status or stats_lib.MISSING}")
    print(f"  Final reward:   {stats_lib.format_reward(stats.final_reward)}")
    print(f"  Steps (A/U/S):  {stats_lib.format_steps(stats.steps)}")
    print(f"  Tokens (P/C/T): {stats_lib.format_tokens(stats.tokens)}")
    print(f"  Tool calls:     {stats_lib.format_tool_sequence(stats.tools)}")

  def _print_step(self, view: StepView) -> None:
    """Prints a single step.

    Args:
      view: What to report about the step.
    """
    label = termcolor.colored(
        f"[{view.step_id}] {view.source.value}",
        _SOURCE_COLORS[view.source],
        attrs=["bold"],
    )
    print(f"{label}: {truncate(view.message, self.max_message_chars)}")
    if view.tool_calls:
      print(f"    tools:  {' -> '.join(view.tool_calls)}")
    if view.tokens.reported:
      print(f"    tokens: {stats_lib.format_tokens(view.tokens)}")
    if view.reward is not None:
      print(f"    reward: {stats_lib.format_reward(view.reward)}")
