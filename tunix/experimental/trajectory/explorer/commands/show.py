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

"""Shows the steps of a single trajectory."""

import dataclasses
import json
import sys
from typing import Any

import simple_parsing
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.explorer.commands import base
from tunix.experimental.trajectory.explorer.commands import formatting
from tunix.experimental.trajectory.explorer.diagnostics import stats


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShowCommand(base.BaseCommand):
  """Prints a trajectory's header, each of its steps, and its tool sequence."""

  trajectory_id: str = simple_parsing.field(
      help="ID of the trajectory to show."
  )
  max_chars: int = simple_parsing.field(
      default=200,
      help=(
          "Cut messages, reasoning, tool arguments and observations to this"
          " many characters each; 0 (or a negative value) shows them in full."
      ),
  )

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    """Shows the trajectory `trajectory_id`.

    Args:
      reader: The trajectory store reader backend.
      output_json: Whether to format output as machine-readable JSON.

    Returns:
      Process exit code: 0 on success, 1 if the trajectory does not exist.
    """
    try:
      [trajectory] = reader.get_trajectories([self.trajectory_id])
    except store.TrajectoryNotFoundError:
      print(
          "trajectory-explore: error: trajectory not found:"
          f" {self.trajectory_id}",
          file=sys.stderr,
      )
      return 1
    trajectory_stats = stats.TrajectoryStats.from_trajectory(trajectory)
    if output_json:
      print(
          json.dumps({
              "trajectory": trajectory.to_json_dict(),
              "stats": trajectory_stats.to_json_dict(),
          })
      )
      return 0
    print(_render_header(trajectory, trajectory_stats))
    for step in trajectory.steps:
      print()
      print(_render_step(step, self.max_chars))
    if trajectory_stats.tool_call_names:
      print()
      print("Tool sequence: " + " -> ".join(trajectory_stats.tool_call_names))
    return 0


def _render_header(
    trajectory: trajectory_lib.Trajectory[Any],
    trajectory_stats: stats.TrajectoryStats,
) -> str:
  """Renders the trajectory's identity and headline figures."""
  lines = [
      f"Trajectory {trajectory.trajectory_id}",
      f"Session: {formatting.format_optional(trajectory.session_id)}",
      f"Agent: {trajectory.agent.name} {trajectory.agent.version}",
      f"Status: {formatting.format_status(trajectory_stats.status)}",
      (
          "Reward:"
          f" {formatting.format_optional(trajectory_stats.total_reward, '.3g')}"
      ),
      f"Steps: {formatting.format_step_counts(trajectory_stats.steps)}",
      f"Tokens: {formatting.format_token_counts(trajectory_stats.tokens)}",
  ]
  if trajectory.subagent_trajectories:
    ids = [
        formatting.format_optional(t.trajectory_id)
        for t in trajectory.subagent_trajectories
    ]
    lines.append("Subagent trajectories: " + ", ".join(ids))
  return "\n".join(lines)


def _render_step(step: trajectory_lib.Step, max_chars: int) -> str:
  """Renders one step: a title line followed by its indented content."""
  title = f"[{step.step_id}] {step.source}"
  if step.model_name:
    title += f"  {step.model_name}"
  step_tokens = stats.TokenCounts.from_metrics(step.metrics)
  if step_tokens is not None:
    title += f"  tokens {formatting.format_token_counts(step_tokens)}"

  lines = [
      title,
      f"  message: {_truncate_to_line(step.message, max_chars)}",
  ]
  if step.reasoning_content:
    reasoning = _truncate_to_line(step.reasoning_content, max_chars)
    lines.append(f"  reasoning: {reasoning}")
  for call in step.tool_calls or ():
    args = _truncate_to_line(
        json.dumps(call.arguments, sort_keys=True), max_chars
    )
    lines.append(f"  tool call: {call.function_name}({args})")
  for result in step.observation.results if step.observation else ():
    content = _truncate_to_line(result.content or "", max_chars)
    lines.append(f"  observation: {content}")
    for ref in result.subagent_trajectory_ref or ():
      # SubagentTrajectoryRef guarantees at least one of these is set.
      if ref.trajectory_id is not None:
        lines.append(f"    subagent: {ref.trajectory_id}")
      else:
        lines.append(f"    subagent (external): {ref.trajectory_path}")
  return "\n".join(lines)


def _truncate_to_line(text: str, max_chars: int) -> str:
  """Collapses `text` onto one line and cuts it to `max_chars` with an ellipsis.

  Args:
    text: The text to shorten.
    max_chars: Maximum number of characters to keep; 0 or a negative value
      keeps everything.

  Returns:
    The single-line text, at most `max_chars` characters long.
  """
  text = " ".join(text.split())
  if max_chars > 0 and len(text) > max_chars:
    return text[: max_chars - 1] + "…"
  return text
