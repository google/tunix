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

"""Tabulates every trajectory in a run alongside the run totals."""

from collections.abc import Sequence
import dataclasses
import json

import simple_parsing
import termcolor
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.explorer.commands import base
from tunix.experimental.trajectory.explorer.commands import stats as stats_lib

_COLUMN_SEPARATOR = "  "
_HEADERS: tuple[str, ...] = (
    "TRAJECTORY",
    "STATUS",
    "REWARD",
    "STEPS (A/U/S)",
    "TOKENS (P/C/T)",
    "TOOLS",
)


def trajectory_ids(
    metadata: Sequence[trajectory_lib.TrajectoryMetadata],
) -> list[str]:
  """Returns the trajectory ID of every metadata record.

  Args:
    metadata: Metadata records read back from a store.

  Returns:
    The IDs, in the order the store returned them.

  Raises:
    ValueError: If a record has no trajectory ID, which no store should write.
  """
  ids = []
  for meta in metadata:
    if meta.trajectory_id is None:
      raise ValueError(
          f"Store returned metadata without a trajectory_id: {meta}"
      )
    ids.append(meta.trajectory_id)
  return ids


def format_table(rows: Sequence[Sequence[str]]) -> list[str]:
  """Left-aligns `rows` into columns, padding all but the last column.

  Args:
    rows: Rows of cells; every row has the same number of cells.

  Returns:
    One line per row, without trailing whitespace.
  """
  widths = [max(len(cell) for cell in column) for column in zip(*rows)]
  return [
      _COLUMN_SEPARATOR.join(
          cell.ljust(width) for cell, width in zip(row, widths)
      ).rstrip()
      for row in rows
  ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SummaryCommand(base.BaseCommand):
  """Summarizes every trajectory in the run: steps, tokens, tools, reward."""

  max_tools: int = simple_parsing.field(
      default=5,
      help=(
          "Truncate each trajectory's tool call sequence to this many calls in"
          " the human-readable table; 0 shows every call. Ignored with --json."
      ),
  )

  def __post_init__(self) -> None:
    if self.max_tools < 0:
      raise ValueError(f"max_tools must be non-negative, got {self.max_tools}.")

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    """Prints per-trajectory statistics and run totals.

    Args:
      reader: The trajectory store reader backend.
      output_json: Whether to format output as machine-readable JSON.

    Returns:
      Process exit code (0 for success).
    """
    ids = trajectory_ids(reader.get_trajectories_metadata())
    run = stats_lib.from_trajectories(reader.get_trajectories(ids))

    if output_json:
      print(
          json.dumps({
              "run": run.to_json_dict(),
              "trajectories": [s.to_json_dict() for s in run.trajectories],
          })
      )
      return 0

    self._print_run_totals(run)
    if not run.trajectories:
      return 0
    print()
    self._print_trajectory_table(run.trajectories)
    return 0

  def _print_run_totals(self, run: stats_lib.RunStats) -> None:
    """Prints the run-level totals above the per-trajectory table.

    Args:
      run: The run statistics.
    """
    print(
        termcolor.colored(
            f"{len(run.trajectories)} trajectories, {run.total_steps} steps",
            attrs=["bold"],
        )
    )
    print(f"Tokens (P/C/T): {stats_lib.format_tokens(run.tokens)}")
    print(
        f"Mean reward:    {stats_lib.format_reward(run.mean_reward)}"
        f" ({len(run.rewards)}/{len(run.trajectories)} reported)"
    )
    tool_counts = run.tool_counts
    tools = (
        ", ".join(f"{name} x{count}" for name, count in tool_counts.items())
        if tool_counts
        else stats_lib.MISSING
    )
    print(f"Tool calls:     {tools}")

  def _print_trajectory_table(
      self, trajectories: Sequence[stats_lib.TrajectoryStats]
  ) -> None:
    """Prints one row of statistics per trajectory.

    Args:
      trajectories: The per-trajectory statistics, in store order.
    """
    rows = [_HEADERS] + [
        (
            s.trajectory_id,
            s.status or stats_lib.MISSING,
            stats_lib.format_reward(s.final_reward),
            stats_lib.format_steps(s.steps),
            stats_lib.format_tokens(s.tokens),
            stats_lib.format_tool_sequence(s.tools, max_shown=self.max_tools),
        )
        for s in trajectories
    ]
    header, *body = format_table(rows)
    print(termcolor.colored(header, attrs=["bold"]))
    for line in body:
      print(line)
