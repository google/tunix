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

"""Summarizes every trajectory in a run."""

import dataclasses
import json

import pandas as pd
from tunix.dist_rl.trajectory import store
from tunix.dist_rl.trajectory.explorer.commands import base
from tunix.dist_rl.trajectory.explorer.commands import formatting
from tunix.dist_rl.trajectory.explorer.diagnostics import stats


@dataclasses.dataclass(frozen=True, kw_only=True)
class SummaryCommand(base.BaseCommand):
  """Prints one line per trajectory plus totals over the run."""

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    """Summarizes all trajectories the reader can see.

    Args:
      reader: The trajectory store reader backend.
      output_json: Whether to format output as machine-readable JSON.

    Returns:
      Process exit code (0 for success).
    """
    # TODO(tunix-dev): `TrajectoryReader.get_trajectories` requires an explicit
    # ID list, so listing the run costs two passes over its metadata (the file
    # and SQL backends re-read every trajectory's metadata in the second call).
    # Extend `get_trajectories` to accept `None` for "all", as
    # `get_trajectories_metadata` already does, and call it directly here.
    ids = [
        m.trajectory_id
        for m in reader.get_trajectories_metadata()
        if m.trajectory_id is not None
    ]
    run_stats = stats.RunStats.from_trajectories(reader.get_trajectories(ids))
    if output_json:
      # `to_dict` yields native Python values (unlike `to_json`, which rounds
      # floats to 10 significant digits); only the nulls need mapping to None.
      records = [
          {k: None if pd.isna(v) else v for k, v in row.items()}
          for row in run_stats.df.to_dict(orient="records")
      ]
      print(
          json.dumps({
              "run": run_stats.to_json_dict(),
              "trajectories": records,
          })
      )
    elif run_stats.num_trajectories == 0:
      print("No trajectories found.")
    else:
      print(_table(run_stats.df), "", *_totals(run_stats), sep="\n")
    return 0


def _table(df: pd.DataFrame) -> str:
  """Renders one row per trajectory; unreported quantities print as `-`."""
  steps = df[["steps_agent", "steps_user", "steps_system", "steps_other"]]
  token_columns = [f"tokens_{f}" for f in stats.TokenCounts._fields]
  tokens = [
      formatting.format_optional(stats.TokenCounts.from_fields(*row))
      for row in df[token_columns].itertuples(index=False)
  ]
  table = pd.DataFrame({
      "trajectory": df["trajectory_id"],
      "status": df["status"],
      "steps": steps.sum(axis=1),
      "agent": df["steps_agent"],
      "env": df["steps_user"] + df["steps_system"],
      "tokens": tokens,
      "tool calls": df["num_tool_calls"],
      "reward": df["total_reward"].map("{:.3g}".format, na_action="ignore"),
  })
  table = table.astype("string").fillna(formatting.MISSING)
  return table.to_string(index=False)


def _totals(run_stats: stats.RunStats) -> list[str]:
  """Returns one line per run-level total."""
  n = run_stats.num_trajectories
  statuses = {s.name: c for s, c in run_stats.status_counts.items()}
  lines = [
      f"Trajectories: {n}",
      f"Steps: {run_stats.steps}",
      (
          f"Tokens: {formatting.format_optional(run_stats.tokens)}"
          f" reported by {run_stats.num_with_tokens}/{n}"
      ),
      (
          f"Statuses: {formatting.format_counts(statuses)}"
          f" reported by {run_stats.num_with_status}/{n}"
      ),
      (
          "Mean reward:"
          f" {formatting.format_optional(run_stats.mean_reward, '.3g')}"
          f" reported by {run_stats.num_with_reward}/{n}"
      ),
  ]
  if run_stats.tool_counts:
    lines.append(
        f"Tool calls: {formatting.format_counts(run_stats.tool_counts)}"
    )
  return lines
