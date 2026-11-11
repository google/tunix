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

"""Schema diagnostics for Tunix RL trajectories."""

from collections.abc import Mapping, Sequence
import types

import pandas as pd
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import formatting
from tunix.dist_rl.trajectory.explorer.diagnostics import schema_diagnostics


class TunixDiagnostics(
    schema_diagnostics.SchemaDiagnostics[trajectory_lib.TunixTrajectoryMetadata]
):
  """Diagnostics for Tunix RL trajectories (`TunixTrajectoryMetadata`).

  The `masked_out` column says whether the overlong filter zeroed the
  trajectory's training masks; it is null until the episode is post-processed.
  """

  COLUMN_DTYPES = types.MappingProxyType({"masked_out": "boolean"})

  def compute_run_summary(
      self, df: pd.DataFrame
  ) -> Mapping[str, schema_diagnostics.JsonValue]:
    """Returns how many trajectories were masked out, and how many report it.

    The keys are `num_masked_out` and `num_with_masked_out`, respectively.
    `num_masked_out` is None when no trajectory reports `masked_out`.

    Args:
      df: This class's `masked_out` column, one row per trajectory of the run.
    """
    masked_out = df["masked_out"]
    num_reported = int(masked_out.notna().sum())
    return {
        "num_masked_out": int(masked_out.sum()) if num_reported else None,
        "num_with_masked_out": num_reported,
    }

  def format_summary_totals(
      self,
      run_summary: Mapping[str, schema_diagnostics.JsonValue],
      num_trajectories: int,
  ) -> Sequence[str]:
    """Returns the `Masked out:` line of the run totals."""
    return (
        (
            "Masked out:"
            f" {formatting.format_optional(run_summary['num_masked_out'])}"
            f" reported by {run_summary['num_with_masked_out']}"
            f"/{num_trajectories}"
        ),
    )
