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
import math
import types

import numpy as np
import pandas as pd
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import formatting
from tunix.dist_rl.trajectory.explorer.diagnostics import schema_diagnostics
from tunix.rl.agentic.agents import agent_types

# The statuses, stripped and upper-cased, of the rollouts that are not valid
# for training: unset (blank), still running, or dropped by the trainer.
_INVALID_STATUSES = frozenset({
    "",
    agent_types.TrajectoryStatus.RUNNING.name,
    *(status.name for status in agent_types.INVALID_TRAJECTORY_STATUSES),
})


class TunixDiagnostics(
    schema_diagnostics.SchemaDiagnostics[trajectory_lib.TunixTrajectory]
):
  """Diagnostics for Tunix RL trajectories (`TunixTrajectory`).

  Columns (`trainable_tokens` and `valid` are hidden):
    masked_out: Whether the overlong filter zeroed the trajectory's training
      masks; null until the episode is post-processed.
    trainable_tokens: The number of assistant tokens the loss masks select, the
      sum of the agent steps' `assistant_masks`; null if no step reports masks.
      The overlong filter zeroes only the masks sent to the trainer, not the
      stored ones, so a masked-out rollout keeps its count.
    valid: Whether the rollout is valid for training, judging from what the
      store records: its status is set and is neither RUNNING nor one of
      `agent_types.INVALID_TRAJECTORY_STATUSES`, it was not masked out, it has
      trainable tokens (if it reports masks), and its reward is set and finite.
      The checks mirror the trainer's `agent_types.is_valid_trajectory_dict`
      and finite-reward mask in `payload_builder.create_trainer_payloads`.
      Never null.
  """

  COLUMN_DTYPES = types.MappingProxyType({
      "masked_out": "boolean",
      "trainable_tokens": "Int64",
      "valid": "boolean",
  })
  HIDDEN_COLUMNS = frozenset({"trainable_tokens", "valid"})

  def compute_columns(
      self, record: trajectory_lib.TunixTrajectory
  ) -> Mapping[str, schema_diagnostics.JsonScalar]:
    """Returns the record's `masked_out`, `trainable_tokens` and `valid`."""
    trainable_tokens = _trainable_tokens(record)
    return {
        "masked_out": record.masked_out,
        "trainable_tokens": trainable_tokens,
        "valid": _is_valid(record, trainable_tokens),
    }

  def compute_run_summary(
      self, df: pd.DataFrame
  ) -> Mapping[str, schema_diagnostics.JsonValue]:
    """Returns how many trajectories were masked out, and how many report it.

    The keys are `num_masked_out` and `num_with_masked_out`, respectively.
    `num_masked_out` is None when no trajectory reports `masked_out`.

    Args:
      df: The run's core columns, then this class's columns, one row per
        Tunix trajectory.
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


def _trainable_tokens(trajectory: trajectory_lib.TunixTrajectory) -> int | None:
  """Returns the sum of the agent steps' `assistant_masks`, or None if unset."""
  masks = [
      step.assistant_masks
      for step in trajectory.steps
      if isinstance(step, trajectory_lib.TunixAgentStep)
      and step.assistant_masks is not None
  ]
  return sum(int(np.sum(m)) for m in masks) if masks else None


def _is_valid(
    trajectory: trajectory_lib.TunixTrajectory, trainable_tokens: int | None
) -> bool:
  """Returns the `valid` column of a trajectory; see `TunixDiagnostics`.

  Never None, unlike the other Tunix columns, so `valid.notna()` selects the
  Tunix rows of a run. An unset status or reward makes the rollout invalid,
  while an unset `masked_out` (an episode that was not post-processed) or
  unreported masks skip their check, as the trainer skips the mask check of
  a rollout without masks.

  Args:
    trajectory: The rollout.
    trainable_tokens: Its `trainable_tokens` column (see `_trainable_tokens`).
  """
  status = trajectory.status
  reward = trajectory.total_reward
  if status is None or reward is None:
    return False
  return (
      status.strip().upper() not in _INVALID_STATUSES
      and not trajectory.masked_out
      and (trainable_tokens is None or trainable_tokens > 0)
      and math.isfinite(reward)
  )
