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
from tunix.rl import advantages
from tunix.rl.agentic.agents import agent_types

_RUNNING_STATUS = agent_types.TrajectoryStatus.RUNNING.name
# The statuses, stripped and upper-cased, of the rollouts that are not valid
# for training: unset (blank), still running, or dropped by the trainer.
_INVALID_STATUSES = frozenset({
    "",
    _RUNNING_STATUS,
    *(status.name for status in agent_types.INVALID_TRAJECTORY_STATUSES),
})
_MIN_VALID = advantages.MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE
# Valid rewards whose sample standard deviation is at most this count as equal.
_ZERO_STD_TOLERANCE = 1e-6
# The regimes of complete prompt groups, in display order, with their labels.
_REGIME_LABELS = types.MappingProxyType({
    "contrastive": f"contrastive (valid>={_MIN_VALID}, std>0)",
    "zero_std_dead": f"zero_std_dead (valid>={_MIN_VALID}, std=0, mean<=0)",
    "zero_std_saturated": "zero_std_saturated (std=0, mean>0)",
    "degenerate": f"degenerate (valid<{_MIN_VALID})",
})
# How many groups each sample table lists.
_NUM_SAMPLE_GROUPS = 5
# The sample tables' run summary keys, in display order, with their titles.
_SAMPLE_TITLES = types.MappingProxyType({
    "contrastive": "Sample Contrastive Groups:",
    "dead_or_degenerate": "Sample Dead / Degenerate Groups:",
})
_SAMPLE_HEADERS = (
    "prompt_id",
    "valid",
    "masked",
    "valid_mean",
    "valid_std",
    "agent steps (min/mean/max)",
    "statuses",
)


class TunixDiagnostics(
    schema_diagnostics.SchemaDiagnostics[trajectory_lib.TunixTrajectory]
):
  """Diagnostics for Tunix RL trajectories (`TunixTrajectory`).

  Columns (all but `masked_out` are hidden):
    masked_out: Whether the overlong filter zeroed the trajectory's training
      masks; null until the episode is post-processed.
    prompt_id: The prompt the rollout answered; the rollouts of a prompt form
      its group, whose rewards set their advantages.
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
      "prompt_id": "string",
      "trainable_tokens": "Int64",
      "valid": "boolean",
  })
  HIDDEN_COLUMNS = frozenset({"prompt_id", "trainable_tokens", "valid"})

  def compute_columns(
      self, record: trajectory_lib.TunixTrajectory
  ) -> Mapping[str, schema_diagnostics.JsonScalar]:
    """Returns the record's value of each column."""
    trainable_tokens = _trainable_tokens(record)
    return {
        "masked_out": record.masked_out,
        "prompt_id": record.prompt_id,
        "trainable_tokens": trainable_tokens,
        "valid": _is_valid(record, trainable_tokens),
    }

  def compute_run_summary(
      self, df: pd.DataFrame
  ) -> Mapping[str, schema_diagnostics.JsonValue]:
    """Returns the masked-out counts and the prompt groups of the run.

    The keys are `num_masked_out` (None when no trajectory reports
    `masked_out`), `num_with_masked_out`, `prompt_groups` (see
    `_prompt_groups`) and `sample_groups` (see `_sample_groups`).

    Args:
      df: The run's core columns, then this class's columns, one row per
        Tunix trajectory.
    """
    masked_out = df["masked_out"]
    num_reported = int(masked_out.notna().sum())
    tunix = df[df["valid"].notna()]
    rollouts = tunix[tunix["prompt_id"].notna()]
    groups = _groups(rollouts)
    return {
        "num_masked_out": int(masked_out.sum()) if num_reported else None,
        "num_with_masked_out": num_reported,
        "prompt_groups": _prompt_groups(groups, len(tunix) - len(rollouts)),
        "sample_groups": _sample_groups(groups, rollouts),
    }

  def format_summary_totals(
      self,
      run_summary: Mapping[str, schema_diagnostics.JsonValue],
      num_trajectories: int,
  ) -> Sequence[str]:
    """Returns the masked-out line, the prompt groups and the sample tables."""
    return (
        (
            "Masked out:"
            f" {formatting.format_optional(run_summary['num_masked_out'])}"
            f" reported by {run_summary['num_with_masked_out']}"
            f"/{num_trajectories}"
        ),
        *_format_prompt_groups(_as_dict(run_summary["prompt_groups"])),
        *_format_sample_groups(_as_dict(run_summary["sample_groups"])),
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


def _groups(rollouts: pd.DataFrame) -> pd.DataFrame:
  """Returns the prompt groups of `rollouts`, one row each, in first-seen order.

  A group with a RUNNING rollout is incomplete. A complete group gets the first
  regime that applies, from the rewards of its valid rollouts:

    degenerate: fewer than `MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE` of them;
    contrastive: rewards that vary (sample standard deviation above 1e-6);
    zero_std_saturated: equal rewards above 0;
    zero_std_dead: equal rewards of 0 or less.

  Args:
    rollouts: The run's Tunix rows that have a `prompt_id`: the core columns,
      then the Tunix columns.

  Returns:
    One row per `prompt_id` (the index): `valid` and `masked`, the numbers of
    valid and masked-out rollouts; `valid_mean` and `valid_std`, the mean and
    sample standard deviation of the valid rollouts' rewards; `steps_min`,
    `steps_mean` and `steps_max`, over the agent steps of every rollout; and
    `regime`, which is "incomplete" for an incomplete group.
  """
  valid = rollouts["valid"].astype(bool)
  running = rollouts["status"].eq(_RUNNING_STATUS).fillna(False).astype(bool)
  groups = (
      pd.DataFrame({
          "prompt_id": rollouts["prompt_id"],
          "running": running,
          "valid": valid,
          "masked": rollouts["masked_out"].fillna(False).astype(bool),
          # Only the rewards of valid rollouts count.
          "reward": rollouts["total_reward"].where(valid),
          "steps": rollouts["steps_agent"],
      })
      .groupby("prompt_id", sort=False)
      .agg(
          running=("running", "any"),
          valid=("valid", "sum"),
          masked=("masked", "sum"),
          valid_mean=("reward", "mean"),
          valid_std=("reward", "std"),
          steps_min=("steps", "min"),
          steps_mean=("steps", "mean"),
          steps_max=("steps", "max"),
      )
  )
  groups["regime"] = np.select(
      [
          groups["running"].to_numpy(dtype=bool),
          (groups["valid"] < _MIN_VALID).to_numpy(dtype=bool),
          (groups["valid_std"] > _ZERO_STD_TOLERANCE).to_numpy(dtype=bool),
          (groups["valid_mean"] > 0).to_numpy(dtype=bool),
      ],
      ["incomplete", "degenerate", "contrastive", "zero_std_saturated"],
      default="zero_std_dead",
  )
  return groups


def _prompt_groups(
    groups: pd.DataFrame, num_ungrouped: int
) -> dict[str, schema_diagnostics.JsonValue]:
  """Counts the run's prompt groups by regime.

  Args:
    groups: The run's prompt groups, from `_groups`.
    num_ungrouped: The number of the run's Tunix rollouts without a `prompt_id`.

  Returns:
    `num_groups`; `num_incomplete`, the number of incomplete groups;
    `num_ungrouped`; and `regimes`, the number of complete groups in each
    regime.
  """
  regimes = groups["regime"]
  regime_counts: dict[str, schema_diagnostics.JsonValue] = {
      regime: int((regimes == regime).sum()) for regime in _REGIME_LABELS
  }
  return {
      "num_groups": len(groups),
      "num_incomplete": int((regimes == "incomplete").sum()),
      "num_ungrouped": num_ungrouped,
      "regimes": regime_counts,
  }


def _sample_groups(
    groups: pd.DataFrame, rollouts: pd.DataFrame
) -> dict[str, schema_diagnostics.JsonValue]:
  """Returns up to 5 contrastive groups and 5 dead or degenerate ones.

  Contrastive groups come by decreasing `valid_std`, dead and degenerate ones
  by decreasing number of masked-out rollouts, and ties by `prompt_id`.

  Args:
    groups: The run's prompt groups, from `_groups`.
    rollouts: The run's Tunix rows that have a `prompt_id`.

  Returns:
    `contrastive` and `dead_or_degenerate`, each a list of groups with their
    `prompt_id`, `valid`, `masked`, `valid_mean`, `valid_std` (None without
    enough valid rollouts), `agent_steps` (`min`, `mean` and `max`), and
    `statuses`, the number of rollouts per status, most frequent first.
  """
  table = groups.reset_index()
  contrastive = table[table["regime"] == "contrastive"].sort_values(
      ["valid_std", "prompt_id"], ascending=[False, True]
  )
  dead_or_degenerate = table[
      table["regime"].isin(["zero_std_dead", "degenerate"])
  ].sort_values(["masked", "prompt_id"], ascending=[False, True])
  return {
      "contrastive": _sample_rows(
          contrastive.head(_NUM_SAMPLE_GROUPS), rollouts
      ),
      "dead_or_degenerate": _sample_rows(
          dead_or_degenerate.head(_NUM_SAMPLE_GROUPS), rollouts
      ),
  }


def _sample_rows(
    groups: pd.DataFrame, rollouts: pd.DataFrame
) -> list[schema_diagnostics.JsonValue]:
  """Returns the run summary rows of a few groups; see `_sample_groups`."""
  rows: list[schema_diagnostics.JsonValue] = []
  for group in groups.to_dict(orient="records"):
    statuses = (
        rollouts.loc[rollouts["prompt_id"] == group["prompt_id"], "status"]
        .value_counts()
        .sort_index()
        .sort_values(ascending=False, kind="stable")
    )
    # Without enough valid rollouts, the mean or standard deviation is NaN.
    mean, std = group["valid_mean"], group["valid_std"]
    rows.append({
        "prompt_id": str(group["prompt_id"]),
        "valid": int(group["valid"]),
        "masked": int(group["masked"]),
        "valid_mean": None if math.isnan(mean) else float(mean),
        "valid_std": None if math.isnan(std) else float(std),
        "agent_steps": {
            "min": int(group["steps_min"]),
            "mean": float(group["steps_mean"]),
            "max": int(group["steps_max"]),
        },
        "statuses": {str(s): int(n) for s, n in statuses.items()},
    })
  return rows


def _format_prompt_groups(
    groups: Mapping[str, schema_diagnostics.JsonValue],
) -> list[str]:
  """Returns the `Prompt Groups:` line, then one line per regime.

  The regime lines give each regime's share of the complete groups, and are
  left out when no group is complete.

  Args:
    groups: The `prompt_groups` of the run summary.
  """
  num_groups = _as_int(groups["num_groups"])
  num_incomplete = _as_int(groups["num_incomplete"])
  lines = [
      f"Prompt Groups: {num_groups} ({num_incomplete} incomplete,"
      f" {groups['num_ungrouped']} ungrouped trajectories)"
  ]
  num_complete = num_groups - num_incomplete
  if not num_complete:
    return lines
  regimes = _as_dict(groups["regimes"])
  label_width = max(map(len, _REGIME_LABELS.values())) + 1
  count_width = len(str(num_complete))
  for regime, label in _REGIME_LABELS.items():
    count = _as_int(regimes[regime])
    lines.append(
        f"  {label + ':':<{label_width}} {count:>{count_width}}"
        f" ({100 * count / num_complete:4.1f}%)"
    )
  return lines


def _format_sample_groups(
    samples: Mapping[str, schema_diagnostics.JsonValue],
) -> list[str]:
  """Returns a titled table for each non-empty list of sample groups.

  Args:
    samples: The `sample_groups` of the run summary.
  """
  lines = []
  for key, title in _SAMPLE_TITLES.items():
    rows = []
    for row in _as_list(samples[key]):
      group = _as_dict(row)
      steps = _as_dict(group["agent_steps"])
      statuses = _as_dict(group["statuses"])
      rows.append([
          str(group["prompt_id"]),
          str(group["valid"]),
          str(group["masked"]),
          formatting.format_optional(group["valid_mean"], ".3f"),
          formatting.format_optional(group["valid_std"], ".3f"),
          (
              f"{steps['min']} /"
              f" {formatting.format_optional(steps['mean'], '.1f')} /"
              f" {steps['max']}"
          ),
          formatting.format_counts(
              {s: _as_int(n) for s, n in statuses.items()}
          ),
      ])
    if rows:
      lines.extend([title, *_format_table(_SAMPLE_HEADERS, rows)])
  return lines


def _format_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> list[str]:
  """Returns `headers`, then `rows`, in left-aligned columns indented by 2."""
  widths = [max(map(len, column)) for column in zip(headers, *rows)]
  return [
      "  "
      + "  ".join(
          cell.ljust(width) for cell, width in zip(row, widths)
      ).rstrip()
      for row in (headers, *rows)
  ]


def _as_dict(
    value: schema_diagnostics.JsonValue,
) -> Mapping[str, schema_diagnostics.JsonValue]:
  """Returns `value`, a mapping of the run summary, or raises TypeError."""
  if not isinstance(value, dict):
    raise TypeError(f"Expected a dict in the run summary, got {value!r}.")
  return value


def _as_list(
    value: schema_diagnostics.JsonValue,
) -> list[schema_diagnostics.JsonValue]:
  """Returns `value`, a list of the run summary, or raises TypeError."""
  if not isinstance(value, list):
    raise TypeError(f"Expected a list in the run summary, got {value!r}.")
  return value


def _as_int(value: schema_diagnostics.JsonValue) -> int:
  """Returns `value`, a count of the run summary, or raises TypeError."""
  if not isinstance(value, int):
    raise TypeError(f"Expected an int in the run summary, got {value!r}.")
  return value
