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

"""Terminal formatting helpers shared by the explorer subcommands."""

from typing import Any

from tunix.experimental.trajectory.explorer.diagnostics import stats

# Rendered in place of a quantity a trajectory does not report.
MISSING = "-"


def format_optional(value: Any, spec: str = "") -> str:
  """Formats `value` with `spec` (as in `format()`), or `MISSING` for None."""
  return MISSING if value is None else format(value, spec)


def format_status(value: stats.TrajectoryStatus | None) -> str:
  """Renders a status by name, or `MISSING` for None."""
  return MISSING if value is None else value.name


def format_step_counts(value: stats.StepCounts) -> str:
  """Formats step counts as `total (agent a, user u, system s, other o)`."""
  return (
      f"{value.total} (agent {value.agent}, user {value.user}, system"
      f" {value.system}, other {value.other})"
  )


def format_token_counts(value: stats.TokenCounts | None) -> str:
  """Formats token counts as `total (prompt+completion)`, or `MISSING`."""
  if value is None:
    return MISSING
  return f"{value.total} ({value.prompt}+{value.completion})"
