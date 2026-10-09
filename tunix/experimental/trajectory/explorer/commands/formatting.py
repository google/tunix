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

"""Terminal rendering of trajectory diagnostics for Explorer subcommands.

`explorer.diagnostics` only derives quantities and reports absent data as None;
this module decides how those quantities look in a terminal column, rendering
absent data as `MISSING` rather than 0 so it is not mistaken for a measurement.
"""

from tunix.experimental.trajectory.explorer.diagnostics import stats as stats_lib

# Column placeholder for a quantity the trajectory does not report.
MISSING = "-"


def format_reward(reward: float | None) -> str:
  """Formats a final reward for a terminal column.

  Args:
    reward: The reward, or None if unreported.

  Returns:
    The reward to two decimal places, or the missing-value placeholder.
  """
  return MISSING if reward is None else f"{reward:.2f}"


def format_tokens(tokens: stats_lib.TokenCounts) -> str:
  """Formats token counts as `prompt/completion/total`.

  Args:
    tokens: The counts to format.

  Returns:
    The formatted counts, or the missing-value placeholder.
  """
  if not tokens.reported:
    return MISSING
  return f"{tokens.prompt}/{tokens.completion}/{tokens.total}"


def format_steps(steps: stats_lib.StepCounts) -> str:
  """Formats step counts as `total (agent/user/system)`.

  Args:
    steps: The counts to format.

  Returns:
    The formatted counts.
  """
  return f"{steps.total} ({steps.agent}/{steps.user}/{steps.system})"


def format_tool_sequence(tools: stats_lib.ToolUsage, max_shown: int = 0) -> str:
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
