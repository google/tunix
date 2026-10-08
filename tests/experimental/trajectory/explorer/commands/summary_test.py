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

"""Unit tests for SummaryCommand execution."""

import json

from absl.testing import absltest
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.explorer.commands import summary
from tunix.experimental.trajectory.explorer.commands import testing

_AGENT = trajectory_lib.Agent(name="agent", version="1.0")


def _tool_call(name: str) -> trajectory_lib.ToolCall:
  return trajectory_lib.ToolCall(
      tool_call_id=f"call_{name}", function_name=name, arguments={}
  )


# Reports a reward, a status, tokens, and three tool calls.
_REWARDED = trajectory_lib.Trajectory(
    trajectory_id="traj_rewarded",
    agent=_AGENT,
    extra={"reward": 1.0, "status": "COMPLETED"},
    steps=[
        trajectory_lib.Step(
            step_id=1, source=trajectory_lib.Source.USER, message="task"
        ),
        trajectory_lib.Step(
            step_id=2,
            source=trajectory_lib.Source.AGENT,
            message="calling tools",
            tool_calls=[_tool_call("search"), _tool_call("read")],
            metrics=trajectory_lib.Metrics(
                prompt_tokens=100, completion_tokens=20
            ),
        ),
        trajectory_lib.Step(
            step_id=3,
            source=trajectory_lib.Source.AGENT,
            message="calling again",
            tool_calls=[_tool_call("search")],
            metrics=trajectory_lib.Metrics(
                prompt_tokens=150, completion_tokens=30
            ),
        ),
    ],
)

# Reports nothing beyond its single step.
_BARE = trajectory_lib.Trajectory(
    trajectory_id="traj_bare",
    agent=_AGENT,
    steps=[
        trajectory_lib.Step(
            step_id=1, source=trajectory_lib.Source.AGENT, message="done"
        )
    ],
)


class SummaryTest(absltest.TestCase):

  def test_json_reports_run_totals_and_each_trajectory(self) -> None:
    mem_store = testing.create_store(trajectories=[_REWARDED, _BARE])

    exit_code, out = testing.execute_command(
        summary.SummaryCommand(), mem_store, output_json=True
    )

    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(
        data["run"],
        {
            "trajectories_count": 2,
            "total_steps": 4,
            "tokens": {
                "prompt": 250,
                "completion": 50,
                "cached": 0,
                "total": 300,
            },
            "mean_reward": 1.0,
            "rewards_reported": 1,
            "tool_counts": {"search": 2, "read": 1},
        },
    )
    by_id = {t["trajectory_id"]: t for t in data["trajectories"]}
    self.assertCountEqual(by_id, ["traj_rewarded", "traj_bare"])
    self.assertEqual(by_id["traj_rewarded"]["final_reward"], 1.0)
    self.assertEqual(by_id["traj_rewarded"]["status"], "COMPLETED")
    self.assertEqual(
        by_id["traj_rewarded"]["tools"]["sequence"],
        ["search", "read", "search"],
    )
    self.assertIsNone(by_id["traj_bare"]["final_reward"])
    self.assertIsNone(by_id["traj_bare"]["tokens"]["total"])

  def test_human_readable_tabulates_each_trajectory(self) -> None:
    mem_store = testing.create_store(trajectories=[_REWARDED, _BARE])

    exit_code, out = testing.execute_command(
        summary.SummaryCommand(), mem_store, output_json=False
    )

    self.assertEqual(exit_code, 0)
    self.assertIn("2 trajectories, 4 steps", out)
    self.assertIn("Tokens (P/C/T): 250/50/300", out)
    self.assertIn("Mean reward:    1.00 (1/2 reported)", out)
    self.assertIn("Tool calls:     search x2, read x1", out)
    lines = out.splitlines()
    rewarded_row = next(l for l in lines if l.startswith("traj_rewarded"))
    self.assertEqual(
        rewarded_row.split(),
        [
            "traj_rewarded",
            "COMPLETED",
            "1.00",
            "3",
            "(2/1/0)",
            "250/50/300",
            "search",
            "->",
            "read",
            "->",
            "search",
        ],
    )
    bare_row = next(l for l in lines if l.startswith("traj_bare"))
    self.assertEqual(
        bare_row.split(), ["traj_bare", "-", "-", "1", "(1/0/0)", "-", "-"]
    )

  def test_max_tools_truncates_the_tool_sequence(self) -> None:
    mem_store = testing.create_store(trajectories=[_REWARDED])

    _, out = testing.execute_command(
        summary.SummaryCommand(max_tools=1), mem_store, output_json=False
    )

    self.assertIn("search -> ... (+2 more)", out)

  def test_empty_store(self) -> None:
    mem_store = testing.create_store(prefill=False)

    exit_code, out = testing.execute_command(
        summary.SummaryCommand(), mem_store, output_json=True
    )

    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(data["run"]["trajectories_count"], 0)
    self.assertEqual(data["trajectories"], [])

  def test_empty_store_human_readable_omits_the_table(self) -> None:
    mem_store = testing.create_store(prefill=False)

    exit_code, out = testing.execute_command(
        summary.SummaryCommand(), mem_store, output_json=False
    )

    self.assertEqual(exit_code, 0)
    self.assertIn("0 trajectories, 0 steps", out)
    self.assertNotIn("TRAJECTORY", out)

  def test_negative_max_tools_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "max_tools"):
      summary.SummaryCommand(max_tools=-1)


class FormatTableTest(absltest.TestCase):

  def test_columns_are_padded_to_the_widest_cell(self) -> None:
    self.assertEqual(
        summary.format_table([("a", "bb", "c"), ("aaa", "b", "")]),
        ["a    bb  c", "aaa  b"],
    )


if __name__ == "__main__":
  absltest.main()
