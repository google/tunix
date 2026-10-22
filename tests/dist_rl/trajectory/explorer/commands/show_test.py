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

"""Unit tests for ShowCommand execution."""

import contextlib
import io
import json
import textwrap
from typing import Any

from absl.testing import absltest
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory.explorer.commands import show
from tunix.dist_rl.trajectory.explorer.commands import testing

_NO_OUTCOME_TRAJECTORY = trajectory_testing.TRAJECTORY_2
_TOOL_CALLING_TRAJECTORY = trajectory_testing.PAIRED_ATIF_TRAJECTORY
_SUCCEEDED_TRAJECTORY = trajectory_testing.PAIRED_TUNIX_TRAJECTORY
_ROLLOUT_TRAJECTORY = trajectory_testing.TUNIX_ROLLOUT_TRAJECTORY


def _trajectory_id(trajectory: trajectory_lib.Trajectory[Any]) -> str:
  """Returns the fixture's ID, which every stored fixture must carry."""
  if trajectory.trajectory_id is None:
    raise ValueError("test fixture has no trajectory_id")
  return trajectory.trajectory_id


class ShowTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.mem_store = testing.create_store(
        trajectories=[_NO_OUTCOME_TRAJECTORY, _TOOL_CALLING_TRAJECTORY]
    )

  def test_unknown_trajectory_exits_1_with_message_on_stderr(self) -> None:
    cmd = show.ShowCommand(trajectory_id="nope")
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
      exit_code, out = testing.execute_command(cmd, self.mem_store)
    self.assertEqual(exit_code, 1)
    self.assertEmpty(out)
    self.assertIn("trajectory not found: nope", stderr.getvalue())

  def test_header_and_steps(self) -> None:
    cmd = show.ShowCommand(trajectory_id=_trajectory_id(_NO_OUTCOME_TRAJECTORY))
    exit_code, out = testing.execute_command(cmd, self.mem_store)
    self.assertEqual(exit_code, 0)
    expected = textwrap.dedent("""\
        Trajectory traj_1002
        Session: -
        Agent: agent_v2 2.0
        Status: -
        Reward: -
        Steps: 5 (agent 3, user 2, system 0, other 0)
        Tokens: -

        [1] user
          message: First step in traj 2

        [2] agent
          message: Second step in traj 2

        [3] user
          message: Third step in traj 2

        [4] agent
          message: Fourth step in traj 2

        [5] agent
          message: Fifth step in traj 2
    """)
    self.assertMultiLineEqual(out, expected)

  def test_tool_calls_observations_and_tokens(self) -> None:
    cmd = show.ShowCommand(
        trajectory_id=_trajectory_id(_TOOL_CALLING_TRAJECTORY)
    )
    exit_code, out = testing.execute_command(cmd, self.mem_store)
    self.assertEqual(exit_code, 0)
    self.assertIn("Status: -", out)
    self.assertIn("Tokens: 150 (100+50)", out)
    self.assertIn("gemini-2.5-pro  tokens 150 (100+50)", out)
    self.assertIn("reasoning: Reasoning trace", out)
    self.assertIn('tool call[call_1]: search({"query": "tunix"})', out)
    self.assertIn("observation[call_1]: search result", out)
    self.assertIn("subagent: sub_traj_ref_1", out)
    self.assertIn("Tool sequence: search", out)
    # A rollout reports only completion tokens, so no prompt count is shown.
    rollout_store = testing.create_store(trajectories=[_ROLLOUT_TRAJECTORY])
    rollout_cmd = show.ShowCommand(
        trajectory_id=_trajectory_id(_ROLLOUT_TRAJECTORY)
    )
    _, rollout_out = testing.execute_command(rollout_cmd, rollout_store)
    self.assertIn("Tokens: 8 completion", rollout_out)
    self.assertIn("[1] agent  tokens 3 completion", rollout_out)
    self.assertIn("[3] agent  tokens 5 completion", rollout_out)

  def test_tool_call_arguments_rendered_with_sorted_keys(self) -> None:
    step = trajectory_testing.STEP_1_1.model_copy(
        update={
            "tool_calls": [
                trajectory_lib.ToolCall(
                    tool_call_id="call_1",
                    function_name="search",
                    # Insertion order is deliberately unsorted.
                    arguments={"query": "tunix", "limit": 3, "cursor": "abc"},
                )
            ]
        }
    )
    trajectory = trajectory_testing.TRAJECTORY_1.model_copy(
        update={"steps": [step]}
    )
    mem_store = testing.create_store(trajectories=[trajectory])
    cmd = show.ShowCommand(trajectory_id=_trajectory_id(trajectory))
    exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertIn(
        'tool call[call_1]: search({"cursor": "abc", "limit": 3, "query": "tunix"})',
        out,
    )

  def test_observations_are_labeled_with_their_source_call_id(self) -> None:
    step = trajectory_testing.STEP_1_1.model_copy(
        update={
            "tool_calls": [
                trajectory_lib.ToolCall(
                    tool_call_id="call_a",
                    function_name="read",
                    arguments={"path": "a.txt"},
                ),
                trajectory_lib.ToolCall(
                    tool_call_id="call_b",
                    function_name="read",
                    arguments={"path": "b.txt"},
                ),
            ],
            "observation": trajectory_lib.Observation(
                results=[
                    # Results arrive in the opposite order to the calls.
                    trajectory_lib.ObservationResult(
                        source_call_id="call_b", content="contents of b"
                    ),
                    trajectory_lib.ObservationResult(
                        source_call_id="call_a", content="contents of a"
                    ),
                    trajectory_lib.ObservationResult(content="unattributed"),
                ]
            ),
        }
    )
    trajectory = trajectory_testing.TRAJECTORY_1.model_copy(
        update={"steps": [step]}
    )
    mem_store = testing.create_store(trajectories=[trajectory])
    cmd = show.ShowCommand(trajectory_id=_trajectory_id(trajectory))
    exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 0)
    expected_lines = textwrap.dedent("""\
        [1] agent
          message: Hello world
          tool call[call_a]: read({"path": "a.txt"})
          tool call[call_b]: read({"path": "b.txt"})
          observation[call_b]: contents of b
          observation[call_a]: contents of a
          observation: unattributed
    """)
    self.assertIn(expected_lines, out)

  def test_path_only_subagent_ref_renders_path(self) -> None:
    step = trajectory_testing.STEP_1_1.model_copy(
        update={
            "observation": trajectory_lib.Observation(
                results=[
                    trajectory_lib.ObservationResult(
                        content="delegated",
                        subagent_trajectory_ref=[
                            trajectory_lib.SubagentTrajectoryRef(
                                trajectory_path="gs://bucket/sub.json"
                            )
                        ],
                    )
                ]
            )
        }
    )
    trajectory = trajectory_testing.TRAJECTORY_1.model_copy(
        update={"steps": [step]}
    )
    mem_store = testing.create_store(trajectories=[trajectory])
    cmd = show.ShowCommand(trajectory_id=_trajectory_id(trajectory))
    exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertIn("subagent (external): gs://bucket/sub.json", out)
    self.assertNotIn("None", out)

  def test_status_and_reward_from_tunix_schema(self) -> None:
    mem_store = testing.create_store(trajectories=[_SUCCEEDED_TRAJECTORY])
    cmd = show.ShowCommand(trajectory_id=_trajectory_id(_SUCCEEDED_TRAJECTORY))
    exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertRegex(out, rf"Status: \S*{_SUCCEEDED_TRAJECTORY.status}")
    self.assertIn(f"Reward: {_SUCCEEDED_TRAJECTORY.total_reward}", out)

  def test_unrecognized_status_exits_1_with_message_on_stderr(self) -> None:
    trajectory = _SUCCEEDED_TRAJECTORY.model_copy(
        update={"status": "not_a_status"}
    )
    mem_store = testing.create_store(trajectories=[trajectory])
    cmd = show.ShowCommand(trajectory_id=_trajectory_id(trajectory))
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
      exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 1)
    self.assertEmpty(out)
    self.assertIn(
        f"error: trajectory {trajectory.trajectory_id}: Unrecognized"
        " trajectory status 'not_a_status'",
        stderr.getvalue(),
    )
    self.assertNotIn("Traceback", stderr.getvalue())

  def test_max_chars_truncates_content(self) -> None:
    cmd = show.ShowCommand(
        trajectory_id=_trajectory_id(_NO_OUTCOME_TRAJECTORY), max_chars=8
    )
    _, out = testing.execute_command(cmd, self.mem_store)
    self.assertIn(
        f"message: {_NO_OUTCOME_TRAJECTORY.steps[0].message[:7]}…", out
    )

  def test_json_output(self) -> None:
    cmd = show.ShowCommand(
        trajectory_id=_trajectory_id(_TOOL_CALLING_TRAJECTORY)
    )
    exit_code, out = testing.execute_command(
        cmd, self.mem_store, output_json=True
    )
    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(
        data["trajectory"]["trajectory_id"],
        _TOOL_CALLING_TRAJECTORY.trajectory_id,
    )
    self.assertLen(
        data["trajectory"]["steps"], len(_TOOL_CALLING_TRAJECTORY.steps)
    )
    self.assertEqual(data["stats"]["tool_call_names"], ["search"])
    self.assertEqual(data["stats"]["tokens"]["total"], 150)


class TruncateToLineTest(absltest.TestCase):

  def test_collapses_whitespace(self) -> None:
    self.assertEqual(show._truncate_to_line("a\n  b\t c", 10), "a b c")

  def test_cuts_with_ellipsis(self) -> None:
    self.assertEqual(show._truncate_to_line("abcdef", 4), "abc…")
    self.assertEqual(show._truncate_to_line("abcd", 4), "abcd")

  def test_zero_keeps_everything(self) -> None:
    self.assertEqual(show._truncate_to_line("a" * 500, 0), "a" * 500)

  def test_negative_keeps_everything(self) -> None:
    # A negative limit must not fall through to Python's negative slicing.
    self.assertEqual(show._truncate_to_line("abcdef", -1), "abcdef")
    self.assertEqual(show._truncate_to_line("a" * 500, -3), "a" * 500)


if __name__ == "__main__":
  absltest.main()
