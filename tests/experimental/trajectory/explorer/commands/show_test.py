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

from absl.testing import absltest
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing
from tunix.experimental.trajectory.explorer.commands import show
from tunix.experimental.trajectory.explorer.commands import testing

_TRAJECTORY = trajectory_lib.Trajectory(
    trajectory_id="traj_show",
    session_id="sess_1",
    agent=trajectory_lib.Agent(
        name="agent", version="1.0", model_name="gemini"
    ),
    extra={"reward": 0.5, "status": "COMPLETED"},
    steps=[
        trajectory_lib.Step(
            step_id=1,
            source=trajectory_lib.Source.USER,
            message="Find   the\nanswer",
        ),
        trajectory_lib.Step(
            step_id=2,
            source=trajectory_lib.Source.AGENT,
            message="Searching",
            tool_calls=[
                trajectory_lib.ToolCall(
                    tool_call_id="call_1", function_name="search", arguments={}
                )
            ],
            metrics=trajectory_lib.Metrics(
                prompt_tokens=10, completion_tokens=5
            ),
        ),
    ],
)


class ShowTest(absltest.TestCase):

  def test_json_reports_stats_and_steps(self) -> None:
    mem_store = testing.create_store(trajectories=[_TRAJECTORY])

    exit_code, out = testing.execute_command(
        show.ShowCommand(trajectory_id="traj_show"),
        mem_store,
        output_json=True,
    )

    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(data["trajectory"]["trajectory_id"], "traj_show")
    self.assertEqual(data["trajectory"]["final_reward"], 0.5)
    self.assertEqual(data["trajectory"]["status"], "COMPLETED")
    self.assertEqual(
        data["steps"],
        [
            {
                "step_id": 1,
                "source": "user",
                "message": "Find   the\nanswer",
                "tool_calls": [],
                "tokens": {
                    "prompt": None,
                    "completion": None,
                    "cached": None,
                    "total": None,
                },
                "reward": None,
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "Searching",
                "tool_calls": ["search"],
                "tokens": {
                    "prompt": 10,
                    "completion": 5,
                    "cached": 0,
                    "total": 15,
                },
                "reward": None,
            },
        ],
    )

  def test_human_readable_shows_header_and_steps(self) -> None:
    mem_store = testing.create_store(trajectories=[_TRAJECTORY])

    exit_code, out = testing.execute_command(
        show.ShowCommand(trajectory_id="traj_show"),
        mem_store,
        output_json=False,
    )

    self.assertEqual(exit_code, 0)
    self.assertIn("Trajectory traj_show", out)
    self.assertIn("Session:        sess_1", out)
    self.assertIn("Agent:          agent 1.0 (model: gemini)", out)
    self.assertIn("Status:         COMPLETED", out)
    self.assertIn("Final reward:   0.50", out)
    self.assertIn("Steps (A/U/S):  2 (1/1/0)", out)
    self.assertIn("Tokens (P/C/T): 10/5/15", out)
    self.assertIn("[1] user", out)
    # Messages are collapsed onto a single line.
    self.assertIn(": Find the answer", out)
    self.assertIn("[2] agent", out)
    self.assertIn("tools:  search", out)
    self.assertIn("tokens: 10/5/15", out)

  def test_max_message_chars_truncates_messages(self) -> None:
    mem_store = testing.create_store(trajectories=[_TRAJECTORY])

    _, out = testing.execute_command(
        show.ShowCommand(trajectory_id="traj_show", max_message_chars=4),
        mem_store,
        output_json=False,
    )

    self.assertIn(": Find...", out)
    self.assertNotIn("answer", out)

  def test_unknown_trajectory_is_a_usage_error(self) -> None:
    mem_store = testing.create_store(prefill=True)
    stderr = io.StringIO()

    with contextlib.redirect_stderr(stderr):
      exit_code, out = testing.execute_command(
          show.ShowCommand(trajectory_id="traj_missing"),
          mem_store,
          output_json=True,
      )

    self.assertEqual(exit_code, 2)
    self.assertEqual(out, "")
    self.assertIn("traj_missing", stderr.getvalue())

  def test_negative_max_message_chars_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "max_message_chars"):
      show.ShowCommand(trajectory_id="t", max_message_chars=-1)


class StepViewTest(absltest.TestCase):

  def test_tunix_env_step_reports_its_reward(self) -> None:
    view = show.step_view(trajectory_testing.TUNIX_ENV_STEP_0)

    self.assertEqual(view.reward, 1.0)

  def test_tunix_agent_step_reports_tools_and_tokens(self) -> None:
    view = show.step_view(trajectory_testing.TUNIX_AGENT_STEP_1)

    self.assertIsNone(view.reward)
    self.assertEqual(view.tool_calls, ("search",))
    self.assertTrue(view.tokens.reported)
    self.assertEqual(view.tokens.total, 150)
    self.assertEqual(view.tokens.cached, 20)


class TruncateTest(absltest.TestCase):

  def test_zero_keeps_the_whole_text(self) -> None:
    self.assertEqual(show.truncate("a  b\nc", 0), "a b c")

  def test_text_within_the_limit_is_kept(self) -> None:
    self.assertEqual(show.truncate("abc", 3), "abc")

  def test_text_over_the_limit_is_truncated(self) -> None:
    self.assertEqual(show.truncate("abcd", 3), "abc...")


if __name__ == "__main__":
  absltest.main()
