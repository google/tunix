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

import contextlib
import io
import json
from typing import Any

from absl.testing import absltest
from tunix.dist_rl.trajectory import store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory.explorer.commands import summary
from tunix.dist_rl.trajectory.explorer.commands import testing

_NO_OUTCOME_TRAJECTORY_1 = trajectory_testing.TRAJECTORY_1
_NO_OUTCOME_TRAJECTORY_2 = trajectory_testing.TRAJECTORY_2
_TOOL_CALLING_TRAJECTORY = trajectory_testing.PAIRED_ATIF_TRAJECTORY
_SUCCEEDED_TRAJECTORY = trajectory_testing.PAIRED_TUNIX_TRAJECTORY


class TrackingReader(store.TrajectoryReader[trajectory_lib.TrajectoryMetadata]):
  """Tracks which trajectory IDs were requested via get_trajectories."""

  def __init__(self, delegate: store.TrajectoryReader[Any]) -> None:
    self._delegate = delegate
    self.requested_ids: list[list[str]] = []

  def get_trajectories_metadata(
      self,
      trajectory_ids: list[str] | None = None,
  ) -> list[trajectory_lib.TrajectoryMetadata]:
    return self._delegate.get_trajectories_metadata(trajectory_ids)

  def get_trajectories(
      self,
      trajectory_ids: list[str],
  ) -> list[trajectory_lib.Trajectory[Any]]:
    self.requested_ids.append(trajectory_ids)
    return self._delegate.get_trajectories(trajectory_ids)

  def get_trajectories_metrics(
      self,
      trajectory_ids: list[str] | None = None,
  ) -> dict[str, trajectory_lib.TrajectoryMetrics]:
    return self._delegate.get_trajectories_metrics(trajectory_ids)


class SummaryTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.cmd = summary.SummaryCommand()

  def test_empty_store(self) -> None:
    mem_store = testing.create_store(prefill=False)
    exit_code, out = testing.execute_command(self.cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertEqual(out.strip(), "No trajectories found.")

  def test_empty_store_json(self) -> None:
    mem_store = testing.create_store(prefill=False)
    exit_code, out = testing.execute_command(
        self.cmd, mem_store, output_json=True
    )
    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(data["run"]["num_trajectories"], 0)
    self.assertEqual(data["trajectories"], [])

  def test_human_readable_lists_every_trajectory_and_totals(self) -> None:
    mem_store = testing.create_store(
        trajectories=[_NO_OUTCOME_TRAJECTORY_1, _NO_OUTCOME_TRAJECTORY_2]
    )
    exit_code, out = testing.execute_command(self.cmd, mem_store)
    self.assertEqual(exit_code, 0)
    lines = out.splitlines()
    # Header plus one row per trajectory; the fixtures report no status,
    # tokens or reward, so those columns are "-". Cells are right-justified.
    self.assertIn("trajectory", lines[0])
    self.assertRegex(
        lines[1],
        rf"^\s*{_NO_OUTCOME_TRAJECTORY_1.trajectory_id}"
        r"\s+-\s+1\s+1\s+0\s+-\s+0\s+-$",
    )
    self.assertRegex(
        lines[2],
        rf"^\s*{_NO_OUTCOME_TRAJECTORY_2.trajectory_id}"
        r"\s+-\s+5\s+3\s+2\s+-\s+0\s+-$",
    )
    self.assertIn("Trajectories: 2", out)
    self.assertIn("Steps: 6 (agent 4, user 2, system 0, other 0)", out)
    self.assertIn("Tokens: - reported by 0/2", out)
    self.assertIn("Statuses: - reported by 0/2", out)
    self.assertIn("Mean reward: - reported by 0/2", out)
    self.assertNotIn("Tool calls:", out)

  def test_human_readable_with_tokens_tools_and_status(self) -> None:
    # The Tunix schema declares `status` and `total_reward`.
    mem_store = testing.create_store(trajectories=[_SUCCEEDED_TRAJECTORY])
    exit_code, out = testing.execute_command(self.cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertRegex(
        out.splitlines()[1],
        rf"^\s*{_SUCCEEDED_TRAJECTORY.trajectory_id}"
        rf"\s+{_SUCCEEDED_TRAJECTORY.status}\s+3\s+1\s+2\s+150 \(100\+50\)"
        rf"\s+1\s+{_SUCCEEDED_TRAJECTORY.total_reward}$",
    )
    self.assertIn("Tokens: 150 (100+50) reported by 1/1", out)
    self.assertIn(
        f"Statuses: {_SUCCEEDED_TRAJECTORY.status} x1 reported by 1/1", out
    )
    self.assertIn(
        f"Mean reward: {_SUCCEEDED_TRAJECTORY.total_reward} reported by 1/1",
        out,
    )
    self.assertIn("Tool calls: search x1", out)

  def test_json_output(self) -> None:
    mem_store = testing.create_store(
        trajectories=[_NO_OUTCOME_TRAJECTORY_1, _TOOL_CALLING_TRAJECTORY]
    )
    exit_code, out = testing.execute_command(
        self.cmd, mem_store, output_json=True
    )
    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(data["run"]["num_trajectories"], 2)
    self.assertEqual(data["run"]["steps"]["total"], 4)
    self.assertEqual(data["run"]["tokens"]["total"], 150)
    self.assertEqual(data["run"]["num_with_tokens"], 1)
    self.assertEqual(data["run"]["status_counts"], {})
    self.assertEqual(data["run"]["tool_counts"], {"search": 1})
    # One flat record per trajectory, the columns of `RunStats.df`, with
    # unreported quantities as null.
    self.assertEqual(
        [t["trajectory_id"] for t in data["trajectories"]],
        [
            _NO_OUTCOME_TRAJECTORY_1.trajectory_id,
            _TOOL_CALLING_TRAJECTORY.trajectory_id,
        ],
    )
    self.assertEqual(
        data["trajectories"][0],
        {
            "trajectory_id": _NO_OUTCOME_TRAJECTORY_1.trajectory_id,
            "status": None,
            "total_reward": None,
            "steps_agent": 1,
            "steps_user": 0,
            "steps_system": 0,
            "steps_other": 0,
            "tokens_prompt": None,
            "tokens_completion": None,
            "tokens_cached": None,
            "num_tool_calls": 0,
        },
    )
    self.assertEqual(data["trajectories"][1]["tokens_prompt"], 100)

  def test_json_output_with_status_and_reward(self) -> None:
    mem_store = testing.create_store(trajectories=[_SUCCEEDED_TRAJECTORY])
    exit_code, out = testing.execute_command(
        self.cmd, mem_store, output_json=True
    )
    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(
        data["run"]["status_counts"], {_SUCCEEDED_TRAJECTORY.status: 1}
    )
    self.assertEqual(
        data["run"]["mean_reward"], _SUCCEEDED_TRAJECTORY.total_reward
    )
    self.assertEqual(
        data["trajectories"][0]["status"], _SUCCEEDED_TRAJECTORY.status
    )
    self.assertEqual(
        data["trajectories"][0]["total_reward"],
        _SUCCEEDED_TRAJECTORY.total_reward,
    )

  def test_json_output_keeps_full_reward_precision(self) -> None:
    # 1/3 has 16 significant digits; a lossy serializer (e.g. pandas'
    # `to_json`, which rounds to 10) would make the row disagree with the
    # run-level mean computed from the very same value.
    trajectory = _SUCCEEDED_TRAJECTORY.model_copy(
        update={"total_reward": 1 / 3}
    )
    mem_store = testing.create_store(trajectories=[trajectory])
    exit_code, out = testing.execute_command(
        self.cmd, mem_store, output_json=True
    )
    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(data["trajectories"][0]["total_reward"], 1 / 3)
    self.assertEqual(data["run"]["mean_reward"], 1 / 3)

  def test_limit_truncates_loaded_trajectories_and_updates_totals(self) -> None:
    mem_store = testing.create_store(
        trajectories=[
            _NO_OUTCOME_TRAJECTORY_1,
            _NO_OUTCOME_TRAJECTORY_2,
            _TOOL_CALLING_TRAJECTORY,
        ]
    )
    cmd = summary.SummaryCommand(limit=2)
    exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertIn(_NO_OUTCOME_TRAJECTORY_1.trajectory_id, out)
    self.assertIn(_NO_OUTCOME_TRAJECTORY_2.trajectory_id, out)
    self.assertNotIn(_TOOL_CALLING_TRAJECTORY.trajectory_id, out)
    self.assertIn("Trajectories: 2 (of 3)", out)

  def test_limit_bounds_get_trajectories_call(self) -> None:
    mem_store = testing.create_store(
        trajectories=[_NO_OUTCOME_TRAJECTORY_1, _NO_OUTCOME_TRAJECTORY_2]
    )
    tracking_reader = TrackingReader(mem_store)
    cmd = summary.SummaryCommand(limit=1)
    exit_code, _ = testing.execute_command(cmd, tracking_reader)
    self.assertEqual(exit_code, 0)
    self.assertEqual(
        tracking_reader.requested_ids,
        [[_NO_OUTCOME_TRAJECTORY_1.trajectory_id]],
    )

  def test_limit_json_output(self) -> None:
    mem_store = testing.create_store(
        trajectories=[_NO_OUTCOME_TRAJECTORY_1, _TOOL_CALLING_TRAJECTORY]
    )
    cmd = summary.SummaryCommand(limit=1)
    exit_code, out = testing.execute_command(cmd, mem_store, output_json=True)
    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(data["run"]["num_trajectories"], 1)
    self.assertEqual(data["run"]["num_total_trajectories"], 2)
    self.assertLen(data["trajectories"], 1)
    self.assertEqual(
        data["trajectories"][0]["trajectory_id"],
        _NO_OUTCOME_TRAJECTORY_1.trajectory_id,
    )

  def test_limit_greater_than_total_shows_normal_totals(self) -> None:
    mem_store = testing.create_store(
        trajectories=[_NO_OUTCOME_TRAJECTORY_1, _NO_OUTCOME_TRAJECTORY_2]
    )
    cmd = summary.SummaryCommand(limit=5)
    exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertIn("Trajectories: 2", out)
    self.assertNotIn("(of 2)", out)

  def test_limit_zero_with_trajectories(self) -> None:
    mem_store = testing.create_store(
        trajectories=[_NO_OUTCOME_TRAJECTORY_1, _NO_OUTCOME_TRAJECTORY_2]
    )
    cmd = summary.SummaryCommand(limit=0)
    exit_code, out = testing.execute_command(cmd, mem_store)
    self.assertEqual(exit_code, 0)
    self.assertIn("Trajectories: 0 (of 2)", out)

    exit_code, json_out = testing.execute_command(
        cmd, mem_store, output_json=True
    )
    self.assertEqual(exit_code, 0)
    data = json.loads(json_out)
    self.assertEqual(data["run"]["num_trajectories"], 0)
    self.assertEqual(data["run"]["num_total_trajectories"], 2)
    self.assertEmpty(data["trajectories"])

  def test_negative_limit_exits_2(self) -> None:
    mem_store = testing.create_store(trajectories=[_NO_OUTCOME_TRAJECTORY_1])
    cmd = summary.SummaryCommand(limit=-1)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
      exit_code = cmd.execute(mem_store)
    self.assertEqual(exit_code, 2)
    self.assertIn(
        "trajectory-explore: error: --limit must be non-negative, got -1",
        stderr.getvalue(),
    )


if __name__ == "__main__":
  absltest.main()
