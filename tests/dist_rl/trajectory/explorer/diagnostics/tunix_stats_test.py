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

"""Unit tests for the Tunix rollout statistics."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import stats as stats_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import tunix_stats

_AGENT = trajectory_lib.Agent(name="a", version="1.0")

Status = tunix_stats.TrajectoryStatus


def _rollout(**kwargs: Any) -> trajectory_lib.TunixTrajectory:
  """Builds a Tunix rollout with the given fields (by default, no steps)."""
  return trajectory_lib.TunixTrajectory(
      trajectory_id="traj_test", agent=_AGENT, **kwargs
  )


def _generic_trajectory() -> trajectory_lib.Trajectory[trajectory_lib.Step]:
  """Builds a base ATIF trajectory, which reports no Tunix figures."""
  return trajectory_lib.Trajectory(
      trajectory_id="traj_test", agent=_AGENT, steps=[]
  )


def _stats(**kwargs: Any) -> tunix_stats.TunixStats:
  return tunix_stats.TunixStats.from_trajectory(_rollout(**kwargs))


class ParseStatusTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("name", "SUCCEEDED"),
      ("lower_case_name", "succeeded"),
  )
  def test_parses_a_name_in_any_case(self, status: str) -> None:
    self.assertIs(tunix_stats.parse_status(status), Status.SUCCEEDED)

  def test_unstated_status_is_none(self) -> None:
    self.assertIsNone(tunix_stats.parse_status(None))

  def test_rejects_an_unknown_name(self) -> None:
    with self.assertRaisesRegex(ValueError, "Unrecognized trajectory status"):
      tunix_stats.parse_status("COMPLETED")


class TunixStatsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("status_name", "SUCCEEDED"),
      ("status_name_in_lower_case", "succeeded"),
  )
  def test_reads_status_and_reward(self, status: str) -> None:
    stats = _stats(status=status, total_reward=0.5)

    self.assertIs(stats.status, Status.SUCCEEDED)
    self.assertEqual(stats.total_reward, 0.5)

  def test_an_unrecognized_status_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "Unrecognized trajectory status"):
      _stats(status="COMPLETED")

  @parameterized.named_parameters(
      ("nan", float("nan")),
      ("inf", float("inf")),
      ("negative_inf", float("-inf")),
  )
  def test_a_non_finite_total_reward_is_unreported(self, value: float) -> None:
    self.assertIsNone(_stats(total_reward=value).total_reward)

  def test_unstated_fields_are_unreported(self) -> None:
    stats = _stats()

    self.assertIsNone(stats.status)
    self.assertIsNone(stats.total_reward)

  def test_json_dict_serializes_every_field(self) -> None:
    stats = _stats(
        status="SUCCEEDED",
        total_reward=1.0,
    )

    self.assertEqual(
        stats.to_json_dict(),
        {
            "status": "SUCCEEDED",
            "total_reward": 1.0,
        },
    )

  def test_json_dict_serializes_unreported_fields_as_null(self) -> None:
    self.assertEqual(
        _stats().to_json_dict(),
        {
            "status": None,
            "total_reward": None,
        },
    )

  def test_row_cells_are_prefixed_and_status_is_a_name(self) -> None:
    stats = _stats(status="SUCCEEDED", total_reward=1.0)

    row = tunix_stats.to_row(stats)

    # `to_row` is the prefixed JSON form, so this also pins the keys of
    # `to_json_dict()` to `TunixStats._fields`, in order.
    self.assertEqual(list(row), [tunix_stats.column(f) for f in stats._fields])
    self.assertEqual(row["tunix_status"], "SUCCEEDED")
    self.assertEqual(row["tunix_total_reward"], 1.0)

  def test_row_cells_of_a_non_tunix_trajectory_are_all_none(self) -> None:
    row = tunix_stats.to_row(None)

    self.assertEqual(
        list(row),
        [tunix_stats.column(f) for f in tunix_stats.TunixStats._fields],
    )
    self.assertTrue(all(value is None for value in row.values()))

  def test_field_dtypes_cover_every_field_in_order(self) -> None:
    # Spelled out separately from `_fields`; keep them in step.
    self.assertEqual(
        list(tunix_stats.FIELD_DTYPES), list(tunix_stats.TunixStats._fields)
    )


class TunixRunStatsTest(parameterized.TestCase):

  def test_aggregates_cover_the_reporting_rollouts(self) -> None:
    # "SUCCEEDED" and "succeeded" must merge into a single bucket: the
    # per-rollout `.upper()` normalisation is what makes the tally correct.
    # The generic trajectory in the same run counts towards nothing.
    run = stats_lib.RunStats.from_trajectories([
        _rollout(status="TIMEOUT", total_reward=0.0),
        _rollout(status="SUCCEEDED", total_reward=1.0),
        _rollout(),
        _rollout(status="succeeded"),
        _generic_trajectory(),
    ])

    self.assertEqual(run.num_trajectories, 5)
    self.assertEqual(
        run.tunix.status_counts, {Status.SUCCEEDED: 2, Status.TIMEOUT: 1}
    )
    # Keyed in first-seen order.
    self.assertEqual(
        list(run.tunix.status_counts), [Status.TIMEOUT, Status.SUCCEEDED]
    )
    self.assertEqual(run.tunix.num_with_status, 3)
    self.assertEqual(run.tunix.mean_reward, 0.5)
    self.assertEqual(run.tunix.num_with_reward, 2)

  def test_json_dict_ranks_statuses_by_count(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _rollout(status="TIMEOUT"),
        _rollout(status="SUCCEEDED"),
        _rollout(status="SUCCEEDED", total_reward=1.0),
    ])

    payload = run.tunix.to_json_dict()

    self.assertEqual(
        payload,
        {
            "status_counts": {"SUCCEEDED": 2, "TIMEOUT": 1},
            "num_with_status": 3,
            "mean_reward": 1.0,
            "num_with_reward": 1,
        },
    )
    # Dict equality ignores order; the most frequent status comes first.
    self.assertEqual(list(payload["status_counts"]), ["SUCCEEDED", "TIMEOUT"])

  @parameterized.named_parameters(
      ("empty_run", 0),
      ("generic_trajectories_only", 2),
  )
  def test_a_run_without_tunix_rollouts_has_empty_aggregates(
      self, num_trajectories: int
  ) -> None:
    run = stats_lib.RunStats.from_trajectories(
        [_generic_trajectory() for _ in range(num_trajectories)]
    )

    self.assertEqual(run.num_trajectories, num_trajectories)
    self.assertEmpty(run.tunix.status_counts)
    self.assertEqual(run.tunix.num_with_status, 0)
    self.assertIsNone(run.tunix.mean_reward)
    self.assertEqual(run.tunix.num_with_reward, 0)
    # The run-level `tunix` object is always present, empty rather than null.
    self.assertEqual(
        run.tunix.to_json_dict(),
        {
            "status_counts": {},
            "num_with_status": 0,
            "mean_reward": None,
            "num_with_reward": 0,
        },
    )


if __name__ == "__main__":
  absltest.main()
