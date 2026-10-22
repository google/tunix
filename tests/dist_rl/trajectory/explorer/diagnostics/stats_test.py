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

"""Unit tests for trajectory statistics derivation."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory.explorer.diagnostics import stats as stats_lib

_AGENT = trajectory_lib.Agent(name="a", version="1.0")


def _trajectory(
    steps: list[trajectory_lib.Step] | None = None,
    **kwargs: Any,
) -> trajectory_lib.Trajectory[trajectory_lib.Step]:
  """Builds a base ATIF trajectory with the given steps and metadata."""
  return trajectory_lib.Trajectory(
      trajectory_id="traj_test",
      agent=_AGENT,
      steps=steps or [],
      **kwargs,
  )


def _step(
    step_id: int,
    source: trajectory_lib.Source,
    **kwargs: Any,
) -> trajectory_lib.Step:
  """Builds a base ATIF step."""
  return trajectory_lib.Step(
      step_id=step_id, source=source, message="m", **kwargs
  )


def _tunix_trajectory(
    trajectory_id: str = "traj_test", **kwargs: Any
) -> trajectory_lib.TunixTrajectory:
  """Builds a Tunix trajectory, which can report `status` and `total_reward`."""
  return trajectory_lib.TunixTrajectory(
      trajectory_id=trajectory_id, agent=_AGENT, **kwargs
  )


class StepCountsTest(absltest.TestCase):

  def test_counts_are_split_by_source(self) -> None:
    trajectory = _trajectory([
        _step(1, trajectory_lib.Source.SYSTEM),
        _step(2, trajectory_lib.Source.USER),
        _step(3, trajectory_lib.Source.AGENT),
        _step(4, trajectory_lib.Source.AGENT),
    ])

    steps = stats_lib.TrajectoryStats.from_trajectory(trajectory).steps

    self.assertEqual(steps, stats_lib.StepCounts(agent=2, user=1, system=1))
    self.assertEqual(steps.total, 4)
    self.assertEqual(steps.env_steps, 2)

  def test_total_is_the_sum_of_the_source_split(self) -> None:
    counts = stats_lib.StepCounts(agent=2, user=2, other=1)

    self.assertEqual(counts.total, 5)
    self.assertEqual(sum(counts), counts.total)
    self.assertEqual(
        counts.to_json_dict(),
        {"total": 5, "agent": 2, "user": 2, "system": 0, "other": 1},
    )

  def test_str_shows_total_and_source_split(self) -> None:
    counts = stats_lib.StepCounts(agent=3, user=2, system=1, other=0)

    self.assertEqual(str(counts), "6 (agent 3, user 2, system 1, other 0)")
    self.assertEqual(f"{counts}", str(counts))


class TokenCountsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "prompt_and_completion",
          stats_lib.TokenCounts(prompt=100, completion=50, cached=20),
          150,
          "150 (100+50)",
      ),
      # Without both prompt and completion there is no total to split.
      (
          "completion_only",
          stats_lib.TokenCounts(completion=50),
          None,
          "50 completion",
      ),
      (
          "prompt_and_cached",
          stats_lib.TokenCounts(prompt=100, cached=20),
          None,
          "100 prompt, 20 cached",
      ),
  )
  def test_total_and_str(
      self,
      counts: stats_lib.TokenCounts,
      expected_total: int | None,
      expected_str: str,
  ) -> None:
    self.assertEqual(counts.total, expected_total)
    self.assertEqual(str(counts), expected_str)
    self.assertEqual(f"{counts}", expected_str)

  @parameterized.named_parameters(
      ("no_metrics", None, None),
      (
          "metrics_without_token_fields",
          trajectory_lib.Metrics(cost_usd=0.01),
          None,
      ),
      (
          "partial_token_fields",
          trajectory_lib.Metrics(prompt_tokens=7),
          stats_lib.TokenCounts(prompt=7, completion=None, cached=None),
      ),
      (
          "all_token_fields",
          trajectory_lib.Metrics(
              prompt_tokens=10, completion_tokens=2, cached_tokens=1
          ),
          stats_lib.TokenCounts(prompt=10, completion=2, cached=1),
      ),
  )
  def test_from_metrics(
      self,
      metrics: trajectory_lib.Metrics | None,
      expected: stats_lib.TokenCounts | None,
  ) -> None:
    self.assertEqual(stats_lib.TokenCounts.from_metrics(metrics), expected)

  def test_final_metrics_are_authoritative(self) -> None:
    trajectory = _trajectory(
        [
            _step(
                1,
                trajectory_lib.Source.AGENT,
                metrics=trajectory_lib.Metrics(
                    prompt_tokens=1, completion_tokens=1
                ),
            )
        ],
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=100,
            total_completion_tokens=20,
            total_cached_tokens=50,
        ),
    )

    tokens = stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens

    expected = stats_lib.TokenCounts(prompt=100, completion=20, cached=50)
    self.assertEqual(tokens, expected)
    self.assertEqual(expected.total, 120)

  @parameterized.named_parameters(
      ("final_metrics_absent", None),
      (
          "final_metrics_carry_no_token_totals",
          trajectory_lib.FinalMetrics(total_steps=3),
      ),
  )
  def test_steps_are_summed_when_final_metrics_report_no_tokens(
      self, final_metrics: trajectory_lib.FinalMetrics | None
  ) -> None:
    trajectory = _trajectory(
        [
            _step(1, trajectory_lib.Source.USER),
            _step(
                2,
                trajectory_lib.Source.AGENT,
                metrics=trajectory_lib.Metrics(
                    prompt_tokens=10, completion_tokens=2, cached_tokens=1
                ),
            ),
            _step(
                3,
                trajectory_lib.Source.AGENT,
                metrics=trajectory_lib.Metrics(
                    prompt_tokens=20, completion_tokens=3
                ),
            ),
        ],
        final_metrics=final_metrics,
    )

    tokens = stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens

    self.assertEqual(
        tokens, stats_lib.TokenCounts(prompt=30, completion=5, cached=1)
    )

  def test_a_measured_zero_is_kept(self) -> None:
    trajectory = _trajectory(
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=0, total_completion_tokens=0
        )
    )

    tokens = stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens

    self.assertEqual(tokens, stats_lib.TokenCounts(prompt=0, completion=0))

  def test_steps_with_metrics_but_no_token_fields_are_unreported(self) -> None:
    trajectory = _trajectory([
        _step(
            1,
            trajectory_lib.Source.AGENT,
            metrics=trajectory_lib.Metrics(cost_usd=0.01),
        )
    ])

    self.assertIsNone(
        stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens
    )


class ToolCallsTest(absltest.TestCase):

  def test_sequence_follows_invocation_order(self) -> None:
    trajectory = _trajectory([
        _step(
            1,
            trajectory_lib.Source.AGENT,
            tool_calls=[
                trajectory_lib.ToolCall(tool_call_id="1", function_name="b"),
                trajectory_lib.ToolCall(tool_call_id="2", function_name="a"),
            ],
        ),
        _step(
            2,
            trajectory_lib.Source.AGENT,
            tool_calls=[
                trajectory_lib.ToolCall(tool_call_id="3", function_name="b")
            ],
        ),
    ])

    names = stats_lib.TrajectoryStats.from_trajectory(
        trajectory
    ).tool_call_names

    self.assertEqual(names, ("b", "a", "b"))


class OutcomeTest(parameterized.TestCase):
  """Tests for `status` and `total_reward`, which are not base ATIF fields."""

  @parameterized.named_parameters(
      ("status_name", "SUCCEEDED"),
      ("status_name_in_lower_case", "succeeded"),
  )
  def test_schemas_that_declare_the_fields_report_them(
      self, status: str
  ) -> None:
    trajectory = _tunix_trajectory(total_reward=0.5, status=status)

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertEqual(stats.status, stats_lib.TrajectoryStatus.SUCCEEDED)
    self.assertEqual(stats.total_reward, 0.5)

  def test_a_status_stated_as_an_enum_member_is_reported(self) -> None:
    # A schema may hold the `TrajectoryStatus` member itself rather than its
    # name; `model_construct` skips validation so the enum is kept as-is.
    trajectory = trajectory_lib.TunixTrajectory.model_construct(
        trajectory_id="traj_test",
        agent=_AGENT,
        steps=[],
        status=stats_lib.TrajectoryStatus.TIMEOUT,
    )

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertEqual(stats.status, stats_lib.TrajectoryStatus.TIMEOUT)
    self.assertEqual(stats.to_row()["status"], "TIMEOUT")

  def test_schemas_that_do_not_declare_the_fields_report_neither(self) -> None:
    trajectory = _trajectory(extra={"status": "SUCCEEDED", "reward": 1.0})

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertIsNone(stats.status)
    self.assertIsNone(stats.total_reward)

  def test_an_unstated_status_is_unreported(self) -> None:
    stats = stats_lib.TrajectoryStats.from_trajectory(_tunix_trajectory())

    self.assertIsNone(stats.status)

  def test_an_unrecognized_status_is_rejected(self) -> None:
    trajectory = _tunix_trajectory(status="COMPLETED")

    with self.assertRaisesRegex(ValueError, "Unrecognized trajectory status"):
      stats_lib.TrajectoryStats.from_trajectory(trajectory)

  @parameterized.named_parameters(
      ("nan", float("nan")),
      ("inf", float("inf")),
      ("negative_inf", float("-inf")),
  )
  def test_a_non_finite_total_reward_is_unreported(self, value: float) -> None:
    trajectory = _tunix_trajectory(total_reward=value)

    self.assertIsNone(
        stats_lib.TrajectoryStats.from_trajectory(trajectory).total_reward
    )


class RunStatsTest(parameterized.TestCase):

  def test_df_has_one_typed_row_per_trajectory(self) -> None:
    reporting = _tunix_trajectory(
        trajectory_id="traj_1",
        steps=[
            trajectory_lib.TunixAgentStep(
                step_id=0,
                source=trajectory_lib.Source.AGENT,
                message="m",
                tool_calls=[
                    trajectory_lib.ToolCall(
                        tool_call_id="0",
                        function_name="a",
                    ),
                    trajectory_lib.ToolCall(
                        tool_call_id="1",
                        function_name="b",
                    ),
                ],
            ),
            trajectory_lib.TunixEnvStep(
                step_id=1, source=trajectory_lib.Source.USER, message="m"
            ),
        ],
        status="SUCCEEDED",
        total_reward=0.5,
        # Every token count is reported: the row is compared with `to_row()`
        # below, and a `pd.NA` cell cannot be compared with None.
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=5,
            total_completion_tokens=2,
            total_cached_tokens=1,
        ),
    )
    silent = _tunix_trajectory(trajectory_id="traj_2")

    run = stats_lib.RunStats.from_trajectories([reporting, silent])

    # The row keys and the schema are spelled out separately; pin them together.
    row = stats_lib.TrajectoryStats.from_trajectory(reporting).to_row()
    self.assertEqual(list(row), list(stats_lib.COLUMN_DTYPES))
    self.assertEqual(run.df.columns.tolist(), list(stats_lib.COLUMN_DTYPES))
    self.assertEqual(
        run.df.dtypes.astype(str).to_dict(), dict(stats_lib.COLUMN_DTYPES)
    )
    self.assertEqual(run.df["trajectory_id"].tolist(), ["traj_1", "traj_2"])

    # Reported quantities land in the right typed column.
    reported = run.df.loc[run.df.trajectory_id == "traj_1"]
    self.assertEqual(reported.squeeze(axis=0).to_dict(), row)
    self.assertEqual(reported["status"].item(), "SUCCEEDED")
    self.assertEqual(reported["total_reward"].item(), 0.5)
    self.assertEqual(reported["tokens_prompt"].item(), 5)
    self.assertEqual(reported["tokens_completion"].item(), 2)
    self.assertEqual(reported["num_tool_calls"].item(), 2)

    # Unreported quantities are nulls, not zeros; counts are always reported.
    unreported = run.df.loc[run.df.trajectory_id == "traj_2"]
    self.assertTrue(unreported["status"].isna().all())
    self.assertTrue(unreported["total_reward"].isna().all())
    self.assertTrue(unreported["tokens_prompt"].isna().all())
    self.assertTrue(unreported["tokens_completion"].isna().all())
    self.assertTrue(unreported["tokens_cached"].isna().all())
    self.assertEqual(unreported["num_tool_calls"].item(), 0)

  def test_empty_run_keeps_the_column_schema(self) -> None:
    run = stats_lib.RunStats.from_trajectories([])

    self.assertEmpty(run.df)
    self.assertEqual(
        run.df.dtypes.astype(str).to_dict(), dict(stats_lib.COLUMN_DTYPES)
    )

  def test_totals_sum_over_trajectories(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _trajectory(
            [_step(i, trajectory_lib.Source.AGENT) for i in range(1, 7)],
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=520,
                total_completion_tokens=90,
                total_cached_tokens=220,
            ),
        ),
        _trajectory(
            [_step(1, trajectory_lib.Source.USER)]
            + [_step(i, trajectory_lib.Source.AGENT) for i in range(2, 5)],
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=180,
                total_completion_tokens=35,
                total_cached_tokens=0,
            ),
        ),
        _trajectory([_step(1, trajectory_lib.Source.SYSTEM)]),
        # A rollout reports only completion tokens: it adds to `completion`
        # and to `num_with_tokens`, but not to `prompt` or `cached`.
        trajectory_testing.TUNIX_ROLLOUT_TRAJECTORY,
    ])

    self.assertEqual(run.num_trajectories, 4)
    self.assertEqual(
        run.steps, stats_lib.StepCounts(agent=11, user=2, system=3)
    )
    self.assertEqual(
        run.tokens,
        stats_lib.TokenCounts(prompt=700, completion=133, cached=220),
    )
    self.assertEqual(run.num_with_tokens, 3)

  def test_mean_reward_averages_only_reporting_trajectories(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _tunix_trajectory(total_reward=1.0),
        _tunix_trajectory(total_reward=0.0),
        _tunix_trajectory(),
    ])

    self.assertEqual(run.df["total_reward"].dropna().tolist(), [1.0, 0.0])
    self.assertEqual(run.mean_reward, 0.5)
    self.assertEqual(run.num_with_reward, 2)

  @parameterized.named_parameters(
      ("empty_run", 0),
      ("trajectories_reporting_nothing", 2),
  )
  def test_quantities_are_unreported_when_no_trajectory_reports_them(
      self, num_trajectories: int
  ) -> None:
    run = stats_lib.RunStats.from_trajectories(
        [_trajectory() for _ in range(num_trajectories)]
    )

    self.assertEqual(run.num_trajectories, num_trajectories)
    self.assertEqual(run.steps.total, 0)
    self.assertIsNone(run.tokens)
    self.assertEqual(run.num_with_tokens, 0)
    self.assertIsNone(run.mean_reward)
    self.assertEqual(run.num_with_reward, 0)
    self.assertEmpty(run.status_counts)
    self.assertEqual(run.num_with_status, 0)
    self.assertEmpty(run.tool_counts)

  def test_status_counts_tally_only_reporting_trajectories(self) -> None:
    # "SUCCEEDED" and "succeeded" must merge into a single bucket: the
    # per-trajectory `.upper()` normalisation is what makes the run-level
    # tally correct.
    run = stats_lib.RunStats.from_trajectories([
        _tunix_trajectory(status="TIMEOUT"),
        _tunix_trajectory(status="SUCCEEDED"),
        _tunix_trajectory(),
        _tunix_trajectory(status="succeeded"),
    ])

    self.assertEqual(
        run.status_counts,
        {
            stats_lib.TrajectoryStatus.SUCCEEDED: 2,
            stats_lib.TrajectoryStatus.TIMEOUT: 1,
        },
    )
    self.assertEqual(
        list(run.status_counts),
        [
            stats_lib.TrajectoryStatus.TIMEOUT,
            stats_lib.TrajectoryStatus.SUCCEEDED,
        ],
    )
    self.assertEqual(run.num_with_status, 3)
    self.assertEqual(
        run.to_json_dict()["status_counts"], {"SUCCEEDED": 2, "TIMEOUT": 1}
    )
    self.assertEqual(
        list(run.to_json_dict()["status_counts"]), ["SUCCEEDED", "TIMEOUT"]
    )

  def test_tool_counts_are_summed_across_the_run(self) -> None:
    def with_tools(*names: str) -> trajectory_lib.Trajectory:
      return _trajectory([
          _step(
              1,
              trajectory_lib.Source.AGENT,
              tool_calls=[
                  trajectory_lib.ToolCall(
                      tool_call_id=str(i), function_name=name
                  )
                  for i, name in enumerate(names)
              ],
          )
      ])

    run = stats_lib.RunStats.from_trajectories(
        [with_tools("b", "c"), with_tools("c", "a")]
    )

    self.assertEqual(run.tool_counts, {"c": 2, "a": 1, "b": 1})
    self.assertEqual(list(run.tool_counts), ["b", "c", "a"])
    self.assertEqual(list(run.to_json_dict()["tool_counts"]), ["c", "b", "a"])

  def test_run_json(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _tunix_trajectory(
            steps=[
                trajectory_lib.TunixAgentStep(
                    step_id=0, source=trajectory_lib.Source.AGENT, message="m"
                )
            ],
            total_reward=1.0,
            status="SUCCEEDED",
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=5, total_completion_tokens=2
            ),
        ),
        _tunix_trajectory(),
    ])

    self.assertEqual(
        run.to_json_dict(),
        {
            "num_trajectories": 2,
            "steps": {
                "total": 1,
                "agent": 1,
                "user": 0,
                "system": 0,
                "other": 0,
            },
            "tokens": {
                "prompt": 5,
                "completion": 2,
                "cached": None,
                "total": 7,
            },
            "num_with_tokens": 1,
            "status_counts": {"SUCCEEDED": 1},
            "num_with_status": 1,
            "mean_reward": 1.0,
            "num_with_reward": 1,
            "tool_counts": {},
        },
    )


class JsonDictTest(absltest.TestCase):

  def test_unreported_quantities_serialize_as_null(self) -> None:
    trajectory = trajectory_lib.TunixTrajectory(agent=_AGENT, steps=[])

    payload = stats_lib.TrajectoryStats.from_trajectory(
        trajectory
    ).to_json_dict()

    self.assertEqual(
        payload,
        {
            "trajectory_id": None,
            "status": None,
            "total_reward": None,
            "steps": {
                "total": 0,
                "agent": 0,
                "user": 0,
                "system": 0,
                "other": 0,
            },
            "tokens": None,
            "tool_call_names": [],
        },
    )
    # A rollout reports only completion tokens; the other counts stay null.
    rollout_payload = stats_lib.TrajectoryStats.from_trajectory(
        trajectory_testing.TUNIX_ROLLOUT_TRAJECTORY
    ).to_json_dict()
    self.assertEqual(
        rollout_payload["tokens"],
        {"prompt": None, "completion": 8, "cached": None, "total": None},
    )

  def test_stats_serialize_every_reported_quantity(self) -> None:
    trajectory = _tunix_trajectory(
        status="SUCCEEDED",
        total_reward=0.85,
        steps=[
            trajectory_lib.TunixAgentStep(
                step_id=0,
                source=trajectory_lib.Source.AGENT,
                message="m",
                tool_calls=[
                    trajectory_lib.ToolCall(
                        tool_call_id="0", function_name="find_files"
                    )
                ],
            ),
            trajectory_lib.TunixEnvStep(
                step_id=1, source=trajectory_lib.Source.USER, message="m"
            ),
        ],
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=520,
            total_completion_tokens=90,
            total_cached_tokens=220,
        ),
    )

    payload = stats_lib.TrajectoryStats.from_trajectory(
        trajectory
    ).to_json_dict()

    self.assertEqual(
        payload,
        {
            "trajectory_id": "traj_test",
            "status": "SUCCEEDED",
            "total_reward": 0.85,
            "steps": {
                "total": 2,
                "agent": 1,
                "user": 1,
                "system": 0,
                "other": 0,
            },
            "tokens": {
                "prompt": 520,
                "completion": 90,
                "cached": 220,
                "total": 610,
            },
            "tool_call_names": ["find_files"],
        },
    )


if __name__ == "__main__":
  absltest.main()
