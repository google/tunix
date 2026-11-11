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

from collections.abc import Mapping
import json
from typing import Any, ClassVar

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory.explorer.diagnostics import schema_diagnostics
from tunix.dist_rl.trajectory.explorer.diagnostics import stats as stats_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import tunix_diagnostics

_AGENT = trajectory_lib.Agent(name="a", version="1.0")
_TUNIX_DIAGNOSTICS = tunix_diagnostics.TunixDiagnostics()


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


class _EvalMetadata(trajectory_lib.TrajectoryMetadata):
  """A hypothetical evaluation schema."""

  EXTENSIONS_KEY: ClassVar[str] = "_acme_eval_extensions"

  verdict: str | None = None
  num_retries: int | None = None
  latency_ms: float | None = None


class _EvalDiagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):

  COLUMN_DTYPES = {
      "verdict": "string",
      "num_retries": "Int64",
      "latency_ms": "Float64",
  }


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
      ("status_name_padded_with_whitespace", " succeeded\n"),
  )
  def test_schemas_that_declare_the_fields_report_them(
      self, status: str
  ) -> None:
    trajectory = _tunix_trajectory(total_reward=0.5, status=status)

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertEqual(stats.status, "SUCCEEDED")
    self.assertEqual(stats.total_reward, 0.5)

  def test_schemas_that_do_not_declare_the_fields_report_neither(self) -> None:
    trajectory = _trajectory(extra={"status": "SUCCEEDED", "reward": 1.0})

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertIsNone(stats.status)
    self.assertIsNone(stats.total_reward)

  @parameterized.named_parameters(
      ("unstated", None),
      ("empty", ""),
      ("blank", " \n"),
  )
  def test_an_unstated_or_blank_status_is_unreported(
      self, status: str | None
  ) -> None:
    stats = stats_lib.TrajectoryStats.from_trajectory(
        _tunix_trajectory(status=status)
    )

    self.assertIsNone(stats.status)

  def test_any_status_is_reported_upper_cased(self) -> None:
    # Statuses are not checked against any vocabulary.
    trajectory = _tunix_trajectory(status="completed")

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertEqual(stats.status, "COMPLETED")

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
    # tally correct. A blank status is not counted, like an unstated one.
    run = stats_lib.RunStats.from_trajectories([
        _tunix_trajectory(status="TIMEOUT"),
        _tunix_trajectory(status="SUCCEEDED"),
        _tunix_trajectory(),
        _tunix_trajectory(status="succeeded"),
        _tunix_trajectory(status=" "),
    ])

    self.assertEqual(run.status_counts, {"SUCCEEDED": 2, "TIMEOUT": 1})
    self.assertEqual(list(run.status_counts), ["TIMEOUT", "SUCCEEDED"])
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
            "num_total_trajectories": 2,
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
            "schema_stats": {},
        },
    )

  def test_from_trajectories_with_num_total_trajectories(self) -> None:
    run = stats_lib.RunStats.from_trajectories(
        [_tunix_trajectory()], num_total_trajectories=10
    )
    self.assertEqual(run.num_trajectories, 1)
    self.assertEqual(run.num_total_trajectories, 10)
    json_dict = run.to_json_dict()
    self.assertEqual(json_dict["num_trajectories"], 1)
    self.assertEqual(json_dict["num_total_trajectories"], 10)


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
            "schema_stats": {},
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
            "schema_stats": {},
        },
    )


class SchemaDiagnosticsTest(parameterized.TestCase):
  """Tests for the schema diagnostics passed as `diagnostics=`."""

  def test_trajectory_stats_carry_the_values_of_active_diagnostics(
      self,
  ) -> None:
    stats = stats_lib.TrajectoryStats.from_trajectory(
        _tunix_trajectory(masked_out=True), diagnostics=(_TUNIX_DIAGNOSTICS,)
    )

    self.assertEqual(stats.schema_stats, {"tunix": {"masked_out": True}})
    self.assertEqual(
        stats.to_json_dict()["schema_stats"], {"tunix": {"masked_out": True}}
    )
    self.assertIs(stats.to_row()["tunix__masked_out"], True)

  def test_diagnostics_skip_trajectories_of_other_schemas(self) -> None:
    stats = stats_lib.TrajectoryStats.from_trajectory(
        _trajectory(), diagnostics=(_TUNIX_DIAGNOSTICS,)
    )

    self.assertEmpty(stats.schema_stats)
    self.assertEqual(stats.to_json_dict()["schema_stats"], {})
    self.assertEqual(list(stats.to_row()), list(stats_lib.COLUMN_DTYPES))

  def test_run_df_gets_a_nullable_column_per_schema_column(self) -> None:
    run = stats_lib.RunStats.from_trajectories(
        [
            _tunix_trajectory("traj_1", masked_out=True),
            _tunix_trajectory("traj_2", masked_out=False),
            _tunix_trajectory("traj_3"),
            _trajectory(),
        ],
        diagnostics=(_TUNIX_DIAGNOSTICS,),
    )

    self.assertEqual(
        run.df.columns.tolist(),
        [*stats_lib.COLUMN_DTYPES, "tunix__masked_out"],
    )
    masked_out = run.df["tunix__masked_out"]
    self.assertEqual(str(masked_out.dtype), "boolean")
    # Trajectories that don't report the value, or aren't of the schema, get
    # nulls.
    self.assertEqual(masked_out.isna().tolist(), [False, False, True, True])
    self.assertEqual(masked_out.dropna().tolist(), [True, False])
    self.assertEqual(
        run.columns_of(_TUNIX_DIAGNOSTICS).columns.tolist(), ["masked_out"]
    )
    self.assertEqual(
        run.schema_stats,
        {"tunix": {"num_masked_out": 1, "num_with_masked_out": 2}},
    )

  def test_trajectories_can_be_a_one_shot_iterator(self) -> None:
    # `trajectories` may be a one-shot iterator, such as a generator.
    run = stats_lib.RunStats.from_trajectories(
        iter([_tunix_trajectory(masked_out=True), _trajectory()]),
        diagnostics=(_TUNIX_DIAGNOSTICS,),
    )

    self.assertEqual(
        run.df["tunix__masked_out"].isna().tolist(), [False, True]
    )

  def test_active_diagnostics_keep_the_order_passed(self) -> None:
    # The first trajectory matches the second diagnostics passed.
    run = stats_lib.RunStats.from_trajectories(
        [
            _EvalMetadata(agent=_AGENT, verdict="pass").create_trajectory(
                steps=[]
            ),
            _tunix_trajectory(masked_out=True),
        ],
        diagnostics=(_TUNIX_DIAGNOSTICS, _EvalDiagnostics()),
    )

    self.assertEqual(
        run.df.columns.tolist()[len(stats_lib.COLUMN_DTYPES) :],
        [
            "tunix__masked_out",
            "acme_eval__verdict",
            "acme_eval__num_retries",
            "acme_eval__latency_ms",
        ],
    )
    self.assertEqual(list(run.schema_stats), ["tunix", "acme_eval"])
    # Each row is null in the other schema's columns.
    self.assertEqual(
        run.df["tunix__masked_out"].isna().tolist(), [True, False]
    )
    self.assertEqual(
        run.df["acme_eval__verdict"].isna().tolist(), [False, True]
    )

  def test_schema_values_serialize_as_json(self) -> None:
    run = stats_lib.RunStats.from_trajectories(
        [_tunix_trajectory(masked_out=False)],
        diagnostics=(_TUNIX_DIAGNOSTICS,),
    )

    payload = json.loads(json.dumps(run.to_json_dict()))
    self.assertEqual(
        payload["schema_stats"],
        {"tunix": {"num_masked_out": 0, "num_with_masked_out": 1}},
    )
    # `summary --json` rows are `df` records: a boolean cell must come out as a
    # plain `bool`, which `json` can serialize (unlike `numpy.bool_`).
    (row,) = run.df.to_dict(orient="records")
    self.assertEqual(json.dumps(row["tunix__masked_out"]), "false")

  def test_columns_are_cast_to_their_nullable_dtypes(self) -> None:
    diagnostics = _EvalDiagnostics()
    run = stats_lib.RunStats.from_trajectories(
        [
            _EvalMetadata(
                agent=_AGENT, verdict="pass", num_retries=2, latency_ms=1.5
            ).create_trajectory(steps=[]),
            _EvalMetadata(agent=_AGENT).create_trajectory(steps=[]),
        ],
        diagnostics=(diagnostics,),
    )

    columns = run.columns_of(diagnostics)
    self.assertEqual(
        columns.dtypes.astype(str).tolist(), ["string", "Int64", "Float64"]
    )
    self.assertEqual(columns.iloc[0].tolist(), ["pass", 2, 1.5])
    self.assertTrue(columns.iloc[1].isna().all())

  def test_integer_columns_with_nulls_keep_large_values_exact(self) -> None:
    diagnostics = _EvalDiagnostics()
    run = stats_lib.RunStats.from_trajectories(
        [
            _EvalMetadata(
                agent=_AGENT, num_retries=2**53 + 1
            ).create_trajectory(steps=[]),
            _EvalMetadata(agent=_AGENT).create_trajectory(steps=[]),
        ],
        diagnostics=(diagnostics,),
    )

    num_retries = run.columns_of(diagnostics)["num_retries"]
    self.assertEqual(num_retries.isna().tolist(), [False, True])
    # 2**53 + 1 is the smallest int a float64 cannot hold exactly.
    self.assertEqual(num_retries.iloc[0], 2**53 + 1)

  @parameterized.named_parameters(
      ("run_of_another_schema", 2),
      ("empty_run", 0),
  )
  def test_diagnostics_that_match_no_trajectory_are_inactive(
      self, num_trajectories: int
  ) -> None:
    run = stats_lib.RunStats.from_trajectories(
        [_trajectory() for _ in range(num_trajectories)],
        diagnostics=(_TUNIX_DIAGNOSTICS,),
    )

    self.assertEmpty(run.schema_stats)
    self.assertEqual(run.df.columns.tolist(), list(stats_lib.COLUMN_DTYPES))
    self.assertEqual(run.to_json_dict()["schema_stats"], {})
    with self.assertRaisesRegex(KeyError, "'tunix' is not active"):
      run.columns_of(_TUNIX_DIAGNOSTICS)

  def test_diagnostics_must_have_distinct_namespaces(self) -> None:
    diagnostics = (_TUNIX_DIAGNOSTICS, tunix_diagnostics.TunixDiagnostics())

    with self.assertRaisesRegex(ValueError, "tunix"):
      stats_lib.TrajectoryStats.from_trajectory(
          _tunix_trajectory(), diagnostics=diagnostics
      )
    # An empty run never reaches the per-trajectory check above, so this pins
    # the up-front check of `from_trajectories` itself.
    with self.assertRaisesRegex(ValueError, "tunix"):
      stats_lib.RunStats.from_trajectories([], diagnostics=diagnostics)

  def test_computed_columns_are_checked(self) -> None:
    class Untyped(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):
      COLUMN_DTYPES = {"verdict": "string"}

      def compute_columns(
          self, record: _EvalMetadata
      ) -> Mapping[str, schema_diagnostics.JsonScalar]:
        del record  # Unused.
        return {"verdict": 1}

    with self.assertRaisesRegex(TypeError, "Untyped"):
      stats_lib.TrajectoryStats.from_trajectory(
          _EvalMetadata(agent=_AGENT).create_trajectory(steps=[]),
          diagnostics=(Untyped(),),
      )


if __name__ == "__main__":
  absltest.main()
