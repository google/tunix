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

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.explorer.diagnostics import stats as stats_lib

_AGENT = trajectory_lib.Agent(name="a", version="1.0")


def _trajectory(
    steps: list[trajectory_lib.Step] | None = None,
    **kwargs,
) -> trajectory_lib.Trajectory:
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
    **kwargs,
) -> trajectory_lib.Step:
  """Builds a base ATIF step."""
  return trajectory_lib.Step(
      step_id=step_id, source=source, message="m", **kwargs
  )


class StepCountsTest(absltest.TestCase):

  def test_counts_are_split_by_source(self) -> None:
    trajectory = _trajectory([
        _step(1, trajectory_lib.Source.SYSTEM),
        _step(2, trajectory_lib.Source.USER),
        _step(3, trajectory_lib.Source.AGENT),
        _step(4, trajectory_lib.Source.AGENT),
    ])

    steps = stats_lib.from_trajectory(trajectory).steps

    self.assertEqual(steps.total, 4)
    self.assertEqual(steps.agent, 2)
    self.assertEqual(steps.user, 1)
    self.assertEqual(steps.system, 1)

  def test_a_trajectory_with_no_steps_counts_zero(self) -> None:
    steps = stats_lib.from_trajectory(_trajectory()).steps

    self.assertEqual(steps.total, 0)


class TokenCountsTest(absltest.TestCase):

  def test_final_metrics_are_authoritative(self) -> None:
    # Per-step metrics disagree; final_metrics is defined as the sum over
    # steps, so the producer's own total wins.
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

    tokens = stats_lib.from_trajectory(trajectory).tokens

    self.assertTrue(tokens.reported)
    self.assertEqual(tokens.prompt, 100)
    self.assertEqual(tokens.completion, 20)
    self.assertEqual(tokens.cached, 50)
    self.assertEqual(tokens.total, 120)

  def test_steps_are_summed_when_final_metrics_are_absent(self) -> None:
    trajectory = _trajectory([
        _step(
            1,
            trajectory_lib.Source.AGENT,
            metrics=trajectory_lib.Metrics(
                prompt_tokens=10, completion_tokens=2, cached_tokens=1
            ),
        ),
        _step(
            2,
            trajectory_lib.Source.AGENT,
            metrics=trajectory_lib.Metrics(
                prompt_tokens=20, completion_tokens=3, cached_tokens=4
            ),
        ),
    ])

    tokens = stats_lib.from_trajectory(trajectory).tokens

    self.assertTrue(tokens.reported)
    self.assertEqual(tokens.prompt, 30)
    self.assertEqual(tokens.completion, 5)
    self.assertEqual(tokens.cached, 5)

  def test_steps_are_summed_when_final_metrics_carry_no_token_totals(
      self,
  ) -> None:
    trajectory = _trajectory(
        [
            _step(
                1,
                trajectory_lib.Source.AGENT,
                metrics=trajectory_lib.Metrics(prompt_tokens=7),
            )
        ],
        # Producers that only count steps must not mask the per-step metrics.
        final_metrics=trajectory_lib.FinalMetrics(total_steps=1),
    )

    tokens = stats_lib.from_trajectory(trajectory).tokens

    self.assertTrue(tokens.reported)
    self.assertEqual(tokens.prompt, 7)

  def test_unreported_tokens_are_distinguished_from_zero(self) -> None:
    # The file backend is routinely filled by producers that record no
    # metrics at all; that must not be reported as a measured zero.
    tokens = stats_lib.from_trajectory(
        _trajectory([_step(1, trajectory_lib.Source.USER)])
    ).tokens

    self.assertFalse(tokens.reported)
    self.assertEqual(
        tokens.to_json_dict(),
        {"prompt": None, "completion": None, "cached": None, "total": None},
    )

  def test_a_measured_zero_is_reported(self) -> None:
    trajectory = _trajectory(
        final_metrics=trajectory_lib.FinalMetrics(total_prompt_tokens=0)
    )

    tokens = stats_lib.from_trajectory(trajectory).tokens

    self.assertTrue(tokens.reported)
    self.assertEqual(tokens.total, 0)

  def test_steps_with_metrics_but_no_token_fields_are_unreported(self) -> None:
    trajectory = _trajectory([
        _step(
            1,
            trajectory_lib.Source.AGENT,
            metrics=trajectory_lib.Metrics(cost_usd=0.01),
        )
    ])

    tokens = stats_lib.from_trajectory(trajectory).tokens

    self.assertFalse(tokens.reported)


class TokenCountsFromMetricsTest(absltest.TestCase):

  def test_absent_metrics_are_unreported(self) -> None:
    self.assertEqual(
        stats_lib.TokenCounts.from_metrics(None), stats_lib.TokenCounts()
    )

  def test_metrics_without_token_fields_are_unreported(self) -> None:
    metrics = trajectory_lib.Metrics(cost_usd=0.01)

    self.assertFalse(stats_lib.TokenCounts.from_metrics(metrics).reported)

  def test_partial_token_fields_default_the_rest_to_zero(self) -> None:
    metrics = trajectory_lib.Metrics(prompt_tokens=7)

    self.assertEqual(
        stats_lib.TokenCounts.from_metrics(metrics),
        stats_lib.TokenCounts(prompt=7, reported=True),
    )

  def test_a_measured_zero_is_reported(self) -> None:
    metrics = trajectory_lib.Metrics(completion_tokens=0)

    self.assertTrue(stats_lib.TokenCounts.from_metrics(metrics).reported)


class TokenCountsSumReportedTest(absltest.TestCase):

  def test_unreported_counts_are_skipped(self) -> None:
    summed = stats_lib.TokenCounts.sum_reported([
        stats_lib.TokenCounts(prompt=1, completion=2, cached=3, reported=True),
        stats_lib.TokenCounts(),
        stats_lib.TokenCounts(prompt=4, completion=5, cached=6, reported=True),
    ])

    self.assertEqual(
        summed,
        stats_lib.TokenCounts(prompt=5, completion=7, cached=9, reported=True),
    )

  def test_nothing_reported_sums_to_unreported(self) -> None:
    summed = stats_lib.TokenCounts.sum_reported([stats_lib.TokenCounts()])

    self.assertFalse(summed.reported)

class ToolUsageTest(absltest.TestCase):

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

    tools = stats_lib.from_trajectory(trajectory).tools

    self.assertEqual(tools.sequence, ("b", "a", "b"))
    self.assertEqual(tools.total_calls, 3)
    self.assertEqual(tools.counts, {"b": 2, "a": 1})

  def test_no_tool_calls(self) -> None:
    tools = stats_lib.from_trajectory(_trajectory()).tools

    self.assertEqual(tools.total_calls, 0)
    self.assertEqual(tools.sequence, ())


class FinalRewardTest(absltest.TestCase):

  def test_tunix_total_reward_wins(self) -> None:
    trajectory = trajectory_lib.TunixTrajectory(
        trajectory_id="t",
        agent=_AGENT,
        total_reward=0.75,
        extra={"reward": 0.1},
    )

    self.assertEqual(stats_lib.from_trajectory(trajectory).final_reward, 0.75)

  def test_root_extra_is_used_by_base_atif_producers(self) -> None:
    trajectory = _trajectory(extra={"reward": 1.0})

    self.assertEqual(stats_lib.from_trajectory(trajectory).final_reward, 1.0)

  def test_final_metrics_extra_is_used(self) -> None:
    trajectory = _trajectory(
        final_metrics=trajectory_lib.FinalMetrics(extra={"reward": 0.5})
    )

    self.assertEqual(stats_lib.from_trajectory(trajectory).final_reward, 0.5)

  def test_environment_step_rewards_sum_to_the_episode_return(self) -> None:
    trajectory = trajectory_lib.TunixTrajectory(
        trajectory_id="t",
        agent=_AGENT,
        steps=[
            trajectory_lib.TunixEnvStep(
                step_id=0,
                source=trajectory_lib.Source.USER,
                message="m",
                reward=0.25,
            ),
            trajectory_lib.TunixEnvStep(
                step_id=1,
                source=trajectory_lib.Source.USER,
                message="m",
                reward=0.5,
            ),
        ],
    )

    self.assertEqual(stats_lib.from_trajectory(trajectory).final_reward, 0.75)

  def test_a_zero_reward_is_reported_rather_than_treated_as_absent(
      self,
  ) -> None:
    trajectory = _trajectory(extra={"reward": 0.0})

    stats = stats_lib.from_trajectory(trajectory)

    self.assertIsNotNone(stats.final_reward)
    self.assertEqual(stats.final_reward, 0.0)

  def test_absent_reward(self) -> None:
    stats = stats_lib.from_trajectory(_trajectory())

    self.assertIsNone(stats.final_reward)


class NonNumericRewardTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("string", "high"),
      # bool is an int subclass, but a flag is not a reward.
      ("bool", True),
      ("none", None),
      ("list", [1.0]),
  )
  def test_non_numeric_rewards_are_ignored(self, value: object) -> None:
    trajectory = _trajectory(extra={"reward": value})

    self.assertIsNone(stats_lib.from_trajectory(trajectory).final_reward)


class StatusTest(absltest.TestCase):

  def test_tunix_status(self) -> None:
    trajectory = trajectory_lib.TunixTrajectory(
        trajectory_id="t", agent=_AGENT, status="FAILED"
    )

    self.assertEqual(stats_lib.from_trajectory(trajectory).status, "FAILED")

  def test_tunix_falls_back_to_root_extra_status(self) -> None:
    trajectory = trajectory_lib.TunixTrajectory(
        trajectory_id="t", agent=_AGENT, extra={"status": "COMPLETED"}
    )

    self.assertEqual(stats_lib.from_trajectory(trajectory).status, "COMPLETED")

  def test_root_extra_status(self) -> None:
    trajectory = _trajectory(extra={"status": "COMPLETED"})

    self.assertEqual(stats_lib.from_trajectory(trajectory).status, "COMPLETED")

  def test_absent_status(self) -> None:
    self.assertIsNone(stats_lib.from_trajectory(_trajectory()).status)


class _CustomTunixTrajectory(trajectory_lib.TunixTrajectory):
  """A TunixTrajectory subclass."""


class TunixSubclassTest(absltest.TestCase):

  def test_tunix_subclass_outcome_is_reported(self) -> None:
    trajectory = _CustomTunixTrajectory(
        trajectory_id="t", agent=_AGENT, total_reward=0.5, status="COMPLETED"
    )

    stats = stats_lib.from_trajectory(trajectory)

    self.assertEqual(stats.final_reward, 0.5)
    self.assertEqual(stats.status, "COMPLETED")


def _tunix_trajectory_read_as_base() -> trajectory_lib.Trajectory:
  """Returns a Tunix trajectory as an unparameterized store may yield it.

  Stores persist base ATIF, packing Tunix fields into
  `extra[TUNIX_EXTENSIONS_KEY]`.
  """
  return (
      trajectory_lib.TunixTrajectoryMetadata(
          trajectory_id="t",
          agent=_AGENT,
          total_reward=0.85,
          status="COMPLETED",
      )
      .to_atif_metadata()
      .create_trajectory()
  )


class PackedExtensionsTest(absltest.TestCase):

  def test_packed_tunix_outcome_is_reported(self) -> None:
    stats = stats_lib.from_trajectory(_tunix_trajectory_read_as_base())

    self.assertEqual(stats.final_reward, 0.85)
    self.assertEqual(stats.status, "COMPLETED")

  def test_a_run_mixing_packed_and_base_trajectories(self) -> None:
    run = stats_lib.from_trajectories([
        _tunix_trajectory_read_as_base(),
        _trajectory(extra={"reward": 0.15}),
        _trajectory(),
    ])

    self.assertEqual(run.rewards, (0.85, 0.15))


class RunStatsTest(absltest.TestCase):

  def test_totals_sum_over_trajectories(self) -> None:
    run = stats_lib.from_trajectories([
        _trajectory(
            [_step(i, trajectory_lib.Source.AGENT) for i in range(1, 7)],
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=520,
                total_completion_tokens=90,
                total_cached_tokens=220,
            ),
        ),
        _trajectory(
            [_step(i, trajectory_lib.Source.AGENT) for i in range(1, 5)],
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=180,
                total_completion_tokens=35,
                total_cached_tokens=0,
            ),
        ),
    ])

    self.assertLen(run.trajectories, 2)
    self.assertEqual(run.total_steps, 10)
    self.assertEqual(run.tokens.prompt, 700)
    self.assertEqual(run.tokens.completion, 125)
    self.assertEqual(run.tokens.cached, 220)
    self.assertEqual(run.tokens.total, 825)

  def test_mean_reward_averages_only_reporting_trajectories(self) -> None:
    run = stats_lib.from_trajectories([
        _trajectory(extra={"reward": 1.0}),
        _trajectory(extra={"reward": 0.0}),
        _trajectory(),
    ])

    self.assertEqual(run.mean_reward, 0.5)
    self.assertLen(run.rewards, 2)

  def test_mean_reward_is_absent_when_nothing_reports_one(self) -> None:
    run = stats_lib.from_trajectories([_trajectory(), _trajectory()])

    self.assertIsNone(run.mean_reward)

  def test_tool_counts_are_ordered_by_frequency_then_name(self) -> None:
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

    run = stats_lib.from_trajectories(
        [with_tools("b", "c"), with_tools("c", "a")]
    )

    self.assertEqual(list(run.tool_counts), ["c", "a", "b"])

  def test_token_totals_are_unreported_when_no_trajectory_reports_any(
      self,
  ) -> None:
    run = stats_lib.from_trajectories([_trajectory()])

    self.assertFalse(run.tokens.reported)

  def test_an_empty_run(self) -> None:
    run = stats_lib.from_trajectories([])

    self.assertEqual(run.total_steps, 0)
    self.assertIsNone(run.mean_reward)
    self.assertEmpty(run.tool_counts)


class JsonDictTest(absltest.TestCase):

  def test_absent_trajectory_id_serializes_as_null(self) -> None:
    trajectory = trajectory_lib.Trajectory(agent=_AGENT, steps=[])

    payload = stats_lib.from_trajectory(trajectory).to_json_dict()

    self.assertIsNone(payload["trajectory_id"])

  def test_unreported_tokens_serialize_as_null(self) -> None:
    payload = stats_lib.from_trajectory(_trajectory()).to_json_dict()

    self.assertEqual(
        payload["tokens"],
        {"prompt": None, "completion": None, "cached": None, "total": None},
    )

  def test_stats_serialize_every_reported_quantity(self) -> None:
    def _agent_step(
        step_id: int, tool: str | None = None
    ) -> trajectory_lib.TunixAgentStep:
      tool_calls = None
      if tool:
        tool_calls = [
            trajectory_lib.ToolCall(
                tool_call_id=str(step_id), function_name=tool
            )
        ]
      return trajectory_lib.TunixAgentStep(
          step_id=step_id,
          source=trajectory_lib.Source.AGENT,
          message="m",
          tool_calls=tool_calls,
      )

    def _env_step(step_id: int) -> trajectory_lib.TunixEnvStep:
      return trajectory_lib.TunixEnvStep(
          step_id=step_id,
          source=trajectory_lib.Source.USER,
          message="m",
      )

    trajectory = trajectory_lib.TunixTrajectory(
        trajectory_id="traj_1001",
        agent=_AGENT,
        status="COMPLETED",
        total_reward=0.85,
        steps=[
            _agent_step(0, "find_files"),
            _env_step(1),
            _agent_step(2, "grep_file"),
            _env_step(3),
            _agent_step(4),
            _env_step(5),
        ],
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=520,
            total_completion_tokens=90,
            total_cached_tokens=220,
        ),
    )

    payload = stats_lib.from_trajectory(trajectory).to_json_dict()

    self.assertEqual(payload["trajectory_id"], "traj_1001")
    self.assertEqual(payload["status"], "COMPLETED")
    self.assertEqual(payload["final_reward"], 0.85)
    self.assertEqual(payload["steps"]["total"], 6)
    self.assertEqual(payload["tokens"]["total"], 610)
    self.assertEqual(payload["tools"]["sequence"], ["find_files", "grep_file"])


if __name__ == "__main__":
  absltest.main()
