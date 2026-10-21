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

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.experimental.orchestrator import step_telemetry


def _rollout_lineage(
    dispatch_s: float,
    finish_s: float,
    worker_id: str = "rollout-0",
    *,
    execution_time_s: float | None = None,
    batch_idx: int | None = None,
) -> lineage.LineageContext:
  """Lineage as a rollout carries it back: dispatched, then finished."""
  dispatch_attrs = {}
  if batch_idx is not None:
    dispatch_attrs["batch_idx"] = batch_idx
  finish_attrs: dict[str, float | str] = {"worker_id": worker_id}
  if execution_time_s is not None:
    finish_attrs["execution_time_s"] = execution_time_s
  return lineage.LineageContext(
      tracking_id="traj",
      events=[
          lineage.LineageEvent(
              component="engine.dispatch",
              operation="rollout",
              timestamp_s=dispatch_s,
              attributes=dispatch_attrs,
          ),
          lineage.LineageEvent(
              component="worker.rollout",
              operation="generate",
              timestamp_s=finish_s,
              attributes=finish_attrs,
          ),
      ],
  )


def _health_item(
    group_index=0,
    *,
    status=datatypes.TrajectoryStatus.SUCCEEDED,
    num_turns=None,
    turn_lengths=None,
    window=None,
    worker_id="rollout-0",
    execution_time_s=None,
    batch_idx=None,
    env_time=None,
    reward_time=None,
):
  """A scored rollout carrying what the rollout health metrics read."""
  traj = {"trajectory_reward": 1.0, "status": status}
  if num_turns is not None:
    traj["conversation_text"] = [{"role": "user", "content": "task"}] + [
        {"role": "assistant", "content": "act"},
        {"role": "user", "content": "observation"},
    ] * num_turns
  if turn_lengths is not None:
    traj["assistant_turn_lengths"] = list(turn_lengths)
  if env_time is not None:
    traj["env_time"] = env_time
  if reward_time is not None:
    traj["reward_time"] = reward_time
  metadata = {}
  if batch_idx is not None:
    metadata["batch_idx"] = batch_idx
  if window is not None:
    metadata["lineage"] = _rollout_lineage(
        *window,
        worker_id=worker_id,
        execution_time_s=execution_time_s,
        batch_idx=batch_idx,
    )
  return datatypes.TrajectoryItem(
      prompt_id="p0", group_index=group_index, traj=traj, metadata=metadata
  )


def _facts(*items):
  return [step_telemetry.RolloutFacts.of(item) for item in items]


class RolloutFactsTest(absltest.TestCase):
  """Covers `RolloutFacts.of`, which reads each rollout once per step."""

  def test_reads_what_the_step_telemetry_needs(self):
    facts = step_telemetry.RolloutFacts.of(
        _health_item(
            1,
            status=datatypes.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED,
            num_turns=2,
            turn_lengths=[5, 57344],
            window=(100.0, 130.0),
            worker_id="w1",
        )
    )

    self.assertEqual(facts.traj_id, "traj_p0_g1")
    self.assertEqual(facts.status, "MAX_CONTEXT_LIMIT_REACHED")
    self.assertEqual(facts.num_turns, 2)
    self.assertEqual(facts.turn_lengths, [5, 57344])
    self.assertEqual(
        facts.window,
        step_telemetry.RolloutWindow(
            dispatch_time_s=100.0, finish_time_s=130.0, worker_id="w1"
        ),
    )
    self.assertEqual(facts.window.duration_s, 30.0)

  def test_missing_inputs_stay_unknown(self):
    # Engine error placeholders carry no conversation and lineage is optional,
    # so these are unknown rather than zero.
    facts = step_telemetry.RolloutFacts.of(_health_item(0))

    self.assertIsNone(facts.num_turns)
    self.assertIsNone(facts.turn_lengths)
    self.assertIsNone(facts.window)

  def test_rollout_without_status_raises(self):
    # `critique_stage` stamps every scored rollout's status, so a missing one
    # is a broken contract rather than an unknown.
    no_status = _health_item(0)
    del no_status.traj["status"]

    with self.assertRaises(KeyError):
      step_telemetry.RolloutFacts.of(no_status)

  def test_lineage_without_a_finish_event_has_no_window(self):
    item = _health_item(0, window=(100.0, 110.0))
    item.metadata["lineage"].events.pop()

    self.assertIsNone(step_telemetry.RolloutFacts.of(item).window)


class RolloutHealthTest(absltest.TestCase):
  """Covers `rollout_health`, the per-step rollout health view."""

  def test_reports_every_status_every_step(self):
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0),
            _health_item(1),
            _health_item(2),
            _health_item(
                3, status=datatypes.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
            ),
        )
    )

    self.assertEqual(health.metrics["rollout/status/succeeded_frac"], 0.75)
    self.assertEqual(health.metrics["rollout/success_rate"], 0.75)
    self.assertEqual(
        health.metrics["rollout/status/max_context_limit_reached_frac"], 0.25
    )
    # Statuses that did not occur still report 0, so series stay continuous.
    self.assertEqual(health.metrics["rollout/status/timeout_frac"], 0.0)
    self.assertLen(
        [k for k in health.metrics if k.startswith("rollout/status/")],
        len(datatypes.TrajectoryStatus),
    )

  def test_counts_rollouts_per_status(self):
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0),
            _health_item(1, status=datatypes.TrajectoryStatus.TIMEOUT),
            _health_item(2, status=datatypes.TrajectoryStatus.TIMEOUT),
        )
    )

    # Every status is listed, in enum order, zeros included.
    self.assertEqual(
        list(health.status_counts),
        [status.name for status in datatypes.TrajectoryStatus],
    )
    self.assertEqual(health.status_counts["SUCCEEDED"], 1)
    self.assertEqual(health.status_counts["TIMEOUT"], 2)
    self.assertEqual(health.status_counts["FAILED"], 0)

  def test_counts_assistant_turns_from_the_conversation(self):
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0, num_turns=2),
            _health_item(1, num_turns=5),
            # No conversation: its turn count is unknown, not zero.
            _health_item(2),
        )
    )

    self.assertEqual(health.metrics["rollout/num_turns_mean"], 3.5)
    self.assertEqual(health.metrics["rollout/num_turns_max"], 5.0)

  def test_pools_turn_lengths_across_rollouts(self):
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0, turn_lengths=[10, 30]),
            _health_item(1, turn_lengths=[20]),
            _health_item(2, turn_lengths=[]),
        )
    )

    self.assertEqual(health.metrics["rollout/turn_length/mean"], 20.0)
    self.assertEqual(health.metrics["rollout/turn_length/p50"], 20.0)
    self.assertAlmostEqual(health.metrics["rollout/turn_length/p99"], 29.8)
    self.assertEqual(health.metrics["rollout/turn_length/max"], 30.0)

  def test_durations_come_from_lineage(self):
    # Durations are 10, 20, 30 and 40 s; the batch runs from the first
    # dispatch (t=100) to the last finish (t=150).
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0, window=(100.0, 110.0)),
            _health_item(1, window=(100.0, 120.0)),
            _health_item(2, window=(105.0, 135.0)),
            _health_item(3, window=(110.0, 150.0)),
            # No lineage: left out rather than counted as instant.
            _health_item(4),
        )
    )

    self.assertEqual(health.metrics["rollout/duration_sec/p50"], 25.0)
    self.assertAlmostEqual(health.metrics["rollout/duration_sec/p90"], 37.0)
    self.assertAlmostEqual(health.metrics["rollout/duration_sec/p99"], 39.7)
    self.assertEqual(health.metrics["rollout/duration_sec/max"], 40.0)
    self.assertEqual(health.metrics["rollout/tail_sec"], 15.0)
    self.assertEqual(health.metrics["rollout/batch_wall_sec"], 50.0)

  def test_separates_worker_execution_from_queue_wait_and_logs_env_reward_time(
      self,
  ):
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(
                0,
                window=(100.0, 130.0),
                execution_time_s=10.0,
                env_time={
                    "reset_latency": 1.0,
                    "step_latency": [2.0, 3.0],
                    "close_latency": 0.5,
                },
                reward_time={"reward_latency": 0.25},
            ),
            _health_item(
                1,
                window=(100.0, 150.0),
                execution_time_s=20.0,
                env_time={
                    "reset_latency": 1.5,
                    "step_latency": [4.0],
                    "close_latency": 1.0,
                },
                reward_time={"reward_latency": 0.75},
            ),
        )
    )

    self.assertEqual(health.metrics["rollout/execution_sec/p50"], 15.0)
    self.assertEqual(health.metrics["rollout/execution_sec/max"], 20.0)
    # Total durations are 30s and 50s; execution times are 10s and 20s, so
    # queue wait times are 20s and 30s.
    self.assertEqual(health.metrics["rollout/queue_wait_sec/p50"], 25.0)
    self.assertEqual(health.metrics["rollout/queue_wait_sec/max"], 30.0)
    self.assertEqual(health.metrics["rollout/env_time_sec/mean"], 6.5)
    self.assertEqual(health.metrics["rollout/reward_time_sec/mean"], 0.5)

  def test_groups_batch_wall_sec_by_batch_idx_cohort(self):
    # Cohort 0 ran in [100, 140] (40s); cohort 1 ran in [300, 330] (30s).
    # Grouping by `batch_idx` reports max(40, 30) = 40s rather than 330 - 100.
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0, window=(100.0, 140.0), batch_idx=0),
            _health_item(1, window=(300.0, 330.0), batch_idx=1),
        )
    )

    self.assertEqual(health.metrics["rollout/batch_wall_sec"], 40.0)

  def test_omits_families_without_inputs(self):
    health = step_telemetry.rollout_health(_facts(_health_item(0)))

    self.assertIn("rollout/status/succeeded_frac", health.metrics)
    self.assertNotIn("rollout/num_turns_mean", health.metrics)
    self.assertNotIn("rollout/turn_length/max", health.metrics)
    self.assertNotIn("rollout/duration_sec/max", health.metrics)
    self.assertNotIn("rollout/batch_wall_sec", health.metrics)

  def test_no_rollouts_report_no_metrics(self):
    health = step_telemetry.rollout_health([])

    self.assertEmpty(health.metrics)
    self.assertEqual(sum(health.status_counts.values()), 0)


class SlowestRolloutsSummaryTest(absltest.TestCase):
  """Covers `slowest_rollouts_summary`, the per-step slowest-rollouts log."""

  def test_lists_the_slowest_rollouts_first(self):
    facts = _facts(
        _health_item(
            0,
            window=(0.0, 10.0),
            num_turns=3,
            turn_lengths=[5, 6, 7],
            worker_id="w0",
        ),
        _health_item(
            1,
            status=datatypes.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED,
            window=(0.0, 300.0),
            execution_time_s=280.0,
            env_time={
                "reset_latency": 1.0,
                "step_latency": [14.0],
                "close_latency": 0.0,
            },
            num_turns=1,
            turn_lengths=[57344],
            worker_id="w1",
        ),
        _health_item(2, window=(0.0, 50.0), worker_id="w2"),
    )

    summary = step_telemetry.slowest_rollouts_summary(facts, k=2)

    self.assertEqual(
        summary,
        "slowest 2 of 3: traj_p0_g1 dur=300.0s exec=280.0s env=15.0s"
        " status=MAX_CONTEXT_LIMIT_REACHED turns=1 max_turn_length=57344"
        " worker=w1 | traj_p0_g2 dur=50.0s status=SUCCEEDED worker=w2",
    )

  def test_returns_none_without_lineage(self):
    self.assertIsNone(
        step_telemetry.slowest_rollouts_summary(_facts(_health_item(0)), k=3)
    )


if __name__ == "__main__":
  absltest.main()
