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
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import lineage
from tunix.dist_rl.orchestrator import step_telemetry


def _rollout_lineage(
    dispatch_s: float,
    finish_s: float,
    worker_id: str = "rollout-0",
    *,
    received_s: float | None = None,
    execution_time_s: float | None = None,
    batch_idx: int | None = None,
    task_name: str = "",
) -> lineage.LineageContext:
  """Lineage a rollout carries back: dispatched, received, then finished."""
  dispatch_attrs = {}
  if batch_idx is not None:
    dispatch_attrs["batch_idx"] = batch_idx
  finish_attrs: dict[str, float | str] = {"worker_id": worker_id}
  if execution_time_s is not None:
    finish_attrs["execution_time_s"] = execution_time_s
  events = [
      lineage.LineageEvent(
          component="engine.dispatch",
          operation="rollout",
          timestamp_s=dispatch_s,
          attributes=dispatch_attrs,
      ),
  ]
  if received_s is not None:
    events.append(
        lineage.LineageEvent(
            component="worker.received",
            operation="generate",
            timestamp_s=received_s,
            attributes={"worker_id": worker_id},
        )
    )
  events.append(
      lineage.LineageEvent(
          component="worker.rollout",
          operation="generate",
          timestamp_s=finish_s,
          attributes=finish_attrs,
      )
  )
  return lineage.LineageContext(
      tracking_id="traj",
      task_name=task_name,
      events=events,
  )


def _health_item(
    group_index=0,
    *,
    prompt_id="p0",
    status=datatypes.TrajectoryStatus.SUCCEEDED,
    num_turns=None,
    turn_lengths=None,
    window=None,
    received_time_s=None,
    worker_id="rollout-0",
    execution_time_s=None,
    batch_idx=None,
    task_name="",
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
        received_s=received_time_s,
        execution_time_s=execution_time_s,
        batch_idx=batch_idx,
        task_name=task_name,
    )
  return datatypes.TrajectoryItem(
      prompt_id=prompt_id, group_index=group_index, traj=traj, metadata=metadata
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
            task_name="gsm8k",
        )
    )

    self.assertEqual(facts.traj_id, "traj_p0_g1")
    self.assertEqual(facts.status, "MAX_CONTEXT_LIMIT_REACHED")
    self.assertEqual(facts.num_turns, 2)
    self.assertEqual(facts.turn_lengths, [5, 57344])
    self.assertEqual(facts.task_name, "gsm8k")
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
    self.assertEqual(facts.task_name, "")

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
    self.assertNotIn("rollout/success_rate", health.metrics)
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

  def test_separates_worker_execution_from_queue_wait_and_logs_env_time(
      self,
  ):
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(
                0,
                window=(100.0, 130.0),
                received_time_s=105.0,
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
                received_time_s=120.0,
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

    self.assertEqual(health.metrics["rollout/dispatch_delay_sec/p50"], 12.5)
    self.assertEqual(health.metrics["rollout/dispatch_delay_sec/max"], 20.0)
    self.assertEqual(health.metrics["rollout/execution_sec/p50"], 15.0)
    self.assertEqual(health.metrics["rollout/execution_sec/max"], 20.0)
    self.assertEqual(health.metrics["rollout/queue_wait_sec/p50"], 25.0)
    self.assertEqual(health.metrics["rollout/queue_wait_sec/max"], 30.0)
    # Item 0: finish=130, received=105, exec=10 -> worker_queue_wait=15s.
    # Item 1: finish=150, received=120, exec=20 -> worker_queue_wait=10s.
    self.assertEqual(health.metrics["rollout/worker_queue_wait_sec/p50"], 12.5)
    self.assertEqual(health.metrics["rollout/worker_queue_wait_sec/max"], 15.0)
    self.assertEqual(health.metrics["rollout/env_time_sec/mean"], 6.5)
    # `reward_time` on the trajectory is not reported as a rollout metric.
    self.assertFalse(
        any(k.startswith("rollout/reward_time_sec") for k in health.metrics)
    )

  def test_rerouted_rollout_uses_latest_worker_received_and_isolates_queue_wait(
      self,
  ):
    item = _health_item(0)
    item.metadata["lineage"] = lineage.LineageContext(
        tracking_id="traj_p0_g0",
        events=[
            lineage.LineageEvent(
                component="engine.dispatch",
                operation="rollout",
                timestamp_s=100.0,
            ),
            lineage.LineageEvent(
                component="worker.received",
                operation="generate",
                timestamp_s=101.0,
                attributes={"worker_id": "dead-worker"},
            ),
            lineage.LineageEvent(
                component="worker.received",
                operation="generate",
                timestamp_s=110.0,
                attributes={"worker_id": "winning-worker"},
            ),
            lineage.LineageEvent(
                component="worker.received",
                operation="generate",
                timestamp_s=140.0,
                attributes={"worker_id": "winning-worker"},
            ),
            lineage.LineageEvent(
                component="worker.rollout",
                operation="generate",
                timestamp_s=165.0,
                attributes={
                    "worker_id": "winning-worker",
                    "execution_time_s": 20.0,
                },
            ),
        ],
    )

    facts = step_telemetry.RolloutFacts.of(item)
    self.assertIsNotNone(facts.window)
    self.assertEqual(facts.window.received_time_s, 140.0)
    self.assertEqual(facts.window.worker_id, "winning-worker")

    health = step_telemetry.rollout_health([facts])
    self.assertEqual(health.metrics["rollout/duration_sec/max"], 65.0)
    self.assertEqual(health.metrics["rollout/dispatch_delay_sec/max"], 40.0)
    self.assertEqual(health.metrics["rollout/execution_sec/max"], 20.0)
    self.assertEqual(health.metrics["rollout/queue_wait_sec/max"], 45.0)
    self.assertEqual(health.metrics["rollout/worker_queue_wait_sec/max"], 5.0)

  def test_ignores_worker_received_from_different_worker_id(self):
    item = _health_item(0)
    item.metadata["lineage"] = lineage.LineageContext(
        tracking_id="traj_p0_g0",
        events=[
            lineage.LineageEvent(
                component="engine.dispatch",
                operation="rollout",
                timestamp_s=100.0,
            ),
            lineage.LineageEvent(
                component="worker.received",
                operation="generate",
                timestamp_s=101.0,
                attributes={"worker_id": "dead-worker"},
            ),
            lineage.LineageEvent(
                component="worker.rollout",
                operation="generate",
                timestamp_s=165.0,
                attributes={
                    "worker_id": "winning-worker",
                    "execution_time_s": 20.0,
                },
            ),
        ],
    )

    facts = step_telemetry.RolloutFacts.of(item)
    self.assertIsNotNone(facts.window)
    self.assertIsNone(facts.window.received_time_s)
    self.assertEqual(facts.window.worker_id, "winning-worker")

    health = step_telemetry.rollout_health([facts])
    self.assertNotIn("rollout/dispatch_delay_sec/max", health.metrics)
    self.assertNotIn("rollout/worker_queue_wait_sec/max", health.metrics)
    self.assertEqual(health.metrics["rollout/queue_wait_sec/max"], 45.0)
    self.assertEqual(health.metrics["rollout/execution_sec/max"], 20.0)

  def test_groups_batch_wall_sec_by_batch_idx_cohort(self):
    # Cohort 0 ran in [100, 140] (40s); cohort 1 ran in [300, 330] (30s).
    # Grouping by `batch_idx` reports max(40, 30) = 40s rather than 330 - 100,
    # and scopes `group_straggler_sec` per `(batch_idx, prompt_id)` cohort.
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0, prompt_id="p0", window=(100.0, 100.0), batch_idx=0),
            _health_item(1, prompt_id="p0", window=(100.0, 140.0), batch_idx=0),
            _health_item(0, prompt_id="p0", window=(300.0, 300.0), batch_idx=1),
            _health_item(1, prompt_id="p0", window=(300.0, 330.0), batch_idx=1),
        )
    )

    self.assertEqual(health.metrics["rollout/batch_wall_sec"], 40.0)
    self.assertEqual(health.metrics["rollout/group_straggler_sec/max"], 40.0)

  def test_computes_group_straggler_sec_from_first_to_last_finish_per_prompt(
      self,
  ):
    # Prompt p0 finishes at 110s and 130s (group lag = 20s).
    # Prompt p1 finishes at 105s and 145s (group lag = 40s).
    # Prompt p2 has only 1 sample (skipped for intra-group lag).
    health = step_telemetry.rollout_health(
        _facts(
            _health_item(0, prompt_id="p0", window=(100.0, 110.0)),
            _health_item(1, prompt_id="p0", window=(100.0, 130.0)),
            _health_item(0, prompt_id="p1", window=(100.0, 105.0)),
            _health_item(1, prompt_id="p1", window=(100.0, 145.0)),
            _health_item(0, prompt_id="p2", window=(100.0, 200.0)),
        )
    )

    self.assertEqual(health.metrics["rollout/group_straggler_sec/mean"], 30.0)
    self.assertEqual(health.metrics["rollout/group_straggler_sec/p50"], 30.0)
    self.assertAlmostEqual(
        health.metrics["rollout/group_straggler_sec/p90"], 38.0
    )
    self.assertAlmostEqual(
        health.metrics["rollout/group_straggler_sec/p99"], 39.8
    )
    self.assertEqual(health.metrics["rollout/group_straggler_sec/max"], 40.0)

  def test_omits_families_without_inputs(self):
    health = step_telemetry.rollout_health(_facts(_health_item(0)))

    self.assertIn("rollout/status/succeeded_frac", health.metrics)
    self.assertNotIn("rollout/num_turns_mean", health.metrics)
    self.assertNotIn("rollout/turn_length/max", health.metrics)
    self.assertNotIn("rollout/duration_sec/max", health.metrics)
    self.assertNotIn("rollout/batch_wall_sec", health.metrics)
    self.assertNotIn("rollout/group_straggler_sec/max", health.metrics)

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
