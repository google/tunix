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

"""Unit tests for worker OpenTelemetry scaffolding and role helpers."""

from unittest import mock

from absl.testing import absltest
from opentelemetry import metrics as otel_api
from opentelemetry.sdk import metrics as sdk_metrics
from opentelemetry.sdk.metrics import export as sdk_metrics_export
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import lineage
from tunix.dist_rl.testing import otel_testing
from tunix.dist_rl.worker import otel_metrics


class WorkerMeterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = sdk_metrics_export.InMemoryMetricReader()
    self.provider = sdk_metrics.MeterProvider(metric_readers=[self.reader])
    self.addCleanup(self.provider.shutdown)

  def test_instruments_stamp_worker_id_and_keep_caller_attributes(self):
    meter = otel_metrics.WorkerMeter("tunix.test", "worker_7", self.provider)
    histogram = meter.histogram(
        "tunix.test.duration", unit="s", description="d"
    )
    counter = meter.counter("tunix.test.items", unit="{item}", description="c")

    histogram.record(0.25, phase="a")
    histogram.record(0.5, phase="a")
    counter.add(3, kind="x")
    counter.add(4)

    self.assertTrue(meter.enabled)
    self.assertIs(meter.provider, self.provider)
    metrics = otel_testing.collect_metrics(self.reader)
    self.assertEqual(
        metrics["tunix.test.duration"],
        [(0.75, 2, {"worker_id": "worker_7", "phase": "a"})],
    )
    self.assertCountEqual(
        metrics["tunix.test.items"],
        [
            (3, None, {"worker_id": "worker_7", "kind": "x"}),
            (4, None, {"worker_id": "worker_7"}),
        ],
    )

  def test_two_workers_on_one_provider_stay_separate_series(self):
    first = otel_metrics.WorkerMeter("tunix.test", "worker_a", self.provider)
    second = otel_metrics.WorkerMeter("tunix.test", "worker_b", self.provider)
    first.counter("tunix.test.items", unit="{item}", description="c").add(1)
    second.counter("tunix.test.items", unit="{item}", description="c").add(5)

    metrics = otel_testing.collect_metrics(self.reader)
    self.assertCountEqual(
        metrics["tunix.test.items"],
        [
            (1, None, {"worker_id": "worker_a"}),
            (5, None, {"worker_id": "worker_b"}),
        ],
    )

  def test_falls_back_to_global_provider_when_none_is_injected(self):
    with mock.patch.object(
        otel_api, "get_meter_provider", return_value=self.provider
    ):
      meter = otel_metrics.WorkerMeter("tunix.test", "worker_g")
    meter.counter("tunix.test.items", unit="{item}", description="c").add(2)

    self.assertIs(meter.provider, self.provider)
    metrics = otel_testing.collect_metrics(self.reader)
    self.assertEqual(
        metrics["tunix.test.items"], [(2, None, {"worker_id": "worker_g"})]
    )

  def test_records_nothing_when_opentelemetry_is_not_installed(self):
    with mock.patch.object(otel_metrics, "otel_api", None):
      meter = otel_metrics.WorkerMeter("tunix.test", "worker_0", self.provider)
      histogram = meter.histogram(
          "tunix.test.duration", unit="s", description="d"
      )
      counter = meter.counter(
          "tunix.test.items", unit="{item}", description="c"
      )
    histogram.record(1.0, phase="a")
    counter.add(1)

    self.assertFalse(meter.enabled)
    self.assertIsNone(meter.provider)
    self.assertIsNone(meter.meter)
    self.assertEmpty(otel_testing.collect_metrics(self.reader))


class RolloutWorkerOtelMetricsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = sdk_metrics_export.InMemoryMetricReader()
    self.provider = sdk_metrics.MeterProvider(metric_readers=[self.reader])
    self.addCleanup(self.provider.shutdown)

  def test_weight_sync_phase_durations_follow_perf_counter_seam(self):
    rec = otel_metrics.RolloutWorkerOtelMetrics(
        worker_id="rollout_3",
        inflight_fn=lambda: 2,
        meter_provider=self.provider,
    )
    self.addCleanup(rec.unregister_inflight)

    # Pin exact monotonic timestamps:
    # - Round 1: pre [10.0 -> 10.15] (0.15s), duplicate pre [10.2 -> 10.25]
    #   (0.05s, keeps pause anchor at 10.0), h2d [10.3 -> 10.7] (0.40s),
    #   post [10.8 -> 10.9] (0.10s) => total_paused = 10.9 - 10.0 = 0.90s.
    # - Unopened abort at 11.0 => records nothing.
    # - Round 2: pre starts at 12.0, abort at 12.35 => aborted_paused = 0.35s.
    ticks = [
        10.0,
        10.15,
        10.2,
        10.25,
        10.3,
        10.7,
        10.8,
        10.9,
        11.0,
        12.0,
        12.35,
    ]
    with mock.patch.object(otel_metrics, "perf_counter_s", side_effect=ticks):
      t_pre = rec.record_pre_weight_sync_start()
      rec.record_weight_sync_phase("pre", t_pre)
      t_dup_pre = rec.record_pre_weight_sync_start()
      rec.record_weight_sync_phase("pre", t_dup_pre)
      t_h2d = otel_metrics.perf_counter_s()
      rec.record_weight_sync_phase("h2d", t_h2d)
      t_post = otel_metrics.perf_counter_s()
      rec.record_post_weight_sync(t_post)
      rec.record_abort_weight_sync()
      rec.record_pre_weight_sync_start()
      rec.record_abort_weight_sync()

    metrics = otel_testing.collect_metrics(self.reader)
    by_phase = {
        attrs["phase"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.worker.weight_sync.duration"]
    }
    self.assertCountEqual(
        by_phase.keys(),
        ["pre", "h2d", "post", "total_paused", "aborted_paused"],
    )
    self.assertAlmostEqual(by_phase["pre"][0], 0.20)
    self.assertEqual(by_phase["pre"][1], 2)
    self.assertAlmostEqual(by_phase["h2d"][0], 0.40)
    self.assertAlmostEqual(by_phase["post"][0], 0.10)
    self.assertAlmostEqual(by_phase["total_paused"][0], 0.90)
    self.assertAlmostEqual(by_phase["aborted_paused"][0], 0.35)
    for _, _, attrs in by_phase.values():
      self.assertEqual(attrs["worker_id"], "rollout_3")
    self.assertEqual(
        metrics["tunix.worker.requests_inflight"],
        [(2, None, {"worker_id": "rollout_3"})],
    )

  def test_extract_lineage_and_validate_trajectory_dict_contracts(self):
    self.assertIsNone(otel_metrics.extract_lineage(None))
    self.assertIsNone(otel_metrics.extract_lineage({}))
    ctx = lineage.LineageContext(tracking_id="t1", task_name="gsm8k")
    self.assertIs(otel_metrics.extract_lineage({"lineage": ctx}), ctx)
    with self.assertRaises(TypeError):
      otel_metrics.extract_lineage({"lineage": "bad"})

    traj, status = otel_metrics.validate_trajectory_dict(
        {"status": "SUCCEEDED", "env_time": {}, "reward_time": {}}
    )
    self.assertEqual(status, "SUCCEEDED")
    self.assertEqual(traj["status"], "SUCCEEDED")
    with self.assertRaises(KeyError):
      otel_metrics.validate_trajectory_dict({"status": "NOT_A_STATUS"})
    with self.assertRaises(TypeError):
      otel_metrics.validate_trajectory_dict(
          datatypes.TrajectoryItem(prompt_id="p", group_index=0, traj={}).traj
          | {"status": 123}
      )


if __name__ == "__main__":
  absltest.main()
