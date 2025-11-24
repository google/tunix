# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from absl.testing import absltest
from tunix.perf import export
from tunix.perf import metrics
from tunix.perf import span
from tunix.perf import trace

patch = mock.patch

PerfMetricsExport = export.PerfMetricsExport
PerfSpanQuery = metrics.PerfSpanQuery
ThreadTimeline = trace.ThreadTimeline
DeviceTimeline = trace.DeviceTimeline


class ExportTest(absltest.TestCase):

  @patch("time.perf_counter")
  def test_export_grpo_metrics_colocated(self, mock_perf_counter):
    # tpu0 span end times
    mock_perf_counter.side_effect = [0.41, 0.61, 0.81, 1.21]

    export_fn = PerfMetricsExport.from_role_to_devices({
        "rollout": ["tpu0"],
        "refer": ["tpu0"],
        "actor": ["tpu0"],
    })
    host_timeline = ThreadTimeline("host", 0.0)
    tpu0_timeline = DeviceTimeline("tpu0", 0.0)
    timelines = {
        "host": host_timeline,
        "tpu0": tpu0_timeline,
    }

    for timeline in timelines.values():
      timeline.span_group_begin("global_step", 0.0)

    for timeline in timelines.values():
      timeline.span_group_begin("mini_batch_step", 0.1)

    for timeline in timelines.values():
      timeline.span_group_begin("micro_batch_steps", 0.2)

    host_timeline.span_begin("rollout", 0.3)
    tpu0_timeline.span("rollout", 0.3, [])  # end 0.41
    host_timeline.span_end(0.4)

    host_timeline.span_begin("refer_inference", 0.5)
    tpu0_timeline.span("refer_inference", 0.5, [])  # end 0.61
    host_timeline.span_end(0.6)

    host_timeline.span_begin("actor_training", 0.7)
    tpu0_timeline.span("actor_training", 0.7, [])  # end 0.81
    host_timeline.span_end(0.8)

    for timeline in timelines.values():
      timeline.span_group_end(0.9)  # micro_batch_steps

    for timeline in timelines.values():
      timeline.span_group_end(1.0)  # mini_batch_step

    host_timeline.span_begin("weight_sync", 1.1)
    tpu0_timeline.span("weight_sync", 1.1, [])  # end 1.21
    host_timeline.span_end(1.2)

    for timeline in timelines.values():
      timeline.span_group_end(1.3)  # global_step

    tpu0_timeline.wait_pending_spans()

    expected_metrics = {
        "perf/global_step_time": 1.3,
        "perf/weight_sync_time": 0.1,
        "perf/sum/rollout_time": 0.11,
        "perf/sum/refer_inference_time": 0.11,
        "perf/sum/actor_train_grad_time": 0.11,
        "perf/mean/rollout_time": 0.11,
        "perf/mean/refer_inference_time": 0.11,
        "perf/mean/actor_train_grad_time": 0.11,
    }
    actual_metrics = {}
    for k, v in export_fn(PerfSpanQuery(timelines, "host")).items():
      actual_metrics[k] = float(v[0])

    self.assertDictAlmostEqual(actual_metrics, expected_metrics)

  @patch("time.perf_counter")
  def test_export_grpo_metrics_rollout_1_actor_2_reference_2(
      self, mock_perf_counter
  ):
    mock_perf_counter.side_effect = [0.41, 0.61, 0.81, 1.21, 1.21]

    export_fn = PerfMetricsExport.from_role_to_devices({
        "rollout": ["tpu0"],
        "refer": ["tpu1"],
        "actor": ["tpu1"],
    })
    host_timeline = ThreadTimeline("host", 0.0)
    tpu0_timeline = DeviceTimeline("tpu0", 0.0)
    tpu1_timeline = DeviceTimeline("tpu1", 0.0)
    timelines = {
        "host": host_timeline,
        "tpu0": tpu0_timeline,
        "tpu1": tpu1_timeline,
    }

    for timeline in timelines.values():
      timeline.span_group_begin("global_step", 0.0)

    for timeline in timelines.values():
      timeline.span_group_begin("mini_batch_step", 0.1)

    for timeline in timelines.values():
      timeline.span_group_begin("micro_batch_steps", 0.2)

    host_timeline.span_begin("rollout", 0.3)
    tpu0_timeline.span("rollout", 0.3, [])  # end 0.41
    host_timeline.span_end(0.4)

    host_timeline.span_begin("refer_inference", 0.5)
    tpu1_timeline.span("refer_inference", 0.5, [])  # end 0.61
    host_timeline.span_end(0.6)

    host_timeline.span_begin("actor_training", 0.7)
    tpu1_timeline.span("actor_training", 0.7, [])  # end 0.81
    host_timeline.span_end(0.8)

    for timeline in timelines.values():
      timeline.span_group_end(0.9)  # micro_batch_steps

    for timeline in timelines.values():
      timeline.span_group_end(1.0)  # mini_batch_step

    host_timeline.span_begin("weight_sync", 1.1)
    tpu0_timeline.span("weight_sync", 1.1, [])  # end 1.21
    tpu1_timeline.span("weight_sync", 1.1, [])  # end 1.21
    host_timeline.span_end(1.2)

    for timeline in timelines.values():
      timeline.span_group_end(1.3)  # global_step

    tpu0_timeline.wait_pending_spans()
    tpu1_timeline.wait_pending_spans()

    expected_metrics = {
        "perf/global_step_time": 1.3,
        "perf/weight_sync_time": 0.1,
        "perf/rollout_idle_time": 0.69,
        "perf/first_micro_batch_rollout_time": 0.41,
        "perf/sum/rollout_time": 0.11,
        "perf/sum/refer_inference_time": 0.11,
        "perf/sum/actor_train_grad_time": 0.11,
        "perf/sum/between_micro_batch_gap_time": 0.0,
        "perf/mean/rollout_time": 0.11,
        "perf/mean/refer_inference_time": 0.11,
        "perf/mean/actor_train_grad_time": 0.11,
        "perf/mean/between_micro_batch_gap_time": 0.0,
    }
    actual_metrics = {}
    for k, v in export_fn(PerfSpanQuery(timelines, "host")).items():
      actual_metrics[k] = float(v[0])

    self.assertDictAlmostEqual(actual_metrics, expected_metrics)

  @patch("time.perf_counter")
  def test_export_grpo_metrics_fully_disaggregated(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.41, 0.61, 0.81, 1.21, 1.21, 1.21]

    export_fn = PerfMetricsExport.from_role_to_devices({
        "rollout": ["tpu0"],
        "refer": ["tpu1"],
        "actor": ["tpu2"],
    })
    host_timeline = ThreadTimeline("host", 0.0)
    tpu0_timeline = DeviceTimeline("tpu0", 0.0)
    tpu1_timeline = DeviceTimeline("tpu1", 0.0)
    tpu2_timeline = DeviceTimeline("tpu2", 0.0)
    timelines = {
        "host": host_timeline,
        "tpu0": tpu0_timeline,
        "tpu1": tpu1_timeline,
        "tpu2": tpu2_timeline,
    }

    for timeline in timelines.values():
      timeline.span_group_begin("global_step", 0.0)

    for timeline in timelines.values():
      timeline.span_group_begin("mini_batch_step", 0.1)

    for timeline in timelines.values():
      timeline.span_group_begin("micro_batch_steps", 0.2)

    host_timeline.span_begin("rollout", 0.3)
    tpu0_timeline.span("rollout", 0.3, [])  # end 0.41
    host_timeline.span_end(0.4)

    host_timeline.span_begin("refer_inference", 0.5)
    tpu1_timeline.span("refer_inference", 0.5, [])  # end 0.61
    host_timeline.span_end(0.6)

    host_timeline.span_begin("actor_training", 0.7)
    tpu2_timeline.span("actor_training", 0.7, [])  # end 0.81
    host_timeline.span_end(0.8)

    for timeline in timelines.values():
      timeline.span_group_end(0.9)  # micro_batch_steps

    for timeline in timelines.values():
      timeline.span_group_end(1.0)  # mini_batch_step

    host_timeline.span_begin("weight_sync", 1.1)
    tpu0_timeline.span("weight_sync", 1.1, [])  # end 1.21
    tpu1_timeline.span("weight_sync", 1.1, [])  # end 1.21
    tpu2_timeline.span("weight_sync", 1.1, [])  # end 1.21
    host_timeline.span_end(1.2)

    for timeline in timelines.values():
      timeline.span_group_end(1.3)  # global_step

    tpu0_timeline.wait_pending_spans()
    tpu1_timeline.wait_pending_spans()
    tpu2_timeline.wait_pending_spans()

    expected_metrics = {
        "perf/global_step_time": 1.3,
        "perf/weight_sync_time": 0.1,
        "perf/rollout_idle_time": 0.69,
        "perf/sum/rollout_time": 0.11,
        "perf/sum/refer_inference_time": 0.11,
        "perf/sum/actor_train_grad_time": 0.11,
        "perf/sum/refer_gap_time": 0.0,
        "perf/sum/actor_gap_time": 0.0,
        "perf/mean/rollout_time": 0.11,
        "perf/mean/refer_inference_time": 0.11,
        "perf/mean/actor_train_grad_time": 0.11,
        "perf/mean/refer_gap_time": 0.0,
        "perf/mean/actor_gap_time": 0.0,
    }
    actual_metrics = {}
    for k, v in export_fn(PerfSpanQuery(timelines, "host")).items():
      actual_metrics[k] = float(v[0])

    self.assertDictAlmostEqual(actual_metrics, expected_metrics)


if __name__ == "__main__":
  absltest.main()
