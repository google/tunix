# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import time

from concurrent import futures

from absl.testing import absltest
from absl.testing import parameterized
from tunix.perf import export
from tunix.perf import trace
from tunix.perf import span

Timeline = trace.Timeline
DeviceTimeline = trace.DeviceTimeline
ThreadTimeline = trace.ThreadTimeline
NoopTracer = trace.NoopTracer
PerfTracer = trace.PerfTracer
patch = mock.patch
Mock = mock.Mock
Span = span.Span
SpanGroup = span.SpanGroup


class TracerTest(parameterized.TestCase):

  def test_span_group(self):
    tracer = PerfTracer()
    for _ in range(2):
      with tracer.span_group("global_step"):
        for _ in range(3):
          with tracer.span_group("mini_batch"):
            for _ in range(2):
              with tracer.span_group("micro_batch"):
                time.sleep(0.1)
    tracer.print()

  def test_span(self):
    tracer = PerfTracer(["tpu0"])
    for _ in range(2):
      with tracer.span_group("global_step"):
        for _ in range(3):
          with tracer.span_group("mini_batch"):
            for _ in range(2):
              with tracer.span_group("micro_batch"):
                with tracer.span("rollout_infer_train", ["tpu0"]):
                  time.sleep(0.1)
            with tracer.span("gradient_update"):
              time.sleep(0.1)
        with tracer.span("weight_sync"):
          time.sleep(0.01)
    tracer.print()

  def test_grpo_workflow(self):
    mesh = {
      "rollout": ["tpu0","tpu1"],
      "refer": ["tpu2","tpu3"],
      "actor": ["tpu4","tpu5"],
    }
    devices = [dev for devlist in mesh.values() for dev in devlist]

    tracer = PerfTracer(devices, export.PerfMetricsExport.from_role_to_devices(mesh))
    executor = futures.ThreadPoolExecutor(max_workers=1)
    for _ in range(3):
      with tracer.span_group("global_step"):
        time.sleep(0.1)
        for _ in range(1):
          with tracer.span_group("mini_batch_step"):
            time.sleep(0.1)
            with tracer.span_group("micro_batch_steps"):
              time.sleep(0.1)
              for _ in range(3):
                def worker():
                  with tracer.span("data_loading"):
                    time.sleep(0.01)
                  with tracer.span("rollout", mesh["rollout"]):
                    time.sleep(0.01)
                  with tracer.span("refer_inference", mesh["refer"]):
                    time.sleep(0.01)
                  with tracer.span("old_actor_inference", mesh["actor"]):
                    time.sleep(0.01)
                future = executor.submit(worker)
                with tracer.span("actor_training", mesh["actor"]):
                  time.sleep(0.01)
                with tracer.span("advantage_compute"): # ???.mesh
                  time.sleep(0.01)
                future.result()
        with tracer.span("weight_sync"):
          time.sleep(0.001)
    tracer.print()

    for k, v in tracer.export().items():
      print(f"{k}: {v}")

if __name__ == "__main__":
  absltest.main()
