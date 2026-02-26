# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for perfetto."""

import os
import tempfile
import time

from absl.testing import absltest
from tunix.perf.experimental import perfetto
from tunix.perf.experimental import tracer


class PerfettoTest(absltest.TestCase):

  def test_create_span_name(self):
    # Test basic span name with global_step
    name = perfetto._create_span_name("my_span", {"global_step": 10})
    self.assertEqual(name, "my_span (step=10)")

    # Test peft_train_step with role
    name = perfetto._create_span_name(
        "peft_train_step", {"global_step": 20, "role": "actor"}
    )
    self.assertEqual(name, "peft_train_step (step=20, role=actor)")

    # Test rollout with group_id and pair_index
    name = perfetto._create_span_name(
        "rollout", {"group_id": 5, "pair_index": 3, "global_step": 100}
    )
    self.assertEqual(name, "rollout (step=100, group_id=5, pair_index=3)")

    # Test rollout with missing pair_index
    name = perfetto._create_span_name("rollout", {"group_id": 5})
    self.assertEqual(name, "rollout (group_id=5)")

    # Test unknown name with extra tags (should ignore specific logic but keep step)
    name = perfetto._create_span_name(
        "unknown_span", {"role": "actor", "global_step": 50}
    )
    self.assertEqual(name, "unknown_span (step=50)")

    # Test no tags
    name = perfetto._create_span_name("simple_span", {})
    self.assertEqual(name, "simple_span")

  # TODO(noghabi): Add more tests for PerfettoTraceWriter.
  def test_perfetto_trace_writer(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      writer = perfetto.PerfettoTraceWriter(trace_dir=tmp_dir)

      # Create some dummy timelines
      t = tracer.Timeline("test_timeline", time.perf_counter())
      s = t.start_span("test_span", time.perf_counter())
      time.sleep(0.001)
      t.stop_span(time.perf_counter())

      timelines = {"test_timeline": t}

      writer.write_timelines(timelines)

      # Check if file was created
      files = os.listdir(tmp_dir)
      self.assertLen(files, 1)
      self.assertTrue(files[0].startswith("perfetto_trace_v2_"))
      self.assertTrue(files[0].endswith(".pb"))

      # We could parse the proto back to verify content, but just existence is good for now.


if __name__ == "__main__":
  absltest.main()
