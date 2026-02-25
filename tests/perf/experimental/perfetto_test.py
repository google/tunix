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
