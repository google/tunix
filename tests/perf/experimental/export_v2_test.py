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

import os
import tempfile
import time
from absl.testing import absltest
from tunix.perf.experimental import export
from tunix.perf.experimental import tracer


class ExportTest(absltest.TestCase):

  def test_perf_metrics_export(self):
    # Backward compatibility check
    with tempfile.TemporaryDirectory() as tmp_dir:
      exporter = export.PerfMetricsExport(trace_dir=tmp_dir)

      # Create dummy timeline
      t = tracer.PerfTracer(export_fn=exporter.export_metrics)
      with t.span("test_span"):
        time.sleep(0.001)
      t.export()

      files = os.listdir(tmp_dir)
      self.assertLen(files, 1)
      self.assertStartsWith(files[0], "perfetto_trace_v2_")

  def test_basic_metrics_export(self):
    with self.assertLogs(level="INFO") as logs:
      export.log_metric_export_fn({})
    self.assertTrue(
        any("=== Exporting Timelines ===" in log for log in logs.output)
    )

  def test_perf_metrics_export_no_trace_dir(self):
    exporter = export.PerfMetricsExport(trace_dir=None)
    # Should not raise exception
    exporter.export_metrics({})


if __name__ == "__main__":
  absltest.main()
