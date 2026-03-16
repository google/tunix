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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from tunix.perf.experimental import export
from tunix.perf.experimental import tracer


class ExportTest(parameterized.TestCase):

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

  @parameterized.named_parameters(
      dict(
          testcase_name="none_dir",
          trace_dir=None,
          expected_dir=export.DEFAULT_TRACE_DIR,
      ),
      dict(
          testcase_name="empty_dir",
          trace_dir="",
          expected_dir=export.DEFAULT_TRACE_DIR,
      ),
      dict(
          testcase_name="custom_dir",
          trace_dir="/my/custom/path",
          expected_dir="/my/custom/path",
      ),
  )
  @mock.patch.object(export.trace_writer_lib, "PerfettoTraceWriter", autospec=True)
  def test_perf_metrics_export_initialization_with_trace_writer_enabled(
      self, mock_writer_cls, trace_dir, expected_dir
  ):
    exporter = export.PerfMetricsExport(
        enable_trace_writer=True, trace_dir=trace_dir
    )
    mock_writer_cls.assert_called_once_with(expected_dir)
    # export_metrics shouldn't crash
    exporter.export_metrics({})

  @mock.patch.object(export.trace_writer_lib, "NoopTraceWriter", autospec=True)
  def test_perf_metrics_export_initialization_with_trace_writer_disabled(
      self, mock_noop_cls
  ):
    exporter = export.PerfMetricsExport(enable_trace_writer=False)
    mock_noop_cls.assert_called_once_with()
    # export_metrics shouldn't crash
    exporter.export_metrics({})


if __name__ == "__main__":
  absltest.main()
