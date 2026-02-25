"""Tests for export."""

import os
import pathlib
import time
from absl.testing import absltest
from tunix.perf.experimental import export
from tunix.perf.experimental import tracer


class ExportTest(absltest.TestCase):

  def test_perf_metrics_export(self):
    # Backward compatibility check
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    exporter = export.PerfMetricsExport(trace_dir=tmp_dir)

    # Create dummy timeline
    t = tracer.PerfTracer(export_fn=exporter.export_metrics)
    with t.span("test_span"):
      time.sleep(0.001)
    t.export()

    files = os.listdir(tmp_dir)
    self.assertLen(files, 1)
    self.assertTrue(files[0].startswith("perfetto_trace_v2_"))


if __name__ == "__main__":
  absltest.main()
