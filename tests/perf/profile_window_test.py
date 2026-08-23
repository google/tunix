#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

from tunix.perf import profile_window
from tunix.perf.experimental import export as perf_export
from tunix.perf.experimental import timeline


class ProfileWindowIntegrationTest(unittest.TestCase):

  def test_real_perfetto_writer_serializes_exactly_one_selected_step(self):
    with tempfile.TemporaryDirectory() as tmp:
      exporter = perf_export.PerfMetricsExport(trace_dir=tmp)
      window = profile_window.single_step_export_fn(
          exporter.export_metrics, target_step=2
      )
      trace = timeline.Timeline("host-1", born=10.0)
      for step in range(4):
        span = trace.start_span("global_step", begin=11.0 + step)
        span.add_tag("step", step)
        trace.stop_span(end=11.5 + step)
        trace.commit_step()
        window({trace.id: trace})
      exporter.shutdown(wait=True)
      artifacts = tuple(Path(tmp).glob("perfetto_trace_v2_*.pb"))
      self.assertEqual(len(artifacts), 1)
      self.assertGreater(artifacts[0].stat().st_size, 0)


if __name__ == "__main__":
  unittest.main()
