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

"""Metrics export functions."""

from absl import logging
from tunix.perf import metrics
from tunix.perf.experimental import perfetto
from tunix.perf.experimental import tracer

MetricsT = metrics.MetricsT
Timeline = tracer.Timeline


class BasicMetricsExport:
  """Basic metrics export that logs the query."""

  @staticmethod
  def log_metrics(timelines: dict[str, Timeline]) -> MetricsT:
    """Logs the export function."""
    logging.info("\n=== Exporting Timelines ===\n %s", timelines)
    return {}


class PerfMetricsExport:
  """Perf metrics export class.

  Exports timelines to a trace writer (Perfetto). [TODO] Calculate aggregated
  metrics from the timelines and export them to a logger (TensorBoard).
  """

  def __init__(self, trace_dir: str | None = None):
    """Initializes the PerfMetricsExport.

    Args:
      trace_dir: The directory to write the Perfetto trace files to. If None,
        the trace writer will not be initialized.
    """

    self._writer = (
        perfetto.PerfettoTraceWriter(trace_dir) if trace_dir else None
    )

  def export_metrics(
      self,
      timelines: dict[str, Timeline],
  ) -> MetricsT:
    """Exports timelines to a Perfetto trace file."""
    if self._writer is not None:
      self._writer.write_timelines(timelines)
    return {}
