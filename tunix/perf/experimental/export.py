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

from __future__ import annotations
from collections.abc import Mapping
from absl import logging
from tunix.perf import metrics
from tunix.perf.experimental import perfetto
from tunix.perf.experimental import tracer

MetricsT = metrics.MetricsT
Timeline = tracer.Timeline

DEFAULT_TRACE_DIR = "/tmp/perf_traces"


def log_metric_export_fn(timelines: Mapping[str, Timeline]) -> MetricsT:
  """Logs the timelines."""
  logging.info("\n=== Exporting Timelines ===\n %s", timelines)
  return {}


# TODO(noghabi): Calculate aggregated metrics from the timelines and export them
# to a logger (TensorBoard).
class PerfMetricsExport:
  """Perf metrics export class.

  Exports timelines to a trace writer (Perfetto).
  """

  def __init__(
      self, *, enable_trace_writer: bool = True, trace_dir: str | None = None
  ):
    """Initializes the instance.

    Args:
      enable_trace_writer: Whether to initialize the trace writer.
      trace_dir: The directory to write the Perfetto trace files to. This is
        only relevant if enable_trace_writer is True. If not provided (None or
        empty string) and enable_trace_writer is True, a default directory is
        used.
    """
    if enable_trace_writer:
      resolved_trace_dir = trace_dir or DEFAULT_TRACE_DIR
      self._writer = perfetto.PerfettoTraceWriter(resolved_trace_dir)
    else:
      # TODO(noghabi): create a NoopTraceWriter.
      self._writer = None

  def export_metrics(
      self,
      timelines: Mapping[str, Timeline],
  ) -> MetricsT:
    """Exports timelines to a Perfetto trace file."""
    if self._writer is not None:
      self._writer.write_timelines(timelines)
    return {}
