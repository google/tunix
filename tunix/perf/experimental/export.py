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
import concurrent.futures
import types
from typing import Self
from absl import logging
from tunix.perf import metrics
from tunix.perf.experimental import trace_writer as trace_writer_lib
from tunix.perf.experimental import tracer

MetricsT = metrics.MetricsT
Timeline = tracer.Timeline

DEFAULT_TRACE_DIR = "/tmp/perf_traces"


def log_metric_export_fn(timelines: Mapping[str, Timeline]) -> MetricsT:
  """Logs the timelines."""
  logging.info("\n=== Exporting Timelines ===\n %s", timelines)
  return {}


class PerfMetricsExport:
  """Perf metrics export class.

  A class for exporting timelines to a trace writer.
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
    self._trace_writer_enabled = enable_trace_writer
    self._writer: trace_writer_lib.TraceWriter
    if enable_trace_writer:
      resolved_trace_dir = trace_dir or DEFAULT_TRACE_DIR
      self._writer = trace_writer_lib.PerfettoTraceWriter(resolved_trace_dir)
      # We need to keep max_workers = 1 to serialize writes
      self._executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1, thread_name_prefix="PerfExport"
      )
    else:
      self._writer = trace_writer_lib.NoopTraceWriter()
      self._executor = None

  def _safe_write(self, timelines: Mapping[str, Timeline]) -> None:
    try:
      self._writer.write_timelines(timelines)
    except Exception:  # pylint: disable=broad-except
      logging.exception("Background trace export failed.")

  def export_metrics(
      self,
      timelines: Mapping[str, Timeline],
  ) -> MetricsT:
    """Exports timelines to a trace file.

    Args:
      timelines: A mapping of names to Timeline objects to be exported.

    Returns:
      An empty dictionary, as this exporter does not produce any aggregated
      metrics.
    """
    # Calculate aggregated metrics
    # TODO(noghabi): Calculate aggregated metrics from the timelines and export
    # them to a logger (TensorBoard).

    # Write timelines to the trace writer executor.
    if self._trace_writer_enabled:
      if self._executor is not None:
        self._executor.submit(self._safe_write, timelines)
      else:
        logging.warning(
            "PerfMetricsExport background worker has been shut down. Cannot"
            " write metrics."
        )
    return {}

  def shutdown(self, wait: bool = True) -> None:
    """Shuts down the background executor safely."""
    if self._executor is not None:
      self._executor.shutdown(wait=wait)
      self._executor = None
      logging.info("PerfMetricsExport background worker shut down.")

  def __enter__(self) -> Self:
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_value: BaseException | None,
      traceback: types.TracebackType | None,
  ) -> None:
    self.shutdown(wait=True)
