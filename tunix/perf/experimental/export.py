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
from typing import Any
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


# TODO(noghabi): Calculate aggregated metrics from the timelines and export them
# to a logger (TensorBoard).
class PerfMetricsExport:
  """Perf metrics export class.

  A class for exporting timelines to a trace writer.
  """

  @classmethod
  def from_cluster_config(
      cls,
      cluster_config: Any,
      enable_trace_writer: bool = True,
      trace_dir: str | None = None,
  ) -> PerfMetricsExport:
    """Creates an instance from a ClusterConfig.

    Args:
      cluster_config: The tunix.rl.rl_cluster.ClusterConfig to extract
        role_to_mesh from.
      enable_trace_writer: Whether to initialize the trace writer.
      trace_dir: The directory to write the Perfetto trace files to.

    Returns:
      A new PerfMetricsExport instance configured with role to device mappings.
    """
    role_to_devices = {}
    if hasattr(cluster_config, "role_to_mesh"):
      for role, mesh in cluster_config.role_to_mesh.items():
        role_name = role.value if hasattr(role, "value") else str(role)
        # Convert JAX devices to strings (or their IDs) if needed, here we just
        # keep them as lists of devices for the trace writer to consume.
        role_to_devices[role_name] = mesh.devices.flatten().tolist()

    return cls(
        enable_trace_writer=enable_trace_writer,
        trace_dir=trace_dir,
        role_to_devices=role_to_devices,
    )

  def __init__(
      self,
      *,
      enable_trace_writer: bool = True,
      trace_dir: str | None = None,
      role_to_devices: Mapping[str, Any] | None = None,
  ):
    """Initializes the instance.

    Args:
      enable_trace_writer: Whether to initialize the trace writer.
      trace_dir: The directory to write the Perfetto trace files to. This is
        only relevant if enable_trace_writer is True. If not provided (None or
        empty string) and enable_trace_writer is True, a default directory is
        used.
      role_to_devices: An optional mapping from role names to their assigned
        devices, passed to the trace writer.
    """
    self._writer: trace_writer_lib.TraceWriter
    if enable_trace_writer:
      resolved_trace_dir = trace_dir or DEFAULT_TRACE_DIR
      self._writer = trace_writer_lib.PerfettoTraceWriter(
          resolved_trace_dir, role_to_devices=role_to_devices
      )
    else:
      self._writer = trace_writer_lib.NoopTraceWriter()

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
    self._writer.write_timelines(timelines)
    return {}
