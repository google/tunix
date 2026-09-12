# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process-level OpenTelemetry metrics setup for the experimental runtime.

``tunix.sft.metrics_logger.MetricsLogger`` double-writes every logged scalar as
an OpenTelemetry gauge once ``MetricsLoggerOptions.enable_opentelemetry`` is
set, but it deliberately does not own the provider: with nothing installed the
API resolves a proxy provider that silently drops every measurement. Each
process that wants to export therefore installs a global ``MeterProvider`` --
an OTLP/gRPC exporter behind a periodic reader, with a ``service.name``
resource so the collector and backend can tell the orchestrator, trainer and
rollout streams apart.

Configuration is environment-driven so the launchers can enable it per
process without code changes:

* ``OTEL_EXPORTER_OTLP_ENDPOINT`` -- enables export when set (for example
  ``http://localhost:4317``; every jobset template runs with
  ``hostNetwork: true``, so a node-local collector is reachable there from a
  pod and from ``launcher.sh`` alike). Unset leaves the default backends only.
* ``OTEL_SERVICE_NAME`` -- overrides the per-process default service name.
* ``OTEL_METRIC_EXPORT_INTERVAL`` -- export interval in milliseconds (default
  10000). Measurements are batched in memory until exported, so a shorter
  interval narrows what a crashing process can lose; ``shutdown`` is also
  registered at exit to drain the last batch.
"""

from __future__ import annotations

import atexit
import logging
import os

_DEFAULT_EXPORT_INTERVAL_MS = 10_000


def setup_metrics(default_service_name: str) -> bool:
  """Installs a global OpenTelemetry MeterProvider if an endpoint is set.

  Args:
    default_service_name: ``service.name`` used unless ``OTEL_SERVICE_NAME``
      overrides it.

  Returns:
    True when a provider was installed and ``enable_opentelemetry`` should be
    passed to ``MetricsLoggerOptions``; False when export is disabled (no
    endpoint) or the SDK / exporter is not installed.
  """
  endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
  if not endpoint:
    return False
  try:
    # pylint: disable=g-import-not-at-top
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    # pylint: enable=g-import-not-at-top
  except ImportError:
    logging.warning(
        "OTEL_EXPORTER_OTLP_ENDPOINT=%s is set but the OpenTelemetry SDK or"
        " OTLP exporter is not installed (pip install 'google-tunix[otel]');"
        " metrics stay on the default backends.",
        endpoint,
    )
    return False

  service_name = (
      os.getenv("OTEL_SERVICE_NAME", "").strip() or default_service_name
  )
  interval_ms = int(
      os.getenv("OTEL_METRIC_EXPORT_INTERVAL", str(_DEFAULT_EXPORT_INTERVAL_MS))
  )
  exporter = OTLPMetricExporter(
      endpoint=endpoint, insecure=endpoint.startswith("http://")
  )
  reader = PeriodicExportingMetricReader(
      exporter, export_interval_millis=interval_ms
  )
  provider = MeterProvider(
      resource=Resource.create({"service.name": service_name}),
      metric_readers=[reader],
  )
  otel_metrics.set_meter_provider(provider)
  # Drain the in-memory batch when the process exits normally.
  atexit.register(provider.shutdown)
  logging.info(
      "OpenTelemetry metrics export enabled: endpoint=%s service.name=%s"
      " interval=%dms",
      endpoint,
      service_name,
      interval_ms,
  )
  return True
