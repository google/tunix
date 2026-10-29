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

"""Process-level OpenTelemetry metrics setup for the dist_rl runtime.

``tunix.sft.metrics_logger.MetricsLogger`` double-writes every logged scalar as
an OpenTelemetry gauge once ``MetricsLoggerOptions.enable_opentelemetry`` is
set, and worker processes can record native OpenTelemetry instruments directly,
but neither owns the provider: with nothing installed the API resolves a proxy
provider that silently drops every measurement. Each process that wants to
export therefore installs a global ``MeterProvider`` -- an OTLP/gRPC exporter
behind a periodic reader, with a ``service.name`` resource and optional process
attributes (such as ``tunix.worker_id``, ``tunix.role``, and ``tunix.run_id``)
so the collector and backend can tell the orchestrator, trainer, and rollout
streams apart.

Configuration is environment-driven so the launchers can enable it per
process without code changes:

* ``OTEL_EXPORTER_OTLP_ENDPOINT`` -- enables export when set, pointing to a
  dedicated collector endpoint (for example ``http://localhost:4317`` under
  ``launcher.sh`` with
  ``tunix/dist_rl/distributed/deployment/yamls/otel-collector.yaml``,
  or a standalone Kubernetes Service DNS such as ``http://${USER}-otel:4317``
  deployed via ``tunix/dist_rl/distributed/deployment/yamls/jobset.otel.yaml``
  on GKE where rollout LeaderWorkerSet pods use isolated pod networking). Unset
  leaves the default backends only.
* ``OTEL_SERVICE_NAME`` -- overrides the per-process default service name.
* ``OTEL_METRIC_EXPORT_INTERVAL`` -- export interval in milliseconds (default
  10000). Measurements are batched in memory until exported, so a shorter
  interval narrows what a crashing process can lose; ``MeterProvider`` also
  registers its ``shutdown`` hook at exit by default to drain the last batch.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import socket
import urllib.parse

from absl import logging

_DEFAULT_EXPORT_INTERVAL_MS = 10_000
_DEFAULT_PROBE_TIMEOUT_S = 1.0


def _parse_host_port(endpoint: str) -> tuple[str, int] | None:
  """Extracts (host, port) from an OTLP endpoint URL or host:port string."""
  parsed = urllib.parse.urlparse(
      endpoint if "://" in endpoint else f"http://{endpoint}"
  )
  if not parsed.hostname:
    return None
  try:
    port = parsed.port or (443 if parsed.scheme == "https" else 4317)
  except ValueError:
    return None
  return parsed.hostname, port


def _warn_if_collector_unreachable(
    endpoint: str,
    service_name: str,
    timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S,
) -> None:
  """Logs a non-blocking warning if the OTLP collector TCP port is unreachable."""
  host_port = _parse_host_port(endpoint)
  if host_port is None:
    logging.warning(
        "[otel] Could not parse host:port from OTEL_EXPORTER_OTLP_ENDPOINT=%r"
        " (service.name=%s); MeterProvider will still be installed.",
        endpoint,
        service_name,
    )
    return
  host, port = host_port
  try:
    with socket.create_connection((host, port), timeout=timeout_s):
      pass
  except OSError as exc:
    logging.warning(
        "[otel] OpenTelemetry collector at %s (%s:%d) is unreachable from"
        " service.name=%s (%s); MeterProvider is still installed and will"
        " retry in the background.",
        endpoint,
        host,
        port,
        service_name,
        exc,
    )


def setup_metrics(
    default_service_name: str,
    resource_attributes: Mapping[str, str] | None = None,
) -> bool:
  """Installs a global OpenTelemetry MeterProvider if an endpoint is set.

  Args:
    default_service_name: ``service.name`` used unless ``OTEL_SERVICE_NAME``
      overrides it.
    resource_attributes: Optional process-level resource attributes (for example
      ``tunix.worker_id``, ``tunix.role``, ``tunix.run_id``) merged into the
      OpenTelemetry ``Resource`` alongside ``service.name``. Empty string values
      are omitted.

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
  except ImportError as exc:
    logging.warning(
        "OTEL_EXPORTER_OTLP_ENDPOINT=%s is set but the OpenTelemetry SDK or"
        " OTLP exporter is not installed (%s; pip install"
        " 'google-tunix[otel]'); metrics stay on the default backends.",
        endpoint,
        exc,
    )
    return False

  if isinstance(otel_metrics.get_meter_provider(), MeterProvider):
    return True

  service_name = (
      os.getenv("OTEL_SERVICE_NAME", "").strip() or default_service_name
  )
  raw_interval = os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "").strip()
  if raw_interval:
    try:
      interval_ms = int(raw_interval)
    except ValueError as exc:
      raise ValueError(
          "OTEL_METRIC_EXPORT_INTERVAL must be a positive integer (ms), got"
          f" {raw_interval!r}."
      ) from exc
    if interval_ms <= 0:
      raise ValueError(
          "OTEL_METRIC_EXPORT_INTERVAL must be a positive integer (ms), got"
          f" {interval_ms}."
      )
  else:
    interval_ms = _DEFAULT_EXPORT_INTERVAL_MS
  attrs: dict[str, str] = (
      {k: v for k, v in resource_attributes.items() if v}
      if resource_attributes
      else {}
  )
  attrs["service.name"] = service_name
  _warn_if_collector_unreachable(endpoint, service_name)
  exporter = OTLPMetricExporter(
      endpoint=endpoint, insecure=endpoint.startswith("http://")
  )
  reader = PeriodicExportingMetricReader(
      exporter, export_interval_millis=interval_ms
  )
  provider = MeterProvider(
      resource=Resource.create(attrs),
      metric_readers=[reader],
  )
  otel_metrics.set_meter_provider(provider)
  logging.info(
      "[otel] OpenTelemetry metrics export enabled: endpoint=%s"
      " service.name=%s interval=%dms attributes=%s",
      endpoint,
      service_name,
      interval_ms,
      attrs,
  )
  return True
