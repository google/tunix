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

"""Process-level OpenTelemetry metrics and event-logs bootstrap for dist_rl.

Installs a global ``MeterProvider`` and ``LoggerProvider`` (exporting to an
OTLP/gRPC collector) and registers ``lineage_otel`` on ``common.lineage``.
Applies three SDK-level policies across orchestrator, trainer, and rollout
processes:

* ``unit="s"`` histograms use log-spaced ``SECONDS_HISTOGRAM_BOUNDARIES`` (5 ms
  to 1 h) instead of the SDK default buckets.
* Synchronous gauges export with delta temporality while counters, histograms,
  and observable gauges stay cumulative.
* Only the process that installed the providers exports. Forked children (such
  as vLLM ``EngineCore`` or ``multiprocessing`` workers) inherit the providers
  and restarted SDK background threads; the exporter guard drops both frozen
  metric copies and any child-emitted records so they never leave under the
  parent's identity over a non-fork-safe gRPC channel.
"""

from __future__ import annotations

import atexit
from collections.abc import Mapping
import logging
import math
import os
import socket
import threading
from typing import Any, cast
import urllib.parse

_logger = logging.getLogger(__name__)

DEFAULT_EXPORT_INTERVAL_MS = 10_000
DEFAULT_EXPORT_TIMEOUT_S = 5.0

# Explicit bucket boundaries of every histogram recorded with ``unit="s"``.
SECONDS_HISTOGRAM_BOUNDARIES: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    3600.0,
)

_installed_pid: int | None = None
_setup_lock = threading.Lock()


class _OwnerProcessExporter:
  """Forwards to an SDK metric or log exporter only in the owning process."""

  def __init__(self, exporter: Any):
    self._exporter = exporter
    self._owner_pid = os.getpid()

  def __getattr__(self, name: str) -> Any:
    if name.startswith("__") and name.endswith("__"):
      raise AttributeError(name)
    # PeriodicExportingMetricReader reads `_preferred_temporality` and
    # `_preferred_aggregation` from its exporter at initialization.
    return getattr(object.__getattribute__(self, "_exporter"), name)

  def export(self, *args: Any, **kwargs: Any) -> Any:
    # Do not log here in a forked child: the SDK restarts its reader and batch
    # threads after `os.fork()`, and Python log handlers bridged to native
    # (C++) log sinks can deadlock or crash on locks inherited from the parent.
    if os.getpid() != self._owner_pid:
      return None
    return self._exporter.export(*args, **kwargs)

  def force_flush(self, *args: Any, **kwargs: Any) -> bool:
    if os.getpid() != self._owner_pid:
      return True
    return self._exporter.force_flush(*args, **kwargs)

  def shutdown(self, *args: Any, **kwargs: Any) -> None:
    if os.getpid() == self._owner_pid:
      self._exporter.shutdown(*args, **kwargs)


def normalize_endpoint(endpoint: str) -> str:
  """Normalizes an OTLP endpoint to ``http(s)://host:port`` or raises ``ValueError``."""
  cleaned = endpoint.strip()
  normalized = cleaned if "://" in cleaned else f"http://{cleaned}"
  try:
    parsed = urllib.parse.urlparse(normalized)
    port = parsed.port
  except ValueError as exc:
    raise ValueError(f"Invalid OTLP endpoint {endpoint!r}: {exc}") from exc
  if parsed.scheme not in ("http", "https"):
    raise ValueError(
        f"OTLP endpoint scheme must be 'http' or 'https', got {endpoint!r}."
    )
  if port is None and parsed.scheme == "https":
    port = 443
  if not parsed.hostname or port is None or port <= 0:
    raise ValueError(
        "OTLP endpoint must include a valid host and explicit port (e.g."
        f" http://localhost:4317), got {endpoint!r}."
    )
  return normalized


def legacy_env_settings() -> tuple[str, int, float]:
  """Reads legacy `OTEL_*` env vars for standalone `argparse` entry points."""
  endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
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
    interval_ms = DEFAULT_EXPORT_INTERVAL_MS

  raw_timeout = os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "").strip()
  if raw_timeout:
    try:
      timeout_s = float(raw_timeout)
    except ValueError as exc:
      raise ValueError(
          "OTEL_EXPORTER_OTLP_TIMEOUT must be a positive finite number (s),"
          f" got {raw_timeout!r}."
      ) from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
      raise ValueError(
          "OTEL_EXPORTER_OTLP_TIMEOUT must be a positive finite number (s),"
          f" got {raw_timeout!r}."
      )
  else:
    timeout_s = DEFAULT_EXPORT_TIMEOUT_S
  return endpoint, interval_ms, timeout_s


def _validate_timing(
    export_interval_ms: int,
    export_timeout_s: float,
) -> tuple[int, float]:
  """Validates positive `export_interval_ms` and finite `export_timeout_s`."""
  if (
      isinstance(export_interval_ms, bool)
      or not isinstance(export_interval_ms, int)
      or export_interval_ms <= 0
  ):
    raise ValueError(
        "export_interval_ms must be a positive integer (ms), got"
        f" {export_interval_ms!r}."
    )
  if (
      isinstance(export_timeout_s, bool)
      or not isinstance(export_timeout_s, (int, float))
      or not math.isfinite(export_timeout_s)
      or export_timeout_s <= 0
  ):
    raise ValueError(
        "export_timeout_s must be a positive finite number (s), got"
        f" {export_timeout_s!r}."
    )
  return export_interval_ms, float(export_timeout_s)


def _build_resource_attrs(
    service_name: str,
    resource_attributes: Mapping[str, str] | None,
) -> dict[str, str]:
  """Builds non-empty resource attributes with ``host.name`` and ``service.name``."""
  attrs = {k: v for k, v in (resource_attributes or {}).items() if v}
  attrs.setdefault("host.name", socket.gethostname())
  attrs["service.name"] = service_name
  return attrs


def _warn_on_service_collision(provider: Any, requested_service: str) -> None:
  """Warns when reusing an already-installed MeterProvider for another role."""
  # pylint: disable-next=protected-access
  existing_service = provider._sdk_config.resource.attributes.get(
      "service.name", requested_service
  )
  if existing_service != requested_service:
    _logger.warning(
        "[otel] MeterProvider is already initialized with service.name=%s;"
        " reusing existing provider for %s (multi-role single-process mode).",
        existing_service,
        requested_service,
    )


def _shutdown_owner(owner: Any) -> None:
  """Unregisters atexit cleanup and shuts down an uncommitted/rejected resource."""
  if owner is not None:
    atexit.unregister(owner.shutdown)
    owner.shutdown()


def _commit_if_accepted(provider: Any, set_fn: Any, get_fn: Any) -> None:
  """Installs ``provider`` globally, or shuts it down if rejected by Once()."""
  if provider is None:
    return
  set_fn(provider)
  if get_fn() is not provider:
    _shutdown_owner(provider)


def _build_logger_provider(
    enabled: bool, resource: Any, endpoint: str, timeout_s: float
) -> Any:
  """Constructs an SDK ``LoggerProvider`` wired to an owner-guarded exporter."""
  if not enabled:
    return None
  # pylint: disable=g-import-not-at-top
  from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
  from opentelemetry.sdk._logs import LoggerProvider
  from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
  # pylint: enable=g-import-not-at-top

  log_owner: Any = None
  try:
    log_exporter = OTLPLogExporter(endpoint=endpoint, timeout=timeout_s)
    log_owner = log_exporter
    provider = LoggerProvider(resource=resource)
    atexit.unregister(provider.shutdown)
    provider.add_log_record_processor(
        BatchLogRecordProcessor(cast(Any, _OwnerProcessExporter(log_exporter)))
    )
    atexit.register(provider.shutdown)
    return provider
  except Exception:
    _shutdown_owner(log_owner)
    raise


def _build_meter_provider(
    enabled: bool,
    resource: Any,
    endpoint: str,
    interval_ms: int,
    timeout_s: float,
) -> Any:
  """Constructs an SDK ``MeterProvider`` with log-spaced s-buckets and delta gauges."""
  if not enabled:
    return None
  # pylint: disable=g-import-not-at-top
  from opentelemetry import metrics as otel_metrics
  from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
  from opentelemetry.sdk import metrics as sdk_metrics
  from opentelemetry.sdk.metrics import MeterProvider
  from opentelemetry.sdk.metrics import view as sdk_view
  from opentelemetry.sdk.metrics.export import AggregationTemporality
  from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
  # pylint: enable=g-import-not-at-top

  metric_owner: Any = None
  try:
    # Synchronous gauges (`_Gauge`) export with delta temporality so each step's
    # scalar is emitted only in the interval in which it was recorded.
    # pylint: disable=protected-access
    metric_exporter = OTLPMetricExporter(
        endpoint=endpoint,
        timeout=timeout_s,
        preferred_temporality={
            sdk_metrics._Gauge: AggregationTemporality.DELTA
        },
    )
    # pylint: enable=protected-access
    metric_owner = metric_exporter
    views = [
        sdk_view.View(
            instrument_type=otel_metrics.Histogram,
            instrument_unit="s",
            aggregation=sdk_view.ExplicitBucketHistogramAggregation(
                boundaries=SECONDS_HISTOGRAM_BOUNDARIES
            ),
        )
    ]
    reader = PeriodicExportingMetricReader(
        cast(Any, _OwnerProcessExporter(metric_exporter)),
        export_interval_millis=interval_ms,
    )
    metric_owner = reader
    return MeterProvider(
        resource=resource, metric_readers=[reader], views=views
    )
  except Exception:
    _shutdown_owner(metric_owner)
    raise


def setup_metrics(
    default_service_name: str,
    resource_attributes: Mapping[str, str] | None = None,
    *,
    endpoint: str | None = None,
    export_interval_ms: int = DEFAULT_EXPORT_INTERVAL_MS,
    export_timeout_s: float = DEFAULT_EXPORT_TIMEOUT_S,
) -> bool:
  """Installs global OpenTelemetry MeterProvider and LoggerProvider when enabled.

  Args:
    default_service_name: ``service.name`` resource attribute identifying the
      process role (for example ``tunix-orchestrator`` or ``tunix-rollout``).
    resource_attributes: Optional process-level resource attributes (such as
      ``tunix.worker_id``, ``tunix.role``, ``tunix.run_id``); empty values are
      omitted.
    endpoint: Explicit OTLP collector endpoint; empty disables export, and
      ``None`` reads ``legacy_env_settings()``.
    export_interval_ms: Metric export interval in milliseconds.
    export_timeout_s: OTLP gRPC export timeout in seconds.

  Returns:
    True when export is active in this process; False when disabled, when the
    OpenTelemetry SDK is unavailable, or in a forked child of the installer.

  Raises:
    ValueError: If the endpoint, interval, or timeout is malformed.
  """
  global _installed_pid
  if endpoint is None:
    raw_endpoint, interval_ms, timeout_s = legacy_env_settings()
  else:
    raw_endpoint = endpoint.strip()
    interval_ms, timeout_s = _validate_timing(
        export_interval_ms, export_timeout_s
    )
  if not raw_endpoint:
    return False
  normalized_endpoint = normalize_endpoint(raw_endpoint)
  try:
    # pylint: disable=g-import-not-at-top
    from opentelemetry import _logs as otel_logs
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from tunix.dist_rl.common import lineage_otel
    # pylint: enable=g-import-not-at-top
    del OTLPLogExporter, OTLPMetricExporter
  except ImportError as exc:
    _logger.warning(
        "[otel] OTLP endpoint %s is set but OpenTelemetry >= 1.39.0 is not"
        " installed (%s; pip install 'google-tunix[otel]'); export stays"
        " disabled.",
        normalized_endpoint,
        exc,
    )
    return False

  current_pid = os.getpid()
  with _setup_lock:
    if _installed_pid is not None and current_pid != _installed_pid:
      # Global OTel providers are set-once so a forked child cannot install its
      # own, and post-fork logging risks deadlocking native (C++) log sinks.
      return False
    existing_meter = otel_metrics.get_meter_provider()
    if _installed_pid == current_pid:
      if isinstance(existing_meter, MeterProvider):
        _warn_on_service_collision(existing_meter, default_service_name)
      return True

    need_meter = not isinstance(existing_meter, MeterProvider)
    if not need_meter:
      _warn_on_service_collision(existing_meter, default_service_name)
    need_log = not isinstance(otel_logs.get_logger_provider(), LoggerProvider)

    attrs = _build_resource_attrs(default_service_name, resource_attributes)
    resource = Resource.create(attrs)
    logger_provider = _build_logger_provider(
        need_log, resource, normalized_endpoint, timeout_s
    )
    try:
      meter_provider = _build_meter_provider(
          need_meter, resource, normalized_endpoint, interval_ms, timeout_s
      )
    except Exception:
      _shutdown_owner(logger_provider)
      raise

    _commit_if_accepted(
        meter_provider,
        otel_metrics.set_meter_provider,
        otel_metrics.get_meter_provider,
    )
    _commit_if_accepted(
        logger_provider,
        otel_logs.set_logger_provider,
        otel_logs.get_logger_provider,
    )
    active_log = otel_logs.get_logger_provider()
    logs_active = isinstance(active_log, LoggerProvider)
    if logs_active:
      lineage_otel.install(active_log)
    metrics_active = isinstance(
        otel_metrics.get_meter_provider(), MeterProvider
    )
    if not (metrics_active or logs_active):
      return False
    _installed_pid = current_pid
    _logger.info(
        "[otel] OpenTelemetry export enabled (metrics=%s, event_logs=%s):"
        " endpoint=%s service.name=%s interval=%dms attributes=%s",
        metrics_active,
        logs_active,
        normalized_endpoint,
        default_service_name,
        interval_ms,
        attrs,
    )
    return True
