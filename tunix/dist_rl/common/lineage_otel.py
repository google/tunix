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

"""OpenTelemetry structured LogRecord sink for ``tunix.dist_rl.common.lineage``."""

from __future__ import annotations

import functools
import math
import numbers
import time
from typing import Any

from opentelemetry import _logs as otel_logs
from tunix.dist_rl.common import lineage

LINEAGE_LOGGER_NAME = "tunix.lineage"
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1
_MAX_TIMESTAMP_S = (_UINT64_MAX // 1_000_000_000) + 1


def _sanitize_scalar(value: Any) -> bool | int | float | str | None:
  """Converts a scalar attribute value into an OTLP-safe primitive."""
  if isinstance(value, bool) or (
      type(value).__module__ == "numpy"
      and type(value).__name__ in ("bool", "bool_")
  ):
    return bool(value)
  if isinstance(value, numbers.Integral):
    ival = int(value)
    return ival if _INT64_MIN <= ival <= _INT64_MAX else None
  if isinstance(value, numbers.Real):
    fval = float(value)
    # OTLP JSON and Cloud Logging reject NaN/Inf numbers; dropping non-finite
    # floats keeps numeric columns type-safe instead of mixing strings.
    return fval if math.isfinite(fval) else None
  if isinstance(value, str):
    return str(value)
  return None


def _sanitize_attr(value: Any) -> Any:
  """Returns an OTel attribute primitive or homogeneous tuple, or ``None``."""
  scalar = _sanitize_scalar(value)
  if scalar is not None:
    return scalar
  if isinstance(value, (list, tuple)):
    items = [s for item in value if (s := _sanitize_scalar(item)) is not None]
    if len(items) == len(value) and len({type(item) for item in items}) <= 1:
      return tuple(items)
  return None


def _resolve_timestamp(timestamp_s: Any) -> tuple[int, float]:
  """Returns an unsigned 64-bit OTLP nanosecond timestamp and finite seconds."""
  if not isinstance(timestamp_s, bool) and isinstance(
      timestamp_s, numbers.Real
  ):
    raw: Any = timestamp_s
    if 0 <= raw <= _MAX_TIMESTAMP_S:
      ts = float(raw)
      ns = max(0, int(ts * 1_000_000_000))
      if ns <= _UINT64_MAX:
        return ns, 0.0 if ts == 0.0 else ts
  fallback_ns = max(0, time.time_ns())
  return fallback_ns, fallback_ns / 1_000_000_000


def _emit(
    logger: otel_logs.Logger,
    ctx: lineage.LineageContext,
    event: lineage.LineageEvent,
) -> None:
  """Emits one structured OpenTelemetry LogRecord on ``logger``."""
  event_name = f"{LINEAGE_LOGGER_NAME}.{event.component}.{event.operation}"
  # Store `event.name` in `attributes` alongside `LogRecord(event_name=...)`
  # because the collector's `googlecloud` exporter maps `LogRecord.attributes`
  # into Cloud Logging `labels`. Keep `timestamp_s` in `body` so Cloud Logging
  # `jsonPayload` retains the float epoch timestamp alongside `timestamp`.
  otel_attrs: dict[str, Any] = {
      "event.name": event_name,
      "tunix.tracking_id": ctx.tracking_id,
      "tunix.task_name": ctx.task_name,
      "tunix.parent_tracking_ids": tuple(ctx.parent_tracking_ids),
      "tunix.component": event.component,
      "tunix.operation": event.operation,
  }
  for key, raw_value in event.attributes.items():
    sanitized = _sanitize_attr(raw_value)
    if sanitized is not None:
      otel_attrs.setdefault(f"tunix.{key}", sanitized)
  timestamp_ns, body_timestamp_s = _resolve_timestamp(event.timestamp_s)
  logger.emit(
      otel_logs.LogRecord(
          timestamp=timestamp_ns,
          observed_timestamp=timestamp_ns,
          severity_number=otel_logs.SeverityNumber.INFO,
          severity_text="INFO",
          body={"timestamp_s": body_timestamp_s},
          attributes=otel_attrs,
          event_name=event_name,
      )
  )


def install(
    logger_provider: otel_logs.LoggerProvider,
) -> lineage.LineageEventSink | None:
  """Binds a ``tunix.lineage`` logger once and registers the lineage sink."""
  logger = logger_provider.get_logger(LINEAGE_LOGGER_NAME)
  return lineage.set_event_sink(functools.partial(_emit, logger))
