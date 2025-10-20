"""Performance tracing primitives with optional OpenTelemetry integration."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Mapping, MutableMapping

try:
  from opentelemetry import trace as otel_trace
except ImportError:  # pragma: no cover - optional dependency
  otel_trace = None  # type: ignore[assignment]

SpanKind = str
_BUSY: SpanKind = "busy"
_BLOCKED: SpanKind = "blocked"
_IDLE: SpanKind = "idle"


@dataclasses.dataclass(frozen=True, slots=True)
class PerformanceMetricsConfig:
  """Configuration for performance instrumentation."""

  enabled: bool = False
  components: tuple[str, ...] | None = None
  sampling_rate: int = 1
  export_jsonl_path: str | None = None
  otel_scope: str | None = None

  def should_sample(self, sequence_id: int | None) -> bool:
    if not self.enabled:
      return False
    rate = max(self.sampling_rate, 1)
    if sequence_id is None:
      return True
    return sequence_id % rate == 0

  def is_component_enabled(self, component: str) -> bool:
    if self.components is None:
      return True
    return component in self.components


class PerfCollector:
  """Collects span data and produces session level metrics."""

  def __init__(self, config: PerformanceMetricsConfig | None = None):
    self._config = config or PerformanceMetricsConfig()
    self._lock = threading.Lock()
    self._jsonl_handle = None
    self._tracer = None
    if self._config.otel_scope and otel_trace is not None:
      self._tracer = otel_trace.get_tracer(self._config.otel_scope)

  def start_session(
      self,
      name: str,
      context_id: Mapping[str, Any] | None = None,
      *,
      sequence_id: int | None = None,
      attributes: Mapping[str, Any] | None = None,
  ) -> PerfSession:
    enabled = self._config.should_sample(sequence_id)
    ctx = dict(context_id or {})
    attrs = dict(attributes or {})
    return PerfSession(
        collector=self,
        name=name,
        context=ctx,
        attributes=attrs,
        enabled=enabled,
    )

  def close(self) -> None:
    if self._jsonl_handle is not None:
      with self._lock:
        if self._jsonl_handle is not None:
          self._jsonl_handle.close()
          self._jsonl_handle = None

  def _should_record_component(self, component: str) -> bool:
    return self._config.is_component_enabled(component)

  def _qualify_span_name(self, session: PerfSession, span_name: str) -> str:
    return f"{session.name}.{span_name}"

  def _record_to_jsonl(self, payload: Mapping[str, Any]) -> None:
    if self._config.export_jsonl_path is None:
      return
    with self._lock:
      if self._jsonl_handle is None:
        self._jsonl_handle = open(  # pylint: disable=consider-using-with
            self._config.export_jsonl_path,
            "a",
            encoding="utf-8",
          )
      self._jsonl_handle.write(json.dumps(payload, ensure_ascii=True))
      self._jsonl_handle.write("\n")
      self._jsonl_handle.flush()

  def _finalize_session(self, session: PerfSession) -> Mapping[str, Any] | None:
    if not session.enabled:
      return None
    summary = session._summarize()  # pylint: disable=protected-access
    if summary is None:
      return None
    self._record_to_jsonl(summary)
    return summary

  @property
  def tracer(self):
    return self._tracer


class _NullSpan:
  """No-op span."""

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    return False


class PerfSpan:
  """Context manager that records a span duration."""

  __slots__ = ("_session", "_name", "_kind", "_attributes", "_start", "_otel_cm")

  def __init__(
      self,
      session: "PerfSession",
      name: str,
      kind: SpanKind,
      attributes: Mapping[str, Any] | None,
  ):
    self._session = session
    self._name = name
    self._kind = kind
    self._attributes = dict(attributes or {})
    self._start = 0.0
    self._otel_cm = contextlib.nullcontext()

  def __enter__(self) -> "PerfSpan":
    if not self._session.enabled or not self._session.collector._should_record_component(self._name):  # pylint: disable=protected-access
      return _NullSpan().__enter__()
    self._start = time.perf_counter()
    tracer = self._session.collector.tracer
    if tracer is not None:
      otel_attrs = {
          "perf.kind": self._kind,
          **self._session.attributes,
          **self._attributes,
      }
      span_name = self._session.collector._qualify_span_name(self._session, self._name)  # pylint: disable=protected-access
      cm = tracer.start_as_current_span(span_name, attributes=otel_attrs)
      self._otel_cm = cm
      self._otel_cm.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not self._session.enabled or not self._session.collector._should_record_component(self._name):  # pylint: disable=protected-access
      return False
    duration = time.perf_counter() - self._start
    self._session._record_span(  # pylint: disable=protected-access
        name=self._name,
        kind=self._kind,
        duration=duration,
        attributes=self._attributes,
    )
    if self._otel_cm is not None:
      self._otel_cm.__exit__(exc_type, exc_val, exc_tb)
    return False


class PerfSession:
  """Tracks spans for a single high level operation."""

  __slots__ = (
      "collector",
      "name",
      "context",
      "attributes",
      "enabled",
      "_start",
      "_busy",
      "_blocked",
      "_explicit_idle",
      "_spans",
      "_components",
      "_finished",
  )

  def __init__(
      self,
      *,
      collector: PerfCollector,
      name: str,
      context: MutableMapping[str, Any],
      attributes: MutableMapping[str, Any],
      enabled: bool,
  ):
    self.collector = collector
    self.name = name
    self.context = context
    self.attributes = attributes
    self.enabled = enabled
    self._start = time.perf_counter() if enabled else 0.0
    self._busy = 0.0
    self._blocked = 0.0
    self._explicit_idle = 0.0
    self._spans: list[dict[str, Any]] = []
    self._components: dict[str, dict[str, float]] = defaultdict(lambda: {_BUSY: 0.0, _BLOCKED: 0.0, _IDLE: 0.0})
    self._finished = False

  def span(
      self, name: str, *, kind: SpanKind = _BUSY, attributes: Mapping[str, Any] | None = None
  ):
    if not self.enabled or not self.collector._should_record_component(name):  # pylint: disable=protected-access
      return contextlib.nullcontext()
    return PerfSpan(self, name, kind, attributes)

  def busy_span(self, name: str, attributes: Mapping[str, Any] | None = None):
    return self.span(name, kind=_BUSY, attributes=attributes)

  def blocked_span(self, name: str, attributes: Mapping[str, Any] | None = None):
    return self.span(name, kind=_BLOCKED, attributes=attributes)

  def idle_span(self, name: str, attributes: Mapping[str, Any] | None = None):
    return self.span(name, kind=_IDLE, attributes=attributes)

  def _record_span(
      self,
      *,
      name: str,
      kind: SpanKind,
      duration: float,
      attributes: Mapping[str, Any] | None,
  ) -> None:
    if not self.enabled:
      return
    if kind == _BUSY:
      self._busy += duration
    elif kind == _BLOCKED:
      self._blocked += duration
    elif kind == _IDLE:
      self._explicit_idle += duration
    self._components[name][kind] += duration
    self._spans.append(
        {
            "name": name,
            "kind": kind,
            "duration_sec": duration,
            "attributes": dict(attributes or {}),
        }
    )

  def finish(self) -> Mapping[str, Any] | None:
    if not self.enabled or self._finished:
      return None
    self._finished = True
    return self.collector._finalize_session(self)  # pylint: disable=protected-access

  def _summarize(self) -> Mapping[str, Any] | None:
    if not self.enabled:
      return None
    wall_time = time.perf_counter() - self._start
    implicit_idle = max(wall_time - (self._busy + self._blocked + self._explicit_idle), 0.0)
    total_idle = self._explicit_idle + implicit_idle
    summary = {
        "session": self.name,
        "context": self.context,
        "attributes": self.attributes,
        "wall_time_sec": wall_time,
        "busy_time_sec": self._busy,
        "blocked_time_sec": self._blocked,
        "idle_time_sec": total_idle,
        "components": {},
        "spans": self._spans,
    }
    for component, totals in self._components.items():
      summary["components"][component] = {
          "busy_time_sec": totals.get(_BUSY, 0.0),
          "blocked_time_sec": totals.get(_BLOCKED, 0.0),
          "idle_time_sec": totals.get(_IDLE, 0.0),
      }
    return summary

  def metric_entries(self, prefix: str = "") -> Dict[str, float]:
    """Returns flat metric entries suitable for MetricsLogger."""
    summary = self._summarize()
    if summary is None:
      return {}
    pre = prefix or "perf"
    metrics = {
        f"{pre}/wall_time_sec": summary["wall_time_sec"],
        f"{pre}/busy_time_sec": summary["busy_time_sec"],
        f"{pre}/blocked_time_sec": summary["blocked_time_sec"],
        f"{pre}/idle_time_sec": summary["idle_time_sec"],
    }
    for component, totals in summary["components"].items():
      metrics[f"{pre}/{component}/busy_time_sec"] = totals["busy_time_sec"]
      metrics[f"{pre}/{component}/blocked_time_sec"] = totals["blocked_time_sec"]
      metrics[f"{pre}/{component}/idle_time_sec"] = totals["idle_time_sec"]
    return metrics
