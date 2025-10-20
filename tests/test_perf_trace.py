"""Tests for the performance tracing helpers."""

import math

import numpy as np

from tunix.perf import trace as perf_trace


def _collector(**overrides):
  config = perf_trace.PerformanceMetricsConfig(enabled=True, **overrides)
  return perf_trace.PerfCollector(config)


def test_metric_entries_include_busy_blocked_idle(monkeypatch):
  collector = _collector()
  session = collector.start_session("unit", {"step": 1})

  session._record_span(  # pylint: disable=protected-access
      name="work", kind="busy", duration=0.12, attributes=None
  )
  session._record_span(  # pylint: disable=protected-access
      name="wait", kind="blocked", duration=0.08, attributes=None
  )

  start = session._start  # pylint: disable=protected-access
  monkeypatch.setattr(
      perf_trace.time, "perf_counter", lambda: start + 0.25
  )

  metrics = session.metric_entries(prefix="test")
  assert math.isclose(metrics["test/busy_time_sec"], 0.12, rel_tol=1e-6)
  assert math.isclose(metrics["test/blocked_time_sec"], 0.08, rel_tol=1e-6)
  assert metrics["test/idle_time_sec"] >= 0.0


def test_sampling_rate_disables_sessions():
  collector = _collector(sampling_rate=2)
  disabled = collector.start_session("unit", {"step": 1}, sequence_id=1)
  assert not disabled.enabled

  enabled = collector.start_session("unit", {"step": 2}, sequence_id=2)
  assert enabled.enabled

  with enabled.busy_span("work"):
    pass
  metrics = enabled.metric_entries()
  assert np.isfinite(metrics["perf/busy_time_sec"])
