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

"""Unit tests for tunix.dist_rl.common.otel_setup."""

import atexit
import importlib
import os
import socket
import sys
import threading
import time
from unittest import mock

from absl.testing import absltest
import numpy as np
from opentelemetry import _logs as otel_logs
from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.otlp.proto.grpc import _log_exporter as otlp_log_exporter
from opentelemetry.exporter.otlp.proto.grpc import metric_exporter as otlp_metric_exporter
from opentelemetry.sdk import _logs as sdk_logs
from opentelemetry.sdk._logs import export as sdk_logs_export
from opentelemetry.sdk.metrics import export as sdk_metrics_export
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.util import _once
from tunix.dist_rl.common import lineage
from tunix.dist_rl.common import otel_setup


def _reset_global_providers() -> None:
  """Resets process-global OpenTelemetry providers and lineage event sink."""
  lineage.set_event_sink(None)
  # pylint: disable=protected-access
  otel_setup._installed_pid = None
  prev_log_provider = otel_logs._internal._LOGGER_PROVIDER
  if isinstance(prev_log_provider, sdk_logs.LoggerProvider):
    atexit.unregister(prev_log_provider.shutdown)
    prev_log_provider.shutdown()
  otel_logs._internal._LOGGER_PROVIDER = None
  otel_logs._internal._LOGGER_PROVIDER_SET_ONCE = _once.Once()
  prev_meter_provider = otel_metrics._internal._METER_PROVIDER
  if isinstance(prev_meter_provider, MeterProvider):
    atexit.unregister(prev_meter_provider.shutdown)
    prev_meter_provider.shutdown(timeout_millis=1500)
  otel_metrics._internal._METER_PROVIDER = None
  otel_metrics._internal._METER_PROVIDER_SET_ONCE = _once.Once()
  # pylint: enable=protected-access


class _RecordingMetricExporter:
  """In-memory metric exporter double preserving SDK preference attributes."""

  def __init__(
      self, *, endpoint: str, timeout: float | None, preferred_temporality
  ):
    self.endpoint = endpoint
    self.timeout = timeout
    self._preferred_temporality = preferred_temporality
    self._preferred_aggregation = None
    self.exports = []
    self.flushes = []
    self.shutdowns = 0

  def reset(self) -> None:
    self.exports.clear()
    self.flushes.clear()
    self.shutdowns = 0

  def export(self, metrics_data, timeout_millis: float = 10_000, **kwargs):
    del timeout_millis, kwargs
    self.exports.append(metrics_data)
    return sdk_metrics_export.MetricExportResult.SUCCESS

  def force_flush(self, timeout_millis: float = 10_000) -> bool:
    self.flushes.append(timeout_millis)
    return True

  def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
    del timeout_millis, kwargs
    self.shutdowns += 1


class _RecordingLogExporter:
  """In-memory log exporter double compatible with opentelemetry-sdk >= 1.39.0."""

  def __init__(self, *, endpoint: str, timeout: float | None):
    self.endpoint = endpoint
    self.timeout = timeout
    self.batches = []
    self.flushes = []
    self.shutdowns = 0

  def reset(self) -> None:
    self.batches.clear()
    self.flushes.clear()
    self.shutdowns = 0

  def export(self, batch):
    self.batches.append(tuple(batch))
    return sdk_logs_export.LogRecordExportResult.SUCCESS

  def force_flush(self, timeout_millis: int = 30_000) -> bool:
    self.flushes.append(timeout_millis)
    return True

  def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
    del timeout_millis, kwargs
    self.shutdowns += 1


class OtelSetupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _reset_global_providers()
    self.addCleanup(_reset_global_providers)

  def test_no_opentelemetry_imported_at_module_top_level(self):
    # Reloading `otel_setup` with all `opentelemetry*` modules blocked must
    # succeed so callers like `resolve_config` work without `[otel]` installed.
    blocked = {
        name: None
        for name in list(sys.modules)
        if name == "opentelemetry" or name.startswith("opentelemetry.")
    }
    blocked["opentelemetry"] = None
    try:
      with mock.patch.dict(sys.modules, blocked):
        reloaded = importlib.reload(otel_setup)
        self.assertEqual(
            reloaded.normalize_endpoint("127.0.0.1:4317"),
            "http://127.0.0.1:4317",
        )
        self.assertFalse(
            reloaded.setup_metrics(default_service_name="x", endpoint="")
        )
    finally:
      importlib.reload(otel_setup)

  def test_disabled_without_endpoint(self):
    with mock.patch.dict(os.environ):
      os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
      self.assertFalse(otel_setup.setup_metrics(default_service_name="x"))

  def test_legacy_env_settings_parses_values_defaults_and_rejects_bad_inputs(
      self,
  ):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
          otel_setup.legacy_env_settings(),
          (
              "",
              otel_setup.DEFAULT_EXPORT_INTERVAL_MS,
              otel_setup.DEFAULT_EXPORT_TIMEOUT_S,
          ),
      )

    with mock.patch.dict(
        os.environ,
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "  127.0.0.1:4317  ",
            "OTEL_METRIC_EXPORT_INTERVAL": "4000",
            "OTEL_EXPORTER_OTLP_TIMEOUT": "2.5",
        },
        clear=True,
    ):
      self.assertEqual(
          otel_setup.legacy_env_settings(), ("127.0.0.1:4317", 4000, 2.5)
      )
      with (
          mock.patch.object(
              otlp_metric_exporter,
              "OTLPMetricExporter",
              side_effect=_RecordingMetricExporter,
          ) as mock_metric,
          mock.patch.object(
              otlp_log_exporter,
              "OTLPLogExporter",
              side_effect=_RecordingLogExporter,
          ) as mock_log,
      ):
        self.assertTrue(
            otel_setup.setup_metrics(default_service_name="tunix-trainer")
        )
      self.assertEqual(
          mock_metric.call_args.kwargs["endpoint"], "http://127.0.0.1:4317"
      )
      self.assertEqual(mock_metric.call_args.kwargs["timeout"], 2.5)
      self.assertEqual(mock_log.call_args.kwargs["timeout"], 2.5)

    for bad_interval in ("abc", "0", "-5"):
      with self.subTest(bad_interval=bad_interval):
        with mock.patch.dict(
            os.environ,
            {"OTEL_METRIC_EXPORT_INTERVAL": bad_interval},
            clear=True,
        ):
          with self.assertRaisesRegex(
              ValueError, "OTEL_METRIC_EXPORT_INTERVAL"
          ):
            otel_setup.legacy_env_settings()

    for bad_timeout in ("abc", "0", "-1", "nan", "inf"):
      with self.subTest(bad_timeout=bad_timeout):
        with mock.patch.dict(
            os.environ,
            {"OTEL_EXPORTER_OTLP_TIMEOUT": bad_timeout},
            clear=True,
        ):
          with self.assertRaisesRegex(ValueError, "OTEL_EXPORTER_OTLP_TIMEOUT"):
            otel_setup.legacy_env_settings()

  def test_disabled_with_warning_when_exporter_missing(self):
    missing = {"opentelemetry.exporter.otlp.proto.grpc.metric_exporter": None}
    with mock.patch.dict(sys.modules, missing):
      with self.assertLogs(level="WARNING") as logs:
        self.assertFalse(
            otel_setup.setup_metrics(
                default_service_name="x", endpoint="http://127.0.0.1:4317"
            )
        )
    self.assertTrue(any("google-tunix[otel]" in line for line in logs.output))
    self.assertNotIsInstance(otel_metrics.get_meter_provider(), MeterProvider)

  def test_end_to_end_setup_views_delta_gauges_lineage_and_fork_suppression(
      self,
  ):
    metric_fakes: list[_RecordingMetricExporter] = []
    log_fakes: list[_RecordingLogExporter] = []

    def _make_metric_exporter(**kwargs):
      fake = _RecordingMetricExporter(**kwargs)
      metric_fakes.append(fake)
      return fake

    def _make_log_exporter(**kwargs):
      fake = _RecordingLogExporter(**kwargs)
      log_fakes.append(fake)
      return fake

    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_make_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=_make_log_exporter,
        ),
    ):
      self.assertTrue(
          otel_setup.setup_metrics(
              default_service_name="tunix-rollout",
              resource_attributes={
                  "tunix.worker_id": "rollout_0",
                  "tunix.role": "rollout",
                  "tunix.run_id": "run_42",
                  "tunix.empty": "",
                  "service.name": "ignored-override",
              },
              endpoint="127.0.0.1:4317",
              export_interval_ms=60_000,
              export_timeout_s=2.5,
          )
      )

    self.assertLen(metric_fakes, 1)
    self.assertLen(log_fakes, 1)
    metric_fake = metric_fakes[0]
    log_fake = log_fakes[0]
    self.assertEqual(metric_fake.endpoint, "http://127.0.0.1:4317")
    self.assertEqual(metric_fake.timeout, 2.5)
    self.assertEqual(log_fake.endpoint, "http://127.0.0.1:4317")
    self.assertEqual(log_fake.timeout, 2.5)

    provider = otel_metrics.get_meter_provider()
    self.assertIsInstance(provider, MeterProvider)
    log_provider = otel_logs.get_logger_provider()
    self.assertIsInstance(log_provider, sdk_logs.LoggerProvider)

    meter = provider.get_meter("tunix.test")
    sec_hist = meter.create_histogram("latency", unit="s")
    tok_hist = meter.create_histogram("tokens", unit="{token}")
    step_gauge = meter.create_gauge("step_scalar", unit="1")
    sec_hist.record(0.3)
    tok_hist.record(42)
    step_gauge.set(7.0)

    def _flush_points():
      metric_fake.exports.clear()
      self.assertTrue(provider.force_flush())
      out = {}
      for batch in metric_fake.exports:
        for rm in batch.resource_metrics:
          for sm in rm.scope_metrics:
            for metric in sm.metrics:
              out[metric.name] = (
                  dict(rm.resource.attributes),
                  list(metric.data.data_points),
              )
      return out

    first = _flush_points()
    res_attrs, latency_pts = first["latency"]
    self.assertEqual(res_attrs["service.name"], "tunix-rollout")
    self.assertEqual(res_attrs["tunix.worker_id"], "rollout_0")
    self.assertEqual(res_attrs["tunix.role"], "rollout")
    self.assertEqual(res_attrs["tunix.run_id"], "run_42")
    self.assertEqual(res_attrs["host.name"], socket.gethostname())
    self.assertNotIn("tunix.empty", res_attrs)
    self.assertEqual(
        tuple(latency_pts[0].explicit_bounds),
        otel_setup.SECONDS_HISTOGRAM_BOUNDARIES,
    )
    self.assertEqual(latency_pts[0].bucket_counts[6], 1)
    self.assertNotEqual(
        tuple(first["tokens"][1][0].explicit_bounds),
        otel_setup.SECONDS_HISTOGRAM_BOUNDARIES,
    )
    self.assertEqual(first["step_scalar"][1][0].value, 7.0)

    # Delta temporality: synchronous gauge is absent on a second flush.
    second = _flush_points()
    self.assertNotIn("step_scalar", second)
    self.assertEqual(sum(second["latency"][1][0].bucket_counts), 1)

    # Lineage event arrives as a structured `tunix.lineage.*` record carrying
    # the process resource attributes and sanitized numpy scalar/array values.
    ctx = lineage.LineageContext(
        tracking_id="traj_p0_g0",
        task_name="deepswe",
        parent_tracking_ids=["p0"],
    )
    ctx.add_event(
        "worker.rollout",
        "generate",
        {
            "worker_id": "rollout_0",
            "batch_idx": np.int64(3),
            "execution_time_s": np.float32(0.5),
            "nan_score": float("nan"),
        },
    )
    self.assertTrue(log_provider.force_flush())
    self.assertNotEmpty(log_fake.batches)
    exported_item = log_fake.batches[-1][0]
    self.assertEqual(
        exported_item.resource.attributes["tunix.run_id"], "run_42"
    )
    rec = exported_item.log_record
    self.assertEqual(
        rec.attributes["event.name"], "tunix.lineage.worker.rollout.generate"
    )
    self.assertEqual(rec.attributes["tunix.tracking_id"], "traj_p0_g0")
    self.assertEqual(rec.attributes["tunix.batch_idx"], 3)
    self.assertAlmostEqual(rec.attributes["tunix.execution_time_s"], 0.5)
    self.assertNotIn("tunix.nan_score", rec.attributes)
    self.assertEqual(rec.body, {"timestamp_s": ctx.events[0].timestamp_s})

    # Real `os.fork()`: child records a metric, adds a lineage event, emits a
    # direct log record on the inherited LoggerProvider, calls setup_metrics,
    # and shuts down; neither fake exporter receives any export or shutdown.
    if hasattr(os, "fork"):
      read_fd, write_fd = os.pipe()
      pid = os.fork()
      if pid == 0:
        try:
          metric_fake.reset()
          log_fake.reset()
          child_setup = otel_setup.setup_metrics(
              default_service_name="tunix-child",
              endpoint="http://127.0.0.1:4317",
          )
          sec_hist.record(1.2)
          ctx.add_event("worker.child", "fork_step", {"turn": 1})
          now_ns = time.time_ns()
          log_provider.get_logger("child.direct").emit(
              otel_logs.LogRecord(
                  timestamp=now_ns,
                  observed_timestamp=now_ns,
                  severity_number=otel_logs.SeverityNumber.INFO,
                  severity_text="INFO",
                  body="child_log",
              )
          )
          provider.force_flush()
          log_provider.force_flush()
          provider.shutdown()
          log_provider.shutdown()
          payload = (
              f"{int(child_setup)}:{len(metric_fake.exports)}:"
              f"{len(log_fake.batches)}:{metric_fake.shutdowns}:"
              f"{log_fake.shutdowns}"
          )
          os.write(write_fd, payload.encode())
        finally:
          os._exit(0)
      os.close(write_fd)
      _, status = os.waitpid(pid, 0)
      child_report = os.read(read_fd, 64).decode()
      os.close(read_fd)
      self.assertEqual(os.waitstatus_to_exitcode(status), 0)
      self.assertEqual(child_report, "0:0:0:0:0")

  def test_idempotent_and_warns_on_role_collision_when_provider_installed(self):
    self.assertTrue(
        otel_setup.setup_metrics(
            default_service_name="tunix-orchestrator",
            endpoint="http://127.0.0.1:1",
            export_interval_ms=60_000,
            export_timeout_s=1.0,
        )
    )
    with mock.patch.object(otel_metrics, "set_meter_provider") as mock_set:
      self.assertTrue(
          otel_setup.setup_metrics(
              default_service_name="tunix-orchestrator",
              endpoint="http://127.0.0.1:1",
          )
      )
      with self.assertLogs(level="WARNING") as logs:
        self.assertTrue(
            otel_setup.setup_metrics(
                default_service_name="tunix-trainer",
                endpoint="http://127.0.0.1:1",
            )
        )
    mock_set.assert_not_called()
    self.assertTrue(
        any(
            "already initialized with service.name=tunix-orchestrator" in line
            for line in logs.output
        )
    )

  def test_concurrent_setup_metrics_installs_providers_once(self):
    entered = threading.Event()
    release = threading.Event()
    metric_fakes: list[_RecordingMetricExporter] = []
    log_fakes: list[_RecordingLogExporter] = []

    def _blocking_metric_exporter(**kwargs):
      entered.set()
      self.assertTrue(release.wait(timeout=5.0))
      fake = _RecordingMetricExporter(**kwargs)
      metric_fakes.append(fake)
      return fake

    def _make_log_exporter(**kwargs):
      fake = _RecordingLogExporter(**kwargs)
      log_fakes.append(fake)
      return fake

    results: list[bool] = []
    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_blocking_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=_make_log_exporter,
        ),
    ):
      t1 = threading.Thread(
          target=lambda: results.append(
              otel_setup.setup_metrics(
                  default_service_name="tunix-orchestrator",
                  endpoint="http://127.0.0.1:4317",
              )
          )
      )
      t2 = threading.Thread(
          target=lambda: results.append(
              otel_setup.setup_metrics(
                  default_service_name="tunix-trainer",
                  endpoint="http://127.0.0.1:4317",
              )
          )
      )
      t1.start()
      self.assertTrue(entered.wait(timeout=5.0))
      t2.start()
      time.sleep(0.05)
      release.set()
      t1.join(timeout=5.0)
      t2.join(timeout=5.0)

    self.assertEqual(results, [True, True])
    self.assertLen(metric_fakes, 1)
    self.assertLen(log_fakes, 1)

  def test_log_setup_failure_rolls_back_without_committing_half_initialized_state(
      self,
  ):
    metric_fakes: list[_RecordingMetricExporter] = []
    log_fakes: list[_RecordingLogExporter] = []

    def _make_metric_exporter(**kwargs):
      fake = _RecordingMetricExporter(**kwargs)
      metric_fakes.append(fake)
      return fake

    def _make_log_exporter(**kwargs):
      fake = _RecordingLogExporter(**kwargs)
      log_fakes.append(fake)
      return fake

    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_make_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=ValueError("bad OTEL_EXPORTER_OTLP_LOGS_CERTIFICATE"),
        ),
    ):
      with self.assertRaisesRegex(
          ValueError, "OTEL_EXPORTER_OTLP_LOGS_CERTIFICATE"
      ):
        otel_setup.setup_metrics(
            default_service_name="tunix-rollout",
            endpoint="http://127.0.0.1:4317",
        )

    self.assertEmpty(metric_fakes)
    self.assertNotIsInstance(otel_metrics.get_meter_provider(), MeterProvider)
    self.assertNotIsInstance(
        otel_logs.get_logger_provider(), sdk_logs.LoggerProvider
    )
    # pylint: disable-next=protected-access
    self.assertIsNone(otel_setup._installed_pid)

    # Also verify rollback when `BatchLogRecordProcessor` or
    # `PeriodicExportingMetricReader` fails after prior resources are built,
    # and confirm a subsequent retry succeeds with both providers wired.
    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_make_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=_make_log_exporter,
        ),
        mock.patch.object(
            sdk_logs_export,
            "BatchLogRecordProcessor",
            side_effect=ValueError("bad OTEL_BLRP_SCHEDULE_DELAY"),
        ),
    ):
      with self.assertRaisesRegex(ValueError, "OTEL_BLRP_SCHEDULE_DELAY"):
        otel_setup.setup_metrics(
            default_service_name="tunix-rollout",
            endpoint="http://127.0.0.1:4317",
        )

    self.assertEmpty(metric_fakes)
    self.assertLen(log_fakes, 1)
    self.assertEqual(log_fakes[0].shutdowns, 1)
    self.assertNotIsInstance(otel_metrics.get_meter_provider(), MeterProvider)

    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_make_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=_make_log_exporter,
        ),
        mock.patch.object(
            sdk_metrics_export,
            "PeriodicExportingMetricReader",
            side_effect=ValueError("bad OTEL_METRIC_EXPORT_INTERVAL"),
        ),
    ):
      with self.assertRaisesRegex(ValueError, "OTEL_METRIC_EXPORT_INTERVAL"):
        otel_setup.setup_metrics(
            default_service_name="tunix-rollout",
            endpoint="http://127.0.0.1:4317",
        )

    self.assertLen(metric_fakes, 1)
    self.assertEqual(metric_fakes[0].shutdowns, 1)
    self.assertLen(log_fakes, 2)
    self.assertEqual(log_fakes[1].shutdowns, 1)
    self.assertNotIsInstance(otel_metrics.get_meter_provider(), MeterProvider)

    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_make_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=_make_log_exporter,
        ),
    ):
      self.assertTrue(
          otel_setup.setup_metrics(
              default_service_name="tunix-rollout",
              endpoint="http://127.0.0.1:4317",
          )
      )
    self.assertIsInstance(otel_metrics.get_meter_provider(), MeterProvider)
    self.assertIsInstance(
        otel_logs.get_logger_provider(), sdk_logs.LoggerProvider
    )
    # pylint: disable-next=protected-access
    self.assertEqual(otel_setup._installed_pid, os.getpid())

  def test_preinstalled_noop_or_sdk_providers_wire_missing_signals_without_leaks(
      self,
  ):
    metric_fakes: list[_RecordingMetricExporter] = []
    log_fakes: list[_RecordingLogExporter] = []

    def _make_metric_exporter(**kwargs):
      fake = _RecordingMetricExporter(**kwargs)
      metric_fakes.append(fake)
      return fake

    def _make_log_exporter(**kwargs):
      fake = _RecordingLogExporter(**kwargs)
      log_fakes.append(fake)
      return fake

    # Case 1: Both global providers pre-locked to NoOp -> rejected providers are
    # shut down immediately, lineage sink stays unbound, and setup returns
    # False.
    otel_metrics.set_meter_provider(otel_metrics.NoOpMeterProvider())
    otel_logs.set_logger_provider(otel_logs.NoOpLoggerProvider())
    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_make_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=_make_log_exporter,
        ),
    ):
      self.assertFalse(
          otel_setup.setup_metrics(
              default_service_name="tunix-rollout",
              endpoint="http://127.0.0.1:4317",
          )
      )
    self.assertEqual(metric_fakes[0].shutdowns, 1)
    self.assertEqual(log_fakes[0].shutdowns, 1)
    # pylint: disable-next=protected-access
    self.assertIsNone(lineage._event_sink)
    # pylint: disable-next=protected-access
    self.assertIsNone(otel_setup._installed_pid)

    # Case 2: External SDK MeterProvider pre-installed before setup_metrics ->
    # reuses existing MeterProvider while still installing LoggerProvider,
    # lineage_otel, and _installed_pid.
    _reset_global_providers()
    metric_fakes.clear()
    log_fakes.clear()
    external_meter = MeterProvider()
    otel_metrics.set_meter_provider(external_meter)
    with (
        mock.patch.object(
            otlp_metric_exporter,
            "OTLPMetricExporter",
            side_effect=_make_metric_exporter,
        ),
        mock.patch.object(
            otlp_log_exporter,
            "OTLPLogExporter",
            side_effect=_make_log_exporter,
        ),
    ):
      self.assertTrue(
          otel_setup.setup_metrics(
              default_service_name="tunix-rollout",
              endpoint="http://127.0.0.1:4317",
          )
      )
    self.assertEmpty(metric_fakes)
    self.assertLen(log_fakes, 1)
    self.assertEqual(log_fakes[0].shutdowns, 0)
    self.assertIs(otel_metrics.get_meter_provider(), external_meter)
    self.assertIsInstance(
        otel_logs.get_logger_provider(), sdk_logs.LoggerProvider
    )
    # pylint: disable-next=protected-access
    self.assertIsNotNone(lineage._event_sink)
    # pylint: disable-next=protected-access
    self.assertEqual(otel_setup._installed_pid, os.getpid())

  def test_normalizes_or_rejects_endpoints_and_intervals(self):
    self.assertEqual(
        otel_setup.normalize_endpoint("https://collector.example.com"),
        "https://collector.example.com",
    )
    self.assertEqual(
        otel_setup.normalize_endpoint("localhost:4317"),
        "http://localhost:4317",
    )
    for bad_endpoint in (
        "http://[::1",
        "http://host:notaport",
        "http://",
        "http://localhost",
        "http://localhost:0",
        "ftp://localhost:4317",
    ):
      with self.subTest(endpoint=bad_endpoint):
        with self.assertRaisesRegex(ValueError, "OTLP endpoint"):
          otel_setup.setup_metrics(
              default_service_name="tunix-rollout", endpoint=bad_endpoint
          )

    for bad_interval in (0, -5, False):
      with self.subTest(interval=bad_interval):
        with self.assertRaisesRegex(ValueError, "export_interval_ms"):
          otel_setup.setup_metrics(
              default_service_name="tunix-trainer",
              endpoint="http://127.0.0.1:1",
              export_interval_ms=bad_interval,
          )

    for bad_timeout in (0.0, -1.0, float("inf"), float("nan"), True):
      with self.subTest(timeout=bad_timeout):
        with self.assertRaisesRegex(ValueError, "export_timeout_s"):
          otel_setup.setup_metrics(
              default_service_name="tunix-trainer",
              endpoint="http://127.0.0.1:1",
              export_timeout_s=bad_timeout,
          )


if __name__ == "__main__":
  absltest.main()
