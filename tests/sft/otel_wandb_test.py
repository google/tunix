"""Tests for the OpenTelemetry to Weights & Biases metric exporter."""

import glob
import json
import os
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
import jax
import metrax.logging as metrax_logging
from opentelemetry.sdk import metrics as otel_sdk_metrics
from opentelemetry.sdk.metrics import export as otel_sdk_export
from tunix.sft import metrics_logger
from tunix.sft import otel_wandb

try:
  import wandb  # pylint: disable=g-import-not-at-top
except ImportError:
  wandb = None


class _FakeWandbRun:
  """Captures wandb.log calls."""

  def __init__(self):
    self.calls = []

  def log(self, data, step=None):
    self.calls.append((dict(data), step))


class WandbMetricsExporterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.run = _FakeWandbRun()
    self.exporter = otel_wandb.WandbMetricsExporter(self.run)
    self.reader = otel_sdk_export.PeriodicExportingMetricReader(
        self.exporter, export_interval_millis=3_600_000
    )
    self.meter_provider = otel_sdk_metrics.MeterProvider(
        metric_readers=[self.reader]
    )
    self.enter_context(
        mock.patch.object(jax.monitoring, "register_scalar_listener")
    )
    self.enter_context(mock.patch.object(jax.monitoring, "record_scalar"))

  def tearDown(self):
    self.meter_provider.shutdown()
    super().tearDown()

  def _temp_dir(self):
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    return temp_dir.name

  def _make_logger(self):
    backend = mock.Mock(spec=metrax_logging.LoggingBackend)
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self._temp_dir(),
        backend_kwargs={"custom_backend": [lambda: backend]},
        enable_opentelemetry=True,
    )
    return metrics_logger.MetricsLogger(
        metrics_logger_options=options,
        otel_meter_provider=self.meter_provider,
    )

  def test_double_write_reaches_wandb_run(self):
    logger = self._make_logger()
    logger.log("actor", "loss", 0.5, metrics_logger.Mode.TRAIN, 3)
    logger.log("actor", "perplexity", 1.65, metrics_logger.Mode.TRAIN, 3)

    self.meter_provider.force_flush()

    self.assertLen(self.run.calls, 1)
    values, step = self.run.calls[0]
    self.assertEqual(step, 3)
    self.assertAlmostEqual(values["actor/train/tunix.training.loss"], 0.5)
    self.assertAlmostEqual(
        values["actor/train/tunix.training.perplexity"], 1.65
    )

  def test_groups_are_logged_in_step_order(self):
    logger = self._make_logger()
    logger.log("critic", "loss", 0.7, metrics_logger.Mode.EVAL, 9)
    logger.log("actor", "loss", 0.5, metrics_logger.Mode.TRAIN, 2)

    self.meter_provider.force_flush()

    steps = [step for _, step in self.run.calls]
    self.assertEqual(steps, sorted(steps))
    logged_keys = set()
    for values, _ in self.run.calls:
      logged_keys.update(values)
    self.assertIn("actor/train/tunix.training.loss", logged_keys)
    self.assertIn("critic/eval/tunix.training.loss", logged_keys)

  def _collect_metrics_data(self):
    """Builds one metric export batch through an in-memory reader."""
    reader = otel_sdk_export.InMemoryMetricReader()
    provider = otel_sdk_metrics.MeterProvider(metric_readers=[reader])
    logger = metrics_logger.MetricsLogger(
        metrics_logger_options=metrics_logger.MetricsLoggerOptions(
            log_dir=self._temp_dir(),
            backend_kwargs={
                "custom_backend": [
                    lambda: mock.Mock(spec=metrax_logging.LoggingBackend)
                ]
            },
            enable_opentelemetry=True,
        ),
        otel_meter_provider=provider,
    )
    logger.log("actor", "loss", 0.5, "train", 1)
    return reader.get_metrics_data()

  def test_export_failure_is_reported_not_raised(self):
    failing_run = mock.Mock()
    failing_run.log.side_effect = RuntimeError("wandb unavailable")
    exporter = otel_wandb.WandbMetricsExporter(failing_run)

    result = exporter.export(self._collect_metrics_data())

    self.assertEqual(result, otel_sdk_export.MetricExportResult.FAILURE)

  def test_resolves_global_wandb_run_lazily(self):
    fake_wandb = mock.Mock()
    fake_wandb.run = _FakeWandbRun()
    exporter = otel_wandb.WandbMetricsExporter()
    metrics_data = self._collect_metrics_data()

    with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
      result = exporter.export(metrics_data)

    self.assertEqual(result, otel_sdk_export.MetricExportResult.SUCCESS)
    self.assertLen(fake_wandb.run.calls, 1)


class _CountingBackend:
  """Legacy Metrax-protocol backend recording jax.monitoring scalars."""

  def __init__(self):
    self.scalars = []

  def log_scalar(self, event, value, **kwargs):
    self.scalars.append((event, value, kwargs.get("step")))

  def close(self):
    pass


@absltest.skipIf(wandb is None, "wandb is not installed")
class WandbOfflineEndToEndTest(absltest.TestCase):
  """Double-writes through the real wandb client using an offline run.

  ``wandb.init(mode="offline")`` exercises the full wandb client (run setup,
  ``log`` semantics, step alignment, transaction-log serialization) without
  needing an account or network access. The test reads the resulting
  ``.wandb`` transaction log back with wandb's own reader and asserts the
  per-step history, while a legacy backend on the same logger proves both
  sides of the double-write.
  """

  def _read_history(self, wandb_dir):
    """Maps step to logged values from the offline run's transaction log."""
    # pylint: disable=g-import-not-at-top
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    # pylint: enable=g-import-not-at-top

    wandb_file = glob.glob(
        os.path.join(wandb_dir, "wandb", "offline-run-*", "run-*.wandb")
    )[0]
    store = datastore.DataStore()
    store.open_for_scan(wandb_file)
    history = {}
    while True:
      data = store.scan_data()
      if data is None:
        break
      record = wandb_internal_pb2.Record()
      record.ParseFromString(data)
      if record.WhichOneof("record_type") == "history":
        items = {}
        for item in record.history.item:
          key = ".".join(item.nested_key) if item.nested_key else item.key
          items[key] = json.loads(item.value_json)
        history[items["_step"]] = items
    return history

  def test_double_write_reaches_real_offline_wandb_run(self):
    wandb_dir = tempfile.TemporaryDirectory()
    self.addCleanup(wandb_dir.cleanup)
    self.enter_context(
        mock.patch.dict(
            os.environ, {"WANDB_MODE": "offline", "WANDB_SILENT": "true"}
        )
    )
    run = wandb.init(
        mode="offline",
        project="tunix-otel-e2e",
        dir=wandb_dir.name,
        settings=wandb.Settings(console="off"),
    )
    reader = otel_sdk_export.PeriodicExportingMetricReader(
        otel_wandb.WandbMetricsExporter(run), export_interval_millis=3_600_000
    )
    provider = otel_sdk_metrics.MeterProvider(metric_readers=[reader])
    self.addCleanup(provider.shutdown)

    legacy_backend = _CountingBackend()
    logger = metrics_logger.MetricsLogger(
        metrics_logger_options=metrics_logger.MetricsLoggerOptions(
            log_dir=os.path.join(wandb_dir.name, "logs"),
            backend_kwargs={"custom_backend": [lambda: legacy_backend]},
            enable_opentelemetry=True,
        ),
        otel_meter_provider=provider,
    )
    # close() clears the process-global scalar listener this logger registers;
    # make sure that happens even when an assertion below fails first.
    self.addCleanup(logger.close)

    for step, loss in enumerate([2.31, 1.87, 1.52], start=1):
      logger.log("actor", "loss", loss, metrics_logger.Mode.TRAIN, step)
      logger.log("actor", "rewards/score mean", 0.1 * step, "train", step)
      provider.force_flush()
    logger.log("actor", "loss", 1.61, metrics_logger.Mode.EVAL, 3)
    provider.force_flush()
    run.finish()

    history = self._read_history(wandb_dir.name)
    self.assertEqual(
        [
            history[step]["actor/train/tunix.training.loss"]
            for step in (1, 2, 3)
        ],
        [2.31, 1.87, 1.52],
    )
    self.assertAlmostEqual(
        history[3]["actor/train/tunix.rewards.score.mean"], 0.3
    )
    self.assertEqual(history[3]["actor/eval/tunix.training.loss"], 1.61)
    # The legacy jax.monitoring path received every scalar in the same run.
    self.assertLen(legacy_backend.scalars, 7)
    self.assertIn(("actor/train/loss", 2.31, 1), legacy_backend.scalars)


if __name__ == "__main__":
  absltest.main()
