"""Metrics logger unittest."""

import collections
import copy
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import jax
import metrax.logging as metrax_logging
import numpy as np
from opentelemetry.sdk import metrics as otel_sdk_metrics
from opentelemetry.sdk.metrics import export as otel_sdk_export
from tunix.sft import metrics_logger
from tunix.utils import env_utils


class MetricLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._temp_dir_obj = tempfile.TemporaryDirectory()
    self.log_dir = self._temp_dir_obj.name
    self.mock_backends = []

    if env_utils.is_internal_env():
      self.mock_backends.append(
          self.enter_context(
              mock.patch.object(metrics_logger, "CluBackend")
          )
      )
    else:
      self.mock_backends.append(
          self.enter_context(
              mock.patch.object(metrics_logger, "TensorboardBackend")
          )
      )
      self.mock_backends.append(
          self.enter_context(
              mock.patch.object(metrics_logger, "WandbBackend")
          )
      )

  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_custom_backends_override_defaults(self, mock_register):
    """Tests that providing a 'backends' list overrides the defaults."""
    mock_backend_instance = mock.Mock(spec=metrax_logging.LoggingBackend)
    mock_factory = mock.Mock(return_value=mock_backend_instance)
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir,
        backend_kwargs={"custom_backend": [mock_factory]},
    )

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    for backend_mock in self.mock_backends:
      backend_mock.assert_not_called()
    mock_factory.assert_called_once()
    self.assertIn(mock_backend_instance, logger._backends)
    self.assertLen(logger._backends, 1)
    mock_register.assert_called_once_with(mock_backend_instance.log_scalar)

    logger.close()
    mock_backend_instance.close.assert_called_once()

  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_defaults_are_used_when_no_backends_provided(self, mock_register):
    """Tests that defaults are created when 'backends' is None."""
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)

    for backend_mock in self.mock_backends:
      backend_mock.assert_called_once()
      self.assertIn(backend_mock.return_value, logger._backends)

    self.assertLen(logger._backends, len(self.mock_backends))
    self.assertEqual(mock_register.call_count, len(self.mock_backends))

    logger.close()
    for backend_mock in self.mock_backends:
      backend_mock.return_value.close.assert_called_once()

  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_logger_handles_missing_wandb_gracefully(self, mock_register):
    """Tests that the logger doesn't crash if wandb is not installed."""
    # wandb is not supported in internal environment.
    if env_utils.is_internal_env():
      return

    mock_tensorboard_instance = self.mock_backends[0].return_value
    mock_wandb_backend = self.mock_backends[1]
    mock_wandb_backend.side_effect = ImportError("W&B not installed")
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    self.assertIn(mock_tensorboard_instance, logger._backends)
    self.assertLen(logger._backends, 1)
    mock_register.assert_called_once_with(mock_tensorboard_instance.log_scalar)

    logger.close()
    mock_tensorboard_instance.close.assert_called_once()

  def test_options_deepcopy_safety(self):
    """Tests that deepcopying options and creating new loggers is safe."""
    options1 = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)
    logger1 = metrics_logger.MetricsLogger(metrics_logger_options=options1)
    for backend_mock in self.mock_backends:
      self.assertEqual(backend_mock.call_count, 1)

    options2 = copy.deepcopy(options1)
    new_log_dir = os.path.join(self.log_dir, "critic")
    options2.log_dir = new_log_dir
    logger2 = metrics_logger.MetricsLogger(metrics_logger_options=options2)

    for backend_mock in self.mock_backends:
      self.assertEqual(backend_mock.call_count, 2)

    if env_utils.is_internal_env():
      self.mock_backends[0].assert_called_with(log_dir=new_log_dir)
    else:
      self.mock_backends[0].assert_called_with(
          log_dir=new_log_dir,
          flush_every_n_steps=100,
      )

    logger1.close()
    logger2.close()

  @mock.patch.object(jax.monitoring, "record_scalar")
  def test_log_metrics(self, mock_record_scalar):
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir, backend_kwargs={"custom_backend": []}
    )
    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    logger.log("test_prefix", "loss", 0.1, metrics_logger.Mode.TRAIN, 1)
    logger.log("test_prefix", "loss", 0.05, metrics_logger.Mode.TRAIN, 2)
    mock_record_scalar.assert_has_calls([
        mock.call("test_prefix/train/loss", 0.1, step=1),
        mock.call("test_prefix/train/loss", 0.05, step=2),
    ])
    self.assertTrue(logger.metric_exists("test_prefix", "loss", "train"))
    self.assertAlmostEqual(
        logger.get_metric("test_prefix", "loss", "train"), 0.075
    )
    history = logger.get_metric_history("test_prefix", "loss", "train")
    self.assertLen(history, 2)
    self.assertAlmostEqual(history[0], 0.1)
    self.assertAlmostEqual(history[1], 0.05)

  @mock.patch.object(jax.monitoring, "record_scalar")
  def test_log_perplexity(self, mock_record_scalar):
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir, backend_kwargs={"custom_backend": []}
    )
    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    logger.log("test_prefix", "perplexity", 10.0, metrics_logger.Mode.EVAL, 1)
    logger.log("test_prefix", "perplexity", 100.0, metrics_logger.Mode.EVAL, 2)
    mock_record_scalar.assert_has_calls([
        mock.call("test_prefix/eval/perplexity", 10.0, step=1),
        mock.call("test_prefix/eval/perplexity", 100.0, step=2),
    ])
    self.assertTrue(logger.metric_exists("test_prefix", "perplexity", "eval"))
    self.assertAlmostEqual(
        logger.get_metric("test_prefix", "perplexity", "eval"), 31.6227766
    )

  @mock.patch.object(jax, "process_index", return_value=1)
  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_no_backends_on_secondary_process(
      self,
      mock_register,
      mock_jax_process_index,
  ):
    del mock_jax_process_index
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)
    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    for backend_mock in self.mock_backends:
      backend_mock.assert_not_called()
    self.assertEmpty(logger._backends)
    mock_register.assert_not_called()
    logger.close()

  @mock.patch.object(env_utils, "is_internal_env", return_value=True)
  def test_raises_when_clu_backend_missing_in_internal_env(
      self, mock_is_internal_env
  ):
    del mock_is_internal_env
    # We need to patch CluBackend to be None in the module.
    with mock.patch.object(metrics_logger, "CluBackend", new=None):
      options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)
      with self.assertRaisesRegex(
          ImportError,
          "Internal environment detected, but CluBackend not available.",
      ):
        options.create_backends()

  def test_backend_kwargs_are_passed_to_backends(self):
    """Tests that backend_kwargs are passed to the backends initialization."""
    mock_wandb = mock.Mock()
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir,
        backend_kwargs={
            "wandb": {
                "resume": "must",
                "id": "12345",
                "settings": mock_wandb.Settings(console="off"),
            },
            "tensorboard": {"flush_interval_s": 10.0},
            "clu": {"some_arg": "value"},
        },
    )

    # Trigger backend creation
    _ = metrics_logger.MetricsLogger(metrics_logger_options=options)

    if env_utils.is_internal_env():
      self.mock_backends[0].assert_called_once_with(
          log_dir=self.log_dir, **options.backend_kwargs["clu"]
      )
    else:
      self.mock_backends[0].assert_called_once_with(
          log_dir=self.log_dir,
          flush_every_n_steps=100,
          **options.backend_kwargs["tensorboard"],
      )
      self.mock_backends[1].assert_called_once_with(
          project="tunix",
          name="",
          resume="must",
          id="12345",
          settings=mock_wandb.Settings.return_value,
      )


def _gauge_points(metrics_data):
  """Maps instrument name to a list of (value, attributes) data points."""
  points = collections.defaultdict(list)
  if metrics_data is None:
    return points
  for resource_metrics in metrics_data.resource_metrics:
    for scope_metrics in resource_metrics.scope_metrics:
      for metric in scope_metrics.metrics:
        for point in metric.data.data_points:
          points[metric.name].append((point.value, dict(point.attributes)))
  return points


class OpenTelemetryDoubleWriteTest(absltest.TestCase):
  """Tests the opt-in OpenTelemetry double-write path."""

  def setUp(self):
    super().setUp()
    self._temp_dir_obj = tempfile.TemporaryDirectory()
    self.log_dir = self._temp_dir_obj.name
    self.backend = mock.Mock(spec=metrax_logging.LoggingBackend)
    self.enter_context(
        mock.patch.object(jax.monitoring, "register_scalar_listener")
    )
    self.mock_record_scalar = self.enter_context(
        mock.patch.object(jax.monitoring, "record_scalar")
    )
    self.reader = otel_sdk_export.InMemoryMetricReader()
    self.meter_provider = otel_sdk_metrics.MeterProvider(
        metric_readers=[self.reader]
    )

  def _make_logger(self, enable_opentelemetry=True):
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir,
        backend_kwargs={"custom_backend": [lambda: self.backend]},
        enable_opentelemetry=enable_opentelemetry,
    )
    return metrics_logger.MetricsLogger(
        metrics_logger_options=options,
        otel_meter_provider=self.meter_provider,
    )

  def test_disabled_by_default_and_legacy_path_unchanged(self):
    logger = self._make_logger(enable_opentelemetry=False)
    logger.log("actor", "loss", 0.5, metrics_logger.Mode.TRAIN, 1)

    self.mock_record_scalar.assert_called_once_with(
        "actor/train/loss", 0.5, step=1
    )
    self.assertEqual(_gauge_points(self.reader.get_metrics_data()), {})
    self.assertAlmostEqual(logger.get_metric("actor", "loss", "train"), 0.5)

  def test_double_write_emits_otel_and_legacy(self):
    logger = self._make_logger()
    logger.log("actor", "loss", 0.5, metrics_logger.Mode.TRAIN, 7)

    # Legacy jax.monitoring write is unchanged.
    self.mock_record_scalar.assert_called_once_with(
        "actor/train/loss", 0.5, step=7
    )
    # OpenTelemetry gauge is additionally emitted.
    points = _gauge_points(self.reader.get_metrics_data())
    expected_attributes = {
        "tunix.training.mode": "train",
        "tunix.metrics.prefix": "actor",
    }
    self.assertEqual(
        points["tunix.training.loss"], [(0.5, expected_attributes)]
    )
    self.assertEqual(points["tunix.training.step"], [(7, expected_attributes)])
    # Local history is unchanged.
    self.assertAlmostEqual(logger.get_metric("actor", "loss", "train"), 0.5)

  def test_dynamic_metric_names_are_normalized(self):
    logger = self._make_logger()
    logger.log("actor", "rewards/score mean", 1.5, "train", 1)

    points = _gauge_points(self.reader.get_metrics_data())
    self.assertIn("tunix.rewards.score.mean", points)

  def test_non_scalar_values_stay_in_history_only(self):
    logger = self._make_logger()
    logger.log("actor", "loss", np.array([0.5, 0.6]), "train", 1)

    points = _gauge_points(self.reader.get_metrics_data())
    self.assertNotIn("tunix.training.loss", points)
    self.assertLen(logger.get_metric_history("actor", "loss", "train"), 1)

  def test_repeated_step_emits_step_gauge_once(self):
    logger = self._make_logger()
    logger.log("actor", "loss", 0.5, "train", 3)
    logger.log("actor", "perplexity", 1.6, "train", 3)

    points = _gauge_points(self.reader.get_metrics_data())
    self.assertLen(points["tunix.training.step"], 1)

  @mock.patch.object(jax, "process_index", return_value=1)
  def test_no_otel_emission_on_secondary_process(self, mock_process_index):
    del mock_process_index
    logger = self._make_logger()
    logger.log("actor", "loss", 0.5, "train", 1)

    self.assertEqual(_gauge_points(self.reader.get_metrics_data()), {})

  def test_missing_otel_api_raises_when_enabled(self):
    with mock.patch.object(metrics_logger, "otel_metrics", new=None):
      with self.assertRaisesRegex(ImportError, "google-tunix\\[otel\\]"):
        self._make_logger()

  def test_close_leaves_meter_provider_running(self):
    logger = self._make_logger()
    logger.log("actor", "loss", 0.5, "train", 1)
    logger.close()

    self.backend.close.assert_called_once()
    logger_after_close = self._make_logger()
    logger_after_close.log("actor", "loss", 0.25, "train", 2)
    points = _gauge_points(self.reader.get_metrics_data())
    self.assertIn(0.25, [value for value, _ in points["tunix.training.loss"]])


if __name__ == "__main__":
  absltest.main()
