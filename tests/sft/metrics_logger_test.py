"""Metrics logger unittest."""

import copy
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import metrax.logging as metrax_logging
from tunix.sft import metrics_logger


class MetricLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._temp_dir_obj = tempfile.TemporaryDirectory()
    self.log_dir = self._temp_dir_obj.name

  @mock.patch("jax.monitoring.register_scalar_listener")
  @mock.patch("tunix.sft.metrics_logger.TensorboardBackend")
  @mock.patch("tunix.sft.metrics_logger.WandbBackend")
  def test_custom_backends_override_defaults(
      self, mock_wandb_backend, mock_tensorboard_backend, mock_register
  ):
    """Tests that providing a 'backends' list overrides the defaults."""
    mock_backend_instance = mock.Mock(spec=metrax_logging.LoggingBackend)
    mock_factory = mock.Mock(return_value=mock_backend_instance)
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir, backend_factories=[mock_factory]
    )

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    mock_tensorboard_backend.assert_not_called()
    mock_wandb_backend.assert_not_called()
    mock_factory.assert_called_once()
    self.assertIn(mock_backend_instance, logger._backends)
    self.assertLen(logger._backends, 1)
    mock_register.assert_called_once_with(mock_backend_instance.log_scalar)

    logger.close()
    mock_backend_instance.close.assert_called_once()

  @mock.patch("jax.monitoring.register_scalar_listener")
  @mock.patch("tunix.sft.metrics_logger.TensorboardBackend")
  @mock.patch("tunix.sft.metrics_logger.WandbBackend")
  def test_defaults_are_used_when_no_backends_provided(
      self, mock_wandb_backend, mock_tensorboard_backend, mock_register
  ):
    """Tests that defaults are created when 'backends' is None."""
    mock_tb_instance = mock_tensorboard_backend.return_value
    mock_wandb_instance = mock_wandb_backend.return_value
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    mock_tensorboard_backend.assert_called_once()
    mock_wandb_backend.assert_called_once()
    self.assertIn(mock_tb_instance, logger._backends)
    self.assertIn(mock_wandb_instance, logger._backends)
    self.assertLen(logger._backends, 2)
    self.assertEqual(mock_register.call_count, 2)

    logger.close()
    mock_tb_instance.close.assert_called_once()
    mock_wandb_instance.close.assert_called_once()

  @mock.patch("jax.monitoring.register_scalar_listener")
  @mock.patch("tunix.sft.metrics_logger.TensorboardBackend")
  @mock.patch("tunix.sft.metrics_logger.WandbBackend")
  def test_logger_handles_missing_wandb_gracefully(
      self, mock_wandb_backend, mock_tensorboard_backend, mock_register
  ):
    """Tests that the logger doesn't crash if wandb is not installed."""
    mock_wandb_backend.side_effect = ImportError("W&B not installed")
    mock_tb_instance = mock_tensorboard_backend.return_value
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    self.assertIn(mock_tb_instance, logger._backends)
    self.assertLen(logger._backends, 1)
    mock_register.assert_called_once_with(mock_tb_instance.log_scalar)

    logger.close()
    mock_tb_instance.close.assert_called_once()

  @mock.patch("tunix.sft.metrics_logger.TensorboardBackend")
  @mock.patch("tunix.sft.metrics_logger.WandbBackend")
  def test_options_deepcopy_safety(self, _, mock_tensorboard_backend):
    """Tests that deepcopying options and creating new loggers is safe."""
    options1 = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)
    logger1 = metrics_logger.MetricsLogger(metrics_logger_options=options1)
    self.assertEqual(mock_tensorboard_backend.call_count, 1)

    options2 = copy.deepcopy(options1)
    new_log_dir = os.path.join(self.log_dir, "critic")
    options2.log_dir = new_log_dir
    logger2 = metrics_logger.MetricsLogger(metrics_logger_options=options2)
    self.assertEqual(mock_tensorboard_backend.call_count, 2)
    mock_tensorboard_backend.assert_called_with(
        log_dir=new_log_dir, flush_every_n_steps=100
    )

    logger1.close()
    logger2.close()


if __name__ == "__main__":
  absltest.main()
