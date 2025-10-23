"""Metrics logger unittest."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from metrax import logging as metrax_logging
import numpy as np
from tunix.sft import metrics_logger


class MetricLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.log_dir = self.create_tempdir().full_path
    except AttributeError:
      self.log_dir = tempfile.TemporaryDirectory().name

  @mock.patch("metrax.logging.logging_backend.datetime")
  @mock.patch("metrax.logging.logging_backend.wandb")
  def test_metrics_logger(self, mock_wandb, mock_datetime):
    fixed_timestamp_str = "2025-07-17_13-56-53"
    mock_datetime.datetime.now.return_value.strftime.return_value = (
        fixed_timestamp_str
    )
    logger = metrics_logger.MetricsLogger(
        metrics_logger.MetricsLoggerOptions(
            log_dir=self.log_dir, flush_every_n_steps=1
        )
    )
    self.assertLen(os.listdir(self.log_dir), 1)
    file_size_before = os.path.getsize(
        os.path.join(self.log_dir, os.listdir(self.log_dir)[0])
    )

    logger.log("loss", np.array(1.0), metrics_logger.Mode.TRAIN, 1)
    logger.log("perplexity", np.exp(1.0), metrics_logger.Mode.TRAIN, 1)
    logger.log("loss", np.array(4.0), "train", 2)
    logger.log("perplexity", np.exp(4.0), "train", 2)
    logger.log("loss", np.array(7.0), metrics_logger.Mode.EVAL, 2)
    logger.log("loss", np.array(10.0), "eval", 2)

    train_loss = logger.get_metric("loss", metrics_logger.Mode.TRAIN)
    self.assertEqual(train_loss, 2.5)
    train_perplexity = logger.get_metric("perplexity", "train")
    self.assertEqual(train_perplexity, np.exp(2.5))

    eval_loss_history = logger.get_metric_history("loss", "eval")
    np.testing.assert_array_equal(eval_loss_history, np.array([7.0, 10.0]))

    self.assertLen(os.listdir(self.log_dir), 1)
    file_size_after = os.path.getsize(
        os.path.join(self.log_dir, os.listdir(self.log_dir)[0])
    )

    self.assertGreater(file_size_after, file_size_before)

    mock_wandb.init.assert_called_once_with(
        project="tunix", name=fixed_timestamp_str, anonymous="allow"
    )
    self.assertEqual(mock_wandb.log.call_count, 6)

    logger.close()
    mock_wandb.finish.assert_called_once()

  # Updated patches to mock the backend classes as used by metrics_logger
  @mock.patch("tunix.sft.metrics_logger.TensorBoardBackend")
  @mock.patch("tunix.sft.metrics_logger.WandbBackend")
  def test_additional_and_default_backends_are_called(
      self, mock_wandb_backend, mock_tensor_board_backend
  ):
    """Tests that default (TB, W&B) and custom backends are all called."""
    mock_tb_instance = mock_tensor_board_backend.return_value
    mock_wandb_instance = mock_wandb_backend.return_value

    mock_additional_backend = mock.Mock(spec=metrax_logging.LoggingBackend)

    logger = metrics_logger.MetricsLogger(
        metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir),
        metric_prefix="test_prefix/",
        additional_backends=[mock_additional_backend],
    )

    logger.log("custom_loss", 1.23, metrics_logger.Mode.TRAIN, step=100)
    expected_args = ("test_prefix/train/custom_loss", 1.23)
    expected_kwargs = {"step": 100}

    mock_tb_instance.log_scalar.assert_called_once_with(
        *expected_args, **expected_kwargs
    )
    mock_wandb_instance.log_scalar.assert_called_once_with(
        *expected_args, **expected_kwargs
    )
    mock_additional_backend.log_scalar.assert_called_once_with(
        *expected_args, **expected_kwargs
    )

    logger.close()

    mock_tb_instance.close.assert_called_once()
    mock_wandb_instance.close.assert_called_once()
    mock_additional_backend.close.assert_called_once()


if __name__ == "__main__":
  absltest.main()
