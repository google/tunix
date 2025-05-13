"""Metrics logger unittest."""

import os
from unittest import mock

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from tunix.sft import metrics_logger


class MetricLoggerTest(absltest.TestCase):

  @mock.patch("tunix.sft.metrics_logger.wandb")
  def test_metrics_logger(self, mock_wandb):
    metrics_logger.wandb = mock_wandb
    log_dir = self.create_tempdir().full_path
    logger = metrics_logger.MetricsLogger(
        metrics_logger.MetricsLoggerOptions(
            log_dir=log_dir, flush_every_n_steps=1
        )
    )
    self.assertLen(os.listdir(log_dir), 1)
    file_size_before = os.path.getsize(
        os.path.join(log_dir, os.listdir(log_dir)[0])
    )

    logger.log("loss", jnp.array(1.0), metrics_logger.Mode.TRAIN, 1)
    logger.log("perplexity", jnp.exp(1.0), metrics_logger.Mode.TRAIN, 1)
    logger.log("loss", jnp.array(4.0), "train", 2)
    logger.log("perplexity", jnp.exp(4.0), "train", 2)
    logger.log("loss", jnp.array(7.0), metrics_logger.Mode.EVAL, 2)
    logger.log("loss", jnp.array(10.0), "eval", 2)

    train_loss = logger.get_metric("loss", metrics_logger.Mode.TRAIN)
    self.assertEqual(train_loss, 2.5)
    train_perplexity = logger.get_metric("perplexity", "train")
    self.assertEqual(train_perplexity, jnp.exp(2.5))

    eval_loss_history = logger.get_metric_history("loss", "eval")
    np.testing.assert_array_equal(eval_loss_history, jnp.array([7.0, 10.0]))

    self.assertLen(os.listdir(log_dir), 1)
    file_size_after = os.path.getsize(
        os.path.join(log_dir, os.listdir(log_dir)[0])
    )

    self.assertGreater(file_size_after, file_size_before)

    mock_wandb.init.assert_called_once_with(
        project="tunix", name="tunix_metrics_logger", anonymous="allow"
    )
    self.assertGreater(mock_wandb.log.call_count, 6)

    logger.close()
    mock_wandb.finish.assert_called_once()


if __name__ == "__main__":
  absltest.main()
