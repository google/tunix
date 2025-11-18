"""Unit tests for `progress_bar`."""

from absl.testing import absltest
import numpy as np
from tunix.sft import metrics_logger
from tunix.sft import progress_bar
import tempfile


class ProgressBarTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.log_dir = self.create_tempdir().full_path
    except Exception:
      self.log_dir = tempfile.TemporaryDirectory().name

    self.metrics_prefix = "test"
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir
    )
    self.metrics_logger = metrics_logger.MetricsLogger(metrics_logging_options)

    self.progress_bar = progress_bar.ProgressBar(
        metrics_logger=self.metrics_logger,
        initial_steps=0,
        max_steps=2,
        metrics_prefix=self.metrics_prefix,
    )

  def test_initial_state(self):
    self.assertDictEqual(self.progress_bar.metrics, {})

  def test_update_metric(self):
    self.metrics_logger.log(
        self.metrics_prefix, "loss", np.array(0.5), metrics_logger.Mode.TRAIN, 1
    )
    self.metrics_logger.log(
        self.metrics_prefix, "loss", np.array(0.6), metrics_logger.Mode.EVAL, 1
    )

    self.progress_bar._update_metric("loss", metrics_logger.Mode.TRAIN)
    self.assertDictEqual(self.progress_bar.metrics, {"test_train_loss": 0.5})
    self.progress_bar._update_metric("loss", metrics_logger.Mode.EVAL)
    self.assertDictEqual(
        self.progress_bar.metrics,
        {"test_train_loss": 0.5, "test_eval_loss": 0.6},
    )

  def test_update_metrics(self):
    # update `metrics_logger`.
    self.metrics_logger.log(
        self.metrics_prefix, "loss", np.array(0.8), metrics_logger.Mode.TRAIN, 2
    )
    self.metrics_logger.log(
        self.metrics_prefix, "loss", np.array(0.9), metrics_logger.Mode.EVAL, 2
    )
    self.metrics_logger.log(
        self.metrics_prefix,
        "perplexity",
        np.array(2.2255),
        metrics_logger.Mode.TRAIN,
        2,
    )
    self.metrics_logger.log(
        self.metrics_prefix,
        "perplexity",
        np.array(2.4596),
        metrics_logger.Mode.EVAL,
        2,
    )

    self.progress_bar.update_metrics(
        ["loss", "perplexity"], metrics_logger.Mode.TRAIN
    )
    exp_output = {"test_train_loss": 0.8, "test_train_perplexity": 2.225}
    self.assertDictEqual(self.progress_bar.metrics, exp_output)

    self.progress_bar.update_metrics(
        ["loss", "perplexity"], metrics_logger.Mode.EVAL
    )
    exp_output.update({"test_eval_loss": 0.9, "test_eval_perplexity": 2.46})
    self.assertDictEqual(self.progress_bar.metrics, exp_output)

  def test_close(self):
    self.assertDictEqual(self.progress_bar.metrics, {})


if __name__ == "__main__":
  absltest.main()
