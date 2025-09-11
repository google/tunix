# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics logger unittest."""

import os
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.sft import metrics_logger


class MetricLoggerTest(absltest.TestCase):

  @mock.patch("tunix.sft.metrics_logger.datetime")
  @mock.patch("tunix.sft.metrics_logger.wandb")
  def test_metrics_logger(self, mock_wandb, mock_datetime):
    fixed_timestamp_str = "2025-07-17_13-56-53"
    mock_datetime.datetime.now.return_value.strftime.return_value = (
        fixed_timestamp_str
    )
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

    self.assertLen(os.listdir(log_dir), 1)
    file_size_after = os.path.getsize(
        os.path.join(log_dir, os.listdir(log_dir)[0])
    )

    self.assertGreater(file_size_after, file_size_before)

    mock_wandb.init.assert_called_once_with(
        project="tunix", name=fixed_timestamp_str, anonymous="allow"
    )
    self.assertEqual(mock_wandb.log.call_count, 6)

    logger.close()
    mock_wandb.finish.assert_called_once()


if __name__ == "__main__":
  absltest.main()
