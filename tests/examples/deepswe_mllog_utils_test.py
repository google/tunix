# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for examples.deepswe.mllog_utils."""

from __future__ import annotations

import os
import shutil
import tempfile
import types

from absl.testing import absltest
from examples.deepswe import mllog_utils


class MllogUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)
    super().tearDown()

  def test_file_logging_with_metric_logger_dir(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=8,
        max_steps=5,
        learning_rate=1e-6,
        b1=0.9,
        b2=0.99,
        weight_decay=0.01,
        max_grad_norm=1.0,
        train_micro_batch_size=1,
        max_prompt_length=4096,
        max_response_length=8192,
        tpu_topology="v5p-64",
        rollout_engine="vllm",
        target_accuracy=0.69,
    )

    mllog_utils.init_start(args)
    mllog_utils.init_print(args, total_devices=64)
    mllog_utils.init_stop()
    mllog_utils.run_start()
    mllog_utils.block_start(args, step=0)
    mllog_utils.log_tracked_stats(
        {"loss": 0.5, "reward": 0.8}, step=1, samples_count=64
    )
    mllog_utils.block_stop(step=1, samples_count=64)
    mllog_utils.run_stop(status="success", samples_count=320)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    self.assertTrue(os.path.exists(expected_log_file))

    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn(":::MLLOG", content)
    self.assertIn('"key": "cache_clear"', content)
    self.assertIn('"key": "init_start"', content)
    self.assertIn('"key": "submission_benchmark"', content)
    self.assertIn('"key": "submission_org"', content)
    self.assertIn('"key": "seed"', content)
    self.assertIn('"key": "run_start"', content)
    self.assertIn('"key": "block_start"', content)
    self.assertIn('"key": "tracked_stats"', content)
    self.assertIn('"key": "block_stop"', content)
    self.assertIn('"key": "run_stop"', content)

  def test_start_and_end_eval(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
    )
    mllog_utils.init_start(args)
    mllog_utils.start_eval(step=2, samples_count=128)
    mllog_utils.end_eval(
        step=2,
        accuracy=0.75,
        samples_count=128,
        validation_time=12.5,
    )

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    self.assertTrue(os.path.exists(expected_log_file))

    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "eval_start"', content)
    self.assertIn('"key": "eval_accuracy"', content)
    self.assertIn("0.75", content)
    self.assertIn('"validation_time": 12.5', content)
    self.assertIn('"key": "eval_stop"', content)

  def test_check_eval_not_early_stop(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=8,
        eval_every_n_steps=2,
        target_accuracy=0.69,
    )
    mllog_utils.init_start(args)

    is_early_stop = mllog_utils.check_eval(
        args,
        step=2,
        eval_accuracy=0.50,
        validation_time=10.0,
    )
    self.assertFalse(is_early_stop)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "block_stop"', content)
    self.assertIn('"key": "eval_start"', content)
    self.assertIn('"validation_time": 10.0', content)
    self.assertIn('"key": "eval_accuracy"', content)
    self.assertIn("0.5", content)
    self.assertIn('"key": "eval_stop"', content)
    self.assertIn('"key": "block_start"', content)

  def test_check_eval_early_stop(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=8,
        eval_every_n_steps=2,
        target_accuracy=0.69,
    )
    mllog_utils.init_start(args)

    is_early_stop = mllog_utils.check_eval(
        args,
        step=4,
        eval_accuracy=0.72,
        validation_time=11.5,
    )
    self.assertTrue(is_early_stop)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "block_stop"', content)
    self.assertIn('"key": "eval_start"', content)
    self.assertIn('"validation_time": 11.5', content)
    self.assertIn('"key": "eval_accuracy"', content)
    self.assertIn("0.72", content)
    self.assertIn('"key": "eval_stop"', content)
    self.assertIn('"key": "run_stop"', content)
    self.assertIn('"status": "success"', content)
    self.assertIn('"key": "train_samples"', content)


if __name__ == "__main__":
  absltest.main()
