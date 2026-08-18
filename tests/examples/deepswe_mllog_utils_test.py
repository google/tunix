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

import json
import os
import shutil
import tempfile
import types
from unittest import mock

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

  def test_end_to_end_mlperf_logging_with_train_configs(self):
    args = types.SimpleNamespace(
        model_version="Qwen3.5-35B-A3B",
        model_source="maxtext",
        model_absolute_path="gs://sanbao-europe/qwen3.5-35b-a3b/scanned/0/items",
        scan_layers=True,
        vllm_utilization=0.4,
        max_response_length=4096,
        max_prompt_length=8192,
        metric_logger_dir=self.test_dir,
        ckpt_dir="none",
        rollout_micro_batch_size=8,
        vllm_reshard_chunk_size=1,
        rollout_mesh_fsdp=8,
        rollout_mesh_tp=4,
        train_mesh_fsdp=16,
        train_mesh_tp=2,
        node_selector_val="cpu-np",
        max_steps=5,
        overlong_filter=False,
        batch_size=64,
        mini_batch_size=64,
        compute_logps_micro_batch_size=16,
        train_micro_batch_size=16,
        temperature=0.7,
        num_generations=2,
        max_turns=20,
        beta=0.001,
        weight_decay=0.1,
        max_grad_norm=0.1,
        logging_level="INFO",
        rcp_logging=True,
        target_accuracy=0.69,
        seed=1,
        learning_rate=1e-6,
        eval_every_n_steps=5,
    )

    mock_train_dataset = [None] * 5480
    rollout_mesh = mock.MagicMock()
    rollout_mesh.shape = {"fsdp": 8, "tp": 4}
    train_mesh = mock.MagicMock()
    train_mesh.shape = {"fsdp": 16, "tp": 2}
    total_devices = 32

    # 1. Initialization phase
    mllog_utils.init_start(args)
    mllog_utils.init_print(
        args,
        train_dataset=mock_train_dataset,
        rollout_mesh=rollout_mesh,
        train_mesh=train_mesh,
        total_devices=total_devices,
    )
    mllog_utils.init_stop()

    # 2. Run and block start
    mllog_utils.run_start()
    mllog_utils.block_start(args, step=0)

    # 3. Training steps with tracked stats
    for step in range(1, args.max_steps + 1):
      samples_count = step * args.batch_size * args.num_generations
      mllog_utils.log_tracked_stats(
          stats={
              "reduced_train_loss": -0.08 + step * 0.01,
              "reward": 0.33 + step * 0.02,
              "grad_norm": 0.04,
              "train_step_time": 100.0,
              "policy_training_time": 25.0,
              "exposed_generation_time": 70.0,
              "weight_sync_time": 2.0,
              "valid_tokens_per_sec_per_gpu": 22.0,
          },
          step=step,
          samples_count=samples_count,
      )

    # 4. Mock evaluation phase with eval_accuracy = 0.70703125
    is_early_stop = mllog_utils.check_eval(
        args,
        step=args.max_steps,
        eval_accuracy=0.70703125,
        validation_time=986.35,
    )
    self.assertTrue(is_early_stop)

    # 5. Validate the generated log file
    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    self.assertTrue(os.path.exists(expected_log_file))

    with open(expected_log_file, "r") as f:
      log_lines = [line.strip() for line in f if line.strip()]

    events = []
    for line in log_lines:
      self.assertTrue(line.startswith(":::MLLOG "))
      json_str = line[len(":::MLLOG ") :]
      events.append(json.loads(json_str))

    event_map = {e["key"]: e for e in events}

    # Verify lifecycle events
    self.assertEqual(event_map["cache_clear"]["value"], True)
    self.assertIn("init_start", event_map)
    self.assertIn("init_stop", event_map)
    self.assertIn("run_start", event_map)
    self.assertIn("block_start", event_map)
    self.assertIn("block_stop", event_map)
    self.assertIn("eval_start", event_map)
    self.assertIn("eval_stop", event_map)
    self.assertIn("run_stop", event_map)
    self.assertEqual(event_map["run_stop"]["metadata"]["status"], "success")

    # Verify hyperparameters and metadata
    self.assertEqual(event_map["submission_benchmark"]["value"], "qwen35_397b_grpo")
    self.assertEqual(event_map["submission_org"]["value"], "Google")
    self.assertEqual(event_map["submission_division"]["value"], "closed")
    self.assertEqual(event_map["submission_status"]["value"], "cloud")
    self.assertEqual(event_map["seed"]["value"], 1)
    self.assertEqual(event_map["max_steps"]["value"], 5)
    self.assertEqual(event_map["global_batch_size"]["value"], 128)
    self.assertEqual(event_map["micro_batch_size"]["value"], 16)
    self.assertEqual(event_map["max_sequence_length"]["value"], 12288)
    self.assertEqual(event_map["train_samples"]["value"], 640)
    self.assertEqual(event_map["tensor_parallelism"]["value"], 2)
    self.assertEqual(event_map["generation_tensor_parallelism"]["value"], 4)
    self.assertEqual(
        event_map["generation_training_rollout_temperature"]["value"], 0.7
    )
    self.assertEqual(event_map["num_prompts_per_step"]["value"], 64)
    self.assertEqual(event_map["num_generations_per_prompt"]["value"], 2)
    self.assertEqual(event_map["target_accuracy"]["value"], 0.69)
    self.assertEqual(event_map["eval_accuracy"]["value"], 0.70703125)


if __name__ == "__main__":
  absltest.main()
