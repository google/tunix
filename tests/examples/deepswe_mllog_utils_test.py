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

"""Tests for tunix.utils.mllog_utils."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import types
from typing import Any
from unittest import mock

import numpy as np
from absl.testing import absltest
from tunix.perf.metrics import MetricsBuffer
from tunix.utils import mllog_utils


def _read_mllog_events(path: str) -> list[dict[str, Any]]:
  with open(path, "r", encoding="utf-8") as f:
    return [
        json.loads(line.split(":::MLLOG ", 1)[1])
        for line in f
        if ":::MLLOG " in line
    ]


@absltest.skipIf(mllog_utils.mllogger is None, "mlperf_logging is not installed")
class MllogUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    if mllog_utils.mllogger is not None:
      for h in list(getattr(mllog_utils.mllogger.logger, "handlers", [])):
        if isinstance(h, logging.FileHandler):
          h.close()
          mllog_utils.mllogger.logger.removeHandler(h)
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
        model_id="",
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
        model_id="",
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

  def test_log_metrics_buffer_with_perf_metrics_buffer(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=4,
        num_generations=2,
    )
    mllog_utils.init_start(args)

    metrics_buffer = MetricsBuffer(
        global_steps=0,
        metrics={
            "loss": ([0.42], None),
            "train_reward": ([0.75, 0.85], np.mean),
            "perf/global_step_time": ([12.5], np.mean),
        },
        mode="train",
    )

    mllog_utils.log_metrics_buffer(metrics_buffer, args=args)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    self.assertTrue(os.path.exists(expected_log_file))

    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "tracked_stats"', content)
    self.assertIn('"loss": 0.42', content)
    self.assertIn('"train_reward": 0.8', content)
    self.assertIn('"step_time": 12.5', content)
    self.assertNotIn('"perf/global_step_time":', content)
    self.assertIn('"step": 1', content)
    self.assertIn('"samples_count": 8', content)

  def test_create_rcp_metrics_logger(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=4,
    )
    mllog_utils.init_start(args)

    logger = mllog_utils.create_rcp_metrics_logger(args)

    metrics_buffer = MetricsBuffer(
        global_steps=2,
        metrics={
            "loss": ([0.15], None),
            "train_reward": ([0.9], np.mean),
        },
        mode="train",
    )

    logger(metrics_buffer)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "tracked_stats"', content)
    self.assertIn('"loss": 0.15', content)
    self.assertIn('"train_reward": 0.9', content)
    self.assertIn('"step": 3', content)
    self.assertIn('"samples_count": 96', content)

  def test_log_metrics_buffer_with_weighted_metric(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=2,
        num_generations=2,
    )
    mllog_utils.init_start(args)

    class DummyWeightedMetric:
      def __init__(self, val):
        self.val = val
      def compute(self):
        return self.val

    metrics_buffer = MetricsBuffer(
        global_steps=0,
        metrics={
            "loss": ([DummyWeightedMetric(0.5)], None),
        },
        mode="train",
    )

    mllog_utils.log_metrics_buffer(metrics_buffer, args=args)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "tracked_stats"', content)
    self.assertIn('"loss": 0.5', content)

  def test_log_metrics_buffer_with_full_tracked_stats(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=16,
    )
    mllog_utils.init_start(args)

    metrics_buffer = MetricsBuffer(
        global_steps=0,
        metrics={
            "step_time": ([1086.7], np.mean),
            "train_reward": ([0.0], np.mean),
            "train_solve": ([0.0], np.mean),
            "adv_abs_mean": ([0.0], np.mean),
            "completion_length": ([4096.0], np.mean),
            "loss": ([-7.5e-06], np.mean),
            "grad_norm": ([0.0858], np.mean),
            "reduced_pg_loss": ([0.0], np.mean),
            "entropy": ([1.815], np.mean),
            "kl": ([-0.0075], np.mean),
            "log_ratio_abs": ([0.521], np.mean),
            "clipfrac": ([0.0], np.mean),
            "generation/prompts/mean_length": ([8192.0], np.mean),
            "trajectory/env_time/step_latency/mean": ([0.0], np.mean),
            "rewards/sum": ([0.0], np.mean),
            "perf/global_step_time": ([1086.7], np.mean),
        },
        mode="train",
    )

    mllog_utils.log_metrics_buffer(metrics_buffer, args=args)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "tracked_stats"', content)
    for expected_key in [
        "step_time",
        "train_reward",
        "train_solve",
        "adv_abs_mean",
        "completion_length",
        "loss",
        "grad_norm",
        "reduced_pg_loss",
        "entropy",
        "kl",
        "log_ratio_abs",
        "clipfrac",
    ]:
      self.assertIn(f'"{expected_key}":', content)

    for excluded_key in [
        "generation/prompts/mean_length",
        "trajectory/env_time/step_latency/mean",
        "rewards/sum",
        "perf/global_step_time",
    ]:
      self.assertNotIn(f'"{excluded_key}":', content)

  def test_create_rcp_metrics_logger_with_rl_engine(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=16,
    )
    mllog_utils.init_start(args)

    # Rollout metrics in metrics_buffer (only original logs, no 12 duplicate logs)
    metrics_buffer = MetricsBuffer(
        global_steps=0,
        metrics={
            "perf/global_step_time": ([1086.7], np.mean),
            "generation/completions/mean_length": ([4096.0], np.mean),
            "trajectory_rewards/mean": ([0.75], np.mean),
        },
        mode="train",
    )

    # Actor trainer metrics in rl_engine.actor_trainer
    mock_trainer_buf = types.SimpleNamespace(
        loss=-7.5e-06,
        additional_metrics={
            "grad_norm": ([0.0858], np.mean),
            "reduced_pg_loss": ([0.0], np.mean),
            "entropy": ([1.815], np.mean),
            "kl": ([-0.0075], np.mean),
            "log_ratio/abs_mean": ([0.521], np.mean),
            "pg_clipfrac": ([0.0], np.mean),
            "advantage/abs_mean": ([0.25], np.mean),
        },
    )
    mock_actor_trainer = types.SimpleNamespace(
        _prev_buffered_train_metrics=mock_trainer_buf,
    )
    mock_rl_engine = types.SimpleNamespace(
        actor_trainer=mock_actor_trainer,
    )

    logger = mllog_utils.create_rcp_metrics_logger(
        args=args, rl_engine=mock_rl_engine
    )
    logger(metrics_buffer)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    self.assertTrue(os.path.exists(expected_log_file))

    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "tracked_stats"', content)
    for expected_key in [
        "step_time",
        "train_reward",
        "train_solve",
        "adv_abs_mean",
        "completion_length",
        "loss",
        "grad_norm",
        "reduced_pg_loss",
        "entropy",
        "kl",
        "log_ratio_abs",
        "clipfrac",
    ]:
      self.assertIn(f'"{expected_key}":', content)

    self.assertIn('"step_time": 1086.7', content)
    self.assertIn('"completion_length": 4096.0', content)
    self.assertIn('"train_reward": 0.75', content)
    self.assertIn('"train_solve": 0.75', content)
    self.assertIn('"adv_abs_mean": 0.25', content)
    self.assertIn('"loss": -7.5e-06', content)
    self.assertIn('"grad_norm": 0.0858', content)
    self.assertIn('"entropy": 1.815', content)
    self.assertIn('"kl": -0.0075', content)
    self.assertIn('"log_ratio_abs": 0.521', content)
    self.assertIn('"clipfrac": 0.0', content)

  def test_train_start(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=8,
        max_steps=5,
    )
    mllog_utils.init_start(args)
    mllog_utils.train_start(args)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    self.assertTrue(os.path.exists(expected_log_file))

    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "init_stop"', content)
    self.assertIn('"key": "run_start"', content)
    self.assertIn('"key": "block_start"', content)

  def test_train_stop(self):
    args = types.SimpleNamespace(
        seed=1,
        metric_logger_dir=self.test_dir,
        batch_size=8,
        num_generations=8,
        max_steps=5,
    )
    mllog_utils.init_start(args)
    mllog_utils.train_start(args)
    mllog_utils.train_stop(args)

    expected_log_file = os.path.join(self.test_dir, "seed_1.out")
    self.assertTrue(os.path.exists(expected_log_file))

    with open(expected_log_file, "r") as f:
      content = f.read()

    self.assertIn('"key": "block_stop"', content)
    # run_stop is emitted by the offline evaluator, not by train_stop.
    self.assertNotIn('"key": "run_stop"', content)

  def test_compute_val_start_step(self):
    self.assertEqual(mllog_utils.compute_val_start_step(256), 18)
    self.assertEqual(mllog_utils.compute_val_start_step(512), 10)
    self.assertEqual(mllog_utils.compute_val_start_step(1024), 7)
    self.assertEqual(mllog_utils.compute_val_start_step(256, 1), 1)
    self.assertEqual(mllog_utils.compute_val_start_step(256, 0), 18)
    with self.assertRaises(ValueError):
      mllog_utils.compute_val_start_step(0)

  def test_append_checkpoint_manifest_upserts_sorted_records(self):
    manifest_path = os.path.join(
        self.test_dir, "mllog", "eval_checkpoints.jsonl"
    )
    for step, ts_ms in ((19, 2000), (18, 1000), (18, 1500)):
      mllog_utils.append_checkpoint_manifest(
          manifest_path,
          {
              "step": step,
              "checkpoint_path": f"gs://ckpt/{step}/model_params",
              "timestamp_ms": ts_ms,
              "samples_count": step * 256,
              "val_start_at": 18,
          },
      )
    with open(manifest_path, "r", encoding="utf-8") as f:
      records = [json.loads(line) for line in f]
    self.assertEqual([r["step"] for r in records], [18, 19])
    self.assertEqual([r["timestamp_ms"] for r in records], [1500, 2000])

  def test_configure_logger_downloads_existing_gcs_log_before_config(self):
    calls = mock.MagicMock()
    fake_mllogger = mock.MagicMock()
    fake_mllogger.logger.handlers = []
    with (
        mock.patch.object(mllog_utils, "mllog", calls.mllog),
        mock.patch.object(mllog_utils, "mllogger", fake_mllogger),
        mock.patch.object(mllog_utils, "_is_master_process", return_value=True),
        mock.patch.object(mllog_utils, "_gcs_target_path", None),
        mock.patch.object(mllog_utils, "_local_log_path", None),
        mock.patch.object(
            mllog_utils, "_download_from_gcs_if_exists", calls.download
        ),
    ):
      mllog_utils.configure_logger(
          metric_logger_dir="gs://b/mllog", seed=42, append=False
      )
      calls.download.assert_not_called()
      calls.reset_mock()

      mllog_utils.configure_logger(
          metric_logger_dir="gs://b/mllog", seed=42, append=True
      )
      local_path = mllog_utils._local_log_path  # pylint: disable=protected-access

    self.assertEqual(
        [c[0] for c in calls.mock_calls], ["download", "mllog.config"]
    )
    calls.download.assert_called_once_with("gs://b/mllog/seed_42.out", local_path)
    self.assertEqual(
        os.path.abspath(calls.mllog.config.call_args.kwargs["filename"]),
        local_path,
    )

  def test_offline_eval_rcp_sequence_converged(self):
    log_dir = self.test_dir
    args = types.SimpleNamespace(batch_size=16, num_generations=16)
    mllog_utils.configure_logger(metric_logger_dir=log_dir, seed=42)
    mllog_utils.train_stop(args, step=19, time_ms=2000)
    self.assertFalse(
        mllog_utils.log_offline_eval_step(
            step=18,
            samples_count=4608,
            eval_accuracy=0.65,
            checkpoint_timestamp_ms=1000,
        )
    )
    self.assertTrue(
        mllog_utils.log_offline_eval_step(
            step=19,
            samples_count=4864,
            eval_accuracy=0.70,
            checkpoint_timestamp_ms=2000,
            is_last_checkpoint=True,
        )
    )

    events = _read_mllog_events(os.path.join(log_dir, "seed_42.out"))
    self.assertEqual(
        [e["key"] for e in events],
        ["block_stop"] + ["eval_start", "eval_accuracy", "eval_stop"] * 2
        + ["run_stop"],
    )
    self.assertEqual(events[0]["time_ms"], 2000)
    self.assertEqual(events[0]["metadata"]["step"], 19)
    self.assertEqual([events[2]["value"], events[5]["value"]], [0.65, 0.70])
    run_stop = events[-1]
    self.assertEqual(run_stop["time_ms"], 2000)
    self.assertEqual(run_stop["metadata"]["status"], "success")
    self.assertEqual(run_stop["metadata"]["samples_count"], 4864)

  def test_offline_eval_rcp_sequence_not_converged(self):
    log_dir = self.test_dir
    mllog_utils.configure_logger(metric_logger_dir=log_dir, seed=42)
    for step, ts_ms in ((18, 1000), (19, 2000), (20, 3000)):
      self.assertFalse(
          mllog_utils.log_offline_eval_step(
              step=step,
              samples_count=step * 256,
              eval_accuracy=0.5,
              checkpoint_timestamp_ms=ts_ms,
              is_last_checkpoint=step == 20,
          )
      )

    events = _read_mllog_events(os.path.join(log_dir, "seed_42.out"))
    run_stops = [e for e in events if e["key"] == "run_stop"]
    self.assertLen(run_stops, 1)
    self.assertEqual(events[-1]["key"], "run_stop")
    self.assertEqual(run_stops[0]["time_ms"], 3000)
    self.assertEqual(run_stops[0]["metadata"]["status"], "aborted")
    self.assertEqual(run_stops[0]["metadata"]["samples_count"], 5120)

  def test_mlperf_6_1_0_init_print_disclosures(self):
    fake_mllogger = mock.MagicMock()
    with (
        mock.patch.object(mllog_utils, "mllogger", fake_mllogger),
        mock.patch.object(mllog_utils, "_is_master_process", return_value=True),
        mock.patch.object(mllog_utils, "_flush_to_gcs_if_needed"),
    ):
      args = types.SimpleNamespace(
          batch_size=16,
          num_generations=16,
          max_prompt_length=4096,
          max_response_length=61440,
          model_id="",
      )
      mllog_utils.init_print(args)
    emitted = {
        c.kwargs["key"]: c.kwargs["value"]
        for c in fake_mllogger.event.call_args_list
    }
    self.assertEqual(emitted["eval_samples"], 251)
    self.assertEqual(emitted["max_sequence_length"], 65536)
    for key in (
        "lowest_numerical_precision_in_linear",
        "lowest_numerical_precision_in_attn",
        "lowest_numerical_precision_in_comm",
    ):
      self.assertEqual(emitted[key], "bfloat16")
    self.assertEqual(emitted["config_filename"], "qwen35_397b_grpo")

  def _finish_training_args(self):
    return types.SimpleNamespace(
        seed=42,
        metric_logger_dir=self.test_dir,
        batch_size=16,
        num_generations=16,
        max_steps=30,
    )

  def test_finish_training_noop_before_train_start(self):
    args = self._finish_training_args()
    mllog_utils.init_start(args)
    mllog_utils.finish_training(args, status="aborted", completed_steps=3)

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    self.assertNotIn("block_stop", [e["key"] for e in events])

  def test_finish_training_uses_checkpoint_ahead_of_step_result(self):
    args = self._finish_training_args()
    mllog_utils.init_start(args)
    mllog_utils.train_start(args, step=0)
    manifest = os.path.join(self.test_dir, "eval_checkpoints.jsonl")
    for step, ts_ms in ((21, 2100), (22, 2200)):
      mllog_utils.append_checkpoint_manifest(
          manifest, {"step": step, "timestamp_ms": ts_ms}
      )
    # Interrupted during step 22's weight sync: the trainer reports 21 steps.
    mllog_utils.finish_training(
        args, status="aborted", completed_steps=21, last_step_time_ms=None
    )
    mllog_utils.finish_training(args, status="aborted", completed_steps=21)

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    block_stops = [e for e in events if e["key"] == "block_stop"]
    self.assertLen(block_stops, 1)
    self.assertEqual(block_stops[0]["metadata"]["step"], 22)
    self.assertEqual(block_stops[0]["metadata"]["samples_count"], 22 * 256)
    self.assertEqual(block_stops[0]["time_ms"], 2200)

  def test_finish_training_success_prefers_step_result(self):
    args = self._finish_training_args()
    mllog_utils.init_start(args)
    mllog_utils.train_start(args, step=0)
    mllog_utils.finish_training(
        args, status="success", completed_steps=30, last_step_time_ms=3000
    )

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    block_stop = next(e for e in events if e["key"] == "block_stop")
    self.assertEqual(block_stop["metadata"]["step"], 30)
    self.assertEqual(block_stop["time_ms"], 3000)

  def test_train_stop_is_idempotent(self):
    args = types.SimpleNamespace(
        seed=42,
        metric_logger_dir=self.test_dir,
        batch_size=16,
        num_generations=16,
        max_steps=30,
    )
    mllog_utils.init_start(args)
    mllog_utils.train_start(args, step=0)
    mllog_utils.train_stop(args, step=22, time_ms=1000)
    mllog_utils.train_stop(args, step=22, time_ms=2000)

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    block_stops = [e for e in events if e["key"] == "block_stop"]
    self.assertLen(block_stops, 1)
    self.assertEqual(block_stops[0]["metadata"]["step"], 22)
    self.assertEqual(block_stops[0]["metadata"]["samples_count"], 5632)
    self.assertEqual(block_stops[0]["time_ms"], 1000)

  def _simulate_new_process(self):
    """Drops mllog file handlers, as a freshly started process would have."""
    for h in list(getattr(mllog_utils.mllogger.logger, "handlers", [])):
      if isinstance(h, logging.FileHandler):
        h.close()
        mllog_utils.mllogger.logger.removeHandler(h)

  def test_training_restart_overwrites_stale_aborted_run_log(self):
    args = types.SimpleNamespace(
        seed=42,
        metric_logger_dir=self.test_dir,
        batch_size=16,
        num_generations=16,
        max_steps=30,
        model_id="",
    )
    # Run 1 is aborted after training started.
    mllog_utils.init_start(args)
    mllog_utils.init_print(args)
    mllog_utils.train_start(args, step=0)
    mllog_utils.train_stop(args, step=0, time_ms=500)

    # Run 2 (a new process) restarts with the same metric_logger_dir and seed.
    self._simulate_new_process()
    mllog_utils.init_start(args)
    mllog_utils.init_print(args)
    mllog_utils.train_start(args, step=0)
    mllog_utils.train_stop(args, step=8, time_ms=1500)

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    self.assertEqual(events[0]["key"], "cache_clear")
    for key in ("init_start", "submission_org", "init_stop", "run_start"):
      self.assertLen([e for e in events if e["key"] == key], 1, key)
    block_stops = [e for e in events if e["key"] == "block_stop"]
    self.assertLen(block_stops, 1)
    self.assertEqual(block_stops[0]["metadata"]["step"], 8)

  def test_offline_eval_auto_closes_unclosed_training_block(self):
    args = types.SimpleNamespace(
        seed=42,
        metric_logger_dir=self.test_dir,
        batch_size=16,
        num_generations=16,
        max_steps=30,
    )
    mllog_utils.init_start(args)
    mllog_utils.train_start(args, step=0)
    mllog_utils.log_tracked_stats(
        {"reduced_train_loss": 0.1, "reward": 0.5},
        step=21,
        samples_count=5376,
    )

    # Training is killed before block_stop; a separate offline evaluation
    # process re-opens the existing log.
    self._simulate_new_process()
    mllog_utils.configure_logger(
        metric_logger_dir=self.test_dir, seed=42, append=True
    )
    mllog_utils.start_eval(step=18, samples_count=4608, time_ms=5000)
    self.assertTrue(
        mllog_utils.log_offline_eval_step(
            step=18,
            samples_count=4608,
            eval_accuracy=0.70,
            checkpoint_timestamp_ms=4000,
            is_last_checkpoint=True,
            emit_start_eval=False,
        )
    )

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    keys = [e["key"] for e in events]
    self.assertEqual(
        keys,
        [
            "cache_clear",
            "init_start",
            "init_stop",
            "run_start",
            "block_start",
            "tracked_stats",
            "block_stop",
            "eval_start",
            "eval_accuracy",
            "eval_stop",
            "run_stop",
        ],
    )
    tracked_stats = events[5]
    block_stop = events[6]
    self.assertEqual(block_stop["metadata"]["step"], 21)
    self.assertEqual(block_stop["metadata"]["samples_count"], 5376)
    self.assertEqual(block_stop["time_ms"], tracked_stats["time_ms"])

  def test_offline_eval_auto_close_in_same_process(self):
    args = types.SimpleNamespace(
        seed=42,
        metric_logger_dir=self.test_dir,
        batch_size=16,
        num_generations=16,
        max_steps=30,
    )
    mllog_utils.init_start(args)
    mllog_utils.train_start(args, step=0)
    mllog_utils.log_tracked_stats({"reward": 0.5}, step=3, samples_count=768)
    mllog_utils.start_eval(step=3, samples_count=768, time_ms=5000)

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    self.assertEqual(
        [e["key"] for e in events][-3:],
        ["tracked_stats", "block_stop", "eval_start"],
    )
    self.assertEqual(events[-2]["metadata"]["samples_count"], 768)
    self.assertEqual(events[-2]["time_ms"], events[-3]["time_ms"])

  def test_auto_close_without_tracked_stats_ignores_block_length(self):
    args = types.SimpleNamespace(
        seed=42,
        metric_logger_dir=self.test_dir,
        batch_size=16,
        num_generations=16,
        max_steps=30,
    )
    mllog_utils.init_start(args)
    # block_start's samples_count is the block length (30 * 256 = 7680).
    mllog_utils.train_start(args, step=0)

    self._simulate_new_process()
    mllog_utils.configure_logger(
        metric_logger_dir=self.test_dir, seed=42, append=True
    )
    mllog_utils.start_eval(step=18, samples_count=4608, time_ms=5000)

    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    block_start = next(e for e in events if e["key"] == "block_start")
    block_stop = next(e for e in events if e["key"] == "block_stop")
    self.assertEqual(block_start["metadata"]["samples_count"], 7680)
    self.assertEqual(block_stop["metadata"]["step"], 0)
    self.assertEqual(block_stop["metadata"]["samples_count"], 4608)
    self.assertEqual(block_stop["time_ms"], 5000)

  def test_offline_eval_tolerates_malformed_log_line(self):
    args = types.SimpleNamespace(
        seed=42,
        metric_logger_dir=self.test_dir,
        batch_size=16,
        num_generations=16,
        max_steps=30,
    )
    mllog_utils.init_start(args)
    mllog_utils.train_start(args, step=0)
    mllog_utils.log_tracked_stats({"reward": 0.5}, step=5, samples_count=1280)
    self._simulate_new_process()
    log_path = os.path.join(self.test_dir, "seed_42.out")
    with open(log_path, "a", encoding="utf-8") as f:
      # Valid JSON that is not an event dict, or has non-dict metadata.
      f.write(":::MLLOG null\n")
      f.write(":::MLLOG []\n")
      f.write(':::MLLOG {"key": "tracked_stats", "metadata": [1]}\n')
      # A truncated trailing line from an interrupted upload.
      f.write(':::MLLOG {"key": "tracked_st\n')

    mllog_utils.configure_logger(
        metric_logger_dir=self.test_dir, seed=42, append=True
    )
    mllog_utils.start_eval(step=5, samples_count=1280, time_ms=5000)

    with open(log_path, "r", encoding="utf-8") as f:
      lines = f.read().splitlines()
    block_stop = json.loads(
        next(l for l in lines if '"block_stop"' in l).split(":::MLLOG ", 1)[1]
    )
    self.assertEqual(block_stop["metadata"]["step"], 5)
    self.assertEqual(block_stop["metadata"]["samples_count"], 1280)

  def test_append_checkpoint_manifest_tolerates_gcs_io_error(self):
    args = self._finish_training_args()
    mllog_utils.init_start(args)
    mllog_utils.train_start(args, step=0)
    manifest = "gs://forbidden-bucket/mllog/eval_checkpoints.jsonl"
    fake_fs = mock.MagicMock()
    fake_fs.exists.side_effect = OSError("Forbidden: scope not authorized")
    fake_fs.open.side_effect = OSError("Forbidden: scope not authorized")
    with (
        mock.patch("fsspec.filesystem", return_value=fake_fs),
        self.assertLogs(level="WARNING") as cm,
    ):
      mllog_utils.append_checkpoint_manifest(
          manifest, {"step": 22, "timestamp_ms": 2200}
      )
    self.assertTrue(
        any("Failed to write checkpoint manifest" in msg for msg in cm.output)
    )
    # In-memory checkpoint progress is still recorded for finish_training.
    mllog_utils.finish_training(
        args, status="aborted", completed_steps=21, last_step_time_ms=None
    )
    events = _read_mllog_events(os.path.join(self.test_dir, "seed_42.out"))
    block_stops = [e for e in events if e["key"] == "block_stop"]
    self.assertLen(block_stops, 1)
    self.assertEqual(block_stops[0]["metadata"]["step"], 22)
    self.assertEqual(block_stops[0]["time_ms"], 2200)

  def test_download_from_gcs_tolerates_io_error(self):
    fake_fs = mock.MagicMock()
    fake_fs.exists.side_effect = OSError("Forbidden")
    with (
        mock.patch("fsspec.filesystem", return_value=fake_fs),
        self.assertLogs(level="WARNING") as cm,
    ):
      mllog_utils._download_from_gcs_if_exists(  # pylint: disable=protected-access
          "gs://forbidden-bucket/mllog/seed_1.out",
          os.path.join(self.test_dir, "seed_1.out"),
      )
    self.assertTrue(
        any("Failed to download mllog file" in msg for msg in cm.output)
    )

  def test_rcp_logging_functions_do_not_raise_on_mllogger_failure(self):
    broken_mllogger = mock.MagicMock()
    broken_mllogger.logger.handlers = []
    broken_mllogger.event.side_effect = OSError("disk full")
    broken_mllogger.start.side_effect = OSError("disk full")
    broken_mllogger.end.side_effect = OSError("disk full")
    args = self._finish_training_args()
    with (
        mock.patch.object(mllog_utils, "mllogger", broken_mllogger),
        mock.patch.object(mllog_utils, "_is_master_process", return_value=True),
    ):
      mllog_utils.init_start(args)
      mllog_utils.init_print(args)
      mllog_utils.train_start(args, step=0)
      mllog_utils.log_rcp_step_stats(
          {"loss": 0.5, "reward": 1.0, "train_step_time": 10.0},
          args=args,
          step=1,
      )
      self.assertTrue(
          mllog_utils.check_eval(args, step=1, eval_accuracy=0.75)
      )
      self.assertTrue(
          mllog_utils.log_offline_eval_step(
              step=1,
              samples_count=256,
              eval_accuracy=0.75,
              target_accuracy=0.69,
          )
      )
      mllog_utils.train_stop(args, step=1)
      mllog_utils.run_stop(status="success", samples_count=256)


if __name__ == "__main__":
  absltest.main()
