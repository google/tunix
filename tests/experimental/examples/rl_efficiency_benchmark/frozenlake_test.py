# Copyright 2026 Google LLC
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

"""CPU-only benchmark regression tests; runnable without TPU/JAX/pytest.

Run directly:
python3 tests/experimental/examples/rl_efficiency_benchmark/frozenlake_test.py
"""

import argparse
import asyncio
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


def load_module(name):
  path = (
      Path(__file__).resolve().parents[4]
      / "tunix/experimental/examples/rl_efficiency_benchmark"
      / (name + ".py")
  )
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


benchmark = load_module("frozenlake")
runtime = load_module("runtime")


class BenchmarkTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.path = Path(self.temp.name)
    self.w = dict(
        batch=8,
        generations=8,
        steps=3,
        warmup=1,
        micro_groups=2,
        prompt=2048,
        response=2048,
        turns=8,
        seed=42,
        dataset_size=10000,
        concurrency=64,
        max_num_seqs=32,
        batched_tokens=8192,
    )

  def test_native_microbatches_and_shared_agentic_mesh(self):
    config = benchmark.agentic_config(self.w, self.path)
    env = benchmark.dist_environment(self.w, self.path, self.path)
    train = config["rl_training_config"]
    for key in ("train_micro_batch_size", "compute_logps_micro_batch_size"):
      self.assertEqual(int(env[key.upper()]), 2)
      self.assertEqual(train[key], self.w["micro_groups"])
    self.assertEqual(config["model_config"]["mesh"]["shape"], "(4,1)")
    self.assertIsNone(config["rollout_model_config"]["mesh"])
    self.assertEqual(config["rollout_model_config"]["same_mesh_as"], "actor")
    self.assertEqual(config["vllm_config"]["data_parallel_size"], 2)
    self.assertEqual(env["CHECKPOINT_SAVE_INTERVAL_STEPS"], "0")
    self.assertIsNone(train["checkpoint_root_directory"])
    self.assertIsNone(train["metrics_logging_options"])
    self.assertIsNone(train["profiler_options"])
    self.assertIsNone(train["actor_optimizer_config"]["schedule_type"])
    self.assertEqual(env["MODEL_NAME"], "gemma-4-e2b")
    self.assertEqual(env["MODEL_ID"], benchmark.MODEL_ID)
    self.assertEqual(config["model_config"]["model_source"], "huggingface")
    self.assertFalse(config["tokenizer_config"]["add_bos"])
    self.assertFalse(config["tokenizer_config"]["add_eos"])
    self.assertEqual(env["MODEL_DTYPE"], config["model_config"]["dtype"])
    self.assertEqual(
        env["MODEL_LOAD_DTYPE"], config["actor_model_config"]["load_dtype"]
    )
    self.assertEqual(env["MODEL_DTYPE"], "bfloat16")
    self.assertEqual(env["MODEL_LOAD_DTYPE"], "float32")
    self.assertTrue(config["agentic_grpo_config"]["exact_token_continuity"])

  def test_split_agentic_mesh_matches_distributed_chip_allocation(self):
    config = benchmark.agentic_config(self.w, self.path, "split2x2")
    self.assertEqual(config["model_config"]["mesh"]["shape"], "(2,1)")
    self.assertEqual(config["actor_model_config"]["mesh"]["shape"], "(2,1)")
    self.assertEqual(config["reference_model_config"]["same_mesh_as"], "actor")
    self.assertEqual(config["rollout_model_config"]["mesh"]["shape"], "(1,2)")
    self.assertIsNone(config["rollout_model_config"]["same_mesh_as"])
    self.assertEqual(config["vllm_config"]["data_parallel_size"], 1)
    self.assertEqual(config["vllm_config"]["tensor_parallel_size"], 2)
    self.assertEqual(
        config["rl_training_config"]["train_micro_batch_size"],
        self.w["micro_groups"],
    )

  def test_matched_microbatch_uses_trajectories_on_distributed(self):
    env = benchmark.dist_environment(
        self.w, self.path, self.path, match_training_microbatch=True
    )
    expected = self.w["micro_groups"] * self.w["generations"]
    self.assertEqual(int(env["TRAIN_MICRO_BATCH_SIZE"]), expected)
    self.assertEqual(int(env["COMPUTE_LOGPS_MICRO_BATCH_SIZE"]), expected)
    self.assertEqual(self.w["batch"] * self.w["generations"] // expected, 4)

  def test_matched_report_rejects_unmatched_observed_microbatch(self):
    keys = (
        "schema", "timing_mode", "training_contract", "workload",
        "versions", "revision", "model_dir", "diff_sha256",
        "benchmark_sha256", "hardware_label", "hostname", "python",
        "model_metadata_sha256", "model_shards", "runtime_environment",
    )
    manifest = dict.fromkeys(keys)
    manifest.update(
        schema=3,
        workload=self.w,
        agentic_chip_layout="split2x2",
        match_training_microbatch=True,
    )
    runs = [self.path / mode for mode in ("agentic", "dist")]
    for run in runs:
      run.mkdir()
      benchmark.write_json(run / "manifest.json", manifest)
      benchmark.write_json(run / "dataset.json", {"sha256": "same"})

    def summary(size):
      shape = [{"prompt": [size, 2048], "completion": [size, 2048]}]
      input_info = {
          "trajectories_per_call": float(size),
          "unique_tensor_shapes": shape,
      }
      calls = {"calls_per_step": 64 / size}
      return {
          "training_input": input_info,
          "actor_log_probs_input": input_info,
          "api_calls": {"training": calls, "actor_log_probs": calls},
      }

    with mock.patch.object(benchmark, "summarize", side_effect=[summary(16), summary(2)]), mock.patch.object(
        benchmark, "compare_summaries", return_value={"parity_checks": {}}
    ):
      with self.assertRaisesRegex(ValueError, "Microbatch mismatch"):
        benchmark.report(runs)

  def record_run(self):
    recorder = runtime.Recorder(self.path / "events.jsonl", clock=lambda: 0)
    for process in ("dist", "trainer-worker", "rollout-worker"):
      recorder.emit("timing_contract", process=process, timing_mode="stages")
    # Initial sync must not count as a training full batch.
    recorder.emit("device_completion", method="weight_install", time=1)
    recorder.finish("weight_sync", 0, 2)
    for step, (start, end) in enumerate(((2, 12), (12, 32), (32, 62))):
      for trajectory in range(64):
        rollout_start = start + 1 + trajectory / 1000
        rollout_end = end - 2 + trajectory / 1000
        recorder.emit(
            "generation",
            step=step,
            start=rollout_start,
            end=rollout_start + 1,
            seconds=1,
            ok=True,
            group_id=trajectory // 8,
            pair_index=trajectory % 8,
            prompt_tokens=10,
            generated_tokens=20,
        )
        recorder.emit(
            "rollout_trajectory",
            exact_token_continuity=True,
            step=step,
            start=rollout_start,
            end=rollout_end,
            seconds=rollout_end - rollout_start,
            ok=True,
            group_id=trajectory // 8,
            pair_index=trajectory % 8,
            prompt_tokens=10,
            generated_tokens=20,
            environment_tokens=5,
            conversation_tokens=25,
            environment_seconds=0.25,
            reward_seconds=0.01,
            reward=1.0,
            turns=2,
            status="SUCCEEDED",
        )
      shapes = [{"prompt": [64, 10], "completion": [64, 20]}]
      recorder.finish(
          "actor_log_probs", start, start + 1, shapes=shapes, ok=True
      )
      for method, rpc_start, rpc_end in (
          ("per_token_logps", start, start + 1),
          ("fwd_bwd", start + 1, end - 3),
          ("update", end - 3, end - 2),
      ):
        recorder.emit(
            "device_completion", method=method, time=(rpc_start + rpc_end) / 2
        )
        recorder.finish("worker_rpc_" + method, rpc_start, rpc_end)
      recorder.finish(
          "training",
          start,
          end - 2,
          trajectories=64,
          shapes=shapes,
      )
      recorder.emit("device_completion", method="weight_install", time=end - 1)
      recorder.finish("weight_sync", end - 2, end)
    recorder.close()
    benchmark.write_json(
        self.path / "manifest.json",
        {
            "schema": 3,
            "timing_mode": "stages",
            "mode": "dist",
            "workload": self.w,
            "chips": 4,
            "training_contract": {
                "compute_dtype": "bfloat16",
                "load_dtype": "float32",
                "exact_token_continuity": True,
            },
        },
    )
    benchmark.write_json(
        self.path / "result.json",
        {"returncode": 0, "total_run_time_seconds": 70},
    )

  def test_sync_inclusive_step_and_per_trajectory_times(self):
    self.record_run()
    result = benchmark.summarize(self.path)
    self.assertEqual(
        result["end_to_end"]["steady_state_step_time_seconds"]["per_step"],
        [20, 30],
    )
    self.assertEqual(
        result["end_to_end"]["steady_state_step_time_seconds"]["median"], 25
    )
    # Concurrent trajectories take 17 or 27 seconds each: mean latency is 22.
    self.assertAlmostEqual(
        result["rollout"]["trajectory_latency_seconds"]["mean"], 22
    )
    self.assertNotIn("trajectories_per_second", result["end_to_end"])
    self.assertEqual(
        result["api_calls"]["weight_sync"]["host_time_per_step_seconds"], 2
    )
    self.assertAlmostEqual(
        result["rollout"]["collection_time_seconds"]["mean"], 44.126 / 2
    )
    self.assertEqual(result["rollout"]["trajectory_latency_seconds"]["p90"], 27)
    self.assertEqual(
        result["rollout"]["average_time_per_trajectory_seconds"],
        {"model_generation_api": 1, "environment": 0.25},
    )
    self.assertEqual(
        result["rollout"]["average_output_tokens_per_trajectory"], 20
    )
    self.assertEqual(result["rollout"]["average_turns_per_trajectory"], 2)
    self.assertEqual(result["rollout"]["average_reward_per_trajectory"], 1)
    self.assertEqual(
        set(result["rollout"]),
        {
            "collection_time_seconds",
            "trajectory_latency_seconds",
            "average_output_tokens_per_trajectory",
            "average_prompt_tokens_per_model_call",
            "average_turns_per_trajectory",
            "average_reward_per_trajectory",
            "average_time_per_trajectory_seconds",
            "trajectory_outcome_percent",
        },
    )
    self.assertEqual(
        result["pipeline"]["rollout_start_delay_seconds"]["median"], 1
    )
    self.assertEqual(result["training_input"]["trajectories_per_call"], 64)
    self.assertEqual(result["training_input"]["trajectories_per_step"], 64)
    self.assertEqual(
        result["training_input"]["padded_token_slots_per_step"], 64 * 30
    )
    self.assertEqual(
        result["actor_log_probs_input"]["padded_token_slots_per_step"],
        64 * 30,
    )

    agentic = json.loads(json.dumps(result))
    agentic["mode"] = "agentic"
    for name in list(agentic["api_calls"]):
      if name.startswith("worker_rpc_"):
        del agentic["api_calls"][name]
    agentic["rollout"]["trajectory_latency_seconds"]["mean"] = 11
    agentic["api_calls"]["training"]["host_time_per_step_seconds"] = 11.5
    result["api_calls"]["worker_rpc_fwd_bwd"] = {
        "calls_per_step": 1,
        "host_time_per_step_seconds": 2,
    }
    comparison = benchmark.compare_summaries([agentic, result])
    step_gap = comparison["metrics"]["end_to_end_step_time_seconds"]
    self.assertEqual(step_gap["agentic"], 25)
    self.assertEqual(step_gap["dist"], 25)
    self.assertEqual(step_gap["dist_relative_to_agentic_percent"], 0)
    self.assertEqual(step_gap["interpretation"], "same")
    latency_gap = comparison["metrics"]["trajectory_latency_mean_seconds"]
    self.assertEqual(latency_gap["dist_relative_to_agentic_percent"], 100)
    self.assertEqual(latency_gap["interpretation"], "dist_slower")
    stages = comparison["pipeline_stages_seconds_per_step"]
    # Sum the microbatch API durations over a full step, without counting
    # nested worker RPCs a second time: (18 + 28) / 2 = 23 seconds.
    self.assertEqual(stages["training"]["dist"], 23)
    self.assertEqual(stages["training"]["agentic"], 11.5)
    self.assertEqual(
        stages["training"]["dist_relative_to_agentic_percent"], 100
    )
    self.assertEqual(stages["training"]["interpretation"], "dist_slower")
    self.assertEqual(stages["actor_log_probs"]["dist"], 1)
    self.assertEqual(stages["weight_sync"]["dist"], 2)
    self.assertAlmostEqual(stages["rollout_collection"]["dist"], 44.126 / 2)
    self.assertEqual(
        set(stages),
        {
            "rollout_collection",
            "actor_log_probs",
            "training",
            "weight_sync",
        },
    )
    self.assertNotIn(
        "end_to_end_trajectories_per_second", comparison["metrics"]
    )
    self.assertNotIn("rollout_trajectories_per_second", comparison["metrics"])
    self.assertEqual(
        comparison["step_time_per_run_seconds"],
        {
            "agentic": [25],
            "dist": [25],
        },
    )
    self.assertTrue(
        comparison["parity_checks"]["training_sequence_lengths_match"]
    )
    rpc_gap = comparison["api_calls"]["worker_rpc_fwd_bwd"]
    self.assertEqual(
        rpc_gap["host_time_per_step_seconds"]["interpretation"],
        "one_stack_only",
    )

  def test_incomplete_and_failed_runs_rejected(self):
    self.record_run()
    manifest = json.loads((self.path / "manifest.json").read_text())
    manifest["workload"]["steps"] = 4
    benchmark.write_json(self.path / "manifest.json", manifest)
    with self.assertRaisesRegex(ValueError, "Incomplete"):
      benchmark.summarize(self.path)
    benchmark.write_json(
        self.path / "result.json",
        {"returncode": 124, "total_run_time_seconds": 70},
    )
    with self.assertRaisesRegex(ValueError, "Failed"):
      benchmark.summarize(self.path)

  def test_comparison_rejects_unequal_or_missing_training_work(self):
    self.record_run()
    agentic = benchmark.summarize(self.path)
    agentic["mode"] = "agentic"
    native_microbatch = copy.deepcopy(agentic)
    native_microbatch["mode"] = "dist"
    for section in ("training_input", "actor_log_probs_input"):
      native_microbatch[section]["trajectories_per_call"] = 2
      native_microbatch[section]["unique_tensor_shapes"][0]["prompt"][0] = 2
      native_microbatch[section]["unique_tensor_shapes"][0]["completion"][0] = 2
    self.assertTrue(
        all(benchmark.compare_summaries([agentic, native_microbatch])[
            "parity_checks"
        ].values())
    )
    for section in ("training_input", "actor_log_probs_input"):
      for field in (
          "padded_token_slots_per_step",
          "trajectories_per_step",
          "unique_tensor_shapes",
      ):
        with self.subTest(section=section, field=field):
          dist = copy.deepcopy(agentic)
          dist["mode"] = "dist"
          if field == "unique_tensor_shapes":
            dist[section][field][0]["completion"][1] += 1
          else:
            dist[section][field] *= 2
          with self.assertRaisesRegex(ValueError, "Training work mismatch"):
            benchmark.compare_summaries([agentic, dist])
      dist = copy.deepcopy(agentic)
      dist["mode"] = "dist"
      dist[section] = None
      with self.assertRaisesRegex(ValueError, "Training work mismatch"):
        benchmark.compare_summaries([agentic, dist])

  def test_comparison_checks_each_run_not_only_medians(self):
    self.record_run()
    dist = benchmark.summarize(self.path)
    agentic = copy.deepcopy(dist)
    agentic["mode"] = "agentic"
    invalid = copy.deepcopy(dist)
    invalid["training_input"]["padded_token_slots_per_step"] *= 2
    with self.assertRaisesRegex(ValueError, "Training work mismatch"):
      benchmark.compare_summaries([agentic, dist, dist, invalid])
    invalid = copy.deepcopy(dist)
    invalid["training_contract"]["compute_dtype"] = "float32"
    with self.assertRaisesRegex(ValueError, "configuration mismatch"):
      benchmark.compare_summaries([agentic, invalid])

  def test_report_rejects_legacy_token_collection(self):
    self.record_run()
    path = self.path / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
      if event["kind"] == "rollout_trajectory":
        event["exact_token_continuity"] = False
    path.write_text("\n".join(json.dumps(event) for event in events))
    with self.assertRaisesRegex(ValueError, "exact token continuity"):
      benchmark.summarize(self.path)

  def test_report_rejects_wrong_generation_identities(self):
    self.record_run()
    path = self.path / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
      if event["kind"] == "generation":
        event["group_id"] += 100
    path.write_text("\n".join(json.dumps(event) for event in events))
    with self.assertRaisesRegex(ValueError, "map one-to-one"):
      benchmark.summarize(self.path)

  def test_training_trajectory_mismatch_rejected(self):
    self.record_run()
    manifest = json.loads((self.path / "manifest.json").read_text())
    manifest["workload"]["generations"] = 2
    benchmark.write_json(self.path / "manifest.json", manifest)
    with self.assertRaisesRegex(ValueError, "work mismatch"):
      benchmark.summarize(self.path)

  def test_failed_sync_is_not_a_completed_batch(self):
    recorder = runtime.Recorder(self.path / "events.jsonl", clock=lambda: 0)
    recorder.finish("training", 0, 1, trajectories=64)
    recorder.finish("weight_sync", 1, 2, ok=False)
    recorder.close()
    events = [
        json.loads(line)
        for line in (self.path / "events.jsonl").read_text().splitlines()
    ]
    self.assertFalse(any(e["kind"] == "training_step" for e in events))

  def test_wrappers_preserve_values_exceptions_and_async_behavior(self):
    class Target:

      def run(self, value):
        return value + 1

      def remote(self, method_name):
        return method_name

      async def fail(self):
        raise ValueError("original exception")

    recorder = runtime.Recorder(self.path / "events.jsonl")
    recorder.wrap(Target, "run", "training", lambda args, kwargs: args[1])
    recorder.wrap(
        Target,
        "remote",
        lambda args, kwargs: "worker_rpc_" + args[1],
    )
    recorder.wrap(Target, "fail", "actor_log_probs")
    self.assertEqual(Target().run(4), 5)
    self.assertEqual(Target().remote("fwd_bwd"), "fwd_bwd")
    with self.assertRaisesRegex(ValueError, "original exception"):
      asyncio.run(Target().fail())
    recorder.close()
    events = [
        json.loads(line)
        for line in (self.path / "events.jsonl").read_text().splitlines()
    ]
    self.assertEqual(events[0]["trained_trajectories"], 4)
    self.assertEqual(events[1]["name"], "worker_rpc_fwd_bwd")
    self.assertFalse(events[2]["ok"])

  def test_timers_include_completion_wait_and_reject_failed_wait(self):
    clock = [0.0]

    class Target:

      def run(self):
        clock[0] += 1  # Dispatch finishes well before the device work.
        return "pending"

      async def async_run(self):
        return self.run()

    def wait(args, kwargs, result):
      self.assertEqual(result, "pending")
      clock[0] += 9

    recorder = runtime.Recorder(
        self.path / "events.jsonl", clock=lambda: clock[0]
    )
    recorder.wrap(Target, "run", "training", wait_fn=wait)
    self.assertEqual(Target().run(), "pending")
    # A failed completion fence must not produce a successful training step.
    recorder.wrap(
        Target,
        "async_run",
        "weight_sync",
        wait_fn=mock.Mock(side_effect=RuntimeError("device failure")),
    )
    with self.assertRaisesRegex(RuntimeError, "device failure"):
      asyncio.run(Target().async_run())
    recorder.close()
    events = [
        json.loads(line)
        for line in (self.path / "events.jsonl").read_text().splitlines()
    ]
    self.assertEqual(events[0]["seconds"], 10)
    self.assertFalse(events[-1]["ok"])
    self.assertFalse(any(e["kind"] == "training_step" for e in events))

  def test_worker_waits_on_updated_state_before_returning_step(self):
    pending = []
    trainer = types.SimpleNamespace(
        model="old parameters",
        optimizer="old optimizer",
        grad_accumulator="gradients",
    )

    class Worker:

      _trainer = trainer

      def fwd_bwd(self):
        pending.append("device work")
        return "queued"

      def per_token_logps(self):
        return types.SimpleNamespace(
            per_token_logps=(
                "updated parameters",
                "updated optimizer",
                "gradients",
            )
        )

      def update(self):
        trainer.model = "updated parameters"
        trainer.optimizer = "updated optimizer"
        pending.append("device work")
        return 7

    def ready(states):
      self.assertEqual(
          states,
          (
              "updated parameters",
              "updated optimizer",
              "gradients",
          ),
      )
      self.assertEqual(pending.pop(), "device work")

    modules = {
        "jax": types.SimpleNamespace(block_until_ready=ready),
        "flax": types.SimpleNamespace(
            nnx=types.SimpleNamespace(state=lambda x: x)
        ),
        "tunix.experimental.worker.trainer_worker": types.SimpleNamespace(
            TrainerWorker=Worker
        ),
    }
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(
        os.environ,
        {
            "TUNIX_BENCHMARK_DIR": str(self.path),
            "TUNIX_BENCHMARK_TIMING": "stages",
        },
    ):
      runtime.install("trainer-worker")
      self.assertEqual(Worker().update(), 7)
      self.assertFalse(pending)
      self.assertEqual(Worker().fwd_bwd(), "queued")
      self.assertFalse(pending)

  def test_timing_modes_cannot_be_mixed_or_rank_unfenced_training(self):
    self.record_run()
    dist = benchmark.summarize(self.path)
    agentic = copy.deepcopy(dist)
    agentic["mode"] = "agentic"
    agentic["timing_mode"] = "pipeline"
    with self.assertRaisesRegex(ValueError, "Timing mode mismatch"):
      benchmark.compare_summaries([dist, agentic])
    dist["timing_mode"] = "pipeline"
    agentic["api_calls"]["training"]["host_time_per_step_seconds"] /= 2
    comparison = benchmark.compare_summaries([dist, agentic])
    self.assertEqual(
        set(comparison["pipeline_stages_seconds_per_step"]),
        {"rollout_collection"},
    )
    self.assertEqual(
        comparison["api_calls"]["training"]["host_time_per_step_seconds"][
            "interpretation"
        ],
        "different_not_ranked",
    )

  def test_missing_worker_hooks_and_old_timing_are_rejected(self):
    self.record_run()
    path = self.path / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in events
            if e.get("process") != "trainer-worker"
        )
    )
    with self.assertRaisesRegex(ValueError, "timing hooks"):
      benchmark.summarize(self.path)
    path = self.path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = 2
    benchmark.write_json(path, manifest)
    with self.assertRaisesRegex(ValueError, "schema 3"):
      benchmark.summarize(self.path)

  def test_completion_evidence_and_timing_corruption_are_rejected(self):
    self.record_run()
    path = self.path / "events.jsonl"
    original = [json.loads(line) for line in path.read_text().splitlines()]
    for case in (
        "missing_wait",
        "late_wait",
        "early_return",
        "bad_duration",
        "trajectory_outside_step",
        "generation_outside_trajectory",
        "timeout",
    ):
      with self.subTest(case=case):
        events = copy.deepcopy(original)
        if case in ("missing_wait", "late_wait"):
          event = next(e for e in events if e.get("method") == "update")
          if case == "missing_wait":
            events.remove(event)
          else:
            event["time"] = 1000
        elif case in ("early_return", "bad_duration"):
          event = next(e for e in events if e.get("name") == "training")
          if case == "early_return":
            event["end"] -= 0.5
            event["seconds"] = event["end"] - event["start"]
          else:
            event["seconds"] += 1
        else:
          event = next(
              e
              for e in events
              if e["kind"] == "rollout_trajectory" and e["step"] == 1
          )
          if case == "timeout":
            event["status"] = "ENV_TIMEOUT"
          elif case == "trajectory_outside_step":
            event["start"] = 0
            event["seconds"] = event["end"] - event["start"]
          else:
            generation = next(
                e
                for e in events
                if e["kind"] == "generation" and e["step"] == 1
            )
            generation["end"] = event["end"] + 1
            generation["seconds"] = generation["end"] - generation["start"]
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
        with self.assertRaises(ValueError):
          benchmark.summarize(self.path)

  def test_agentic_requires_completed_waits(self):
    events = [
        dict(
            kind="span",
            name=name,
            start=0,
            end=1,
            seconds=1,
            device_complete=True,
        )
        for name in ("training", "actor_log_probs", "weight_sync")
    ]
    benchmark.validate_completion(events, "agentic", "stages")
    for event in events:
      event["device_complete"] = False
      with self.assertRaisesRegex(ValueError, "completion evidence"):
        benchmark.validate_completion(events, "agentic", "stages")
      event["device_complete"] = True

  def test_unobserved_outcome_is_zero_percent(self):
    self.record_run()
    dist = benchmark.summarize(self.path)
    agentic = copy.deepcopy(dist)
    agentic["mode"] = "agentic"
    dist["rollout"]["trajectory_outcome_percent"] = {
        "SUCCEEDED": 75.0,
        "MAX_STEPS_REACHED": 25.0,
    }
    outcomes = benchmark.compare_summaries([agentic, dist])[
        "trajectory_outcome_percent"
    ]
    self.assertEqual(outcomes["MAX_STEPS_REACHED"]["agentic"], 0.0)
    self.assertEqual(outcomes["MAX_STEPS_REACHED"]["dist_minus_agentic"], 25.0)
    self.assertEqual(
        outcomes["MAX_STEPS_REACHED"]["interpretation"], "different_not_ranked"
    )
    self.assertIsNone(
        outcomes["MAX_STEPS_REACHED"]["dist_relative_to_agentic_percent"]
    )

  def test_rollout_token_fields_use_actual_outputs(self):
    output = types.SimpleNamespace(
        tokens=[[1, 2, 3], [4]],
        prompt_lengths=[5, 7],
        left_padded_prompt_tokens=[[0] * 99],
    )
    self.assertEqual(runtime._generation_counts(output), (12, 4))
    engine = types.SimpleNamespace(
        exact_token_continuity=True,
        env=types.SimpleNamespace(
            task={},
            extra_kwargs={
                "benchmark_policy_version": 3,
                "group_id": "prompt-1",
                "pair_index": 2,
            },
        ),
        agent=types.SimpleNamespace(
            trajectory=types.SimpleNamespace(steps=[1, 2])
        ),
    )
    fields = runtime._trajectory_fields(
        engine,
        {
            "prompt_tokens": [1, 2, 3],
            "conversation_tokens": [4, 5, 6, 7],
            "conversation_masks": [1, 1, 0, 0],
            "env_time": {"reset": 0.1, "steps": [0.2, 0.3]},
            "reward_time": {"reward": 0.05},
            "trajectory_reward": 1.0,
            "status": "SUCCEEDED",
        },
    )
    self.assertEqual(fields["step"], 3)
    self.assertEqual(fields["prompt_tokens"], 3)
    self.assertEqual(fields["generated_tokens"], 2)
    self.assertEqual(fields["environment_tokens"], 2)
    self.assertAlmostEqual(fields["environment_seconds"], 0.6)
    self.assertEqual(fields["reward"], 1.0)
    self.assertEqual(fields["turns"], 2)

  def test_prepare_and_report_reject_mismatched_data(self):
    model = self.path / "model"
    model.mkdir()
    (model / "weights.safetensors").touch()
    (model / "config.json").write_text("{}")
    runs = []
    for mode in ("dist", "agentic"):
      output = self.path / mode
      args = argparse.Namespace(
          **self.w,
          mode=mode,
          output=output,
          model_dir=model,
          cache_root=self.path / "cache",
          hardware="v5p-4",
          timing_mode="stages",
          agentic_chip_layout="shared4",
          execute=False,
          timeout=60
      )
      with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
          benchmark.subprocess, "Popen", wraps=benchmark.subprocess.Popen
      ) as popen:
        benchmark.prepare(args)
        # Provenance collection can run git/platform commands; planning must
        # never start the training launcher or agentic entry point.
        for call in popen.call_args_list:
          command = str(call.args[0])
          self.assertNotIn("launcher.sh", command)
          self.assertNotIn("frozenlake_agentic", command)
      manifest = json.loads((output / "manifest.json").read_text())
      self.assertEqual(manifest["environment"]["WANDB_MODE"], "disabled")
      if mode == "dist":
        self.assertTrue(manifest["command"][-1].endswith("/run_gemma4_e2b.sh"))
        self.assertTrue(manifest["environment"]["PYTHON_BIN"].endswith("/python.sh"))
      benchmark.write_json(
          output / "dataset.json", {"sha256": mode, "rows": 24}
      )
      runs.append(output)
    with self.assertRaisesRegex(ValueError, "Dataset"):
      benchmark.report(runs)
    # Preparation must not overwrite a prior run directory.
    with self.assertRaises(FileExistsError):
      benchmark.prepare(args)

  def test_dataset_hash_preserves_order(self):
    rows = [{"seed": 1}, {"seed": 2}]
    digests = []
    for idx, ordered in enumerate((rows, list(reversed(rows)))):
      output = self.path / str(idx)
      output.mkdir()
      with mock.patch.dict(
          runtime.os.environ, {"TUNIX_BENCHMARK_DIR": str(output)}
      ):
        runtime.record_dataset(ordered)
      digests.append(
          json.loads((output / "dataset.json").read_text())["sha256"]
      )
    self.assertNotEqual(*digests)


if __name__ == "__main__":
  unittest.main()
