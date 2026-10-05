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
python3 tests/experimental/examples/rl_efficiency_benchmark/benchmark_test.py
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


benchmark = load_module("benchmark")
runtime = load_module("runtime")

# The smallest workload that satisfies benchmark.load_workload.
FAKE_WORKLOAD = """
import sys

MODULE = "fake_dist.fake_benchmark"
RESPONSE_RESERVE = 16
DEFAULTS = dict(
    batch=8,
    generations=8,
    steps=3,
    warmup=1,
    micro_groups=2,
    prompt=2048,
    response=2048,
    seed=42,
)
# Records the launch environment, then exits like a failed run.
CHILD = (
    "import json, os, pathlib, sys;"
    " pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)));"
    " sys.exit(3)"
)


def validate(workload):
  if workload["seed"] < 0:
    raise ValueError("seed must be non-negative.")


def dist_launch(workload, output, model_dir):
  command = [sys.executable, "-c", CHILD, str(output / "launched.json")]
  return command, {
      "STEPS": str(workload["steps"]),
      "MODEL_DTYPE": "bfloat16",
      "MODEL_LOAD_DTYPE": "float32",
  }


def agentic_config(workload, model_dir):
  return {
      "data_module": MODULE,
      "model_config": {"dtype": "bfloat16"},
      "actor_model_config": {"load_dtype": "float32"},
  }
"""


def read_events(path):
  return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_events(path, events):
  Path(path).write_text("".join(json.dumps(e) + "\n" for e in events))


class BenchmarkTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
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
        seed=42,
    )

  def fake_workload(self):
    path = self.path / "fake_benchmark.py"
    path.write_text(FAKE_WORKLOAD)
    return benchmark.load_workload(path)

  def model_dir(self):
    model = self.path / "model"
    model.mkdir()
    (model / "weights.safetensors").touch()
    (model / "config.json").write_text("{}")
    return model

  def run_args(self, mode, output, model, **overrides):
    return argparse.Namespace(**{
        **self.w,
        "mode": mode,
        "output": output,
        "model_dir": model,
        "cache_root": self.path / "cache",
        "hardware": "v5p-4",
        "timing_mode": "stages",
        "execute": False,
        "timeout": 60,
        **overrides,
    })

  def record_run(self):
    """Records a three-step dist run: 64 trajectories per step, 1 warmup."""
    recorder = runtime.Recorder(self.path / "events.jsonl", clock=lambda: 0)
    for process in ("dist", "trainer-worker", "rollout-worker"):
      recorder.emit("timing_contract", process=process, timing_mode="stages")
    # Initial sync must not count as a training full batch.
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
            prompt_tokens=10,
            generated_tokens=20,
        )
        recorder.emit(
            "rollout_trajectory",
            step=step,
            start=rollout_start,
            end=rollout_end,
            seconds=rollout_end - rollout_start,
            ok=True,
            exact_token_continuity=True,
            environment_seconds=0.25,
            reward=1.0,
            turns=2,
            status="SUCCEEDED",
        )
      shapes = [{"prompt": [64, 10], "completion": [64, 20]}]
      recorder.finish("actor_log_probs", start, start + 1, shapes=shapes)
      recorder.finish(
          "training", start + 1, end - 2, trajectories=64, shapes=shapes
      )
      recorder.finish("weight_sync", end - 2, end)
    recorder.close()
    benchmark.write_json(
        self.path / "manifest.json",
        {
            "mode": "dist",
            "timing_mode": "stages",
            "workload": self.w,
            "training_contract": {
                "compute_dtype": "bfloat16",
                "load_dtype": "float32",
            },
        },
    )
    benchmark.write_json(self.path / "result.json", {"returncode": 0})

  def test_summary_and_comparison(self):
    self.record_run()
    result = benchmark.summarize(self.path)
    self.assertEqual(result["step_time_seconds"]["per_step"], [20, 30])
    self.assertEqual(result["step_time_seconds"]["median"], 25)
    self.assertEqual(result["warmup_step_time_seconds"], [10])
    rollout = result["rollout"]
    # Concurrent trajectories take 17 or 27 seconds each: mean latency is 22.
    self.assertAlmostEqual(rollout["trajectory_latency_seconds"]["mean"], 22)
    self.assertEqual(rollout["trajectory_latency_seconds"]["p90"], 27)
    self.assertAlmostEqual(
        rollout["collection_time_seconds"]["mean"], 44.126 / 2
    )
    self.assertEqual(rollout["output_tokens_per_trajectory"], 20)
    self.assertEqual(rollout["prompt_tokens_per_model_call"], 10)
    self.assertEqual(rollout["turns_per_trajectory"], 2)
    self.assertEqual(rollout["reward_per_trajectory"], 1)
    self.assertEqual(rollout["generation_seconds_per_trajectory"], 1)
    self.assertEqual(rollout["environment_seconds_per_trajectory"], 0.25)
    self.assertEqual(
        result["api_calls"]["weight_sync"]["host_time_per_step_seconds"], 2
    )
    self.assertEqual(result["work"]["training"]["calls_per_step"], 1)

    agentic = copy.deepcopy(result)
    agentic["mode"] = "agentic"
    agentic["rollout"]["trajectory_latency_seconds"]["mean"] = 11
    result["api_calls"]["rollout_poll"] = {"calls_per_step": 1}
    comparison = benchmark.compare([agentic, result])
    self.assertEqual(comparison["run_count"], {"agentic": 1, "dist": 1})
    self.assertEqual(
        comparison["step_time_per_run_seconds"], {"agentic": [25], "dist": [25]}
    )
    metrics = comparison["metrics"]
    step = metrics["step_time_seconds.mean"]
    self.assertEqual((step["agentic"], step["dist"]), (25, 25))
    self.assertEqual(step["dist_relative_to_agentic_percent"], 0)
    latency = metrics["rollout.trajectory_latency_seconds.mean"]
    self.assertEqual(latency["dist_minus_agentic"], 11)
    self.assertEqual(latency["dist_relative_to_agentic_percent"], 100)
    one_stack = metrics["api_calls.rollout_poll.calls_per_step"]
    self.assertIsNone(one_stack["agentic"])
    self.assertIsNone(one_stack["dist_minus_agentic"])

  def test_comparison_rejects_mismatched_runs(self):
    self.record_run()
    dist = benchmark.summarize(self.path)
    agentic = copy.deepcopy(dist)
    agentic["mode"] = "agentic"
    for key, change in (
        ("work", lambda s: s["work"]["training"]["shapes"].append("[1]")),
        ("training_contract", lambda s: s["training_contract"].clear()),
        ("timing_mode", lambda s: s.update(timing_mode="pipeline")),
    ):
      with self.subTest(key=key):
        invalid = copy.deepcopy(dist)
        change(invalid)
        # Every run is checked, not only the per-stack medians.
        with self.assertRaisesRegex(ValueError, f"differ in {key}"):
          benchmark.compare([agentic, dist, invalid])
    with self.assertRaisesRegex(ValueError, "at least one agentic"):
      benchmark.compare([dist, dist])

  def test_invalid_runs_rejected(self):
    self.record_run()
    events = read_events(self.path / "events.jsonl")
    manifest = json.loads((self.path / "manifest.json").read_text())

    def trajectory(e):
      return e["kind"] == "rollout_trajectory" and e["step"] == 1

    for case, message, edit_events, edit_manifest in (
        ("steps", "Incomplete", None, {"steps": 4}),
        ("generations", "work mismatch", None, {"generations": 2}),
        (
            "missing_hooks",
            "timing hooks",
            lambda e: e.get("process") == "trainer-worker",
            None,
        ),
        ("timeout", "timed-out", trajectory, None),
        ("continuity", "exact token continuity", trajectory, None),
    ):
      with self.subTest(case=case):
        changed = copy.deepcopy(events)
        if case == "missing_hooks":
          changed = [e for e in changed if not edit_events(e)]
        elif case == "timeout":
          next(filter(edit_events, changed))["status"] = "ENV_TIMEOUT"
        elif case == "continuity":
          next(filter(edit_events, changed))["exact_token_continuity"] = False
        write_events(self.path / "events.jsonl", changed)
        benchmark.write_json(
            self.path / "manifest.json",
            {**manifest, "workload": {**self.w, **(edit_manifest or {})}},
        )
        with self.assertRaisesRegex(ValueError, message):
          benchmark.summarize(self.path)
    benchmark.write_json(self.path / "result.json", {"returncode": 124})
    with self.assertRaisesRegex(ValueError, "Failed"):
      benchmark.summarize(self.path)

  def test_failed_sync_is_not_a_completed_batch(self):
    recorder = runtime.Recorder(self.path / "events.jsonl", clock=lambda: 0)
    recorder.finish("training", 0, 1, trajectories=64)
    recorder.finish("weight_sync", 1, 2, ok=False)
    recorder.close()
    events = read_events(self.path / "events.jsonl")
    self.assertFalse(any(e["kind"] == "training_step" for e in events))

  def test_wrappers_time_completion_waits_and_preserve_behavior(self):
    clock = [0.0]

    class Target:

      def run(self, value):
        clock[0] += 1  # Dispatch finishes well before the device work.
        return value + 1

      async def fail(self):
        raise ValueError("original exception")

    def wait(args, kwargs, result):
      self.assertEqual(result, 5)
      clock[0] += 9

    recorder = runtime.Recorder(
        self.path / "events.jsonl", clock=lambda: clock[0]
    )
    recorder.wrap(
        Target,
        "run",
        "training",
        trajectories_fn=lambda args, kwargs: args[1],
        wait_fn=wait,
    )
    recorder.wrap(Target, "fail", "weight_sync")
    self.assertEqual(Target().run(4), 5)
    with self.assertRaisesRegex(ValueError, "original exception"):
      asyncio.run(Target().fail())
    recorder.close()
    run, fail = read_events(self.path / "events.jsonl")
    self.assertEqual(run["seconds"], 10)
    self.assertEqual(run["trained_trajectories"], 4)
    self.assertFalse(fail["ok"])

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

      def per_token_logps(self):  # Also fenced; not exercised here.
        raise NotImplementedError

      def update(self):
        trainer.model = "updated parameters"
        trainer.optimizer = "updated optimizer"
        pending.append("device work")
        return 7

    def ready(states):
      self.assertEqual(
          states, ("updated parameters", "updated optimizer", "gradients")
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
            "TUNIX_BENCHMARK_RESPONSE_RESERVE": "0",
        },
    ):
      runtime.install("trainer-worker")
      self.assertEqual(Worker().update(), 7)
      self.assertFalse(pending)
      self.assertEqual(Worker().fwd_bwd(), "queued")
      self.assertFalse(pending)

  def test_rollout_fields_use_actual_outputs(self):
    output = types.SimpleNamespace(
        tokens=[[1, 2, 3], [4]],
        prompt_lengths=[5, 7],
        left_padded_prompt_tokens=types.SimpleNamespace(shape=(1, 99)),
    )
    self.assertEqual(runtime._generation_counts(output), (12, 4))
    output.prompt_lengths = None
    self.assertEqual(runtime._generation_counts(output), (99, 4))
    engine = types.SimpleNamespace(
        exact_token_continuity=True,
        env=types.SimpleNamespace(
            task={"policy_version": 1},
            extra_kwargs={"benchmark_policy_version": 3},
        ),
        env_time={
            "reset_latency": 0.1,
            "step_latency": [0.2, 0.3],
            "close_latency": 0.0,
        },
        agent=types.SimpleNamespace(
            trajectory=types.SimpleNamespace(
                steps=[1, 2],
                reward=1,
                status=types.SimpleNamespace(name="SUCCEEDED"),
            )
        ),
    )
    self.assertEqual(runtime._step(engine), 3)
    fields = runtime._trajectory_fields(engine)
    self.assertAlmostEqual(fields.pop("environment_seconds"), 0.6)
    self.assertEqual(
        fields,
        {
            "exact_token_continuity": True,
            "turns": 2,
            "reward": 1.0,
            "status": "SUCCEEDED",
        },
    )

  def test_prepare_and_report(self):
    workload = self.fake_workload()
    model = self.model_dir()
    runs = []
    for mode in ("dist", "agentic"):
      output = self.path / mode
      args = self.run_args(mode, output, model)
      with contextlib.redirect_stdout(io.StringIO()):
        benchmark.prepare(args, workload)
      # Planning must never start training.
      self.assertFalse((output / "launcher.log").exists())
      manifest = json.loads((output / "manifest.json").read_text())
      self.assertEqual(
          manifest["training_contract"],
          {"compute_dtype": "bfloat16", "load_dtype": "float32"},
      )
      environment = manifest["environment"]
      self.assertEqual(environment["WANDB_MODE"], "disabled")
      self.assertEqual(environment["TUNIX_BENCHMARK_RESPONSE_RESERVE"], "16")
      if mode == "dist":
        self.assertEqual(environment["STEPS"], "3")
      else:
        self.assertEqual(
            manifest["command"][:3], [sys.executable, "-m", workload.MODULE]
        )
        config = json.loads((output / "agentic.json").read_text())
        self.assertEqual(config["data_module"], workload.MODULE)
      benchmark.write_json(output / "dataset.json", {"sha256": mode})
      runs.append(output)
    with self.assertRaisesRegex(ValueError, "Dataset"):
      benchmark.report(runs)
    # Preparation must not overwrite a prior run directory.
    with self.assertRaises(FileExistsError):
      benchmark.prepare(args, workload)

  def test_execute_records_failed_launch(self):
    workload = self.fake_workload()
    output = self.path / "dist"
    args = self.run_args("dist", output, self.model_dir(), execute=True)
    with mock.patch.dict(
        os.environ, {"STEPS": "99"}
    ), contextlib.redirect_stdout(io.StringIO()):
      with self.assertRaisesRegex(RuntimeError, r"Training failed \(3\)"):
        benchmark.prepare(args, workload)
    # Workload overrides take precedence over the caller's environment.
    launched = json.loads((output / "launched.json").read_text())
    self.assertEqual(launched["STEPS"], "3")
    result = json.loads((output / "result.json").read_text())
    self.assertEqual(result["returncode"], 3)

  def test_invalid_workload_flags_rejected_before_output(self):
    workload = self.fake_workload()
    output = self.path / "run"
    for overrides, message in (
        ({"response": 16}, "16-token"),
        ({"batch": 7}, "divisible"),
        ({"warmup": 3}, "steps > warmup"),
        ({"seed": -1}, "seed must be non-negative"),
    ):
      with self.subTest(**overrides):
        args = self.run_args("dist", output, self.path / "model", **overrides)
        with self.assertRaisesRegex(ValueError, message):
          benchmark.prepare(args, workload)
    self.assertFalse(output.exists())

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
