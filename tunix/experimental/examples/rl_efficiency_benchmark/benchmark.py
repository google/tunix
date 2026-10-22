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

"""Prepare, run, and compare distributed-vs-agentic RL benchmarks.

Planning and reporting need only the Python standard library. Training uses
the repository's normal TPU dependencies. The workload module
(`frozenlake_dist/frozenlake_benchmark.py`) supplies the training recipe for
both stacks.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
WORKLOAD = (
    Path(__file__).resolve().parents[1]
    / "frozenlake_dist"
    / "frozenlake_benchmark.py"
)
COMPLETED_STATUSES = (
    "SUCCEEDED",
    "MAX_STEPS_REACHED",
    "MAX_CONTEXT_LIMIT_REACHED",
)


def write_json(path, value):
  Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_workload(path=WORKLOAD):
  """Loads a workload module by path, without importing the tunix package.

  A workload imports only the standard library at module level and defines
  MODULE (run as `python -m MODULE BASE_CONFIG override_config_file=FILE` for
  agentic training), DEFAULTS (integer workload flags), RESPONSE_RESERVE,
  validate(workload), dist_launch(workload, output, model_dir) and
  agentic_config(workload, model_dir).

  Args:
    path: The workload module file.

  Returns:
    The loaded workload module.
  """
  spec = importlib.util.spec_from_file_location("rl_benchmark_workload", path)
  workload = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(workload)
  return workload


def prepare(args, workload):
  """Writes the run manifest and, with --execute, trains and summarizes."""
  w = {key: vars(args)[key] for key in workload.DEFAULTS}
  if w["warmup"] < 0 or w["steps"] <= w["warmup"] or w["generations"] < 2:
    raise ValueError("Require steps > warmup >= 0 and generations >= 2.")
  if (
      w["batch"] <= 0
      or w["micro_groups"] <= 0
      or w["batch"] % w["micro_groups"]
  ):
    raise ValueError("Require positive batch divisible by micro_groups.")
  if w["response"] <= workload.RESPONSE_RESERVE:
    raise ValueError(
        f"response must exceed the {workload.RESPONSE_RESERVE}-token"
        " environment reserve."
    )
  workload.validate(w)
  output = args.output.resolve()
  model_dir = args.model_dir.resolve()
  if (
      not any(model_dir.glob("*.safetensors"))
      or not (model_dir / "config.json").is_file()
  ):
    raise ValueError("--model-dir must contain safetensors and config.json.")
  # A new directory prevents accidental checkpoint resume or mixed-run events.
  output.mkdir(parents=True, exist_ok=False)
  cache = args.cache_root.resolve() / args.mode
  env = {
      "WANDB_MODE": "disabled",
      "TUNIX_BENCHMARK_DIR": str(output),
      "TUNIX_BENCHMARK_TIMING": args.timing_mode,
      "TUNIX_BENCHMARK_RESPONSE_RESERVE": str(workload.RESPONSE_RESERVE),
      "PYTHON_BIN": sys.executable,
      "PYTHONUNBUFFERED": "1",
      "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
      "JAX_LOG_COMPILES": "1",
      "VLLM_CACHE_ROOT": str(cache / "vllm"),
      "VLLM_XLA_CACHE_PATH": str(cache / "vllm/xla_cache"),
      "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
  }
  if args.mode == "dist":
    command, dist_env = workload.dist_launch(w, output, model_dir)
    env.update(dist_env)
    dtypes = (env["MODEL_DTYPE"], env["MODEL_LOAD_DTYPE"])
  else:
    config = workload.agentic_config(w, model_dir)
    write_json(output / "agentic.json", config)
    dtypes = (
        config["model_config"]["dtype"],
        config["actor_model_config"]["load_dtype"],
    )
    env.update({
        "JAX_PLATFORMS": "tpu,cpu",
        "TPU_VISIBLE_DEVICES": "0,1,2,3",
        "TPU_VISIBLE_CHIPS": "0,1,2,3",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,4,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "LIBTPU_INIT_ARGS": (
            "--deepsea_chips_per_host_bounds=1,4,1 --deepsea_host_bounds=1,1,1"
        ),
    })
    command = [
        sys.executable,
        "-m",
        workload.MODULE,
        "tunix/cli/base_agentic_config.yaml",
        f"override_config_file={output / 'agentic.json'}",
    ]
  manifest = {
      "mode": args.mode,
      "timing_mode": args.timing_mode,
      "workload": w,
      "training_contract": dict(zip(("compute_dtype", "load_dtype"), dtypes)),
      "hardware": args.hardware,
      "model_dir": str(model_dir),
      "command": command,
      "environment": env,
  }
  write_json(output / "manifest.json", manifest)
  print(json.dumps(manifest, indent=2))
  if not args.execute:
    print("Plan only. Re-run with --execute and a NEW --output directory.")
    return
  start = time.monotonic()
  with (output / "launcher.log").open("x") as log:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, **env},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
      code = process.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
      # Kill the whole session: workers can outlive their launcher.
      os.killpg(process.pid, signal.SIGKILL)
      process.wait()
      code = 124
  write_json(
      output / "result.json",
      {"returncode": code, "total_run_time_seconds": time.monotonic() - start},
  )
  if code:
    raise RuntimeError(
        f"Training failed ({code}); inspect {output}/launcher.log and worker"
        " logs."
    )
  write_json(output / "summary.json", summarize(output))


def _stats(values):
  return {
      "mean": statistics.mean(values),
      "median": statistics.median(values),
      "per_step": values,
  }


def rollout_summary(events, step_ids, expected):
  """Aggregates trajectory and model-call events of the measured steps."""
  trajectories = [
      e
      for e in events
      if e["kind"] == "rollout_trajectory" and e["step"] in step_ids
  ]
  generations = [
      e for e in events if e["kind"] == "generation" and e["step"] in step_ids
  ]
  if any(e["status"] not in COMPLETED_STATUSES for e in trajectories):
    raise ValueError("Run contains incomplete or timed-out trajectories.")
  if not all(e["exact_token_continuity"] for e in trajectories):
    raise ValueError(
        "Benchmark requires exact token continuity in both stacks."
    )
  collection_times = []
  for step in sorted(step_ids):
    in_step = [e for e in trajectories if e["step"] == step]
    if len(in_step) != expected:
      raise ValueError(
          f"Rollout step {step} has {len(in_step)} trajectories; expected"
          f" {expected}."
      )
    collection_times.append(
        max(e["end"] for e in in_step) - min(e["start"] for e in in_step)
    )
  count = len(trajectories)
  latencies = [e["seconds"] for e in trajectories]
  return {
      "collection_time_seconds": _stats(collection_times),
      "trajectory_latency_seconds": {
          "mean": statistics.mean(latencies),
          "p90": statistics.quantiles(latencies, n=10, method="inclusive")[-1],
      },
      "output_tokens_per_trajectory": (
          sum(e["generated_tokens"] for e in generations) / count
      ),
      "prompt_tokens_per_model_call": (
          sum(e["prompt_tokens"] for e in generations) / len(generations)
      ),
      "turns_per_trajectory": statistics.mean(e["turns"] for e in trajectories),
      "reward_per_trajectory": statistics.mean(
          e["reward"] for e in trajectories
      ),
      "generation_seconds_per_trajectory": (
          sum(e["seconds"] for e in generations) / count
      ),
      "environment_seconds_per_trajectory": (
          sum(e["environment_seconds"] for e in trajectories) / count
      ),
  }


def api_call_summary(spans, first, last, step_count):
  """Normalizes host-observed API time by measured training steps."""
  durations = {}
  for e in spans:
    if e["end"] > first and e["start"] < last:
      durations.setdefault(e["name"], []).append(
          min(e["end"], last) - max(e["start"], first)
      )
  return {
      name: {
          "calls_per_step": len(values) / step_count,
          "host_time_per_step_seconds": sum(values) / step_count,
      }
      for name, values in sorted(durations.items())
  }


def work_summary(spans, step_ids):
  """Returns per-step call counts and tensor shapes of the large modules."""
  work = {}
  for name in ("training", "actor_log_probs"):
    calls = [e for e in spans if e["name"] == name and e["step"] in step_ids]
    work[name] = {
        "calls_per_step": len(calls) / len(step_ids),
        "shapes": sorted(
            {json.dumps(shape) for e in calls for shape in e["shapes"]}
        ),
    }
  return work


def summarize(directory):
  """Validates one run and returns its statistics."""
  directory = Path(directory)
  manifest = json.loads((directory / "manifest.json").read_text())
  if json.loads((directory / "result.json").read_text())["returncode"]:
    raise ValueError(f"Failed run: {directory}")
  events = [
      json.loads(line)
      for path in sorted(directory.glob("events*.jsonl"))
      for line in path.read_text().splitlines()
  ]
  mode, timing_mode = manifest["mode"], manifest["timing_mode"]
  processes = (
      {"dist", "trainer-worker", "rollout-worker"}
      if mode == "dist"
      else {"agentic"}
  )
  contracts = [e for e in events if e["kind"] == "timing_contract"]
  if {e["process"] for e in contracts} != processes or any(
      e["timing_mode"] != timing_mode for e in contracts
  ):
    raise ValueError("Missing or mismatched timing hooks; rerun all workers.")
  if not all(e.get("ok", True) for e in events):
    raise ValueError("Run contains failed calls; inspect events.")
  w = manifest["workload"]
  expected = w["batch"] * w["generations"]
  steps = [e for e in events if e["kind"] == "training_step"]
  if len(steps) != w["steps"] or any(
      e["trained_trajectories"] != expected for e in steps
  ):
    raise ValueError(
        "Incomplete run or training work mismatch; no valid speed comparison."
    )
  steady = steps[w["warmup"] :]
  step_ids = {e["step"] for e in steady}
  spans = [e for e in events if e["kind"] == "span"]
  return {
      "mode": mode,
      "timing_mode": timing_mode,
      "training_contract": manifest["training_contract"],
      "step_time_seconds": _stats([e["seconds"] for e in steady]),
      "warmup_step_time_seconds": [e["seconds"] for e in steps[: w["warmup"]]],
      "rollout": rollout_summary(events, step_ids, expected),
      "api_calls": api_call_summary(
          spans, steady[0]["start"], steady[-1]["end"], len(steady)
      ),
      "work": work_summary(spans, step_ids),
  }


def _flatten(value, prefix=""):
  """Returns the numeric leaves of nested dicts, keyed by dotted path."""
  if isinstance(value, dict):
    return {
        path: leaf
        for key, item in value.items()
        for path, leaf in _flatten(item, f"{prefix}{key}.").items()
    }
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return {prefix[:-1]: value}
  return {}


def compare(summaries):
  """Compares the per-stack medians of every numeric metric."""
  for key in ("timing_mode", "training_contract", "work"):
    if any(s[key] != summaries[0][key] for s in summaries):
      raise ValueError(f"Runs differ in {key}; no valid speed comparison.")
  runs = {
      mode: [_flatten(s) for s in summaries if s["mode"] == mode]
      for mode in ("agentic", "dist")
  }
  if not all(runs.values()):
    raise ValueError(
        "Comparison requires at least one agentic and one dist run."
    )
  metrics = {}
  for key in sorted(
      {key for flat in runs["agentic"] + runs["dist"] for key in flat}
  ):
    medians = {}
    for mode, flats in runs.items():
      values = [flat[key] for flat in flats if key in flat]
      medians[mode] = statistics.median(values) if values else None
    agentic, dist = medians["agentic"], medians["dist"]
    both = agentic is not None and dist is not None
    metrics[key] = {
        **medians,
        "dist_minus_agentic": dist - agentic if both else None,
        "dist_relative_to_agentic_percent": (
            100 * (dist / agentic - 1) if both and agentic else None
        ),
    }
  return {
      "timing_mode": summaries[0]["timing_mode"],
      "run_count": {mode: len(flats) for mode, flats in runs.items()},
      "step_time_per_run_seconds": {
          mode: [flat["step_time_seconds.mean"] for flat in flats]
          for mode, flats in runs.items()
      },
      "metrics": metrics,
  }


def report(runs):
  """Checks that RUNS are matched and compares them."""
  manifests = [
      json.loads((Path(p) / "manifest.json").read_text()) for p in runs
  ]
  for key in ("workload", "hardware", "model_dir"):
    if any(m[key] != manifests[0][key] for m in manifests):
      raise ValueError(f"Runs differ in {key}; rerun matched configurations.")
  datasets = [json.loads((Path(p) / "dataset.json").read_text()) for p in runs]
  if any(d != datasets[0] for d in datasets):
    raise ValueError("Dataset order/content mismatch.")
  return compare([summarize(p) for p in runs])


def main():
  workload = load_workload()
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="action", required=True)
  run = commands.add_parser("run", description=workload.__doc__)
  run.add_argument("--mode", choices=("dist", "agentic"), required=True)
  run.add_argument("--output", type=Path, required=True)
  run.add_argument("--model-dir", type=Path, required=True)
  run.add_argument("--cache-root", type=Path, required=True)
  run.add_argument(
      "--hardware",
      required=True,
      help="Hardware label, e.g. v5p-4; confirm device inventory in logs.",
  )
  run.add_argument("--execute", action="store_true")
  run.add_argument(
      "--timing-mode",
      choices=("stages", "pipeline"),
      default="stages",
      help=(
          "stages: wait for module completion (default); pipeline: preserve"
          " intra-step overlap and compare end-to-end time only."
      ),
  )
  run.add_argument("--timeout", type=int, default=172800)
  for flag, default in workload.DEFAULTS.items():
    run.add_argument("--" + flag.replace("_", "-"), type=int, default=default)
  commands.add_parser("report").add_argument("runs", nargs="+", type=Path)
  args = parser.parse_args()
  if args.action == "run":
    prepare(args, workload)
  else:
    print(json.dumps(report(args.runs), indent=2))


if __name__ == "__main__":
  main()
