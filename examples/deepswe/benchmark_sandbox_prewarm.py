#!/usr/bin/env python3
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

"""Standalone Benchmark for Asynchronous Sandbox & Workspace Pre-warming.

Measures Turn 0 blocking latency under:
1. Baseline: Synchronous SandboxFleet acquisition and OpenHands workspace setup.
2. Pre-warmed: Background sandbox pre-acquisition and setup overlapped with
   simulated prior-step training / model inference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

BENCHMARK_TASKS: List[Dict[str, Any]] = [
    {
        "instance_id": "numpy-05aa44d53f4f",
        "repo": "numpy/numpy",
        "repo_name": "numpy/numpy",
        "docker_image": "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/numpy_final:05aa44d53f4f9528847a0c014fe4bda5caa5fd3d",
        "problem_statement": "Fix numpy array scalar indexing bug.",
        "base_commit": "05aa44d53f4f9528847a0c014fe4bda5caa5fd3d",
        "commit_hash": "05aa44d53f4f9528847a0c014fe4bda5caa5fd3d",
        "parsed_commit": json.dumps({
            "file_diffs": [],
            "old_commit_hash": "05aa44d53f4f9528847a0c014fe4bda5caa5fd3d",
            "new_commit_hash": "05aa44d53f4f9528847a0c014fe4bda5caa5fd3d",
            "commit_message": "benchmark stub",
        }),
        "parsed_commit_content": json.dumps({
            "file_diffs": [],
            "old_commit_hash": "05aa44d53f4f9528847a0c014fe4bda5caa5fd3d",
            "new_commit_hash": "05aa44d53f4f9528847a0c014fe4bda5caa5fd3d",
            "commit_message": "benchmark stub",
        }),
        "execution_result_content": "{}",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
        "version": "1.26",
    },
    {
        "instance_id": "aiohttp-07429ed0084b",
        "repo": "aio-libs/aiohttp",
        "repo_name": "aio-libs/aiohttp",
        "docker_image": "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/aiohttp_final:07429ed0084b3cdd636b178d4eda6944a1809069",
        "problem_statement": "Fix aiohttp client payload handler.",
        "base_commit": "07429ed0084b3cdd636b178d4eda6944a1809069",
        "commit_hash": "07429ed0084b3cdd636b178d4eda6944a1809069",
        "parsed_commit": json.dumps({
            "file_diffs": [],
            "old_commit_hash": "07429ed0084b3cdd636b178d4eda6944a1809069",
            "new_commit_hash": "07429ed0084b3cdd636b178d4eda6944a1809069",
            "commit_message": "benchmark stub",
        }),
        "parsed_commit_content": json.dumps({
            "file_diffs": [],
            "old_commit_hash": "07429ed0084b3cdd636b178d4eda6944a1809069",
            "new_commit_hash": "07429ed0084b3cdd636b178d4eda6944a1809069",
            "commit_message": "benchmark stub",
        }),
        "execution_result_content": "{}",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
        "version": "3.8",
    },
]


def run_benchmark():
  parser = argparse.ArgumentParser(
      description="Benchmark Sandbox Pre-warming vs Synchronous Reset"
  )
  parser.add_argument(
      "--scaffold",
      default="openhands",
      choices=["openhands", "r2egym"],
      help="Agent scaffold environment",
  )
  parser.add_argument(
      "--num_tasks",
      type=int,
      default=2,
      help="Number of tasks to evaluate",
  )
  parser.add_argument(
      "--simulated_overlap_sec",
      type=float,
      default=30.0,
      help="Simulated inter-step training / rollout overlap time in seconds",
  )
  parser.add_argument(
      "--namespace",
      default=os.getenv("SANDBOX_NAMESPACE", "default"),
      help="Kubernetes namespace for sandboxes",
  )
  args = parser.parse_args()

  try:
    from examples.deepswe import sandbox_utils, swe_env
  except ImportError:
    logging.error(
        "Failed to import deepswe modules. Ensure PYTHONPATH includes the tunix root."
    )
    sys.exit(1)

  tasks = BENCHMARK_TASKS[: min(args.num_tasks, len(BENCHMARK_TASKS))]
  logging.info(
      "Initializing SandboxFleet in namespace '%s' for %d task(s)...",
      args.namespace,
      len(tasks),
  )

  fleet = sandbox_utils.init_global_fleet(
      tasks=tasks,
      scaffold=args.scaffold,
      namespace=args.namespace,
  )

  # Prime warm pools with 2 replicas so both Baseline and Pre-warm have ready sandboxes
  images = [task["docker_image"] for task in tasks]
  logging.info(
      "Priming SandboxWarmPools for %d image(s) (%s) with 2 replicas...",
      len(images),
      images,
  )
  fleet.warm_images(images, replicas_override=2, wait=True)
  logging.info("SandboxWarmPools successfully primed and ready!")

  results: List[Dict[str, Any]] = []

  # =========================================================================
  # Phase 1: Synchronous Baseline (Status Quo)
  # =========================================================================
  logging.info("\n" + "=" * 80)
  logging.info("PHASE 1: Synchronous Baseline (Status Quo)")
  logging.info("fleet.acquire() and OpenHands workspace setup happen on Turn 0")
  logging.info("=" * 80)

  # For Baseline, disable automatic pre-warming during __init__ so it measures synchronous reset
  old_prewarm_env = os.environ.get("TUNIX_PREWARM_SANDBOX", "1")
  os.environ["TUNIX_PREWARM_SANDBOX"] = "0"

  for task in tasks:
    task_id = task["instance_id"]
    logging.info(">>> [Baseline] Initializing SWEEnv for %s...", task_id)
    env = swe_env.SWEEnv(
        entry=task,
        scaffold=args.scaffold,
        use_agent_sandbox=True,
        fleet=fleet,
    )

    logging.info(">>> [Baseline] Invoking env.reset() (Turn 0 blocking)...")
    t0 = time.perf_counter()
    obs, info = env.reset()
    reset_latency = time.perf_counter() - t0

    logging.info(
        "[Baseline] %s env.reset() finished in %.2fs (Turn 0 blocked!)",
        task_id,
        reset_latency,
    )

    results.append({
        "task_id": task_id,
        "mode": "Synchronous Baseline",
        "overlap_sec": 0.0,
        "turn0_latency_sec": reset_latency,
        "total_setup_sec": reset_latency,
    })

    logging.info(">>> [Baseline] Closing environment and releasing sandbox...")
    env.close()
    time.sleep(2)

  # Restore pre-warm environment
  os.environ["TUNIX_PREWARM_SANDBOX"] = old_prewarm_env

  # =========================================================================
  # Phase 2: Asynchronous Background Pre-warming
  # =========================================================================
  logging.info("\n" + "=" * 80)
  logging.info("PHASE 2: Asynchronous Background Pre-warming")
  logging.info("Sandboxes and workspaces are pre-acquired in background thread")
  logging.info("=" * 80)

  for task in tasks:
    task_id = task["instance_id"]
    logging.info(">>> [Prewarm] Initializing SWEEnv for %s...", task_id)
    t_start = time.perf_counter()
    env = swe_env.SWEEnv(
        entry=task,
        scaffold=args.scaffold,
        use_agent_sandbox=True,
        fleet=fleet,
    )

    # env.prewarm() was dispatched automatically or natively
    if not getattr(env, "_prewarm_future", None):
      logging.info(">>> [Prewarm] Calling native env.prewarm()...")
      env.prewarm()

    logging.info(
        ">>> [Prewarm] Simulating %.1fs of inter-step model inference / training...",
        args.simulated_overlap_sec,
    )
    time.sleep(args.simulated_overlap_sec)

    logging.info(">>> [Prewarm] Invoking env.reset() (Turn 0 dispatch)...")
    t0 = time.perf_counter()
    obs, info = env.reset()
    turn0_latency = time.perf_counter() - t0
    total_setup_sec = time.perf_counter() - t_start

    logging.info(
        "[Prewarm] %s env.reset() returned in %.4fs (total setup %.2fs)!",
        task_id,
        turn0_latency,
        total_setup_sec,
    )

    results.append({
        "task_id": task_id,
        "mode": "Background Pre-warm",
        "overlap_sec": args.simulated_overlap_sec,
        "turn0_latency_sec": turn0_latency,
        "total_setup_sec": total_setup_sec,
    })

    logging.info(">>> [Prewarm] Closing environment and releasing sandbox...")
    env.close()
    time.sleep(2)

  # =========================================================================
  # Benchmark Summary Table
  # =========================================================================
  print("\n" + "=" * 90)
  print(f"{'Task ID':<28} | {'Mode':<22} | {'Turn 0 Delay (s)':<18} | {'Total Time (s)':<15}")
  print("-" * 90)
  baseline_times = {}
  prewarm_times = {}

  for r in results:
    tid = r["task_id"]
    mode = r["mode"]
    t0_lat = r["turn0_latency_sec"]
    tot_time = r["total_setup_sec"]
    print(f"{tid:<28} | {mode:<22} | {t0_lat:<18.3f} | {tot_time:<15.3f}")

    if mode == "Synchronous Baseline":
      baseline_times[tid] = t0_lat
    else:
      prewarm_times[tid] = t0_lat

  print("=" * 90)

  print("\n=== LATENCY REDUCTION SUMMARY ===")
  for tid in baseline_times:
    if tid in prewarm_times:
      b = baseline_times[tid]
      p = prewarm_times[tid]
      saved = b - p
      reduction = ((b - p) / b) * 100 if b > 0 else 0.0
      print(
          f"Task {tid}:\n"
          f"  Baseline Turn 0 Delay: {b:.2f}s\n"
          f"  Prewarm Turn 0 Delay:  {p:.4f}s\n"
          f"  Idle Time Saved:       {saved:.2f}s ({reduction:.1f}% reduction)\n"
      )


if __name__ == "__main__":
  run_benchmark()
