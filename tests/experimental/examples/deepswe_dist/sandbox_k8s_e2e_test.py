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

"""End-to-end test exercising DeepSWE dataset loading and sandbox lifecycle on Kubernetes.

Verifies the full pipeline:
  1. Dataset loading (DeepSWE R2E-Gym dataset)
  2. Fleet initialization
  3. Lookahead dynamic pre-warming with PrewarmDatasetIterator
  4. Sandbox checkout / acquire via SWEEnv(use_agent_sandbox=True)
  5. Live command execution inside the warm sandbox pod
  6. Sandbox release back to fleet on env.close()
  7. Dynamic retirement / unwarming of finished batch pools as iterator advances
  8. Clean fleet teardown
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
import sys
import threading
import time
from typing import Any

from absl.testing import absltest

# pylint: disable=g-import-not-at-top
try:
  from tunix.experimental.examples.deepswe_dist import deepswe
except ImportError:
  try:
    import deepswe
  except ImportError:
    deepswe = None

try:
  from examples.deepswe import swe_env
except ImportError:
  try:
    from examples.deepswe import swe_env
  except ImportError:
    import swe_env

try:
  from examples.deepswe import sandbox_utils
except ImportError:
  try:
    from examples.deepswe import sandbox_utils
  except ImportError:
    import sandbox_utils


class FakeFleet:
  """Mock SandboxFleet tracking calls for local verification without a live GKE cluster."""

  def __init__(self):
    self._lock = threading.Lock()
    self.warm_calls: list[tuple[str, int | None, bool]] = []
    self.set_replicas_calls: list[tuple[str, int]] = []
    self.unwarm_calls: list[str] = []
    self.active_pools: dict[str, int] = {}
    self.acquired_tasks: list[Any] = []
    self.released_handles: list[Any] = []

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    with self._lock:
      self.warm_calls.append((image, replicas_override, wait))
      self.active_pools[image] = replicas_override or 1

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    with self._lock:
      self.set_replicas_calls.append((image, replicas))
      self.active_pools[image] = replicas

  def unwarm_image(self, image: str) -> None:
    with self._lock:
      self.unwarm_calls.append(image)
      self.active_pools.pop(image, None)

  def acquire(self, task: Any) -> Any:
    with self._lock:
      self.acquired_tasks.append(task)

    class FakeHandle:

      def __init__(self, t):
        self.task = t
        self.pod_name = f"pod-{getattr(t, 'image', 'default')}-claim"

    return FakeHandle(task)

  def release(self, handle: Any) -> None:
    with self._lock:
      self.released_handles.append(handle)

  def teardown(self) -> None:
    with self._lock:
      self.active_pools.clear()


def create_synthetic_dataset(
    num_samples: int = 12,
    default_image: str = "numpy_final:25010c16edbe54ac9449ccac9d8f1da0c2d97cb0",
) -> list[dict[str, Any]]:
  """Creates synthetic DeepSWE task records for testing."""
  test_images = [
      default_image,
      "pandas_final:82a102b393f0b9c6c1f19e1f89fc23544c22f2ba",
      "sympy_final:b39e65839b1a5ef8c8db182239d22c95e54d7e97",
      "scikit-learn_final:c6e5e8e4a9e224e75d691e843c9ba811b7a2bb03",
  ]
  samples = []
  for i in range(num_samples):
    img = test_images[i % len(test_images)]
    samples.append({
        "instance_id": f"task-sample-{i}",
        "docker_image": img,
        "repo_name": f"test/repo-{i}",
        "prompt": f"Fix issue #{i} in repository",
        "metadata": {
            "env_config": {
                "entry": {
                    "docker_image": img,
                }
            }
        },
    })
  return samples


def run_pipeline_e2e(
    dataset: list[Any],
    fleet: Any,
    batch_size: int = 1,
    num_generations: int = 2,
    max_steps: int = 3,
    is_mock: bool = False,
    scaffold: str = "r2egym",
    minimum_step_time_in_second: float = 0.0,
    sampling_rate: float = 1.0,
) -> dict[str, Any]:
  """Runs the complete pre-warm, acquire, execute, release, and unwarm cycle."""
  logging.info("=== [DeepSWE E2E Test] Starting Sandbox Lifecycle Run ===")
  logging.info(
      "Parameters: batch_size=%d, num_generations=%d, max_steps=%d,"
      " is_mock=%s, scaffold=%s, minimum_step_time_in_second=%.2f,"
      " sampling_rate=%.2f",
      batch_size,
      num_generations,
      max_steps,
      is_mock,
      scaffold,
      minimum_step_time_in_second,
      sampling_rate,
  )

  # 1. Initialize PrewarmDatasetIterator
  logging.info("1. Initializing PrewarmDatasetIterator...")
  iterator = sandbox_utils.PrewarmDatasetIterator(
      dataset,
      fleet=fleet,
      num_generations=num_generations,
      batch_size=batch_size,
      unwarm_on_exhaustion=True,
      scaffold=scaffold,
  )
  logging.info(
      "   [OK] Priming complete. Active warm pools: %s",
      getattr(fleet, "active_pools", {}),
  )

  stats = {
      "steps_executed": 0,
      "batches_sampled": 0,
      "sandboxes_acquired": 0,
      "sandboxes_released": 0,
      "pools_unwarmed": 0,
      "avg_acquire_time_sec": 0.0,
      "avg_exec_time_sec": 0.0,
  }

  acquire_durations: list[float] = []
  exec_durations: list[float] = []
  print_lock = threading.Lock()

  def _execute_single_rollout(
      step: int,
      item_idx: int,
      batch_item: Any,
      gen_idx: int,
      total_items: int,
      n_gens: int,
  ) -> dict[str, Any]:
    try:
      if is_mock:
        task = sandbox_utils.normalize_tasks_for_fleet(
            [batch_item], scaffold=scaffold
        )[0]
        t_acq_start = time.perf_counter()
        handle = fleet.acquire(task)
        t_acq = time.perf_counter() - t_acq_start

        t_exec_start = time.perf_counter()
        step_res = (
            f"AUTHORS.rst LICENSE README.rst setup.py (gen={gen_idx})",
            0.0,
            True,
            {"max_steps": 1},
        )
        t_exec = time.perf_counter() - t_exec_start

        fleet.release(handle)
      else:
        t_acq_start = time.perf_counter()
        env = swe_env.SWEEnv(
            batch_item,
            fleet=fleet,
            use_agent_sandbox=True,
            scaffold=scaffold,
            max_steps=1,
        )
        _ = env.reset()
        t_acq = time.perf_counter() - t_acq_start

        t_exec_start = time.perf_counter()
        param_name = "command" if scaffold == "openhands" else "cmd"
        action = (
            "<function=execute_bash>\n"
            f"<parameter={param_name}>ls</parameter>\n"
            "</function>"
        )
        step_res = env.step(action)
        t_exec = time.perf_counter() - t_exec_start

        env.close()

      # Print out immediately for the first generation of each item
      if gen_idx == 0:
        obs_text = (
            step_res[0]
            if isinstance(step_res, tuple)
            else getattr(step_res, "observation", str(step_res))
        )
        with print_lock:
          logging.info(
              "      [Batch Item %d/%d, Gen 0/%d] Sandbox acquired in %.3fs,"
              " command executed in %.3fs",
              item_idx + 1,
              total_items,
              n_gens,
              t_acq,
              t_exec,
          )
          logging.info(
              "      [OK] step_res (first 128 chars): %s",
              str(step_res)[:128],
          )
          print(
              f"      >>> [Batch Item {item_idx + 1}/{total_items},"
              f" Gen 0/{n_gens}] acq={t_acq:.3f}s, exec={t_exec:.3f}s",
              flush=True,
          )
          print(
              "      >>> [OK] step_res (first 128 chars):"
              f" {str(step_res)[:128]}",
              flush=True,
          )
          print(
              "      >>> [OK] observation (first 128 chars):"
              f" {repr(str(obs_text)[:128])}",
              flush=True,
          )

      return {
          "item_idx": item_idx,
          "gen_idx": gen_idx,
          "t_acq": t_acq,
          "t_exec": t_exec,
          "step_res": step_res,
      }
    except Exception as e:
      with print_lock:
        logging.error(
            "      [FAIL] Step %d Item %d/%d Gen %d/%d rollout failed: %s",
            step,
            item_idx + 1,
            total_items,
            gen_idx,
            n_gens,
            e,
        )
        print(
            f"      [FAIL] Step {step} Item {item_idx + 1}/{total_items} Gen"
            f" {gen_idx}/{n_gens} rollout failed: {e}",
            flush=True,
        )
      raise

  for step in range(max_steps):
    step_start_time = time.perf_counter()
    logging.info("--- [Step %d] Fetching next batch from iterator ---", step)
    batch_items = []
    for b_idx in range(batch_size):
      try:
        batch_items.append(next(iterator))
      except StopIteration:
        logging.info(
            "   Dataset iterator exhausted at step %d, item %d", step, b_idx
        )
        break

    if not batch_items:
      logging.info("   Dataset iterator exhausted at step %d", step)
      break

    stats["steps_executed"] += 1
    logging.info(
        "   [OK] Step %d fetched batch of %d items", step, len(batch_items)
    )

    should_test_batch = (sampling_rate >= 1.0) or (
        sampling_rate > 0.0 and random.random() < sampling_rate
    )

    if not should_test_batch:
      logging.info(
          "   [Step %d] Skipping sandbox acquire & execution"
          " (sampling_rate=%.2f).",
          step,
          sampling_rate,
      )
      print(
          f"   [Step {step}] Skipping sandbox acquire & execution"
          f" (sampling_rate={sampling_rate:.2f})",
          flush=True,
      )
    else:
      stats["batches_sampled"] += 1
      rollout_jobs = []
      for item_idx, batch_item in enumerate(batch_items):
        for gen_idx in range(num_generations):
          rollout_jobs.append((item_idx, batch_item, gen_idx))

      logging.info(
          "   [Step %d] Fanning out %d parallel sandbox acquire & execute tasks"
          " (%d items * %d generations)...",
          step,
          len(rollout_jobs),
          len(batch_items),
          num_generations,
      )
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=max(1, len(rollout_jobs))
      ) as executor:
        futures = [
            executor.submit(
                _execute_single_rollout,
                step,
                item_idx,
                b_item,
                gen_idx,
                len(batch_items),
                num_generations,
            )
            for item_idx, b_item, gen_idx in rollout_jobs
        ]
        results = [f.result() for f in futures]

      results.sort(key=lambda r: (r["item_idx"], r["gen_idx"]))
      step_acq_times = []
      step_exec_times = []
      for res in results:
        t_acq = res["t_acq"]
        t_exec = res["t_exec"]

        acquire_durations.append(t_acq)
        exec_durations.append(t_exec)
        step_acq_times.append(t_acq)
        step_exec_times.append(t_exec)
        stats["sandboxes_acquired"] += 1
        stats["sandboxes_released"] += 1

      step_avg_acq = (
          sum(step_acq_times) / len(step_acq_times) if step_acq_times else 0.0
      )
      step_avg_exec = (
          sum(step_exec_times) / len(step_exec_times)
          if step_exec_times
          else 0.0
      )
      logging.info(
          "   [Step %d Thread Summary] %d parallel rollouts: avg_acquire=%.3fs,"
          " avg_exec=%.3fs",
          step,
          len(results),
          step_avg_acq,
          step_avg_exec,
      )
      print(
          f"   [Step {step} Thread Summary] {len(results)} parallel rollouts:"
          f" avg_acquire={step_avg_acq:.3f}s, avg_exec={step_avg_exec:.3f}s",
          flush=True,
      )

    logging.info(
        "   - Active warm pools after step %d: %s",
        step,
        getattr(fleet, "active_pools", {}),
    )

    step_duration = time.perf_counter() - step_start_time
    if (
        minimum_step_time_in_second > 0
        and step < max_steps - 1
        and step_duration < minimum_step_time_in_second
    ):
      sleep_secs = minimum_step_time_in_second - step_duration
      logging.info(
          "   [Step %d] Finished in %.2fs (< %.2fs"
          " minimum_step_time_in_second). Waiting %.2fs before processing the"
          " next step...",
          step,
          step_duration,
          minimum_step_time_in_second,
          sleep_secs,
      )
      print(
          f"   [Step {step}] Finished in {step_duration:.2f}s (<"
          f" {minimum_step_time_in_second:.2f}s). Waiting {sleep_secs:.2f}s"
          " before processing the next step...",
          flush=True,
      )
      time.sleep(sleep_secs)

  # 5. Teardown and cleanup
  logging.info("5. Tearing down iterator and cleaning up fleet...")
  iterator.close()
  if not is_mock:
    sandbox_utils.teardown_global_fleet()
  else:
    fleet.teardown()

  stats["pools_unwarmed"] = len(getattr(iterator, "unwarm_calls", [])) or len(
      getattr(fleet, "unwarm_calls", [])
  )
  stats["avg_acquire_time_sec"] = (
      sum(acquire_durations) / len(acquire_durations)
      if acquire_durations
      else 0.0
  )
  stats["avg_exec_time_sec"] = (
      sum(exec_durations) / len(exec_durations) if exec_durations else 0.0
  )
  logging.info(
      "   [OK] Timing: avg_acquire_time=%.3fs, avg_exec_time=%.3fs",
      stats["avg_acquire_time_sec"],
      stats["avg_exec_time_sec"],
  )
  print(
      "\n=== Performance Timers ==="
      f"\n  Average Sandbox Acquire Time: {stats['avg_acquire_time_sec']:.3f}s"
      f"\n  Average Command Exec Time:    {stats['avg_exec_time_sec']:.3f}s\n",
      flush=True,
  )
  logging.info(
      "   [OK] Teardown complete. Active pools: %s",
      getattr(fleet, "active_pools", {}),
  )
  logging.info("=== [DeepSWE E2E Test] Pipeline Run Complete: %s ===", stats)
  return stats


def str_to_bool(v: Any) -> bool:
  """Converts string representations of boolean values to bool."""
  if isinstance(v, bool):
    return v
  if str(v).lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif str(v).lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def main(argv: list[str]) -> None:
  parser = argparse.ArgumentParser(description="DeepSWE Sandbox K8s E2E Test.")
  parser.add_argument(
      "--dataset_name", type=str, default="R2E-Gym/R2E-Gym-Subset"
  )
  parser.add_argument("--dataset_split", type=str, default="train")
  parser.add_argument("--dataset_path", type=str, default="")
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--num_generations", type=int, default=2)
  parser.add_argument("--max_steps", type=int, default=3)
  parser.add_argument("--namespace", type=str, default="rl-tunix-swebench")
  parser.add_argument(
      "--scaffold",
      type=str,
      default="r2egym",
      choices=["r2egym", "sweagent", "openhands"],
      help="Scaffold harness to test ('r2egym', 'sweagent', 'openhands').",
  )
  parser.add_argument(
      "--dry_run", action="store_true", help="Run with mock fleet."
  )
  parser.add_argument(
      "--synthetic_dataset",
      action="store_true",
      help="Use synthetic samples.",
  )
  parser.add_argument(
      "--node_selector_key",
      type=str,
      default=os.environ.get(
          "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
      ),
      help="Node selector key for sandbox placement.",
  )
  parser.add_argument(
      "--node_selector_val",
      type=str,
      default=os.environ.get("NODE_SELECTOR_VAL", "sandbox-cpu-pool"),
      help="Node selector value (nodepool name) for sandbox placement.",
  )
  parser.add_argument(
      "--minimum_step_time_in_second",
      type=float,
      default=0.0,
      help=(
          "Minimum duration in seconds for each step. If a step finishes"
          " faster than this, sleeps the difference before processing the"
          " next step."
      ),
  )
  parser.add_argument(
      "--sampling_rate",
      type=float,
      default=1.0,
      help=(
          "Sampling rate in [0.0, 1.0] for randomly choosing batches to test"
          " sandbox acquire and execution (default: 1.0). Once a batch is"
          " selected, all generations in the group are tested."
      ),
  )
  parser.add_argument(
      "--shuffle",
      type=str_to_bool,
      nargs="?",
      const=True,
      default=True,
      help="Whether to shuffle the dataset (default: True).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for shuffling (default: 42).",
  )
  args, _ = parser.parse_known_args(argv[1:])

  if args.seed is not None:
    random.seed(args.seed)

  os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] [DeepSWEE2E] %(message)s",
      force=True,
  )

  dataset = None
  if not args.synthetic_dataset and not args.dry_run and deepswe is not None:
    try:
      logging.info(
          "Loading dataset %s (split=%s, shuffle=%s, seed=%d)...",
          args.dataset_name,
          args.dataset_split,
          args.shuffle,
          args.seed,
      )
      dataset = deepswe.load_deepswe_dataset(
          dataset_name=args.dataset_name,
          dataset_split=args.dataset_split,
          dataset_path=args.dataset_path or None,
          shuffle=args.shuffle,
          seed=args.seed,
      )
      logging.info("Loaded %d dataset samples.", len(dataset))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Dataset load note (%s), falling back to synthetic dataset.", e
      )
      dataset = None

  if dataset is None:
    logging.info("Creating synthetic dataset samples for E2E test...")
    dataset = create_synthetic_dataset(
        num_samples=max(6, args.max_steps * args.batch_size * 2)
    )
    if args.shuffle:
      random.Random(args.seed).shuffle(dataset)

  if args.dry_run:
    fleet = FakeFleet()
    is_mock = True
  else:
    node_sel = None
    if args.node_selector_key and args.node_selector_val:
      node_sel = {args.node_selector_key: args.node_selector_val}
    logging.info(
        "Initializing global fleet in namespace '%s' targeting node"
        " selector: %s (scaffold=%s)",
        args.namespace,
        node_sel,
        args.scaffold,
    )
    fleet = sandbox_utils.init_global_fleet(
        tasks=dataset,
        max_concurrency=128,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        namespace=args.namespace,
        scaffold=args.scaffold,
        node_selector=node_sel,
    )
    is_mock = False

  stats = run_pipeline_e2e(
      dataset=dataset,
      fleet=fleet,
      batch_size=args.batch_size,
      num_generations=args.num_generations,
      max_steps=args.max_steps,
      is_mock=is_mock,
      scaffold=args.scaffold,
      minimum_step_time_in_second=args.minimum_step_time_in_second,
      sampling_rate=args.sampling_rate,
  )

  expected_sandboxes = args.max_steps * args.batch_size * args.num_generations
  assert (
      stats["steps_executed"] == args.max_steps
  ), f"Executed {stats['steps_executed']} steps, expected {args.max_steps}"
  if args.sampling_rate >= 1.0:
    assert stats["sandboxes_acquired"] == expected_sandboxes, (
        f"Acquired {stats['sandboxes_acquired']} sandboxes, expected"
        f" {expected_sandboxes}"
    )
  else:
    assert stats["sandboxes_acquired"] <= expected_sandboxes, (
        f"Acquired {stats['sandboxes_acquired']} sandboxes, expected <="
        f" {expected_sandboxes}"
    )
  assert stats["sandboxes_acquired"] == stats["sandboxes_released"], (
      f"Acquired {stats['sandboxes_acquired']} but released"
      f" {stats['sandboxes_released']}!"
  )
  assert stats["pools_unwarmed"] > 0, "No warm pools were unwarmed!"
  print(
      "\nPerformance Summary:"
      f"\n  Average Sandbox Acquire Time: {stats['avg_acquire_time_sec']:.3f}s"
      f"\n  Average Command Exec Time:    {stats['avg_exec_time_sec']:.3f}s"
  )
  print("\nALL CHECKS PASSED: DeepSWE Agent Sandbox E2E Verified Successfully!")


class DeepSWESandboxE2ETest(absltest.TestCase):
  """Unit test case enabling test execution."""

  def test_e2e_pipeline_lifecycle_mock(self):
    batch_size = 1
    num_generations = 2
    max_steps = 3
    dataset = create_synthetic_dataset(num_samples=12)
    fleet = FakeFleet()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        scaffold="r2egym",
    )
    expected_sandboxes = max_steps * batch_size * num_generations
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertEqual(stats["sandboxes_acquired"], expected_sandboxes)
    self.assertEqual(stats["sandboxes_released"], expected_sandboxes)
    self.assertGreater(stats["pools_unwarmed"], 0)
    self.assertIn("avg_acquire_time_sec", stats)
    self.assertIn("avg_exec_time_sec", stats)
    self.assertEqual(fleet.active_pools, {})

  def test_e2e_pipeline_lifecycle_mock_openhands(self):
    batch_size = 1
    num_generations = 2
    max_steps = 3
    dataset = create_synthetic_dataset(num_samples=12)
    fleet = FakeFleet()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        scaffold="openhands",
    )
    expected_sandboxes = max_steps * batch_size * num_generations
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertEqual(stats["sandboxes_acquired"], expected_sandboxes)
    self.assertEqual(stats["sandboxes_released"], expected_sandboxes)
    self.assertGreater(stats["pools_unwarmed"], 0)
    self.assertIn("avg_acquire_time_sec", stats)
    self.assertIn("avg_exec_time_sec", stats)
    self.assertEqual(fleet.active_pools, {})

  def test_e2e_pipeline_minimum_step_time(self):
    batch_size = 1
    num_generations = 1
    max_steps = 2
    dataset = create_synthetic_dataset(num_samples=6)
    fleet = FakeFleet()
    t_start = time.perf_counter()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        minimum_step_time_in_second=0.1,
    )
    elapsed = time.perf_counter() - t_start
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertGreaterEqual(elapsed, 0.1)

  def test_e2e_pipeline_sampling_rate_zero(self):
    batch_size = 1
    num_generations = 2
    max_steps = 3
    dataset = create_synthetic_dataset(num_samples=6)
    fleet = FakeFleet()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        sampling_rate=0.0,
    )
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertEqual(stats["batches_sampled"], 0)
    self.assertEqual(stats["sandboxes_acquired"], 0)
    self.assertEqual(stats["sandboxes_released"], 0)
    self.assertGreater(stats["pools_unwarmed"], 0)


if __name__ == "__main__":
  if (
      "--dry_run" in sys.argv
      or "--batch_size" in sys.argv
      or "--run_as_job" in sys.argv
  ):
    main(sys.argv)
  else:
    absltest.main()
