"""Minimal DeepSWE env probe.

Loads one R2E-Gym example, creates a single ``SWEEnv``, and calls ``reset()``
to isolate whether ``RepoEnv(...)`` / ``get_task_instruction()`` can complete.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Any

from datasets import load_dataset


def _setup_paths() -> None:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
  workspace_root = os.path.dirname(repo_root)
  candidate_roots = [
      script_dir,
      repo_root,
      workspace_root,
      os.path.join(repo_root, "tunix"),
      os.path.join(repo_root, "pathways-utils"),
      os.path.join(repo_root, "r2egym"),
      os.path.join(workspace_root, "tunix"),
      os.path.join(workspace_root, "pathways-utils"),
      os.path.join(workspace_root, "r2egym"),
  ]

  for root in candidate_roots:
    if root not in sys.path:
      sys.path.insert(0, root)


_setup_paths()

try:
  import pathwaysutils  # pytype: disable=import-error
except ImportError:
  pathwaysutils = None

try:
  import tunix  # pytype: disable=import-error  # noqa: F401
  import r2egym  # pytype: disable=import-error  # noqa: F401
  print("✅ tunix / r2egym import succeeded for debug_single_env")
except ImportError as exc:
  print(f"❌ debug_single_env import bootstrap failed: {exc}")

if pathwaysutils is not None and os.getenv("JAX_PLATFORMS", None) == "proxy":
  pathwaysutils.initialize()

from r2egym_runtime_patch import apply_repoenv_kubernetes_watch_patch
from swe_env import SWEEnv


def _transform_entry(entry: dict[str, Any]) -> dict[str, Any]:
  out = dict(entry)
  for k, v in out.items():
    if isinstance(v, list):
      out[k] = json.dumps(v)
  return out


def _make_timeout_handler(timeout_secs: int):
  def _handler(signum, frame):
    del signum, frame
    raise TimeoutError(f"Timed out after {timeout_secs}s")

  return _handler


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Create exactly one DeepSWE SWEEnv and call reset()."
  )
  parser.add_argument(
      "--dataset_name",
      type=str,
      default="R2E-Gym/R2E-Gym-Subset",
  )
  parser.add_argument("--split", type=str, default="train")
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument(
      "--cache_dir",
      type=str,
      default=os.path.join(os.getcwd(), "dataset_cache"),
  )
  parser.add_argument("--backend", type=str, default="kubernetes")
  parser.add_argument(
      "--node_selector_key",
      type=str,
      default="cloud.google.com/gke-nodepool",
  )
  parser.add_argument(
      "--node_selector_val",
      type=str,
      default="deepswe-cpu-pool",
  )
  parser.add_argument(
      "--scaffold",
      type=str,
      default="r2egym",
      choices=["r2egym", "sweagent"],
  )
  parser.add_argument("--step_timeout", type=int, default=90)
  parser.add_argument("--reward_timeout", type=int, default=300)
  parser.add_argument("--max_steps", type=int, default=1)
  parser.add_argument("--group_id", type=int, default=0)
  parser.add_argument("--pair_index", type=int, default=0)
  parser.add_argument("--reset_timeout_secs", type=int, default=300)
  parser.add_argument(
      "--stack_dump_secs",
      type=int,
      default=60,
      help="Dump Python thread stacks every N seconds while blocked.",
  )
  parser.add_argument(
      "--verbose",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  args = parser.parse_args()

  logging.basicConfig(
      stream=sys.stdout,
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      force=True,
  )

  os.environ["NODE_SELECTOR_KEY"] = args.node_selector_key
  os.environ["NODE_SELECTOR_VAL"] = args.node_selector_val
  logging.info(
      "Using Kubernetes node selector: %s=%s",
      os.environ["NODE_SELECTOR_KEY"],
      os.environ["NODE_SELECTOR_VAL"],
  )

  apply_repoenv_kubernetes_watch_patch()

  os.makedirs(args.cache_dir, exist_ok=True)

  logging.info(
      "Loading dataset=%s split=%s index=%d cache_dir=%s",
      args.dataset_name,
      args.split,
      args.index,
      args.cache_dir,
  )
  ds = load_dataset(
      args.dataset_name,
      split=args.split,
      cache_dir=args.cache_dir,
      trust_remote_code=True,
  )
  logging.info("Dataset size=%d", len(ds))

  raw_entry = ds[int(args.index)]
  entry = _transform_entry(raw_entry)
  problem = str(entry.get("problem_statement", ""))
  logging.info(
      "Selected example repo=%r docker_image=%r problem_preview=%r",
      entry.get("repo_name"),
      entry.get("docker_image"),
      problem.replace("\n", " ")[:200],
  )

  env = SWEEnv(
      entry,
      group_id=args.group_id,
      pair_index=args.pair_index,
      step_timeout=args.step_timeout,
      reward_timeout=args.reward_timeout,
      backend=args.backend,
      verbose=args.verbose,
      scaffold=args.scaffold,
      max_steps=args.max_steps,
  )

  if args.stack_dump_secs > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(
        args.stack_dump_secs, repeat=True, file=sys.stderr
    )

  stop_watchdog = threading.Event()

  def _watchdog():
    t0 = time.perf_counter()
    while not stop_watchdog.wait(30):
      logging.warning(
          "env.reset() still blocked after %.1fs",
          time.perf_counter() - t0,
      )

  watchdog_thread = threading.Thread(
      target=_watchdog, name="env-reset-watchdog", daemon=True
  )
  watchdog_thread.start()

  previous_handler = signal.getsignal(signal.SIGALRM)
  signal.signal(
      signal.SIGALRM, _make_timeout_handler(args.reset_timeout_secs)
  )
  signal.alarm(args.reset_timeout_secs)

  try:
    logging.info(
        "Calling env.reset() with timeout=%ss backend=%s scaffold=%s",
        args.reset_timeout_secs,
        args.backend,
        args.scaffold,
    )
    t0 = time.perf_counter()
    observation, info = env.reset()
    dt = time.perf_counter() - t0
    logging.info(
        "env.reset() completed in %.2fs obs_chars=%d info_keys=%s",
        dt,
        len(str(observation)),
        sorted(info.keys()),
    )
    print()
    print("=== OBSERVATION PREVIEW ===")
    print(str(observation)[:2000])
    print("=== END OBSERVATION PREVIEW ===")
  except TimeoutError:
    logging.exception("env.reset() timed out")
    raise
  finally:
    stop_watchdog.set()
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous_handler)
    if args.stack_dump_secs > 0:
      faulthandler.cancel_dump_traceback_later()
    try:
      env.close()
    except Exception:
      logging.exception("env.close() failed")


if __name__ == "__main__":
  main()
