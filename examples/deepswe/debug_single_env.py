"""Minimal DeepSWE env probe.

Loads one R2E-Gym example, creates a single ``SWEEnv``, and calls ``reset()``
to isolate whether ``RepoEnv(...)`` / ``get_task_instruction()`` can complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Any

from datasets import load_dataset


def _setup_paths() -> None:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  workdir = os.getcwd()
  tunix_root = os.path.join(workdir, "tunix")
  pathways_root = os.path.join(workdir, "pathways-utils")
  r2egym_root = os.path.join(workdir, "r2egym")

  for root in [script_dir, workdir, tunix_root, pathways_root, r2egym_root]:
    if root not in sys.path:
      sys.path.insert(0, root)


_setup_paths()

try:
  import pathwaysutils  # pytype: disable=import-error
except ImportError:
  pathwaysutils = None

if pathwaysutils is not None and os.getenv("JAX_PLATFORMS", None) == "proxy":
  pathwaysutils.initialize()

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
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous_handler)
    try:
      env.close()
    except Exception:
      logging.exception("env.close() failed")


if __name__ == "__main__":
  main()
