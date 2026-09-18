#!/usr/bin/env python3
"""Install and attest the pinned FrozenLake runtime from local wheels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


EXPECTED_WHEELS = {
    "gymnasium-1.3.0-py3-none-any.whl": (
        "6b8c159a8540dcbcb221722d7efda24d78ebbcbc3bd2ea1c2611aa2a34471fc2"
    ),
    "farama_notifications-0.0.6-py3-none-any.whl": (
        "f84839188efa1ce5bb361c2a84881b2dc2c0d0d7fb661ff00421820170930935"
    ),
}

PROTECTED_DISTRIBUTIONS = (
    "numpy",
    "jax",
    "jaxlib",
    "libtpu",
    "flax",
    "optax",
    "qwix",
    "vllm-tpu",
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _versions() -> dict[str, str | None]:
  versions = {}
  for name in PROTECTED_DISTRIBUTIONS:
    try:
      versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      versions[name] = None
  return versions


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--gymnasium-wheel", required=True, type=Path)
  parser.add_argument("--farama-wheel", required=True, type=Path)
  parser.add_argument("--report", required=True, type=Path)
  args = parser.parse_args()
  if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"expected Python 3.12, got {sys.version}")
  if args.report.exists():
    raise FileExistsError(f"refusing to overwrite {args.report}")

  wheels = (args.gymnasium_wheel, args.farama_wheel)
  evidence = []
  for path in wheels:
    expected = EXPECTED_WHEELS.get(path.name)
    if expected is None:
      raise RuntimeError(f"unregistered wheel: {path.name}")
    actual = _sha256(path)
    if actual != expected:
      raise RuntimeError(f"wheel SHA-256 mismatch: {path.name}")
    evidence.append({"name": path.name, "sha256": actual})

  before = _versions()
  subprocess.check_call([
      sys.executable,
      "-m",
      "pip",
      "install",
      "--disable-pip-version-check",
      "--no-deps",
      str(args.farama_wheel),
      str(args.gymnasium_wheel),
  ])
  after = _versions()
  if after != before:
    raise RuntimeError(f"protected stack changed: before={before} after={after}")

  import gymnasium as gym

  if gym.__version__ != "1.3.0":
    raise RuntimeError(f"unexpected gymnasium version: {gym.__version__}")
  env = gym.make("FrozenLake-v1", is_slippery=False)
  observation, _ = env.reset(seed=0)
  next_observation, reward, terminated, truncated, _ = env.step(0)
  env.close()
  if (
      int(observation) != 0
      or int(next_observation) != 0
      or float(reward) != 0.0
      or bool(terminated)
      or bool(truncated)
  ):
    raise RuntimeError("FrozenLake deterministic runtime smoke changed")

  report = {
      "verdict": "PASS",
      "python": ".".join(map(str, sys.version_info[:3])),
      "gymnasium": gym.__version__,
      "wheels": evidence,
      "protected_stack_before": before,
      "protected_stack_after": after,
  }
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print("[P41.FROZENLAKE] RUNTIME_PASS " + json.dumps(report, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
