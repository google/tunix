#!/usr/bin/env python3
"""Runs one exact Pallas interpret case with the Qwen3-32B 128 tile."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "canon-zero-tim/src/engine_shims/p22_pallas_matmul.py"
SPEC = importlib.util.spec_from_file_location("p34_pallas_matmul", PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import canonical Pallas matmul")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def main() -> None:
  x = (np.arange(128 * 128, dtype=np.int32).reshape(128, 128) % 5 - 2) / 16
  y = (np.arange(128 * 128, dtype=np.int32).reshape(128, 128) % 7 - 3) / 16
  actual = np.asarray(module.matmul(
      jnp.asarray(x, jnp.bfloat16),
      jnp.asarray(y, jnp.bfloat16),
      interpret=True,
      block_m=128,
      block_n=128,
      block_k=128,
  ))
  expected = np.asarray(jnp.asarray(
      np.asarray(x, np.float64) @ np.asarray(y, np.float64),
      dtype=jnp.bfloat16,
  ))
  if not np.array_equal(actual, expected):
    raise AssertionError("P34 Pallas 128 known-answer mismatch")
  print("P34_PALLAS_128_INTERPRET_PASS shape=128x128x128 exact=1")


if __name__ == "__main__":
  main()
