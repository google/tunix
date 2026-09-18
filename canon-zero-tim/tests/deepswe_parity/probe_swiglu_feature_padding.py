#!/usr/bin/env python3
"""Exact Pallas-interpret and custom-VJP gate for model-pinned SwiGLU padding."""

from __future__ import annotations

import argparse
import os


_REQUIRED_ENV = (
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_SWIGLU",
    "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_SWIGLU_MPAD",
    "CANON_PALLAS_CANONICAL_VJP",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--feature", type=int, required=True)
  parser.add_argument("--padded-feature", type=int, required=True)
  parser.add_argument("--model", required=True)
  args = parser.parse_args()

  for name in _REQUIRED_ENV:
    os.environ[name] = "1"

  import jax
  import jax.numpy as jnp
  import numpy as np
  from p22xj_padded_swiglu import padded_feature_extent
  from p22xj_padded_swiglu import swiglu as padded_swiglu
  from p22xk_vjp_ops import canonical_swiglu
  from p22xk_vjp_ops import swiglu as promoted_swiglu

  if padded_feature_extent(args.feature) != args.padded_feature:
    raise AssertionError("model-pinned feature extent mismatch")
  try:
    padded_feature_extent(args.feature + 1)
  except ValueError:
    pass
  else:
    raise AssertionError("unregistered feature width was admitted")

  rows = 129
  count = rows * args.feature
  gate_values = (
      (np.arange(count, dtype=np.int32) % 29) - 14
  ).reshape(rows, args.feature) / 32
  up_values = (
      (np.arange(count, dtype=np.int32) % 31) - 15
  ).reshape(rows, args.feature) / 64
  gate = jnp.asarray(gate_values, jnp.bfloat16)
  up = jnp.asarray(up_values, jnp.bfloat16)

  def forward(g, u):
    return padded_swiglu(g, u, interpret=True)

  actual = forward(gate, up)
  expected = canonical_swiglu(gate, up)
  if not np.array_equal(np.asarray(actual), np.asarray(expected)):
    raise AssertionError("feature-padded SwiGLU forward mismatch")

  cotangent = jnp.asarray(
      np.where(np.arange(count).reshape(rows, args.feature) % 3, 1, -1),
      jnp.bfloat16,
  )
  _, promoted_pullback = jax.vjp(
      lambda g, u: promoted_swiglu(g, u, forward=forward), gate, up
  )
  _, canonical_pullback = jax.vjp(canonical_swiglu, gate, up)
  actual_grads = promoted_pullback(cotangent)
  expected_grads = canonical_pullback(cotangent)
  if not all(
      np.array_equal(np.asarray(actual_grad), np.asarray(expected_grad))
      for actual_grad, expected_grad in zip(actual_grads, expected_grads)
  ):
    raise AssertionError("feature-padded SwiGLU custom-VJP mismatch")

  row_extent = 256
  print(
      "SWIGLU_FEATURE_PADDING_INTERPRET_PASS "
      f"model={args.model} shape={rows}x{args.feature} "
      f"padded={row_extent}x{args.padded_feature} "
      "forward_exact=1 vjp_exact=1 negative=1"
  )


if __name__ == "__main__":
  main()
