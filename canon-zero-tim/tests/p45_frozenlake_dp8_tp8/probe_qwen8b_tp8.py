#!/usr/bin/env python3
"""Seven-site shape plus exact canonical forward/VJP gate for Qwen3-8B TP8."""

from __future__ import annotations

import os


_REQUIRED_ENV = (
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_SWIGLU",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_SWIGLU_MPAD",
    "CANON_PALLAS_CANONICAL_VJP",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
)

_MODEL_ENV = {
    "CANON_QWEN3_HIDDEN_SIZE": "4096",
    "CANON_QWEN3_INTERMEDIATE_SIZE": "12288",
    "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
    "CANON_QWEN3_NUM_KV_HEADS": "8",
    "CANON_QWEN3_HEAD_DIM": "128",
    "CANON_QWEN3_TP_SIZE": "8",
}


def _values(shape, *, modulus: int, divisor: int):
  import numpy as np

  count = int(np.prod(shape))
  return (
      (np.arange(count, dtype=np.int32) % modulus) - modulus // 2
  ).reshape(shape) / divisor


def main() -> None:
  for name in _REQUIRED_ENV:
    os.environ[name] = "1"
  os.environ.update(_MODEL_ENV)

  import jax
  import jax.numpy as jnp
  import numpy as np
  import p22xf_contract as model
  from p22xi_padded_matmul import matmul as padded_matmul
  from p22xi_padded_matmul import padded_matmul_extents
  from p22xk_vjp_ops import canonical_matmul
  from p22xk_vjp_ops import matmul as promoted_matmul

  model.preflight(require_enabled=True)
  model.validate_manifest(model.SITES)

  # Abstract forward/VJP traces cover every exact production-local K/N shape
  # without allocating seven full gradient matrices on the CPU gate.
  completed = 0
  for site in model.SITES:
    k, n = site.k_local, site.n_local
    if padded_matmul_extents(k, n, block_k=128, block_n=128) != (k, n):
      raise AssertionError(f"{site.family} unexpectedly requires padding")
    x = jax.ShapeDtypeStruct((1, k), jnp.bfloat16)
    y = jax.ShapeDtypeStruct((k, n), jnp.bfloat16)
    cotangent = jax.ShapeDtypeStruct((1, n), jnp.bfloat16)
    out = jax.eval_shape(canonical_matmul, x, y)

    def pullback(a, b, cot):
      _, pb = jax.vjp(canonical_matmul, a, b)
      return pb(cot)

    dx, dy = jax.eval_shape(pullback, x, y, cotangent)
    if out.shape != (1, n) or dx.shape != (1, k) or dy.shape != (k, n):
      raise AssertionError(
          f"{site.family} abstract forward/VJP shapes invalid: "
          f"{out.shape}/{dx.shape}/{dy.shape}"
      )
    completed += 1
    print(
        "P45_QWEN8B_TP8_SITE_PASS "
        f"site={site.family} local_shape=1x{k}x{n} "
        "padding=none forward_shape=1 vjp_shapes=1",
        flush=True,
    )

  # One exact Pallas-interpret tile executes the same production wrapper and
  # custom VJP instead of proving only abstract shape admission.
  rows = k = n = 128
  x = jnp.asarray(_values((rows, k), modulus=5, divisor=64), jnp.bfloat16)
  y = jnp.asarray(_values((k, n), modulus=7, divisor=128), jnp.bfloat16)

  def forward(a, b):
    return padded_matmul(
        a, b, interpret=True, block_k=128, block_n=128
    )

  actual = forward(x, y)
  expected = canonical_matmul(x, y)
  if not np.array_equal(np.asarray(actual), np.asarray(expected)):
    raise AssertionError("Qwen3-8B TP8 Pallas forward differs from canonical")

  cotangent = jnp.asarray(
      _values((rows, n), modulus=3, divisor=32), jnp.bfloat16
  )
  _, actual_pullback = jax.vjp(
      lambda a, b: promoted_matmul(a, b, forward=forward), x, y
  )
  _, expected_pullback = jax.vjp(canonical_matmul, x, y)
  actual_grads = actual_pullback(cotangent)
  expected_grads = expected_pullback(cotangent)
  if not all(
      np.array_equal(np.asarray(actual_grad), np.asarray(expected_grad))
      for actual_grad, expected_grad in zip(actual_grads, expected_grads)
  ):
    raise AssertionError("Qwen3-8B TP8 canonical custom VJP differs")

  print(
      "P45_QWEN8B_TP8_FORWARD_VJP_PASS "
      "pallas_interpret=1 shape=128x128x128 forward_exact=1 vjp_exact=1",
      flush=True,
  )
  print(
      "P45_QWEN8B_TP8_PROBE_PASS "
      f"sites={completed}/7 padding=none tp=8",
      flush=True,
  )


if __name__ == "__main__":
  main()
