#!/usr/bin/env python3
"""Exact forward/VJP and real-TPU lowering gate for Qwen3-4B matmul padding."""

from __future__ import annotations

import argparse
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
    "CANON_QWEN3_HIDDEN_SIZE": "2560",
    "CANON_QWEN3_INTERMEDIATE_SIZE": "9728",
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
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=("interpret", "tpu"), required=True)
  args = parser.parse_args()

  for name in _REQUIRED_ENV:
    os.environ[name] = "1"
  os.environ.update(_MODEL_ENV)

  import jax
  import jax.numpy as jnp
  import numpy as np
  from p22xi_padded_matmul import matmul as padded_matmul
  from p22xi_padded_matmul import padded_matmul_extents
  from p22xk_vjp_ops import canonical_matmul
  from p22xk_vjp_ops import matmul as promoted_matmul

  if args.mode == "tpu":
    devices = jax.devices()
    if len(devices) != 4 or any("TPU v5" not in d.device_kind for d in devices):
      raise AssertionError(f"expected four direct TPU v5 devices, got {devices}")
    # Match the first remote P44 matmul failure's semantic M exactly.  This
    # keeps the target-shaped grid in the real-TPU gate instead of proving
    # only that a single BM tile lowers.
    rows = 4096
    # Seven projection sites collapse to these five unique local MKN shapes:
    # q; k/v; o; gate/up; and down.  Exercise every shape so the one-host gate
    # detects a second Mosaic block-spec failure before another remote launch.
    cases = (
        (2560, 512, "q", 2560, 512),
        (2560, 128, "kv", 2560, 128),
        (512, 2560, "o", 512, 2560),
        (2560, 1216, "output", 2560, 1280),
        (1216, 2560, "contract", 1280, 2560),
    )
  else:
    rows = 3
    cases = (
        (128, 1216, "output", 128, 1280),
        (1216, 128, "contract", 1280, 128),
    )

  completed = 0
  for k, n, label, expected_kp, expected_np in cases:
    kp, npadded = padded_matmul_extents(
        k, n, block_k=128, block_n=128
    )
    expected_extents = (expected_kp, expected_np)
    if (kp, npadded) != expected_extents:
      raise AssertionError(
          f"{label} padding mismatch: {(kp, npadded)} != {expected_extents}"
      )

    x = jnp.asarray(_values((rows, k), modulus=5, divisor=64), jnp.bfloat16)
    y = jnp.asarray(_values((k, n), modulus=7, divisor=128), jnp.bfloat16)

    def forward(a, b):
      return padded_matmul(
          a,
          b,
          interpret=args.mode == "interpret",
          block_k=128,
          block_n=128,
      )

    forward_call = jax.jit(forward) if args.mode == "tpu" else forward
    actual = forward_call(x, y)
    expected = canonical_matmul(x, y)
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
      raise AssertionError(f"{label} padded matmul forward mismatch")

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
      raise AssertionError(f"{label} padded matmul custom-VJP mismatch")
    completed += 1
    print(
        "MATMUL_DIM_PADDING_CASE_PASS "
        f"mode={args.mode} label={label} shape={rows}x{k}x{n} "
        f"padded=128x{kp}x{npadded} forward_exact=1 vjp_exact=1",
        flush=True,
    )

  negatives = 0
  for k, n in ((1217, 128), (128, 1217)):
    try:
      padded_matmul_extents(k, n, block_k=128, block_n=128)
    except ValueError:
      negatives += 1
  if negatives != 2:
    raise AssertionError(f"matmul padding negatives={negatives}, expected 2")

  print(
      "MATMUL_DIM_PADDING_PASS "
      f"mode={args.mode} cases={completed}/{len(cases)} "
      "forward_exact=1 vjp_exact=1 "
      f"negatives={negatives}/2 devices={len(jax.devices())}",
      flush=True,
  )


if __name__ == "__main__":
  main()
