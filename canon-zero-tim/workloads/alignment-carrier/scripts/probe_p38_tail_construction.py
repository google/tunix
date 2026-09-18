#!/usr/bin/env python3
"""Exercise one canonical log-softmax callable in two outer TPU programs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp

from tunix.rl import canonical_logsoftmax


def _different_elements(left, right):
  left_bits = jax.lax.bitcast_convert_type(left, jnp.uint32)
  right_bits = jax.lax.bitcast_convert_type(right, jnp.uint32)
  return jnp.count_nonzero(left_bits != right_bits)


def _flip_one_bit(value):
  bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
  bits = bits.at[0, 0].set(bits[0, 0] ^ jnp.uint32(1))
  return jax.lax.bitcast_convert_type(bits, jnp.float32)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  devices = jax.devices()
  if len(devices) != 4 or jax.default_backend() != "tpu":
    raise RuntimeError("P38 tail construction requires exactly four TPU devices")
  if os.environ.get(canonical_logsoftmax.ENV) != "1":
    raise RuntimeError(f"{canonical_logsoftmax.ENV}=1 is required")

  rows = canonical_logsoftmax.PRODUCTION_M
  vocab = canonical_logsoftmax.PRODUCTION_V
  columns = jnp.arange(vocab, dtype=jnp.int32)[None, :]
  row_ids = jnp.arange(rows, dtype=jnp.int32)[:, None]
  logits = (
      jnp.mod(columns * jnp.int32(13) + row_ids * jnp.int32(17), 1021)
      .astype(jnp.float32)
      / jnp.float32(97.0)
      - jnp.float32(5.0)
  )
  logits.block_until_ready()

  shared_tail = canonical_logsoftmax.log_softmax
  decode_tail = shared_tail
  prefill_tail = shared_tail
  if decode_tail is not prefill_tail:
    raise RuntimeError("tail aliases are not the same Python function object")

  @jax.jit
  def decode_outer(value):
    return decode_tail(value)

  @jax.jit
  def prefill_outer(value, metadata):
    return prefill_tail(value), jnp.sum(metadata, dtype=jnp.int32)

  metadata = jnp.arange(rows, dtype=jnp.int32)
  decode_output = decode_outer(logits)
  decode_output.block_until_ready()
  prefill_output, metadata_sum = prefill_outer(logits, metadata)
  prefill_output.block_until_ready()
  metadata_sum.block_until_ready()

  compare = jax.jit(_different_elements)
  differing = int(compare(decode_output, prefill_output))
  negative_output = jax.jit(_flip_one_bit)(prefill_output)
  negative_differing = int(compare(decode_output, negative_output))
  verdict = (
      "PASS_CONSTRUCTION_ONLY"
      if differing == 0 and negative_differing == 1
      else "FAIL"
  )
  report = {
      "schema_version": 1,
      "verdict": verdict,
      "claim_scope": "direct-attached-tail-construction-only",
      "shape": [rows, vocab],
      "dtype": str(decode_output.dtype),
      "device_count": len(devices),
      "backend": jax.default_backend(),
      "same_python_callable": decode_tail is prefill_tail,
      "outer_programs": 2,
      "differing_elements": differing,
      "total_elements": int(decode_output.size),
      "negative_control_differing_elements": negative_differing,
      "metadata_sum": int(metadata_sum),
  }
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"[P38.TAIL] {json.dumps(report, sort_keys=True)}", flush=True)
  if verdict != "PASS_CONSTRUCTION_ONLY":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
