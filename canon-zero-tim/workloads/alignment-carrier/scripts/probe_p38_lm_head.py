#!/usr/bin/env python3
"""Screen Qwen3-8B lm_head M=16/M=256 invariance on one v5p host.

This is an operator construction gate.  It uses the real checkpoint lm_head
weight and the same dot-algorithm preset as CANON_MM_ALGO, but it does not
reproduce the Pathways decode/prefill executable envelopes.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np


DECODE_M = 16
PREFILL_M = 256
HIDDEN = 4096
VOCAB = 151936
PRESET_NAME = "BF16_BF16_F32"


def classify(rows: list[dict[str, Any]], negative_differing: int) -> str:
  if negative_differing != 1:
    return "FAIL_NEGATIVE_CONTROL"
  default_red = any(row["default_differing_elements"] for row in rows)
  algorithm_red = any(row["algorithm_differing_elements"] for row in rows)
  if algorithm_red:
    return "ALGORITHM_NOT_SUFFICIENT"
  if default_red:
    return "ALGORITHM_ELIMINATES_OPERATOR_DRIFT"
  return "BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE"


def _bf16_numpy(tensor: Any) -> np.ndarray:
  import ml_dtypes
  import torch

  bits = tensor.contiguous().view(torch.uint16).numpy()
  return bits.view(ml_dtypes.bfloat16)


def _load_weight(
    model: Path,
    sharding: NamedSharding,
    hidden_size: int = HIDDEN,
    weight_key: str = "lm_head.weight",
) -> jax.Array:
  from safetensors import safe_open

  index = json.loads((model / "model.safetensors.index.json").read_text())
  shard_name = index["weight_map"].get(weight_key)
  if not shard_name:
    raise RuntimeError(f"checkpoint has no {weight_key}")
  shard_path = model / shard_name

  with safe_open(shard_path, framework="pt", device="cpu") as handle:
    source_shape = tuple(handle.get_slice(weight_key).get_shape())
  hidden_size = int(hidden_size)
  if source_shape != (VOCAB, hidden_size):
    raise RuntimeError(
        "unexpected lm_head.weight shape: "
        f"{source_shape} != {(VOCAB, hidden_size)}"
    )

  def callback(index: tuple[slice, ...]) -> np.ndarray:
    hidden_slice, vocab_slice = index
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
      source = handle.get_slice(weight_key)[vocab_slice, hidden_slice]
    # Safetensors stores [V,D]; JaxLmHead consumes [D,V].
    return _bf16_numpy(source).T

  weight = jax.make_array_from_callback(
      (hidden_size, VOCAB), sharding, callback
  )
  weight.block_until_ready()
  if weight.dtype != jnp.bfloat16:
    raise RuntimeError(f"unexpected lm_head dtype: {weight.dtype}")
  return weight


def _different_elements(left: jax.Array, right: jax.Array) -> jax.Array:
  left_bits = jax.lax.bitcast_convert_type(left, jnp.uint16)
  right_bits = jax.lax.bitcast_convert_type(right, jnp.uint16)
  return jnp.count_nonzero(left_bits != right_bits)


def _max_abs(left: jax.Array, right: jax.Array) -> jax.Array:
  return jnp.max(jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32)))


def _flip_one_bit(value: jax.Array) -> jax.Array:
  bits = jax.lax.bitcast_convert_type(value, jnp.uint16)
  bits = bits.at[0, 0].set(bits[0, 0] ^ jnp.uint16(1))
  return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)


def _lowering_receipt(lowered: Any) -> dict[str, Any]:
  text = lowered.as_text()
  dot_lines = [
      line.strip() for line in text.splitlines()
      if "dot_general" in line or "algorithm =" in line
  ]
  return {
      "stablehlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
      "dot_lines": dot_lines,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--seeds", type=int, default=4)
  args = parser.parse_args()
  if args.seeds < 1:
    raise RuntimeError("--seeds must be positive")

  devices = jax.devices()
  if jax.default_backend() != "tpu" or len(devices) != 4:
    raise RuntimeError("P38 lm_head probe requires exactly four TPU devices")
  mesh = Mesh(np.asarray(devices), ("model",))
  replicated = NamedSharding(mesh, P(None, None))
  vocab_sharded = NamedSharding(mesh, P(None, "model"))
  weight = _load_weight(args.model, vocab_sharded)

  @partial(
      jax.jit,
      in_shardings=(replicated, vocab_sharded),
      out_shardings=vocab_sharded,
  )
  def default_lm_head(hidden: jax.Array, kernel: jax.Array) -> jax.Array:
    return jnp.einsum("TD,DV->TV", hidden, kernel)

  preset = getattr(jax.lax.DotAlgorithmPreset, PRESET_NAME)

  @partial(
      jax.jit,
      in_shardings=(replicated, vocab_sharded),
      out_shardings=vocab_sharded,
  )
  def algorithm_lm_head(hidden: jax.Array, kernel: jax.Array) -> jax.Array:
    return jnp.einsum("TD,DV->TV", hidden, kernel, precision=preset)

  compare = jax.jit(_different_elements)
  max_abs = jax.jit(_max_abs)
  rows: list[dict[str, Any]] = []
  lowerings: dict[str, dict[str, Any]] = {}
  last_default_prefill = None
  for seed in range(args.seeds):
    key = jax.random.PRNGKey(seed + 1701)
    hidden = jax.random.normal(key, (PREFILL_M, HIDDEN), dtype=jnp.float32)
    hidden = jax.device_put(hidden.astype(jnp.bfloat16), replicated)
    decode_hidden = hidden[:DECODE_M]

    if not lowerings:
      lowerings = {
          "default_decode": _lowering_receipt(
              default_lm_head.lower(decode_hidden, weight)
          ),
          "default_prefill": _lowering_receipt(
              default_lm_head.lower(hidden, weight)
          ),
          "algorithm_decode": _lowering_receipt(
              algorithm_lm_head.lower(decode_hidden, weight)
          ),
          "algorithm_prefill": _lowering_receipt(
              algorithm_lm_head.lower(hidden, weight)
          ),
      }

    default_decode = default_lm_head(decode_hidden, weight)
    default_prefill = default_lm_head(hidden, weight)
    algorithm_decode = algorithm_lm_head(decode_hidden, weight)
    algorithm_prefill = algorithm_lm_head(hidden, weight)
    for value in (
        default_decode, default_prefill, algorithm_decode, algorithm_prefill
    ):
      value.block_until_ready()

    default_prefill_rows = default_prefill[:DECODE_M]
    algorithm_prefill_rows = algorithm_prefill[:DECODE_M]
    row = {
        "seed": seed,
        "default_differing_elements": int(
            compare(default_decode, default_prefill_rows)
        ),
        "default_max_abs": float(
            max_abs(default_decode, default_prefill_rows)
        ),
        "algorithm_differing_elements": int(
            compare(algorithm_decode, algorithm_prefill_rows)
        ),
        "algorithm_max_abs": float(
            max_abs(algorithm_decode, algorithm_prefill_rows)
        ),
        "decode_intervention_differing_elements": int(
            compare(default_decode, algorithm_decode)
        ),
        "prefill_intervention_differing_elements": int(
            compare(default_prefill_rows, algorithm_prefill_rows)
        ),
    }
    rows.append(row)
    print(f"[P38.LM_HEAD] seed={seed} {json.dumps(row, sort_keys=True)}", flush=True)
    last_default_prefill = default_prefill_rows

  assert last_default_prefill is not None
  negative = jax.jit(_flip_one_bit)(last_default_prefill)
  negative.block_until_ready()
  negative_differing = int(compare(last_default_prefill, negative))
  verdict = classify(rows, negative_differing)
  report = {
      "schema_version": 1,
      "verdict": verdict,
      "claim_scope": "onehost-real-weight-operator-screen-only",
      "backend": jax.default_backend(),
      "device_count": len(devices),
      "decode_shape": [DECODE_M, HIDDEN],
      "prefill_shape": [PREFILL_M, HIDDEN],
      "weight_shape": [HIDDEN, VOCAB],
      "output_dtype": str(last_default_prefill.dtype),
      "weight_dtype": str(weight.dtype),
      "weight_sharding": str(weight.sharding),
      "algorithm_preset": PRESET_NAME,
      "negative_control_differing_elements": negative_differing,
      "lowerings": lowerings,
      "seeds": rows,
  }
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"[P38.LM_HEAD] {json.dumps(report, sort_keys=True)}", flush=True)
  if verdict in {"FAIL_NEGATIVE_CONTROL", "ALGORITHM_NOT_SUFFICIENT"}:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
