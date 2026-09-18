#!/usr/bin/env python3
"""Rehearse the P38 KV fingerprint on one four-chip v5p host.

This validates the observer primitive and its TP4 sharding. It does not
reproduce the production carrier or admit a 64-chip launch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "src/engine_shims/p38_kv_fingerprint.py"
SPEC = importlib.util.spec_from_file_location("p38_kv_fingerprint", MODULE)
assert SPEC is not None and SPEC.loader is not None
fingerprint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fingerprint)


def _tree_bytes_digest(tree) -> str:
  digest = hashlib.sha256()
  for leaf in jax.tree.leaves(jax.device_get(tree)):
    array = np.ascontiguousarray(np.asarray(leaf))
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.view(np.uint8))
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--layers", type=int, default=36)
  parser.add_argument("--pages", type=int, default=12)
  parser.add_argument("--selected-pages", type=int, default=9)
  parser.add_argument("--page-size", type=int, default=256)
  parser.add_argument("--kv-heads", type=int, default=8)
  parser.add_argument("--head-dim", type=int, default=128)
  parser.add_argument("--output")
  args = parser.parse_args()

  devices = jax.devices()
  if len(devices) != 4 or any(device.platform != "tpu" for device in devices):
    raise RuntimeError(
        f"P38 one-host rehearsal requires exactly four TPU devices: {devices}"
    )
  if args.kv_heads % len(devices):
    raise RuntimeError("P38 KV heads must be divisible by TP4")
  if not 0 < args.selected_pages <= args.pages:
    raise RuntimeError("P38 selected page count is out of range")

  mesh = Mesh(np.asarray(devices), ("model",))
  sharding = NamedSharding(mesh, P(None, None, "model", None, None))
  cache_shape = (
      args.pages,
      args.page_size,
      args.kv_heads,
      2,
      args.head_dim,
  )
  host_cache = np.zeros(cache_shape, dtype=jnp.bfloat16)
  caches = tuple(
      jax.device_put(
          (host_cache + np.float32(layer % 7)).astype(jnp.bfloat16),
          sharding,
      )
      for layer in range(args.layers)
  )
  physical_pages = np.arange(args.selected_pages, dtype=np.int32)
  global_pages = fingerprint.global_page_indices(
      cache_shape, 1, 0, physical_pages
  )
  valid_tokens = np.full(
      args.selected_pages, args.page_size, dtype=np.int32
  )
  valid_tokens[-1] = max(1, args.page_size // 2)
  fingerprint.validate_kv_fingerprint_contract(
      (args.selected_pages, *cache_shape[1:]),
      jnp.bfloat16,
      valid_tokens,
  )
  read_bytes = fingerprint.estimate_fingerprint_read_bytes(
      [cache_shape] * args.layers, args.selected_pages
  )

  observer = jax.jit(fingerprint.fingerprint_kv_cache_layer_prefixes)
  endpoint = jax.jit(lambda value: value * jnp.float32(1.25) - 0.5)
  endpoint_input = jax.device_put(np.linspace(-1, 1, 4096, dtype=np.float32))
  endpoint_before = jax.device_get(endpoint(endpoint_input))

  compile_start = time.monotonic()
  first = observer(
      caches, jnp.asarray(global_pages)
  )
  jax.block_until_ready(first)
  compile_seconds = time.monotonic() - compile_start

  steady_start = time.monotonic()
  repeat = observer(
      caches, jnp.asarray(global_pages)
  )
  jax.block_until_ready(repeat)
  steady_seconds = time.monotonic() - steady_start
  endpoint_after = jax.device_get(endpoint(endpoint_input))

  transfer_start = time.monotonic()
  repeat_host = jax.device_get(repeat)
  host_transfer_seconds = time.monotonic() - transfer_start
  output_bytes = sum(
      int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(repeat_host)
  )
  base_digest = _tree_bytes_digest(first)
  repeat_digest = _tree_bytes_digest(repeat_host)
  endpoint_exact = np.array_equal(endpoint_before, endpoint_after)

  # Use a normal non-zero BF16 value. Flipping +0's low bit creates a
  # subnormal that TPU arithmetic may flush before the observer sees it,
  # which is not a valid end-to-end negative control.
  poisoned_layer = args.layers - 2
  bits = jax.lax.bitcast_convert_type(caches[poisoned_layer], jnp.uint16)
  poisoned_bits = bits.at[0, 0, 0, 0, 0].set(bits[0, 0, 0, 0, 0] ^ 1)
  poisoned_cache = jax.lax.bitcast_convert_type(poisoned_bits, jnp.bfloat16)
  poisoned_caches = list(caches)
  poisoned_caches[poisoned_layer] = poisoned_cache
  negative = observer(
      tuple(poisoned_caches),
      jnp.asarray(global_pages),
  )
  jax.block_until_ready(negative)
  negative_digest = _tree_bytes_digest(negative)

  result = {
      "schema": "p38-kv-prefix-table-onehost-v1",
      "status": "PASS" if (
          base_digest == repeat_digest
          and endpoint_exact
          and negative_digest != base_digest
      ) else "FAIL",
      "scope": "observer-primitive-not-production-carrier",
      "devices": [str(device) for device in devices],
      "cache_shape": list(cache_shape),
      "layers": args.layers,
      "selected_pages": args.selected_pages,
      "valid_tokens": valid_tokens.tolist(),
      "sharding": str(sharding),
      "read_bytes": read_bytes,
      "output_bytes": output_bytes,
      "compile_seconds": compile_seconds,
      "steady_seconds": steady_seconds,
      "host_transfer_seconds": host_transfer_seconds,
      "repeat_exact": base_digest == repeat_digest,
      "endpoint_exact": bool(endpoint_exact),
      "negative_control_detected": negative_digest != base_digest,
      "fingerprint_sha256": base_digest,
      "repeat_sha256": repeat_digest,
      "negative_sha256": negative_digest,
      "claim_ceiling": [
          "This rehearses the observer primitive and TP4 sharding only.",
          "It does not reproduce the production A-B carrier.",
          "A compact fingerprint is not a cryptographic content proof.",
      ],
  }
  encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.output:
    Path(args.output).write_text(encoded, encoding="utf-8")
  print(encoded, end="")
  if result["status"] != "PASS":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
