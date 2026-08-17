"""Bounded P38 terminal discriminator primitives and durable records.

The device primitive deliberately observes already-materialized logits.  It
does not own lm_head, sampling transforms, or production logprob evaluation.
The returned integer summaries use only exact bitwise/modular operations;
floating block statistics are diagnostic values used to split an lm_head
input/output difference from a vocabulary-reduction difference.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np


P38_TERMINAL_BLOCK_SIZE = 256
P38_TERMINAL_ROW_BUCKET = 4
P38_TERMINAL_SIGNATURE_FIELDS = (
    "xor",
    "sum",
    "weighted_sum",
    "first",
    "middle",
    "last",
)


def fingerprint_terminal_rows(
    raw_rows: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Return bounded block signatures and reduction checkpoints.

  Shapes are ``[selected_rows, vocab_blocks, ...]`` except for the final two
  row scalars.  The vocabulary is padded only inside this observer.  Padding
  values are excluded from floating reductions and are zero in bit summaries.
  """
  if raw_rows.ndim != 2 or raw_rows.shape[0] != P38_TERMINAL_ROW_BUCKET:
    raise ValueError(
        "P38 terminal rows must use the fixed shared observer bucket: "
        f"expected={P38_TERMINAL_ROW_BUCKET} actual={raw_rows.shape}")
  rows = raw_rows.astype(jnp.float32)
  vocab = int(rows.shape[1])
  pad = (-vocab) % P38_TERMINAL_BLOCK_SIZE
  bit_rows = jax.lax.bitcast_convert_type(rows, jnp.uint32)
  if pad:
    bit_rows = jnp.pad(bit_rows, ((0, 0), (0, pad)), constant_values=0)
    float_rows = jnp.pad(
        rows, ((0, 0), (0, pad)), constant_values=-jnp.inf)
  else:
    float_rows = rows
  blocks = bit_rows.reshape(
      bit_rows.shape[0], -1, P38_TERMINAL_BLOCK_SIZE)
  float_blocks = float_rows.reshape(
      float_rows.shape[0], -1, P38_TERMINAL_BLOCK_SIZE)
  weights = jnp.arange(
      1, P38_TERMINAL_BLOCK_SIZE + 1, dtype=jnp.uint32)
  signatures = jnp.stack((
      jnp.bitwise_xor.reduce(blocks, axis=-1),
      jnp.sum(blocks, axis=-1, dtype=jnp.uint32),
      jnp.sum(blocks * weights, axis=-1, dtype=jnp.uint32),
      blocks[..., 0],
      blocks[..., P38_TERMINAL_BLOCK_SIZE // 2],
      blocks[..., -1],
  ), axis=-1)
  row_max = jnp.max(rows, axis=-1)
  block_max = jnp.max(float_blocks, axis=-1)
  block_exp_sum = jnp.sum(
      jnp.exp(float_blocks - row_max[:, None, None]),
      axis=-1,
      dtype=jnp.float32,
  )
  observer_log_normalizer = row_max + jnp.log(
      jnp.sum(block_exp_sum, axis=-1, dtype=jnp.float32))
  return (
      signatures,
      block_max,
      block_exp_sum,
      row_max,
      observer_log_normalizer,
  )


def fingerprint_terminal_pair(
    raw_rows: jax.Array,
    processed_rows: jax.Array,
) -> tuple[jax.Array, ...]:
  """Observe raw and processed rows inside one shared fixed-shape program."""
  return (
      *fingerprint_terminal_rows(raw_rows),
      *fingerprint_terminal_rows(processed_rows),
  )


def write_terminal_record(
    directory: str,
    state: dict[str, int],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping,
    max_bytes: int,
) -> tuple[int, str]:
  """Persist one exclusive, bounded, and self-checking terminal record."""
  record_index = int(state.get("records", 0))
  current_bytes = int(state.get("bytes", 0))
  root = Path(directory)
  root.mkdir(parents=True, exist_ok=True)
  base = root / f"p38_terminal_{record_index:06d}"
  npz_path = Path(str(base) + ".npz")
  json_path = Path(str(base) + ".json")
  if npz_path.exists() or json_path.exists():
    raise RuntimeError(f"P38 terminal evidence collision at {base}")

  payload = io.BytesIO()
  np.savez(payload, **arrays)
  npz_bytes = payload.getvalue()
  npz_sha256 = hashlib.sha256(npz_bytes).hexdigest()
  record = {
      **metadata,
      "array_keys": sorted(arrays),
      "npz_sha256": npz_sha256,
      "record_index": record_index,
      "schema": "p38-terminal-discriminator-v1",
      "signature_fields": P38_TERMINAL_SIGNATURE_FIELDS,
      "vocab_block_size": P38_TERMINAL_BLOCK_SIZE,
  }
  json_bytes = (
      json.dumps(record, sort_keys=True, indent=2, default=str) + "\n"
  ).encode()
  new_bytes = len(npz_bytes) + len(json_bytes)
  if current_bytes + new_bytes > int(max_bytes):
    raise RuntimeError("P38 terminal evidence exceeded its output byte bound")

  for path, content in ((npz_path, npz_bytes), (json_path, json_bytes)):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
      if os.write(descriptor, content) != len(content):
        raise RuntimeError(f"P38 terminal write was incomplete: {path.name}")
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
  if hashlib.sha256(npz_path.read_bytes()).hexdigest() != npz_sha256:
    raise RuntimeError("P38 terminal NPZ self-check failed")
  state["records"] = record_index + 1
  state["bytes"] = current_bytes + new_bytes
  return record_index, npz_sha256
