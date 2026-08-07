#!/usr/bin/env python3
"""Proves global M512 is split to local M256 across engine data=2."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import canonical_qwen3_adapter


def main() -> None:
  if jax.device_count() != 4:
    raise RuntimeError(f"expected four CPU devices, got {jax.device_count()}")
  mesh = jax.make_mesh(
      (2, 2),
      ("data", "model"),
      devices=jax.devices(),
      axis_types=(jax.sharding.AxisType.Auto,) * 2,
  )
  traced_shapes = []

  def local_log_softmax(value):
    traced_shapes.append(tuple(value.shape))
    if value.shape != (256, 7):
      raise RuntimeError(f"expected local log-softmax M256, got {value.shape}")
    return jax.nn.log_softmax(value, axis=-1)

  def gather(logprobs, token_ids, max_logprobs):
    del max_logprobs
    return (jnp.take_along_axis(logprobs, token_ids[:, None], axis=-1),)

  previous = canonical_qwen3_adapter.canonical_logsoftmax.log_softmax
  canonical_qwen3_adapter.canonical_logsoftmax.log_softmax = local_log_softmax
  try:
    with jax.set_mesh(mesh):
      function = canonical_qwen3_adapter._make_canonical_compute_and_gather(
          gather, mesh
      )
      logits = jnp.arange(512 * 7, dtype=jnp.float32).reshape(512, 7) / 100
      tokens = jnp.arange(512, dtype=jnp.int32) % 7
      actual = function(logits, tokens, 1)[0]
      expected = jnp.take_along_axis(
          jax.nn.log_softmax(logits, axis=-1), tokens[:, None], axis=-1
      )
      np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
  finally:
    canonical_qwen3_adapter.canonical_logsoftmax.log_softmax = previous
  if traced_shapes != [(256, 7)]:
    raise RuntimeError(f"unexpected traced local shapes: {traced_shapes}")
  shards = sorted(
      (
          int(shard.index[0].start or 0),
          int(shard.index[0].stop or 512),
          int(shard.device.id),
      )
      for shard in actual.addressable_shards
  )
  expected_shards = [
      (0, 16 * 16, 0),
      (0, 16 * 16, 1),
      (16 * 16, 32 * 16, 2),
      (16 * 16, 32 * 16, 3),
  ]
  if shards != expected_shards:
    raise RuntimeError(f"unexpected result shards: {shards}")
  q_len = jnp.asarray([129, 256], jnp.int32)
  kv_len = jnp.asarray([385, 512], jnp.int32)
  block_tables, seq_lens, query_start, distribution = (
      canonical_qwen3_adapter._canonical_dp_attention_metadata_arrays(
          data_size=2,
          max_num_reqs=32,
          blocks_per_req=25,
          q_len=q_len,
          kv_len=kv_len,
      )
  )
  cache = jnp.zeros((50, 2), jnp.bfloat16)

  def inspect_local(cache_local, tables_local, seq_local, query_local, dist_local):
    expected_shapes = ((25, 2), (400,), (16,), (17,), (3,))
    actual_shapes = tuple(
        value.shape
        for value in (
            cache_local,
            tables_local,
            seq_local,
            query_local,
            dist_local,
        )
    )
    if actual_shapes != expected_shapes:
      raise RuntimeError(
          f"unexpected local RPA metadata shapes: {actual_shapes}"
      )
    return jnp.asarray(
        [
            tables_local[0],
            tables_local[24],
            seq_local[0],
            query_local[0],
            query_local[1],
            dist_local[0],
            dist_local[1],
            dist_local[2],
        ],
        jnp.int32,
    )[None, :]

  with jax.set_mesh(mesh):
    inspect = jax.shard_map(
        inspect_local,
        mesh=mesh,
        in_specs=(
            jax.sharding.PartitionSpec("data", None),
            jax.sharding.PartitionSpec("data"),
            jax.sharding.PartitionSpec("data"),
            jax.sharding.PartitionSpec("data"),
            jax.sharding.PartitionSpec("data"),
        ),
        out_specs=jax.sharding.PartitionSpec("data", None),
        check_vma=False,
    )
    metadata_result = inspect(
        cache, block_tables, seq_lens, query_start, distribution
    )
  expected_metadata = np.asarray(
      [
          [0, 24, 385, 0, 129, 0, 0, 1],
          [0, 24, 512, 0, 256, 0, 0, 1],
      ],
      np.int32,
  )
  np.testing.assert_array_equal(
      np.asarray(metadata_result), expected_metadata
  )
  try:
    canonical_qwen3_adapter._canonical_dp_attention_metadata_arrays(
        data_size=2,
        max_num_reqs=33,
        blocks_per_req=25,
        q_len=q_len,
        kv_len=kv_len,
    )
  except canonical_qwen3_adapter.FunctionalMappingError:
    pass
  else:
    raise RuntimeError("odd max_num_reqs negative control was accepted")
  print(
      "P32.D2.LOGPROB_ROW_SHARD_PASS "
      "global_m=512 local_m=256 data=2 model=2 values_exact=1 "
      "cache_pages=50/25 query_offsets=34/17 distribution=6/3",
      flush=True,
  )


if __name__ == "__main__":
  os.environ["CANON_P32_DP2TP2"] = "1"
  main()
