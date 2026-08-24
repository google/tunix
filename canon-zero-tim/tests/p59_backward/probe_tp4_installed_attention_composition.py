#!/usr/bin/env python3
"""Exercise the installed RPA boundary inside P59's real DP/TP carrier."""

from __future__ import annotations

import importlib.util
import os


TP_SIZE = int(os.environ.get("P59_TEST_TP_SIZE", "4"))
if TP_SIZE not in (4, 8):
  raise RuntimeError(f"installed-attention probe supports TP4/TP8, got TP{TP_SIZE}")

os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
os.environ["CANON_RPA_VJP2"] = "1"
os.environ["CANON_VJP2_MAX_SEQS"] = "1"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import canon_shim_root  # noqa: E402
from tunix.rl import canonical_qwen3_adapter  # noqa: E402


def _load_installed_attention():
  path = canon_shim_root.resolve("attn_iface_patched.py")
  spec = importlib.util.spec_from_file_location(
      "_p59_installed_attention_interface", path
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load installed attention interface from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _inputs(mesh, *, wrong_cache: bool = False):
  dp_size = 2
  tokens_per_rank = 4
  head_dim = 128
  local_q_heads = 4
  global_q_heads = local_q_heads * TP_SIZE
  global_kv_heads = 8
  local_kv_heads = global_kv_heads // TP_SIZE
  global_cache_heads = global_kv_heads + (TP_SIZE if wrong_cache else 0)

  def put(value, spec):
    return jax.device_put(
        value, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
    )

  token_count = dp_size * tokens_per_rank
  q = put(
      (jnp.arange(token_count * global_q_heads * head_dim, dtype=jnp.float32)
       .reshape(token_count, global_q_heads, head_dim) / 8192).astype(jnp.bfloat16),
      ("data", "model", None),
  )
  k = put(
      jnp.full((token_count, global_kv_heads, head_dim), 0.125, jnp.bfloat16),
      ("data", "model", None),
  )
  v = put(
      jnp.full((token_count, global_kv_heads, head_dim), -0.25, jnp.bfloat16),
      ("data", "model", None),
  )
  cache = put(
      jnp.zeros((4, 4, global_cache_heads, 2, head_dim), jnp.bfloat16),
      ("data", None, "model", None, None),
  )
  kv_lens = put(jnp.asarray([4, 4], jnp.int32), ("data",))
  page_indices = put(jnp.asarray([0, 1, 0, 1], jnp.int32), ("data",))
  cu_q_lens = put(jnp.asarray([0, 4, 0, 4], jnp.int32), ("data",))
  distribution = put(
      jnp.asarray([0, 0, 1, 0, 0, 1], jnp.int32), ("data",)
  )
  return (
      q,
      k,
      v,
      cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      local_kv_heads,
  )


def _fake_rpa(q, k, v, cache, *_metadata, **_kwargs):
  if k.shape != v.shape or int(k.shape[1]) != int(cache.shape[2]):
    raise ValueError(
        "installed fake RPA K/V/cache shape mismatch: "
        f"q={q.shape} k={k.shape} v={v.shape} cache={cache.shape}"
    )
  if int(q.shape[1]) % int(k.shape[1]):
    raise ValueError(f"installed fake RPA invalid GQA ratio: {q.shape}/{k.shape}")
  factor = int(q.shape[1]) // int(k.shape[1])
  output = q + jnp.repeat(k + v, factor, axis=1)
  return output, cache


def _p59_vjp(attention, engine_mesh, outer_mesh, values):
  q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution, _ = values

  def local_vjp(lq, lk, lv, lcache, llens, lpages, lcu, ldist):
    def primal(q_arg, k_arg, v_arg, cache_arg):
      output, _ = attention.sharded_ragged_paged_attention(
          engine_mesh,
          q_arg,
          k_arg,
          v_arg,
          cache_arg,
          llens,
          lpages,
          lcu,
          ldist,
          None,
          sm_scale=1.0 / np.sqrt(128),
      )
      return output

    output, pullback = jax.vjp(primal, lq, lk, lv, lcache)
    return pullback(jnp.ones_like(output))

  mapped = jax.shard_map(
      local_vjp,
      mesh=outer_mesh,
      in_specs=(
          jax.sharding.PartitionSpec("data", "model", None),
          jax.sharding.PartitionSpec("data", "model", None),
          jax.sharding.PartitionSpec("data", "model", None),
          jax.sharding.PartitionSpec("data", None, "model", None, None),
          jax.sharding.PartitionSpec("data"),
          jax.sharding.PartitionSpec("data"),
          jax.sharding.PartitionSpec("data"),
          jax.sharding.PartitionSpec("data"),
      ),
      out_specs=(
          jax.sharding.PartitionSpec("data", "model", None),
          jax.sharding.PartitionSpec("data", "model", None),
          jax.sharding.PartitionSpec("data", "model", None),
          jax.sharding.PartitionSpec("data", None, "model", None, None),
      ),
      axis_names=frozenset(("data", "model")),
      check_vma=False,
  )
  with canonical_qwen3_adapter._p59_localize_engine_shard_maps(
      outer_mesh, "zt_tr_dp_parallel_installed_attention"
  ):
    return jax.jit(mapped)(
        q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution
    )


def _ordinary_gqa(attention, engine_mesh):
  # Flag presence alone must not skip the stock global GQA expansion. Two
  # global KV heads cannot be partitioned over TP4/TP8; success proves the
  # stock path repeated them to exactly TP_SIZE before the inner shard_map.
  tokens = 8
  q = jnp.zeros((tokens, TP_SIZE, 128), jnp.bfloat16)
  k = jnp.ones((tokens, 2, 128), jnp.bfloat16)
  v = jnp.ones((tokens, 2, 128), jnp.bfloat16)
  cache = jnp.zeros((4, 4, TP_SIZE, 2, 128), jnp.bfloat16)
  output, _ = attention.sharded_ragged_paged_attention(
      engine_mesh,
      q,
      k,
      v,
      cache,
      jnp.asarray([4, 4], jnp.int32),
      jnp.asarray([0, 1, 0, 1], jnp.int32),
      jnp.asarray([0, 4, 0, 4], jnp.int32),
      jnp.asarray([0, 0, 1, 0, 0, 1], jnp.int32),
      None,
      sm_scale=1.0 / np.sqrt(128),
  )
  jax.block_until_ready(output)
  if output.shape != q.shape:
    raise AssertionError(f"ordinary GQA output shape changed: {output.shape}")


def main() -> None:
  device_count = 2 * TP_SIZE
  if len(jax.devices()) < device_count:
    raise RuntimeError(
        f"P59 installed-attention probe requires {device_count} devices"
    )
  devices = np.asarray(jax.devices()[:device_count])
  outer_mesh = jax.sharding.Mesh(
      devices.reshape(2, TP_SIZE), ("data", "model")
  )
  engine_mesh = jax.sharding.Mesh(
      devices.reshape(2, 1, 1, 1, TP_SIZE, 1),
      ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp"),
  )
  attention = _load_installed_attention()
  attention.ragged_paged_attention = _fake_rpa

  values = _inputs(outer_mesh)
  gradients = _p59_vjp(attention, engine_mesh, outer_mesh, values)
  jax.block_until_ready(gradients)
  expected_shapes = tuple(value.shape for value in values[:4])
  if tuple(value.shape for value in gradients) != expected_shapes:
    raise AssertionError(
        "installed attention VJP shape changed: "
        f"{tuple(value.shape for value in gradients)} != {expected_shapes}"
    )

  try:
    _p59_vjp(attention, engine_mesh, outer_mesh, _inputs(outer_mesh, wrong_cache=True))
  except ValueError as error:
    if "P59 local attention cache shape mismatch" not in str(error):
      raise
  else:
    raise AssertionError("P59 local attention wrong-cache negative did not fire")

  _ordinary_gqa(attention, engine_mesh)
  print(
      f"P59_TP{TP_SIZE}_INSTALLED_ATTENTION_PASS "
      f"topology=DP2xTP{TP_SIZE} local_kv_heads={values[-1]} "
      "rpa_vjp2=1 wrong_cache_negative=1 ordinary_global_gqa=1 "
      "optimizer_commits=0",
      flush=True,
  )


if __name__ == "__main__":
  main()
