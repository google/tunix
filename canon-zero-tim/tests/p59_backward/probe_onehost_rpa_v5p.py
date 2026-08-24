#!/usr/bin/env python3
"""Real-v5p P59 DP2xTP2 RPA VJP plus ordinary DP1xTP4 control."""

from __future__ import annotations

import importlib.util
import os


os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
os.environ["CANON_RPA_VJP2"] = "1"
os.environ["CANON_VJP2_MAX_SEQS"] = "1"
os.environ["CANON_RPA_D"] = "128,512,128,512"
os.environ["CANON_RPA_P"] = "128,512,128,512"
os.environ["CANON_RPA_M"] = "128,512,128,512"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import canon_shim_root  # noqa: E402
from tunix.rl import canonical_qwen3_adapter  # noqa: E402


_HEAD_DIM = 128
_TOKENS_PER_DP_RANK = 256
_PAGE_SIZE = 256
_PAGES_PER_DP_RANK = 9


def _load_attention():
  path = canon_shim_root.resolve("attn_iface_patched.py")
  spec = importlib.util.spec_from_file_location(
      "_p59_onehost_attention_interface", path
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load installed attention interface from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _put(value, mesh, spec):
  return jax.device_put(
      value,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec)),
  )


def _p59_inputs(mesh, *, wrong_cache: bool = False):
  # This bounded geometry deliberately has one TP-local KV head. The old stock
  # prelude therefore saw 1 < TP2 and repeated it a second time, reproducing
  # the exact class of Attempt-3 failure while fitting four physical chips.
  dp_size = 2
  tp_size = 2
  token_count = dp_size * _TOKENS_PER_DP_RANK
  global_q_heads = 8
  global_kv_heads = 2
  global_cache_heads = global_kv_heads + (tp_size if wrong_cache else 0)
  q = _put(
      (jnp.arange(token_count * global_q_heads * _HEAD_DIM, dtype=jnp.float32)
       .reshape(token_count, global_q_heads, _HEAD_DIM) / 65536)
      .astype(jnp.bfloat16),
      mesh,
      ("data", "model", None),
  )
  k = _put(
      jnp.full((token_count, global_kv_heads, _HEAD_DIM), 0.125, jnp.bfloat16),
      mesh,
      ("data", "model", None),
  )
  v = _put(
      jnp.full((token_count, global_kv_heads, _HEAD_DIM), -0.25, jnp.bfloat16),
      mesh,
      ("data", "model", None),
  )
  cache = _put(
      jnp.zeros(
          (
              dp_size * _PAGES_PER_DP_RANK,
              _PAGE_SIZE,
              global_cache_heads,
              2,
              _HEAD_DIM,
          ),
          jnp.bfloat16,
      ),
      mesh,
      ("data", None, "model", None, None),
  )
  kv_lens = _put(
      jnp.asarray([_TOKENS_PER_DP_RANK] * dp_size, jnp.int32),
      mesh,
      ("data",),
  )
  pages = list(range(_PAGES_PER_DP_RANK)) * dp_size
  page_indices = _put(jnp.asarray(pages, jnp.int32), mesh, ("data",))
  cu_q_lens = _put(
      jnp.asarray([0, _TOKENS_PER_DP_RANK] * dp_size, jnp.int32),
      mesh,
      ("data",),
  )
  distribution = _put(
      jnp.asarray([0, 0, 1] * dp_size, jnp.int32), mesh, ("data",)
  )
  return q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution


def _run_p59_vjp(attention, engine_mesh, outer_mesh, values):
  q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution = values

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
          sm_scale=1.0 / np.sqrt(_HEAD_DIM),
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
      outer_mesh, "zt_tr_dp_parallel_onehost_rpa"
  ):
    return jax.jit(mapped)(
        q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution
    )


def _check_gradients(gradients, primals):
  if tuple(value.shape for value in gradients) != tuple(
      value.shape for value in primals[:4]
  ):
    raise AssertionError("P59 one-host RPA VJP gradient shapes changed")
  finite = tuple(
      bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(value)))))
      for value in gradients
  )
  if not all(finite):
    raise AssertionError(f"P59 one-host RPA VJP non-finite gradients: {finite}")
  norms = tuple(
      float(
          np.asarray(
              jax.device_get(
                  jnp.sqrt(jnp.sum(jnp.square(value.astype(jnp.float32))))
              )
          )
      )
      for value in gradients
  )
  if not all(np.isfinite(value) for value in norms):
    raise AssertionError(f"P59 one-host RPA VJP invalid norms: {norms}")
  return norms


def _run_ordinary_tp4(attention, devices):
  # The same four devices are rearranged as DP1xTP4. With no outer manual P59
  # map, two global KV heads must follow the unchanged stock expansion to four.
  mesh = jax.sharding.Mesh(
      devices.reshape(1, 1, 1, 1, 4, 1),
      ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp"),
  )
  q = jnp.zeros((_TOKENS_PER_DP_RANK, 4, _HEAD_DIM), jnp.bfloat16)
  k = jnp.full((_TOKENS_PER_DP_RANK, 2, _HEAD_DIM), 0.125, jnp.bfloat16)
  v = jnp.full((_TOKENS_PER_DP_RANK, 2, _HEAD_DIM), -0.25, jnp.bfloat16)
  cache = jnp.zeros(
      (_PAGES_PER_DP_RANK, _PAGE_SIZE, 4, 2, _HEAD_DIM), jnp.bfloat16
  )
  output, updated = attention.sharded_ragged_paged_attention(
      mesh,
      q,
      k,
      v,
      cache,
      jnp.asarray([_TOKENS_PER_DP_RANK], jnp.int32),
      jnp.arange(_PAGES_PER_DP_RANK, dtype=jnp.int32),
      jnp.asarray([0, _TOKENS_PER_DP_RANK], jnp.int32),
      jnp.asarray([0, 0, 1], jnp.int32),
      None,
      sm_scale=1.0 / np.sqrt(_HEAD_DIM),
  )
  jax.block_until_ready((output, updated))
  if output.shape != q.shape or updated.shape != cache.shape:
    raise AssertionError(
        f"ordinary TP4 RPA output changed: {output.shape}/{updated.shape}"
    )


def main() -> None:
  devices = np.asarray(jax.devices("tpu"))
  if devices.size != 4:
    raise RuntimeError(f"one-host RPA gate requires exactly four TPU devices: {devices}")
  kinds = tuple(str(device.device_kind) for device in devices.flat)
  if set(kinds) != {"TPU v5"}:
    raise RuntimeError(f"one-host RPA gate requires v5p devices: {kinds}")
  attention = _load_attention()

  outer_mesh = jax.sharding.Mesh(devices.reshape(2, 2), ("data", "model"))
  engine_mesh = jax.sharding.Mesh(
      devices.reshape(2, 1, 1, 1, 2, 1),
      ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp"),
  )
  values = _p59_inputs(outer_mesh)
  gradients = _run_p59_vjp(attention, engine_mesh, outer_mesh, values)
  jax.block_until_ready(gradients)
  norms = _check_gradients(gradients, values)

  try:
    _run_p59_vjp(
        attention,
        engine_mesh,
        outer_mesh,
        _p59_inputs(outer_mesh, wrong_cache=True),
    )
  except ValueError as error:
    if "P59 local attention cache shape mismatch" not in str(error):
      raise
  else:
    raise AssertionError("P59 one-host wrong-cache negative did not fire")

  _run_ordinary_tp4(attention, devices)
  print(
      "P59_RPA_ONEHOST_V5P_PASS "
      "p59_topology=DP2xTP2 ordinary_topology=DP1xTP4 "
      "real_rpa=2 rpa_vjp2=1 local_kv_heads=1 wrong_cache_negative=1 "
      "ordinary_global_gqa=1 optimizer_commits=0 "
      "gradient_norms=" + ",".join(f"{value:.8g}" for value in norms),
      flush=True,
  )


if __name__ == "__main__":
  main()
