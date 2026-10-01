# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical Qwen3 model integrating Zero-TIM batch-invariant kernels.

This module provides `Qwen3Canon`, a drop-in counterpart to
`tunix.models.qwen3.model.Qwen3` that replaces individual operations with
Zero-TIM kernels from `tunix.experimental.zero_tim_kernel` to eliminate
training-inference mismatch (TIM):

1. `use_canon_rmsnorm`: Fixed-BF=128 left-to-right f32 accumulation RMSNorm
   (`pallas_rmsnorm` + `canonical_vjp.rmsnorm`) for `input_layernorm`,
   `post_attention_layernorm`, `q_norm`, `k_norm`, and `final_norm`.
2. `use_canon_qkv_proj`: Fixed-tile padded matmul + canonical VJP
   (`padded_matmul` + `canonical_vjp.matmul`) for `q_proj`, `k_proj`, `v_proj`.
3. `use_canon_rope`: Barriered float32 RoPE rotation preventing FMA fusion drift
   between decode (`T=1`), prefill (`T=L`), and `fwd+bwd`.
4. `use_canon_attention`: Fixed KV-block (`bkv_csz=16`) online softmax attention
   in `float32` using `exp2` (`LOG2_E`) with strict no-op masking on inactive
   KV blocks (`rpa_diff` semantics), making KV-cached decode bitwise identical
   to cacheless prefill and learner forward.
5. `use_canon_o_proj`: Fixed-tile padded matmul + canonical VJP for `o_proj`.
6. `use_canon_mlp_proj`: Fixed-tile padded matmul + canonical VJP for
   `gate_proj`, `up_proj`, and `down_proj`.
7. `use_canon_swiglu`: Padded Pallas SwiGLU (`padded_swiglu` +
   `canonical_vjp.swiglu`).
8. `use_fixed_lm_head`: P38 `FIXED_M=256` padded/chunked output head with
   ascending-chunk `lax.scan` VJP (`fixed_lm_head`).
9. `use_canon_logsoftmax`: Fixed 1024-tile 3-stage log-softmax / gathered
   logprobs (`canonical_logsoftmax`).
10. `use_fixed_order_reduce`: Rank-0-to-(TP-1) f32 TP reduction
    (`fixed_order_reduce`) when running over a tensor-parallel mesh.

All 10 switches are individually configurable via `CanonKernelConfig` to
support fine-grained per-kernel numeric and performance ablation studies while
sharing the exact same `nnx.Param` state tree as stock `Qwen3`.
"""

from __future__ import annotations

import dataclasses
from functools import partial
import os
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp
from jax.interpreters import pxla
from jax.sharding import PartitionSpec as P
import jaxtyping
from tunix.experimental.zero_tim_kernel import canonical_logsoftmax
from tunix.experimental.zero_tim_kernel import canonical_vjp
from tunix.experimental.zero_tim_kernel import fixed_lm_head
from tunix.experimental.zero_tim_kernel import fixed_order_reduce
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul
from tunix.experimental.zero_tim_kernel import padded_swiglu
from tunix.experimental.zero_tim_kernel import pallas_rmsnorm
from tunix.experimental.zero_tim_kernel import rpa_diff
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.qwen3 import model as qwen3_stock
from tunix.utils import compat

K_MASK = qwen3_stock.K_MASK
LayerCache = qwen3_stock.LayerCache
Cache = qwen3_stock.Cache
RematConfig = qwen3_stock.RematConfig
ShardingConfig = qwen3_stock.ShardingConfig
ModelConfig = qwen3_stock.ModelConfig
shard = qwen3_stock.shard

LOG2_E = 1.4426950408889634

ABLATION_SWITCH_NAMES: tuple[str, ...] = (
    'use_canon_rmsnorm',
    'use_canon_qkv_proj',
    'use_canon_rope',
    'use_canon_attention',
    'use_canon_o_proj',
    'use_canon_mlp_proj',
    'use_canon_swiglu',
    'use_fixed_lm_head',
    'use_canon_logsoftmax',
    'use_fixed_order_reduce',
)


@dataclasses.dataclass(slots=True, frozen=True)
class CanonKernelConfig:
  """Per-kernel switches for Zero-TIM canonical execution and ablation tests."""

  use_canon_rmsnorm: bool = True
  use_canon_qkv_proj: bool = True
  use_canon_rope: bool = True
  use_canon_attention: bool = True
  use_canon_o_proj: bool = True
  use_canon_mlp_proj: bool = True
  use_canon_swiglu: bool = True
  use_fixed_lm_head: bool = True
  use_canon_logsoftmax: bool = True
  use_fixed_order_reduce: bool = True
  attention_kv_block_size: int = 16
  interpret: bool | None = None

  def resolve_interpret(self) -> bool:
    if self.interpret is not None:
      return self.interpret
    return jax.default_backend() == 'cpu'

  @classmethod
  def all_enabled(
      cls,
      *,
      attention_kv_block_size: int = 16,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    return cls(
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )

  @classmethod
  def all_disabled(
      cls,
      *,
      attention_kv_block_size: int = 16,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    kwargs = {name: False for name in ABLATION_SWITCH_NAMES}
    return cls(
        **kwargs,
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )

  @classmethod
  def only(
      cls,
      switch_name: str,
      *,
      attention_kv_block_size: int = 16,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    """Enables ONLY `switch_name` (Add-One-In ablation)."""
    if switch_name not in ABLATION_SWITCH_NAMES:
      raise ValueError(
          f'Unknown switch {switch_name!r}; valid: {ABLATION_SWITCH_NAMES}'
      )
    kwargs = {name: name == switch_name for name in ABLATION_SWITCH_NAMES}
    return cls(
        **kwargs,
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )

  @classmethod
  def without(
      cls,
      switch_name: str,
      *,
      attention_kv_block_size: int = 16,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    """Enables all canonical kernels EXCEPT `switch_name` (Leave-One-Out)."""
    if switch_name not in ABLATION_SWITCH_NAMES:
      raise ValueError(
          f'Unknown switch {switch_name!r}; valid: {ABLATION_SWITCH_NAMES}'
      )
    kwargs = {name: name != switch_name for name in ABLATION_SWITCH_NAMES}
    return cls(
        **kwargs,
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )


def _round_up(x: int, tile: int) -> int:
  return ((x + tile - 1) // tile) * tile


def resolve_or_build_contract(
    config: ModelConfig,
    tp_size: int = 1,
) -> model_contracts.ModelContract:
  """Returns a registered `ModelContract` or builds a compatible one."""
  for contract in model_contracts.CONTRACTS.values():
    if (
        contract.hidden_size == config.embed_dim
        and contract.intermediate_size == config.hidden_dim
        and contract.num_attention_heads == config.num_heads
        and contract.num_kv_heads == config.num_kv_heads
        and contract.head_dim == config.head_dim
        and contract.tp_size == tp_size
        and config.vocab_size == model_contracts.VOCAB_SIZE
    ):
      return contract

  block_m = 128
  if config.embed_dim >= 1024 and tp_size <= 4:
    block_n = 256
    block_k = 256
  else:
    block_n = 128
    block_k = 128

  q_width = (config.num_heads * config.head_dim) // tp_size
  kv_width = (config.num_kv_heads * config.head_dim) // tp_size
  mlp_width = config.hidden_dim // tp_size
  hidden = config.embed_dim
  local_vocab = config.vocab_size // tp_size

  k_widths = {hidden, q_width, mlp_width}
  n_widths = {q_width, kv_width, hidden, mlp_width, local_vocab}

  k_pad: dict[int, int] = {}
  for w in k_widths:
    for tile in (block_k, fixed_lm_head.BK):
      if w % tile != 0:
        k_pad[w] = _round_up(w, tile)

  n_pad: dict[int, int] = {}
  for w in n_widths:
    for tile in (block_n, fixed_lm_head.BN):
      if w % tile != 0:
        n_pad[w] = _round_up(w, tile)

  sw_pad: dict[int, int] = {}
  if mlp_width % model_contracts.SWIGLU_BF != 0:
    sw_pad[mlp_width] = _round_up(mlp_width, model_contracts.SWIGLU_BF)

  return model_contracts.ModelContract(
      name=f'qwen3_d{hidden}_f{config.hidden_dim}_tp{tp_size}',
      block_m=block_m,
      block_n=block_n,
      block_k=block_k,
      matmul_k_padding=k_pad,
      matmul_n_padding=n_pad,
      swiglu_feature_padding=sw_pad,
      hidden_size=hidden,
      intermediate_size=config.hidden_dim,
      num_attention_heads=config.num_heads,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
      tp_size=tp_size,
  )


def _active_tp_size_and_mesh(
    tp_axis: str = 'tp',
) -> tuple[int, jax.sharding.Mesh | None]:
  """Returns `(tp_size, mesh)` if a physical mesh with `tp_axis` is active."""
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh is not None and not mesh.empty and tp_axis in mesh.shape:
    return int(mesh.shape[tp_axis]), mesh
  return 1, None


def _tp_sum_with_replicated_bwd(
    partial: jax.Array, axis_name: str, count: int, *, mode: str
) -> jax.Array:
  """Runs `fixed_order_tp_sum` in forward and identity VJP on replicated cotangent."""

  @jax.custom_vjp
  def op(p):
    return fixed_order_reduce.fixed_order_tp_sum(p, axis_name, count, mode=mode)

  def fwd(p):
    return op(p), None

  def bwd(_, cotangent):
    return (cotangent.astype(partial.dtype),)

  op.defvjp(fwd, bwd)
  return op(partial)


def _canon_matmul_2d(
    x_2d: jax.Array,
    w_2d: jax.Array,
    *,
    contract: model_contracts.ModelContract,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> jax.Array:
  """Runs 2D bf16 fixed-tile padded matmul with canonical VJP."""
  x_bf16 = x_2d.astype(jnp.bfloat16)
  w_bf16 = w_2d.astype(jnp.bfloat16)
  bm = contract.block_m if block_m is None else block_m
  bn = contract.block_n if block_n is None else block_n
  bk = contract.block_k if block_k is None else block_k

  def forward(a: jax.Array, b: jax.Array) -> jax.Array:
    return padded_matmul.matmul(
        a,
        b,
        contract=contract,
        interpret=interpret,
        shape_invariant_numerics=True,
        block_m=bm,
        block_n=bn,
        block_k=bk,
    )

  out_bf16 = canonical_vjp.matmul(
      x_bf16, w_bf16, forward=forward, contract=contract
  )
  return out_bf16.astype(out_dtype)


def _canon_rmsnorm_nd_local(
    x: jax.Array,
    w: jax.Array,
    *,
    norm_eps: float,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
) -> jax.Array:
  """Runs fixed-BF=128 Pallas RMSNorm on a local (per-rank) array."""
  orig_shape = x.shape
  f = int(orig_shape[-1])
  m = int(x.size // f)
  x_2d = x.reshape(m, f).astype(jnp.bfloat16)
  w_1d = w.reshape(f).astype(jnp.bfloat16)

  bm = pallas_rmsnorm.BM
  bf = pallas_rmsnorm.BF
  mp = _round_up(m, bm)

  if f % bf != 0:
    xf = x_2d.astype(jnp.float32)
    wf = w_1d.astype(jnp.float32)
    rms = lax.rsqrt(
        jnp.mean(xf * xf, axis=-1, keepdims=True) + jnp.float32(norm_eps)
    )
    return (xf * rms * wf[None, :]).astype(out_dtype).reshape(orig_shape)

  if mp != m:
    x_padded = jnp.pad(x_2d, ((0, mp - m), (0, 0)), constant_values=0)
  else:
    x_padded = x_2d

  def forward(a: jax.Array, weight: jax.Array) -> jax.Array:
    return pallas_rmsnorm.rmsnorm(
        a,
        weight,
        epsilon=norm_eps,
        interpret=interpret,
        shape_invariant_numerics=True,
    )

  out_padded = canonical_vjp.rmsnorm(
      x_padded, w_1d, epsilon=norm_eps, forward=forward
  )
  out_2d = out_padded[:m, :]
  return out_2d.astype(out_dtype).reshape(orig_shape)


def _canon_rmsnorm_nd(
    x: jax.Array,
    w: jax.Array,
    *,
    norm_eps: float,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
) -> jax.Array:
  """Runs fixed-BF=128 Pallas RMSNorm with row padding and TP shard_map support."""
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if mesh is not None and tp_size > 1:
    if x.ndim == 4:

      def local_4d(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
        return fixed_order_reduce.p59_column_parallel(
            lambda a, weight: _canon_rmsnorm_nd_local(
                a,
                weight,
                norm_eps=norm_eps,
                interpret=interpret,
                out_dtype=out_dtype,
            ),
            x_loc,
            w_loc,
            replicated=(False, True),
            axis_name='tp',
            count=tp_size,
        )

      return jax.shard_map(
          local_4d,
          mesh=mesh,
          in_specs=(P(None, None, 'tp', None), P(None)),
          out_specs=P(None, None, 'tp', None),
          check_vma=False,
      )(x, w)
    if x.ndim == 3:

      def local_3d(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
        return _canon_rmsnorm_nd_local(
            x_loc,
            w_loc,
            norm_eps=norm_eps,
            interpret=interpret,
            out_dtype=out_dtype,
        )

      return jax.shard_map(
          local_3d,
          mesh=mesh,
          in_specs=(P(None, None, None), P(None)),
          out_specs=P(None, None, None),
          check_vma=False,
      )(x, w)
  return _canon_rmsnorm_nd_local(
      x, w, norm_eps=norm_eps, interpret=interpret, out_dtype=out_dtype
  )


def _apply_rope_canonical_local(
    inputs: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
  """Local per-rank body of canonical RoPE with query-length padding to 64."""
  l = int(inputs.shape[1])
  lp = _round_up(l, 64)
  if lp != l:
    pad_l = lp - l
    inp_pad = jnp.pad(
        inputs, ((0, 0), (0, pad_l), (0, 0), (0, 0)), constant_values=0
    )
    pos_pad = jnp.pad(positions, ((0, 0), (0, pad_l)), constant_values=0)
  else:
    inp_pad = inputs
    pos_pad = positions

  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = jnp.float32(rope_theta) ** fraction

  sinusoid_inp = (
      pos_pad.astype(jnp.float32)[..., jnp.newaxis]
      / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp).astype(inputs.dtype).astype(jnp.float32)
  cos = jnp.cos(sinusoid_inp).astype(inputs.dtype).astype(jnp.float32)

  inp_f32 = inp_pad.astype(jnp.float32)
  first_half, second_half = jnp.split(inp_f32, 2, axis=-1)
  # Separate products via optimization_barrier so XLA cannot fuse `a*cos - b*sin`
  # into `fmsub` in one compilation unit and separate `mul + sub` in another.
  first_cos = lax.optimization_barrier(first_half * cos)
  second_sin = lax.optimization_barrier(second_half * sin)
  second_cos = lax.optimization_barrier(second_half * cos)
  first_sin = lax.optimization_barrier(first_half * sin)

  first_part = (first_cos - second_sin).astype(inputs.dtype)
  second_part = (second_cos + first_sin).astype(inputs.dtype)
  out_pad = jnp.concatenate([first_part, second_part], axis=-1)
  return out_pad[:, :l, :, :]


def apply_rope_canonical(
    inputs: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
  """Applies RoPE with barriered f32 arithmetic and TP shard_map support."""
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if mesh is not None and tp_size > 1 and int(inputs.shape[2]) % tp_size == 0:
    return jax.shard_map(
        lambda x_loc, p_loc: _apply_rope_canonical_local(
            x_loc, p_loc, head_dim=head_dim, rope_theta=rope_theta
        ),
        mesh=mesh,
        in_specs=(P(None, None, 'tp', None), P(None, None)),
        out_specs=P(None, None, 'tp', None),
        check_vma=False,
    )(inputs, positions)
  return _apply_rope_canonical_local(
      inputs, positions, head_dim=head_dim, rope_theta=rope_theta
  )


def _canonical_online_attention_local(
    query_proj: jax.Array,  # [B, T, QH, D]
    key_proj: jax.Array,  # [B, S, KH, D]
    value_proj: jax.Array,  # [B, S, KH, D]
    *,
    scale: float,
    attn_mask: jax.Array | None,  # [B, T, S]
    segment_ids: jax.Array | None = None,  # [B, T]
    kv_block_size: int = 16,
) -> jax.Array:
  """Local per-rank body of fixed KV-block online softmax attention."""
  b, t, qh, d = query_proj.shape
  _, s, kh, _ = key_proj.shape
  g = qh // kh
  out_dtype = query_proj.dtype

  q_grouped = query_proj.reshape(b, t, kh, g, d).transpose(0, 2, 3, 1, 4)
  k_grouped = key_proj.transpose(0, 2, 1, 3)  # [B, KH, S, D]
  v_grouped = value_proj.transpose(0, 2, 1, 3)  # [B, KH, S, D]

  if attn_mask is not None:
    mask_bts = attn_mask.astype(jnp.bool_)
  else:
    mask_bts = jnp.ones((b, t, s), dtype=jnp.bool_)

  if segment_ids is not None and t == s:
    seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    mask_bts = jnp.logical_and(mask_bts, seg_mask)

  # Pad query length `t` to a fixed multiple of 64 (analogous to RPA v3's
  # `bq_csz=128` / `MIN_TOKEN_BUCKET`) so that `G * tp >= 128` in both
  # Decode (`t=1`) and Prefill (`t=64`), preventing XLA TPU from transposing
  # the LHS/RHS operands of the MXU `dot_general` (`qk` and `pv`) when `t=1`.
  tp = _round_up(t, 64)
  if tp != t:
    pad_t = tp - t
    q_grouped = jnp.pad(
        q_grouped,
        ((0, 0), (0, 0), (0, 0), (0, pad_t), (0, 0)),
        constant_values=0,
    )
    mask_bts = jnp.pad(
        mask_bts, ((0, 0), (0, pad_t), (0, 0)), constant_values=False
    )

  bkv = int(kv_block_size)
  sp = _round_up(s, bkv)
  num_kv_blocks = sp // bkv

  if sp != s:
    pad_s = sp - s
    k_grouped = jnp.pad(
        k_grouped, ((0, 0), (0, 0), (0, pad_s), (0, 0)), constant_values=0
    )
    v_grouped = jnp.pad(
        v_grouped, ((0, 0), (0, 0), (0, pad_s), (0, 0)), constant_values=0
    )
    mask_bts = jnp.pad(
        mask_bts, ((0, 0), (0, 0), (0, pad_s)), constant_values=False
    )

  k_blocks = k_grouped.reshape(b, kh, num_kv_blocks, bkv, d).transpose(
      2, 0, 1, 3, 4
  )
  v_blocks = v_grouped.reshape(b, kh, num_kv_blocks, bkv, d).transpose(
      2, 0, 1, 3, 4
  )
  mask_blocks = mask_bts.reshape(b, tp, num_kv_blocks, bkv).transpose(
      2, 0, 1, 3
  )[
      :, :, None, None, :, :
  ]  # [num_kv_blocks, B, 1, 1, tp, bkv]

  q_f32 = q_grouped.astype(jnp.float32) * jnp.float32(scale * LOG2_E)
  init_m = jnp.full((b, kh, g, tp), -jnp.inf, dtype=jnp.float32)
  init_l = jnp.zeros((b, kh, g, tp), dtype=jnp.float32)
  init_acc = jnp.zeros((b, kh, g, tp, d), dtype=jnp.float32)

  def scan_kv_block(carry, kv_inputs):
    m_prev, l_prev, acc_prev = carry
    k_blk, v_blk, mask_blk = kv_inputs
    qk = jnp.einsum(
        'bhgtd,bhsd->bhgts',
        q_f32,
        k_blk.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    qk_masked = jnp.where(mask_blk, qk, -jnp.inf)
    blk_valid_any = jnp.any(mask_blk, axis=-1)  # [B, 1, 1, tp]

    m_curr = jnp.max(qk_masked, axis=-1)  # [B, KH, G, tp]
    m_cand = jnp.maximum(m_prev, m_curr)
    m_safe = jnp.where(jnp.isneginf(m_cand), jnp.float32(0.0), m_cand)

    alpha = jnp.where(
        jnp.isneginf(m_prev),
        jnp.float32(0.0),
        jnp.exp2(m_prev - m_safe),
    )
    p = jnp.where(
        mask_blk,
        jnp.exp2(qk_masked - m_safe[..., None]),
        jnp.float32(0.0),
    )
    l_blk = jnp.sum(p, axis=-1, dtype=jnp.float32)
    l_cand = l_prev * alpha + l_blk

    pv = jnp.einsum(
        'bhgts,bhsd->bhgtd',
        p,
        v_blk.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    acc_cand = acc_prev * alpha[..., None] + pv

    m_next = jnp.where(blk_valid_any, m_cand, m_prev)
    l_next = jnp.where(blk_valid_any, l_cand, l_prev)
    acc_next = jnp.where(blk_valid_any[..., None], acc_cand, acc_prev)
    return (m_next, l_next, acc_next), None

  (_, l_final, acc_final), _ = lax.scan(
      scan_kv_block,
      (init_m, init_l, init_acc),
      (k_blocks, v_blocks, mask_blocks),
  )

  denom = jnp.where(l_final == 0.0, jnp.float32(1.0), l_final)[..., None]
  out = (acc_final / denom).astype(out_dtype)
  return out[:, :, :, :t, :].transpose(0, 3, 1, 2, 4).reshape(b, t, qh, d)


def canonical_online_attention(
    query_proj: jax.Array,  # [B, T, QH, D]
    key_proj: jax.Array,  # [B, S, KH, D]
    value_proj: jax.Array,  # [B, S, KH, D]
    *,
    scale: float,
    attn_mask: jax.Array | None,  # [B, T, S]
    segment_ids: jax.Array | None = None,  # [B, T]
    kv_block_size: int = 16,
) -> jax.Array:
  """Fixed KV-block (`bkv_csz=16`) online softmax attention (`rpa_diff`)."""
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if mesh is not None and tp_size > 1 and int(key_proj.shape[2]) % tp_size == 0:
    mask_spec = P(None, None, None) if attn_mask is not None else None
    seg_spec = P(None, None) if segment_ids is not None else None

    def local_attn(q_loc, k_loc, v_loc, m_loc, s_loc):
      return _canonical_online_attention_local(
          q_loc,
          k_loc,
          v_loc,
          scale=scale,
          attn_mask=m_loc,
          segment_ids=s_loc,
          kv_block_size=kv_block_size,
      )

    return jax.shard_map(
        local_attn,
        mesh=mesh,
        in_specs=(
            P(None, None, 'tp', None),
            P(None, None, 'tp', None),
            P(None, None, 'tp', None),
            mask_spec,
            seg_spec,
        ),
        out_specs=P(None, None, 'tp', None),
        check_vma=False,
    )(query_proj, key_proj, value_proj, attn_mask, segment_ids)
  return _canonical_online_attention_local(
      query_proj,
      key_proj,
      value_proj,
      scale=scale,
      attn_mask=attn_mask,
      segment_ids=segment_ids,
      kv_block_size=kv_block_size,
  )


def _fixed_m_lm_head_2d_local(
    x_bf16: jax.Array,  # [M, D]
    w_bf16: jax.Array,  # [D, V_local]
    *,
    contract: model_contracts.ModelContract,
    interpret: bool,
) -> jax.Array:
  """Local per-rank P38 `FIXED_M=256` padded/chunked LM head."""
  m, hidden = map(int, x_bf16.shape)
  _, vocab = map(int, w_bf16.shape)
  local_matmul = fixed_lm_head.canonical_local_matmul(
      contract, interpret=interpret
  )
  fixed_m = fixed_lm_head.FIXED_M
  bm = fixed_lm_head.BM
  bn = fixed_lm_head.BN if vocab >= fixed_lm_head.BN else contract.block_n
  bk = fixed_lm_head.BK if hidden >= 1024 else contract.block_k

  def run_fixed_chunk(a_chunk: jax.Array, weight: jax.Array) -> jax.Array:
    return local_matmul(
        a_chunk,
        weight,
        block_m=bm,
        block_n=bn,
        block_k=bk,
        shape_invariant_numerics=True,
    )

  if m < fixed_m:
    a_padded = jnp.pad(x_bf16, ((0, fixed_m - m), (0, 0)), constant_values=0)
    return run_fixed_chunk(a_padded, w_bf16)[:m, :]
  if m == fixed_m:
    return run_fixed_chunk(x_bf16, w_bf16)

  mp = _round_up(m, fixed_m)
  if mp != m:
    a_all = jnp.pad(x_bf16, ((0, mp - m), (0, 0)), constant_values=0)
  else:
    a_all = x_bf16

  def chunked_forward(a_learner: jax.Array, weight: jax.Array) -> jax.Array:
    a_chunks = a_learner.reshape((-1, fixed_m, hidden))
    return lax.map(
        lambda chunk: run_fixed_chunk(chunk, weight), a_chunks
    ).reshape((mp, vocab))

  @jax.custom_vjp
  def chunked_vjp(a_learner: jax.Array, weight: jax.Array) -> jax.Array:
    return chunked_forward(a_learner, weight)

  def chunked_fwd(a_learner: jax.Array, weight: jax.Array):
    return chunked_forward(a_learner, weight), (a_learner, weight)

  def chunked_bwd(residual, cotangent):
    a_learner, weight = residual
    a_chunks = a_learner.reshape((-1, fixed_m, hidden))
    cot_chunks = cotangent.reshape((-1, fixed_m, vocab))

    def accumulate(w_cot, values):
      a_chunk, out_cot = values
      _, pullback = jax.vjp(run_fixed_chunk, a_chunk, weight)
      a_cot, chunk_w_cot = pullback(out_cot)
      return w_cot + chunk_w_cot, a_cot

    w_cot, a_cot_chunks = lax.scan(
        accumulate,
        jnp.zeros_like(weight),
        (a_chunks, cot_chunks),
    )
    return a_cot_chunks.reshape(a_learner.shape), w_cot

  chunked_vjp.defvjp(chunked_fwd, chunked_bwd)
  return chunked_vjp(a_all, w_bf16)[:m, :]


def _fixed_m_lm_head_2d(
    x_2d: jax.Array,  # [M, D]
    w_dv: jax.Array,  # [D, V]
    *,
    contract: model_contracts.ModelContract,
    interpret: bool,
    endpoint: str,
) -> jax.Array:
  """Runs the P38 `FIXED_M=256` padded/chunked LM head on `[M, D] @ [D, V]`."""
  del endpoint
  x_bf16 = x_2d.astype(jnp.bfloat16)
  w_bf16 = w_dv.astype(jnp.bfloat16)
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if mesh is not None and tp_size > 1 and int(w_bf16.shape[1]) % tp_size == 0:

    def local_head(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
      return fixed_order_reduce.p59_column_parallel(
          lambda a, weight: _fixed_m_lm_head_2d_local(
              a, weight, contract=contract, interpret=interpret
          ),
          x_loc,
          w_loc,
          replicated=(True, False),
          axis_name='tp',
          count=tp_size,
      )

    return jax.shard_map(
        local_head,
        mesh=mesh,
        in_specs=(P(None, None), P(None, 'tp')),
        out_specs=P(None, 'tp'),
        check_vma=False,
    )(x_bf16, w_bf16)

  return _fixed_m_lm_head_2d_local(
      x_bf16, w_bf16, contract=contract, interpret=interpret
  )


class CanonEinsum(nnx.Module):
  """Einsum module supporting both stock `jnp.einsum` and Zero-TIM matmul."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.dtype = dtype
    self.w = nnx.Param(
        nnx.initializers.glorot_uniform()(
            rngs.params(), shape, dtype=param_dtype
        ),
        sharding=sharding,
    )

  @jax.named_scope('einsum')
  def __call__(
      self,
      x: jaxtyping.ArrayLike,
      *,
      use_canon: bool = False,
      use_fixed_order_reduce: bool = False,
      contract: model_contracts.ModelContract | None = None,
      interpret: bool = False,
  ) -> jaxtyping.Array:
    x_arr = jnp.astype(x, self.dtype)
    w_arr = jnp.astype(self.w.value, self.dtype)

    if self.einsum_str == 'BTNH,NHD->BTD':
      b, t, n_heads, head_dim = x_arr.shape
      _, _, d = w_arr.shape
      tp_size, mesh = _active_tp_size_and_mesh('tp')

      def local_mm(a2: jax.Array, w2: jax.Array) -> jax.Array:
        if use_canon and contract is not None:
          return _canon_matmul_2d(
              a2,
              w2,
              contract=contract,
              interpret=interpret,
              out_dtype=jnp.bfloat16,
          )
        return jnp.dot(a2.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)).astype(
            jnp.bfloat16
        )

      if not use_canon and not use_fixed_order_reduce:
        return jnp.einsum(self.einsum_str, x_arr, w_arr)

      if mesh is not None and tp_size > 1:

        def local_o_proj(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
          a2 = x_loc.reshape(b * t, -1).astype(jnp.bfloat16)
          w2 = w_loc.reshape(a2.shape[1], d).astype(jnp.bfloat16)
          partial = local_mm(a2, w2)
          if use_fixed_order_reduce:
            acc = _tp_sum_with_replicated_bwd(
                partial, 'tp', tp_size, mode=fixed_order_reduce.GATHER
            )
          else:
            acc = lax.psum(partial, 'tp')
          return acc.astype(self.dtype).reshape(b, t, d)

        return jax.shard_map(
            local_o_proj,
            mesh=mesh,
            in_specs=(P(None, None, 'tp', None), P('tp', None, None)),
            out_specs=P(None, None, None),
            check_vma=False,
        )(x_arr, w_arr)

      if use_fixed_order_reduce and n_heads >= 2 and n_heads % 2 == 0:
        tp_shards = 2
        h_per_tp = n_heads // tp_shards
        x_tp = x_arr.reshape(b * t, tp_shards, h_per_tp * head_dim)
        w_tp = w_arr.reshape(tp_shards, h_per_tp * head_dim, d)
        acc = jnp.zeros((b * t, d), dtype=jnp.float32)
        for r in range(tp_shards):
          part = local_mm(x_tp[:, r, :], w_tp[r, :, :])
          acc = acc + part.astype(jnp.float32)
        return acc.astype(self.dtype).reshape(b, t, d)

      if not use_canon or contract is None:
        return jnp.einsum(self.einsum_str, x_arr, w_arr)

      x_2d = x_arr.reshape(b * t, n_heads * head_dim)
      w_2d = w_arr.reshape(n_heads * head_dim, d)
      out_2d = _canon_matmul_2d(
          x_2d,
          w_2d,
          contract=contract,
          interpret=interpret,
          out_dtype=self.dtype,
      )
      return out_2d.reshape(b, t, d)

    if not use_canon or contract is None:
      return jnp.einsum(self.einsum_str, x_arr, w_arr)

    if self.einsum_str in ('BTD,DNH->BTNH', 'BSD,DKH->BSKH'):
      b, t, d = x_arr.shape
      _, n_heads, head_dim = w_arr.shape
      tp_size, mesh = _active_tp_size_and_mesh('tp')
      if mesh is not None and tp_size > 1 and n_heads % tp_size == 0:

        def local_qkv(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
          n_h_loc = int(w_loc.shape[1])

          def _mm(a_3d: jax.Array, weight_3d: jax.Array) -> jax.Array:
            out_2d = _canon_matmul_2d(
                a_3d.reshape(b * t, d),
                weight_3d.reshape(d, n_h_loc * head_dim),
                contract=contract,
                interpret=interpret,
                out_dtype=self.dtype,
            )
            return out_2d.reshape(b, t, n_h_loc, head_dim)

          return fixed_order_reduce.p59_column_parallel(
              _mm,
              x_loc,
              w_loc,
              replicated=(True, False),
              axis_name='tp',
              count=tp_size,
          )

        return jax.shard_map(
            local_qkv,
            mesh=mesh,
            in_specs=(P(None, None, None), P(None, 'tp', None)),
            out_specs=P(None, None, 'tp', None),
            check_vma=False,
        )(x_arr, w_arr)

      x_2d = x_arr.reshape(b * t, d)
      w_2d = w_arr.reshape(d, n_heads * head_dim)
      out_2d = _canon_matmul_2d(
          x_2d,
          w_2d,
          contract=contract,
          interpret=interpret,
          out_dtype=self.dtype,
      )
      return out_2d.reshape(b, t, n_heads, head_dim)

    if self.einsum_str == 'BTD,DV->BTV':
      b, t, d = x_arr.shape
      _, vocab = w_arr.shape
      x_2d = x_arr.reshape(b * t, d)
      out_2d = _fixed_m_lm_head_2d(
          x_2d,
          w_arr,
          contract=contract,
          interpret=interpret,
          endpoint='untied_lm_head',
      )
      return out_2d.astype(self.dtype).reshape(b, t, vocab)

    return jnp.einsum(self.einsum_str, x_arr, w_arr)


class CanonEmbedder(nnx.Module):
  """Embedder module supporting stock decode and P38 fixed_lm_head decode."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (vocab_size, embed_dim)
        ),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config
    self.dtype = dtype

  @jax.named_scope('embedder_encode')
  def encode(
      self,
      x: jaxtyping.ArrayLike,
      *,
      use_fixed_order_reduce: bool = False,
  ) -> jaxtyping.Array:
    x_ids = jnp.asarray(x, dtype=jnp.int32)
    tp_size, mesh = _active_tp_size_and_mesh('tp')
    if (
        use_fixed_order_reduce
        and mesh is not None
        and tp_size > 1
        and int(self.input_embedding.value.shape[0]) % tp_size == 0
    ):
      orig_shape = x_ids.shape
      flat_ids = x_ids.reshape(-1)
      table = jnp.astype(self.input_embedding.value, self.dtype)

      def local_embed(
          ids_local: jax.Array, table_local: jax.Array
      ) -> jax.Array:
        j = lax.axis_index('tp')
        vloc = table_local.shape[0]
        li = ids_local - j * vloc
        ok = (li >= 0) & (li < vloc)
        part = jnp.where(
            ok[:, None],
            jnp.take(table_local, jnp.clip(li, 0, vloc - 1), axis=0),
            jnp.zeros((), table_local.dtype),
        )
        return _tp_sum_with_replicated_bwd(
            part, 'tp', tp_size, mode=fixed_order_reduce.RING
        )

      out_2d = jax.shard_map(
          local_embed,
          mesh=mesh,
          in_specs=(P(None), P('tp', None)),
          out_specs=P(None, None),
          check_vma=False,
      )(flat_ids, table)
      return out_2d.astype(self.dtype).reshape(
          *orig_shape, self.input_embedding.value.shape[1]
      )
    out = self.input_embedding[(x_ids,)]
    out = jnp.astype(out, self.dtype)
    out = shard(out, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    return out

  @jax.named_scope('embedder_decode')
  def decode(
      self,
      x: jaxtyping.ArrayLike,
      *,
      use_fixed_lm_head: bool = False,
      contract: model_contracts.ModelContract | None = None,
      interpret: bool = False,
  ) -> jaxtyping.Array:
    x_arr = jnp.astype(x, self.dtype)
    w_arr = jnp.astype(self.input_embedding.value, self.dtype)
    if not use_fixed_lm_head or contract is None:
      return jnp.dot(x_arr, w_arr.T)

    b, t, d = x_arr.shape
    vocab = w_arr.shape[0]
    x_2d = x_arr.reshape(b * t, d)
    out_2d = _fixed_m_lm_head_2d(
        x_2d,
        w_arr.T,
        contract=contract,
        interpret=interpret,
        endpoint='tied_embed',
    )
    return out_2d.astype(self.dtype).reshape(b, t, vocab)


class CanonRMSNorm(nnx.Module):
  """RMSNorm layer supporting both stock JAX and P22.XH Pallas RMSNorm."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(
            rngs.params(), dim, param_dtype  # pyrefly: ignore[bad-argument-type]
        ),  # pyrefly: ignore[bad-argument-type]
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope('rms_norm')
  def __call__(
      self,
      x: jaxtyping.Array,
      *,
      use_canon: bool = False,
      interpret: bool = False,
  ) -> jaxtyping.Array:
    if not use_canon:
      x_f32 = jnp.astype(x, jnp.float32)
      rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.norm_eps)
      return jnp.astype(
          jnp.astype(self.w.value, jnp.float32) * (x_f32 / rms), self.dtype
      )
    return _canon_rmsnorm_nd(
        x,
        self.w.value,
        norm_eps=self.norm_eps,
        interpret=interpret,
        out_dtype=self.dtype,
    )


class CanonAttention(nnx.Module):
  """Attention module with Zero-TIM kernel switches."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig,
      contract: model_contracts.ModelContract,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.canon_config = canon_config
    self.contract = contract
    self.q_proj = CanonEinsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_proj = CanonEinsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.v_proj = CanonEinsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.o_proj = CanonEinsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_norm = CanonRMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_norm = CanonRMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    seq_len = x.shape[1]
    cc = self.canon_config
    interpret = cc.resolve_interpret()

    q_raw = self.q_proj(
        x,
        use_canon=cc.use_canon_qkv_proj,
        contract=self.contract,
        interpret=interpret,
    )
    k_raw = self.k_proj(
        x,
        use_canon=cc.use_canon_qkv_proj,
        contract=self.contract,
        interpret=interpret,
    )
    value_proj = self.v_proj(
        x,
        use_canon=cc.use_canon_qkv_proj,
        contract=self.contract,
        interpret=interpret,
    )

    query_proj = self.q_norm(
        q_raw, use_canon=cc.use_canon_rmsnorm, interpret=interpret
    )
    key_proj = self.k_norm(
        k_raw, use_canon=cc.use_canon_rmsnorm, interpret=interpret
    )

    query_proj = shard(
        query_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
    )
    key_proj = shard(
        key_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
    )
    value_proj = shard(
        value_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
    )

    if cc.use_canon_rope:
      query_proj = apply_rope_canonical(
          query_proj,
          segment_pos,
          head_dim=self.head_dim,
          rope_theta=self.config.rope_theta,
      )
      key_proj = apply_rope_canonical(
          key_proj,
          segment_pos,
          head_dim=self.head_dim,
          rope_theta=self.config.rope_theta,
      )
    else:
      query_proj = qwen3_stock.apply_rope(
          query_proj,
          segment_pos,
          head_dim=self.head_dim,
          rope_theta=self.config.rope_theta,
      )
      key_proj = qwen3_stock.apply_rope(
          key_proj,
          segment_pos,
          head_dim=self.head_dim,
          rope_theta=self.config.rope_theta,
      )

    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )
      cache_value_proj = value_proj
      cache_key_proj = key_proj
    else:
      cache_value_proj = value_proj
      cache_key_proj = key_proj

    b, t, qh, d = query_proj.shape
    _, _, kh, _ = key_proj.shape

    if cc.use_canon_attention:
      qkv = canonical_online_attention(
          query_proj,
          key_proj,
          value_proj,
          scale=self.scale,
          attn_mask=attn_mask,
          segment_ids=segment_ids,
          kv_block_size=cc.attention_kv_block_size,
      )
    else:
      query_grouped = query_proj.reshape((b, t, kh, qh // kh, d))
      attn = (
          jnp.einsum('BTHGD,BSHD->BHGTS', query_grouped, key_proj) * self.scale
      )
      if attn_mask is not None:
        attn = jnp.where(attn_mask[:, None, None, :, :], attn, K_MASK)
      if segment_ids is not None:
        seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
        attn = jnp.where(seg_mask[:, None, None, :, :], attn, K_MASK)
      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )
      qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
      qkv = qkv.reshape((b, t, qh, d))

    outputs = self.o_proj(
        qkv,
        use_canon=cc.use_canon_o_proj,
        use_fixed_order_reduce=cc.use_fixed_order_reduce,
        contract=self.contract,
        interpret=interpret,
    )
    outputs = shard(
        outputs, self.shd_config.act_btd  # pyrefly: ignore[bad-argument-type]
    )

    if cache is not None:
      new_cache = {
          'v': cache_value_proj,
          'k': cache_key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, outputs

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, segment_pos, cache, attn_mask, segment_ids=segment_ids
      )
    return self.block(x, segment_pos, cache, attn_mask, segment_ids=segment_ids)

  @property
  def head_dim(self):
    return self.o_proj.shape[1]

  @property
  def num_heads(self):
    return self.q_proj.shape[0]

  @property
  def num_kv_heads(self):
    return self.k_proj.shape[1]


class CanonMLP(nnx.Module):
  """MLP module with Zero-TIM `padded_matmul` and `padded_swiglu` switches."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig,
      contract: model_contracts.ModelContract,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.canon_config = canon_config
    self.contract = contract
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_fd,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def _linear_proj(
      self,
      x: jax.Array,
      linear_mod: nnx.Linear,
      *,
      use_canon: bool,
      use_fixed_order_reduce: bool = False,
      contract_parallel: bool = False,
      interpret: bool,
  ) -> jax.Array:
    b, t, d_in = x.shape
    w = jnp.astype(linear_mod.kernel.value, self.config.dtype)
    d_out = w.shape[1]
    tp_size, mesh = _active_tp_size_and_mesh('tp')

    def local_mm(a2: jax.Array, w2: jax.Array) -> jax.Array:
      if use_canon:
        return _canon_matmul_2d(
            a2,
            w2,
            contract=self.contract,
            interpret=interpret,
            out_dtype=jnp.bfloat16,
        )
      return jnp.dot(a2.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)).astype(
          jnp.bfloat16
      )

    if contract_parallel:
      if not use_canon and not use_fixed_order_reduce:
        return linear_mod(x)
      if mesh is not None and tp_size > 1:

        def local_down_proj(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
          a2 = x_loc.reshape(b * t, -1).astype(jnp.bfloat16)
          w2 = w_loc.astype(jnp.bfloat16)
          partial = local_mm(a2, w2)
          if use_fixed_order_reduce:
            acc = _tp_sum_with_replicated_bwd(
                partial, 'tp', tp_size, mode=fixed_order_reduce.GATHER
            )
          else:
            acc = lax.psum(partial, 'tp')
          return acc.astype(self.config.dtype).reshape(b, t, d_out)

        return jax.shard_map(
            local_down_proj,
            mesh=mesh,
            in_specs=(P(None, None, 'tp'), P('tp', None)),
            out_specs=P(None, None, None),
            check_vma=False,
        )(x, w)
      if use_fixed_order_reduce and d_in >= 256 and d_in % 2 == 0:
        tp_shards = 2
        k_per_tp = d_in // tp_shards
        x_tp = x.reshape(b * t, tp_shards, k_per_tp)
        w_tp = w.reshape(tp_shards, k_per_tp, d_out)
        acc = jnp.zeros((b * t, d_out), dtype=jnp.float32)
        for r in range(tp_shards):
          part = local_mm(x_tp[:, r, :], w_tp[r, :, :])
          acc = acc + part.astype(jnp.float32)
        return acc.astype(self.config.dtype).reshape(b, t, d_out)

    if not use_canon:
      return linear_mod(x)

    if mesh is not None and tp_size > 1 and d_out % tp_size == 0:

      def local_col_proj(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
        d_out_loc = int(w_loc.shape[1])

        def _mm(a_3d: jax.Array, weight_2d: jax.Array) -> jax.Array:
          out_2d = _canon_matmul_2d(
              a_3d.reshape(b * t, d_in),
              weight_2d,
              contract=self.contract,
              interpret=interpret,
              out_dtype=self.config.dtype,
          )
          return out_2d.reshape(b, t, d_out_loc)

        return fixed_order_reduce.p59_column_parallel(
            _mm,
            x_loc,
            w_loc,
            replicated=(True, False),
            axis_name='tp',
            count=tp_size,
        )

      return jax.shard_map(
          local_col_proj,
          mesh=mesh,
          in_specs=(P(None, None, None), P(None, 'tp')),
          out_specs=P(None, None, 'tp'),
          check_vma=False,
      )(x, w)

    out_2d = _canon_matmul_2d(
        x.reshape(b * t, d_in),
        w,
        contract=self.contract,
        interpret=interpret,
        out_dtype=self.config.dtype,
    )
    return out_2d.reshape(b, t, d_out)

  def block(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    cc = self.canon_config
    interpret = cc.resolve_interpret()

    gate = self._linear_proj(
        x,
        self.gate_proj,
        use_canon=cc.use_canon_mlp_proj,
        interpret=interpret,
    )
    up = self._linear_proj(
        x,
        self.up_proj,
        use_canon=cc.use_canon_mlp_proj,
        interpret=interpret,
    )

    if cc.use_canon_swiglu:
      b, t, f = gate.shape
      tp_size, mesh = _active_tp_size_and_mesh('tp')

      def swiglu_fwd(g: jax.Array, u: jax.Array) -> jax.Array:
        return padded_swiglu.swiglu(
            g,
            u,
            contract=self.contract,
            interpret=interpret,
            shape_invariant_numerics=True,
        )

      if mesh is not None and tp_size > 1 and f % tp_size == 0:

        def local_swiglu(g_loc: jax.Array, u_loc: jax.Array) -> jax.Array:
          f_loc = int(g_loc.shape[-1])
          g_2d = g_loc.reshape(b * t, f_loc).astype(jnp.bfloat16)
          u_2d = u_loc.reshape(b * t, f_loc).astype(jnp.bfloat16)
          act_2d = canonical_vjp.swiglu(g_2d, u_2d, forward=swiglu_fwd)
          return act_2d.astype(self.config.dtype).reshape(b, t, f_loc)

        activations = jax.shard_map(
            local_swiglu,
            mesh=mesh,
            in_specs=(P(None, None, 'tp'), P(None, None, 'tp')),
            out_specs=P(None, None, 'tp'),
            check_vma=False,
        )(gate, up)
      else:
        gate_2d = gate.reshape(b * t, f).astype(jnp.bfloat16)
        up_2d = up.reshape(b * t, f).astype(jnp.bfloat16)
        act_2d = canonical_vjp.swiglu(gate_2d, up_2d, forward=swiglu_fwd)
        activations = act_2d.astype(self.config.dtype).reshape(b, t, f)
    else:
      activations = nnx.silu(gate) * up

    activations = shard(
        activations, self.shd_config.act_btf  # pyrefly: ignore[bad-argument-type]
    )
    outputs = self._linear_proj(
        activations,
        self.down_proj,
        use_canon=cc.use_canon_mlp_proj,
        use_fixed_order_reduce=cc.use_fixed_order_reduce,
        contract_parallel=True,
        interpret=interpret,
    )
    return outputs

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(state, x)
    return self.block(x)  # pyrefly: ignore[bad-argument-type]


class CanonDecoderLayer(nnx.Module):
  """DecoderLayer with Zero-TIM kernel switches."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig,
      contract: model_contracts.ModelContract,
  ):
    self.config = config
    self.canon_config = canon_config
    self.contract = contract
    self.input_layernorm = CanonRMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.attn = CanonAttention(
        config=config,
        rngs=rngs,
        canon_config=canon_config,
        contract=contract,
    )
    self.post_attention_layernorm = CanonRMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if config.num_experts is None:
      self.mlp = CanonMLP(
          config=config,
          rngs=rngs,
          canon_config=canon_config,
          contract=contract,
      )
    else:
      self.mlp = qwen3_stock.MoELayer(
          config=config,
          rngs=rngs,
      )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    cc = self.canon_config
    interpret = cc.resolve_interpret()
    inputs_normalized = self.input_layernorm(
        x, use_canon=cc.use_canon_rmsnorm, interpret=interpret
    )
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
        segment_ids=segment_ids,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(
        attn_output, use_canon=cc.use_canon_rmsnorm, interpret=interpret
    )
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.DECODER
        or self.config.remat_config == RematConfig.DECODER.value
    ):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, segment_pos, cache, attn_mask, segment_ids=segment_ids
      )
    return self.block(x, segment_pos, cache, attn_mask, segment_ids=segment_ids)


class Qwen3Canon(BackendMappingMixin, nnx.Module):
  """Canonical Qwen3 model with configurable Zero-TIM kernel switches."""

  BACKEND_PACKAGE_PATH = qwen3_stock.__name__

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig = CanonKernelConfig.all_enabled(),
      contract: model_contracts.ModelContract | None = None,
  ):
    self.config = config
    self.canon_config = canon_config
    tp_size, _ = _active_tp_size_and_mesh('tp')
    self.contract = (
        resolve_or_build_contract(config, tp_size=tp_size)
        if contract is None
        else contract
    )
    self.embedder = CanonEmbedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=self.config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        CanonDecoderLayer(
            config=config,
            rngs=rngs,
            canon_config=canon_config,
            contract=self.contract,
        )
        for _ in range(config.num_layers)
    ])
    self.final_norm = CanonRMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=self.config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = CanonEinsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=self.config.shd_config.emb_dv,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

  def set_canon_config(self, canon_config: CanonKernelConfig) -> None:
    """Updates `canon_config` across the model and all decoder layers in-place."""
    self.canon_config = canon_config
    for layer in self.layers:
      layer.canon_config = canon_config
      layer.attn.canon_config = canon_config
      if isinstance(layer.mlp, CanonMLP):
        layer.mlp.canon_config = canon_config

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    """Initializes the KV cache for the model."""
    config = self.config
    shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    return {
        f'layer_{i}': {
            'k': jnp.zeros(shape, dtype=config.dtype),
            'v': jnp.zeros(shape, dtype=config.dtype),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
        for i in range(config.num_layers)
    }

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
      output_hidden_states: bool = False,
      segment_ids: jaxtyping.Array | None = None,  # [B, L]
      skip_lm_head: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    new_cache = None if cache is None else {}
    cc = self.canon_config
    x = self.embedder.encode(
        input_tokens, use_fixed_order_reduce=cc.use_fixed_order_reduce
    )

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
          segment_ids=segment_ids,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    cc = self.canon_config
    interpret = cc.resolve_interpret()
    x = self.final_norm(x, use_canon=cc.use_canon_rmsnorm, interpret=interpret)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)

    if skip_lm_head:
      return x, new_cache

    logits = self.compute_final_logits(x)
    return logits, new_cache  # pytype: disable=bad-return-type

  def compute_final_logits(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    """Computes final float32 logits from normalized hidden states."""
    cc = self.canon_config
    interpret = cc.resolve_interpret()
    if self.config.use_tied_embedding:
      logits = self.embedder.decode(
          x,
          use_fixed_lm_head=cc.use_fixed_lm_head,
          contract=self.contract,
          interpret=interpret,
      )
    else:
      logits = self.lm_head(
          x,
          use_canon=cc.use_fixed_lm_head,
          contract=self.contract,
          interpret=interpret,
      )
    return jnp.astype(logits, jnp.float32)

  def compute_log_softmax(
      self,
      logits: jaxtyping.Array,
  ) -> jaxtyping.Array:
    """Computes log-probabilities over the vocabulary."""
    return compute_log_softmax(
        logits,
        use_canon_logsoftmax=self.canon_config.use_canon_logsoftmax,
        interpret=self.canon_config.resolve_interpret(),
    )

  def compute_token_logprobs(
      self,
      logits: jaxtyping.Array,
      token_ids: jaxtyping.Array,
  ) -> jaxtyping.Array:
    """Computes gathered log-probabilities for specific `token_ids`."""
    return compute_token_logprobs(
        logits,
        token_ids,
        use_canon_logsoftmax=self.canon_config.use_canon_logsoftmax,
        interpret=self.canon_config.resolve_interpret(),
    )

  def get_model_input(self):
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'input_tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
    }


def compute_log_softmax(
    logits: jaxtyping.Array,
    *,
    use_canon_logsoftmax: bool = True,
    interpret: bool | None = None,
) -> jax.Array:
  """Computes full-vocabulary log-softmax (stock or Zero-TIM canonical).

  Handles arbitrary batch/sequence dimensions by flattening to 2D `[M, V]`,
  padding `M` to a valid `ROW_BUCKET_ALIGN=8` bucket (or chunking by
  `PRODUCTION_M=256`), and invoking `canonical_logsoftmax.log_softmax`.

  Args:
    logits: Float array `[..., V]`.
    use_canon_logsoftmax: Whether to use Zero-TIM `canonical_logsoftmax`.
    interpret: Optional Pallas interpret mode override.

  Returns:
    Log-probabilities of the same shape as `logits` in `float32`.
  """
  logits_f32 = jnp.astype(logits, jnp.float32)
  if not use_canon_logsoftmax:
    return jax.nn.log_softmax(logits_f32, axis=-1)

  if interpret is None:
    interpret = jax.default_backend() == 'cpu'
  vocab = int(logits_f32.shape[-1])

  os.environ.setdefault(canonical_logsoftmax.ENV, '1')
  orig_shape = logits_f32.shape
  m = int(logits_f32.size // vocab)
  flat = logits_f32.reshape(m, vocab)

  row_align = 1 if interpret else canonical_logsoftmax.ROW_BUCKET_ALIGN
  prod_m = canonical_logsoftmax.PRODUCTION_M
  tp_size, mesh = _active_tp_size_and_mesh('tp')

  def _run_2d(mat_2d: jax.Array) -> jax.Array:
    return canonical_logsoftmax.log_softmax(
        mat_2d, interpret=interpret, strict_vocab=False
    )

  if mesh is not None and tp_size > 1:
    run_2d = jax.shard_map(
        _run_2d,
        mesh=mesh,
        in_specs=P(None, None),
        out_specs=P(None, None),
        check_vma=False,
    )
  else:
    run_2d = _run_2d

  if m <= prod_m:
    mp = _round_up(m, row_align)
    if mp != m:
      padded = jnp.pad(
          flat, ((0, mp - m), (0, 0)), constant_values=jnp.float32(0.0)
      )
    else:
      padded = flat
    out_2d = run_2d(padded)[:m, :]
    return out_2d.reshape(orig_shape)

  mp = _round_up(m, prod_m)
  if mp != m:
    padded = jnp.pad(
        flat, ((0, mp - m), (0, 0)), constant_values=jnp.float32(0.0)
    )
  else:
    padded = flat
  chunks = padded.reshape(-1, prod_m, vocab)
  out_chunks = lax.map(run_2d, chunks)
  return out_chunks.reshape(mp, vocab)[:m, :].reshape(orig_shape)


def compute_token_logprobs(
    logits: jaxtyping.Array,
    token_ids: jaxtyping.Array,
    *,
    use_canon_logsoftmax: bool = True,
    interpret: bool | None = None,
) -> jax.Array:
  """Computes per-token log-probabilities for `token_ids` (shape `[...]`).

  When `use_canon_logsoftmax=True`, uses
  `canonical_logsoftmax.gathered_logprobs`
  in forward-only mode and `compute_log_softmax` + `take_along_axis` under
  autodiff so gradients flow cleanly through the custom VJP.

  Args:
    logits: Logits tensor `[..., V]`.
    token_ids: Integer token IDs `[...]`.
    use_canon_logsoftmax: Whether to use Zero-TIM `canonical_logsoftmax`.
    interpret: Optional Pallas interpret mode override.

  Returns:
    Per-token log-probabilities `float32[...]` matching `token_ids.shape`.
  """
  log_probs = compute_log_softmax(
      logits,
      use_canon_logsoftmax=use_canon_logsoftmax,
      interpret=interpret,
  )
  return jnp.take_along_axis(
      log_probs, token_ids.astype(jnp.int32)[..., None], axis=-1
  )[..., 0]


def get_tp_sharding_config(tp_axis: str = 'tp') -> ShardingConfig:
  """Returns TP-sharded weights and activations with replicated residual stream."""
  return ShardingConfig(
      emb_vd=P(tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      emb_dv=P(None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      q_weight_dnh=P(None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      kv_weight_dnh=P(None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      o_weight_nhd=P(tp_axis, None, None),  # pyrefly: ignore[bad-argument-type]
      ffw_weight_df=P(None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      ffw_weight_fd=P(tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      rms_norm_weight=P(None),  # pyrefly: ignore[bad-argument-type]
      act_btd=P(None, None, None),  # pyrefly: ignore[bad-argument-type]
      act_btf=P(None, None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      act_btnh=P(None, None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      score_weight_d1=P(None, None),  # pyrefly: ignore[bad-argument-type]
      exp_weight_edf=P(None, None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      exp_weight_efd=P(None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
  )


def init_non_zero_mlp_weights(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    *,
    seed: int = 42,
) -> None:
  """Initializes synthetic model weights with realistic fan-in scaling.

  Stock `Qwen3` initializes MLP linear kernels to zeros (expecting checkpoint
  restore) and 3D `Einsum` weights with default `glorot_uniform` (which treats
  the leading dimension as a spatial receptive field, yielding `stddev ~ 0.007`
  relative to `embedder` `stddev = 1.0`). Populating balanced fan-in scaled
  weights ensures that every sub-layer (`qkv_proj`, `o_proj`, `mlp_proj`,
  `swiglu`, `lm_head`) contributes at realistic magnitude to the residual
  stream in `bfloat16`.

  Args:
    model: A `Qwen3` or `Qwen3Canon` instance to initialize in-place.
    seed: PRNG seed.
  """
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  shd_cfg = model.config.shd_config

  def _place(arr: jax.Array, spec) -> jax.Array:
    if mesh is None or tp_size <= 1:
      return arr
    pspec = spec if isinstance(spec, P) else P(*spec)
    return jax.device_put(arr, jax.sharding.NamedSharding(mesh, pspec))

  key = jax.random.PRNGKey(seed)
  emb_Key, key = jax.random.split(key)
  emb_w = model.embedder.input_embedding.value
  emb_scale = (float(emb_w.shape[1]) ** -0.5) * 0.8
  model.embedder.input_embedding.value = _place(
      (
          jax.random.normal(emb_Key, emb_w.shape, dtype=jnp.float32) * emb_scale
      ).astype(emb_w.dtype),
      shd_cfg.emb_vd,
  )

  if hasattr(model, 'lm_head'):
    lm_key, key = jax.random.split(key)
    lm_w = model.lm_head.w.value
    lm_scale = float(lm_w.shape[0]) ** -0.5
    model.lm_head.w.value = _place(
        (
            jax.random.normal(lm_key, lm_w.shape, dtype=jnp.float32) * lm_scale
        ).astype(lm_w.dtype),
        shd_cfg.emb_dv,
    )

  model.final_norm.w.value = _place(model.final_norm.w.value, P(None))

  for layer in model.layers:
    layer.input_layernorm.w.value = _place(
        layer.input_layernorm.w.value, P(None)
    )
    layer.post_attention_layernorm.w.value = _place(
        layer.post_attention_layernorm.w.value, P(None)
    )
    layer.attn.q_norm.w.value = _place(layer.attn.q_norm.w.value, P(None))
    layer.attn.k_norm.w.value = _place(layer.attn.k_norm.w.value, P(None))

    k_q, k_k, k_v, k_o, key = jax.random.split(key, 5)
    for proj_mod, k_proj, fan_in, w_spec in (
        (
            layer.attn.q_proj,
            k_q,
            layer.attn.q_proj.w.value.shape[0],
            shd_cfg.q_weight_dnh,
        ),
        (
            layer.attn.k_proj,
            k_k,
            layer.attn.k_proj.w.value.shape[0],
            shd_cfg.kv_weight_dnh,
        ),
        (
            layer.attn.v_proj,
            k_v,
            layer.attn.v_proj.w.value.shape[0],
            shd_cfg.kv_weight_dnh,
        ),
        (
            layer.attn.o_proj,
            k_o,
            layer.attn.o_proj.w.value.shape[0]
            * layer.attn.o_proj.w.value.shape[1],
            shd_cfg.o_weight_nhd,
        ),
    ):
      w_val = proj_mod.w.value
      scale = (float(fan_in) ** -0.5) * 0.8
      proj_mod.w.value = _place(
          (
              jax.random.normal(k_proj, w_val.shape, dtype=jnp.float32) * scale
          ).astype(w_val.dtype),
          w_spec,
      )

    if hasattr(layer.mlp, 'gate_proj') and isinstance(
        layer.mlp.gate_proj, nnx.Linear
    ):
      k1, k2, k3, key = jax.random.split(key, 4)
      gate_w = layer.mlp.gate_proj.kernel.value
      up_w = layer.mlp.up_proj.kernel.value
      down_w = layer.mlp.down_proj.kernel.value
      scale_in = (float(gate_w.shape[0]) ** -0.5) * 0.8
      scale_down = (float(down_w.shape[0]) ** -0.5) * 0.8
      layer.mlp.gate_proj.kernel.value = _place(
          (
              jax.random.normal(k1, gate_w.shape, dtype=jnp.float32) * scale_in
          ).astype(gate_w.dtype),
          shd_cfg.ffw_weight_df,
      )
      layer.mlp.up_proj.kernel.value = _place(
          (
              jax.random.normal(k2, up_w.shape, dtype=jnp.float32) * scale_in
          ).astype(up_w.dtype),
          shd_cfg.ffw_weight_df,
      )
      layer.mlp.down_proj.kernel.value = _place(
          (
              jax.random.normal(k3, down_w.shape, dtype=jnp.float32)
              * scale_down
          ).astype(down_w.dtype),
          shd_cfg.ffw_weight_fd,
      )


def copy_weights(
    src_model: qwen3_stock.Qwen3 | Qwen3Canon,
    dst_model: qwen3_stock.Qwen3 | Qwen3Canon,
) -> None:
  """Copies all `nnx.Param` weights from `src_model` into `dst_model`."""
  _, src_params = nnx.split(src_model, nnx.Param)
  nnx.update(dst_model, src_params)


@partial(
    jax.jit,
    static_argnames=('max_new_tokens', 'cache_size', 'use_canon_logsoftmax'),
)
def forward_sample_with_logprobs(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    prompt_tokens: jax.Array,  # [B, L_prompt]
    *,
    max_new_tokens: int,
    cache_size: int,
    use_canon_logsoftmax: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs KV-cached prefill + greedy decode sampling and returns logprobs.

  Args:
    model: `Qwen3` or `Qwen3Canon` model.
    prompt_tokens: Input prompt tokens `[B, L_prompt]`.
    max_new_tokens: Number of tokens to generate autoregressively.
    cache_size: Total KV cache length (`>= L_prompt + max_new_tokens`).
    use_canon_logsoftmax: Whether to use `canonical_logsoftmax` for scoring.

  Returns:
    Tuple of `(full_tokens, decode_step_logprobs, prefill_prompt_logprobs)`:
      - `full_tokens`: `[B, L_prompt + max_new_tokens]` int32 token IDs
      - `decode_step_logprobs`: `[B, max_new_tokens]` float32 logprobs computed
        at each step during KV-cached decoding (Path A: Rollout / Sampler)
      - `prefill_prompt_logprobs`: `[B, L_prompt - 1]` float32 logprobs of
        prompt tokens `[1..L_prompt-1]` computed during prefill
  """
  b, l_prompt = prompt_tokens.shape
  cache = model.init_cache(b, cache_size, model.config.dtype)
  for layer_cache in cache.values():
    layer_cache['k'] = shard(
        layer_cache['k'], model.config.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
    )
    layer_cache['v'] = shard(
        layer_cache['v'], model.config.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
    )

  # 1. Prefill over prompt_tokens [B, L_prompt] with KV cache [B, L_prompt, cache_size]
  prompt_pos = jnp.broadcast_to(
      jnp.arange(l_prompt, dtype=jnp.int32)[None, :], (b, l_prompt)
  )
  cache_positions = jnp.arange(cache_size, dtype=jnp.int32)[None, None, :]
  prefill_mask = jnp.logical_and(
      cache_positions <= prompt_pos[:, :, None],
      cache_positions < l_prompt,
  )
  prefill_logits, cache = model(prompt_tokens, prompt_pos, cache, prefill_mask)

  if l_prompt > 1:
    prefill_prompt_logprobs = compute_token_logprobs(
        prefill_logits[:, :-1, :],
        prompt_tokens[:, 1:],
        use_canon_logsoftmax=use_canon_logsoftmax,
    )
  else:
    prefill_prompt_logprobs = jnp.zeros((b, 0), dtype=jnp.float32)

  # First generated token comes from the last prompt position's logits
  last_logits = prefill_logits[:, -1:, :]  # [B, 1, V]
  first_next_token = jnp.argmax(last_logits, axis=-1).astype(
      jnp.int32
  )  # [B, 1]
  first_logprob = compute_token_logprobs(
      last_logits,
      first_next_token,
      use_canon_logsoftmax=use_canon_logsoftmax,
  )  # [B, 1]

  if max_new_tokens == 1:
    full_tokens = jnp.concatenate([prompt_tokens, first_next_token], axis=1)
    return full_tokens, first_logprob, prefill_prompt_logprobs

  # 2. Autoregressive Decode loop for steps 1 .. max_new_tokens - 1
  def decode_step(carry, step_idx):
    curr_cache, prev_token = carry
    cur_pos = jnp.full((b, 1), l_prompt + step_idx, dtype=jnp.int32)
    step_mask = cache_positions <= cur_pos[:, :, None]
    step_logits, next_cache = model(prev_token, cur_pos, curr_cache, step_mask)
    next_token = jnp.argmax(step_logits, axis=-1).astype(jnp.int32)
    step_logprob = compute_token_logprobs(
        step_logits,
        next_token,
        use_canon_logsoftmax=use_canon_logsoftmax,
    )
    return (next_cache, next_token), (next_token[:, 0], step_logprob[:, 0])

  _, (rest_tokens, rest_logprobs) = lax.scan(
      decode_step,
      (cache, first_next_token),
      jnp.arange(max_new_tokens - 1, dtype=jnp.int32),
  )
  # rest_tokens, rest_logprobs have shape [max_new_tokens - 1, B] -> [B, max_new_tokens - 1]
  gen_tokens = jnp.concatenate([first_next_token, rest_tokens.T], axis=1)
  decode_logprobs = jnp.concatenate([first_logprob, rest_logprobs.T], axis=1)
  full_tokens = jnp.concatenate([prompt_tokens, gen_tokens], axis=1)
  return full_tokens, decode_logprobs, prefill_prompt_logprobs


@partial(jax.jit, static_argnames=('use_canon_logsoftmax',))
def score_sequence_prefill(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    full_tokens: jax.Array,  # [B, L_total]
    *,
    use_canon_logsoftmax: bool = False,
) -> jax.Array:
  """Computes next-token logprobs over `full_tokens` via cacheless prefill (Path B)."""
  b, l_total = full_tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(l_total, dtype=jnp.int32)[None, :], (b, l_total)
  )
  idx = jnp.arange(l_total, dtype=jnp.int32)
  causal_mask = jnp.broadcast_to(
      (idx[:, None] >= idx[None, :])[None, :, :], (b, l_total, l_total)
  )
  logits, _ = model(full_tokens, positions, None, causal_mask)
  return compute_token_logprobs(
      logits[:, :-1, :],
      full_tokens[:, 1:],
      use_canon_logsoftmax=use_canon_logsoftmax,
  )


@partial(jax.jit, static_argnames=('use_canon_logsoftmax',))
def score_sequence_fwd_bwd(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    full_tokens: jax.Array,  # [B, L_total]
    *,
    use_canon_logsoftmax: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Computes next-token logprobs inside a `nnx.value_and_grad` trace (Path C).

  In RL training (GRPO/PPO), the learner computes `old_logprobs` /
  `current_logprobs` inside a forward+backward autodiff compilation unit.
  Without Zero-TIM custom VJPs and canonical kernels, XLA re-fuses the primal
  forward graph differently when paired with a backward pass.

  Args:
    model: `Qwen3` or `Qwen3Canon` model.
    full_tokens: Sequence tokens `[B, L_total]`.
    use_canon_logsoftmax: Whether to use `canonical_logsoftmax`.

  Returns:
    Tuple `(token_logprobs, grad_norm)`:
      - `token_logprobs`: `[B, L_total - 1]` float32 next-token logprobs
      - `grad_norm`: scalar float32 L2 norm of parameter gradients
  """
  b, l_total = full_tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(l_total, dtype=jnp.int32)[None, :], (b, l_total)
  )
  idx = jnp.arange(l_total, dtype=jnp.int32)
  causal_mask = jnp.broadcast_to(
      (idx[:, None] >= idx[None, :])[None, :, :], (b, l_total, l_total)
  )

  graphdef, params, other_state = nnx.split(model, nnx.Param, ...)

  def loss_fn(param_state):
    m = nnx.merge(graphdef, param_state, other_state)
    logits, _ = m(full_tokens, positions, None, causal_mask)
    token_lps = compute_token_logprobs(
        logits[:, :-1, :],
        full_tokens[:, 1:],
        use_canon_logsoftmax=use_canon_logsoftmax,
    )
    loss = -jnp.mean(token_lps)
    return loss, token_lps

  (_, token_logprobs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
  grad_leaves = jax.tree_util.tree_leaves(grads)
  grad_sq = sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in grad_leaves)
  return token_logprobs, jnp.sqrt(grad_sq)
