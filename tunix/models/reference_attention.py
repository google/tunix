# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical pure-XLA reference scaled dot-product attention for Tunix models."""

import math
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
import numpy as np

AxisSpec = str | tuple[str, ...] | None
DEFAULT_MASK_VALUE: float = -0.7 * float(np.finfo(np.dtype("float32")).max)


def extract_token_mask(
    attention_mask: jaxtyping.Array | None,
    seq_len: int,
    *,
    segment_ids: jaxtyping.Array | None = None,
    dtype: jax.typing.DTypeLike = jnp.bool_,
) -> jaxtyping.Array | None:
  """Extracts per-token validity mask `(B, T)` from `segment_ids` or `attention_mask`.

  For 3D/4D `attention_mask` with `seq_len > 1`, assumes initial prefill
  (`end_index == 0`) so that query position `t` corresponds to key column `t`
  along the diagonal of `attention_mask[:, :seq_len, :seq_len]`.

  Args:
    attention_mask: Boolean mask of shape `(B, T)`, `(B, T, S)`, or `(B, 1, T,
      S)`.
    seq_len: Query sequence length `T`.
    segment_ids: Optional explicit token segment IDs of shape `(B, T)`, which
      take precedence over `attention_mask` when provided.
    dtype: Output dtype for the returned mask (defaults to `jnp.bool_`).

  Returns:
    Validity mask of shape `(B, T)` with `dtype`, or `None` if both
    `segment_ids` and `attention_mask` are `None`.
  """
  if segment_ids is not None:
    return segment_ids[:, :seq_len].astype(dtype)
  if attention_mask is None:
    return None
  if attention_mask.ndim == 2:
    return attention_mask[:, :seq_len].astype(dtype)
  mask = (
      attention_mask[:, 0, :, :] if attention_mask.ndim == 4 else attention_mask
  )
  out = (
      jnp.any(mask, axis=-1)
      if seq_len == 1
      else jnp.diagonal(mask[:, :seq_len, :seq_len], axis1=1, axis2=2)
  )
  return out.astype(dtype)


def _axis_shards(
    axis: AxisSpec,
    mesh_shape: dict[str, int],
) -> int:
  """Returns the product of mesh axis sizes for `axis`, or `0` if invalid."""
  if not axis:
    return 0
  names = (axis,) if isinstance(axis, str) else tuple(axis)
  if not set(names).issubset(mesh_shape):
    return 0
  return math.prod(map(mesh_shape.__getitem__, names))


def resolve_splash_sharding(
    act_btnh: tuple[AxisSpec, ...],
    batch_size: int,
    kv_heads: int,
) -> tuple[
    shd.Mesh | shd.AbstractMesh,
    AxisSpec,
    AxisSpec,
    AxisSpec,
    AxisSpec,
    AxisSpec,
    int,
    int,
]:
  """Resolves active mesh and validated sharding axes for TPU Splash Attention."""
  # TODO(tunix-dev): Remove deprecated pxla.thread_resources.env.physical_mesh
  # lookup once callers migrate to jax.set_mesh and shd.get_abstract_mesh().
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty:
    mesh = shd.get_abstract_mesh()
  shd_b, shd_t, shd_n, shd_h = act_btnh
  shape = mesh.shape
  b_shards = _axis_shards(shd_b, shape)
  t_shards = _axis_shards(shd_t, shape)
  n_shards = _axis_shards(shd_n, shape)
  h_shards = _axis_shards(shd_h, shape)
  shd_b = shd_b if b_shards > 0 and batch_size % b_shards == 0 else None
  shd_t = shd_t if t_shards > 0 else None
  shd_n = shd_n if n_shards > 0 else None
  shd_h = shd_h if h_shards > 0 else None
  head_shards = max(1, n_shards)
  q_seq_shards = max(1, t_shards)
  shd_n_kv = (
      shd_n if shd_n is not None and kv_heads % head_shards == 0 else None
  )
  return (
      mesh,
      shd_b,
      shd_t,
      shd_n,
      shd_h,
      shd_n_kv,
      head_shards,
      q_seq_shards,
  )


def reference_attention(
    query: jaxtyping.Array,
    key: jaxtyping.Array,
    value: jaxtyping.Array,
    *,
    scale: float | jaxtyping.Array | None = None,
    attn_mask: jaxtyping.Array | None = None,
    segment_ids: jaxtyping.Array | None = None,
    is_causal: bool = False,
    local_window_size: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    softmax_dtype: jax.typing.DTypeLike | None = jnp.float32,
) -> jaxtyping.Array:
  """Computes pure-XLA MHA/GQA attention over `[B, T, N, H]` tensors.

  Serves as the single canonical reference implementation shared across both
  model eager fallback paths (`Qwen3`, `Qwen2`, `Qwen3.5`, `Gemma`, `Gemma3`,
  `Gemma4`, `Llama3`) and GPU/TPU flash-attention parity tests.

  Args:
    query: Query tensor of shape `[B, T, N, H]`.
    key: Key tensor of shape `[B, S, K, H]`, where `N % K == 0`.
    value: Value tensor of shape `[B, S, K, V]`.
    scale: Multiplicative logit scale (`query.shape[-1] ** -0.5` when `None`).
    attn_mask: Optional boolean attention mask of shape `[B, T, S]` or `[B, 1 |
      N, T, S]`.
    segment_ids: Optional per-token segment IDs of shape `[B, T]` (`S == T`).
    is_causal: Whether to apply an additional lower-triangular causal mask.
    local_window_size: Optional `(left_window, right_window)` sliding window.
    logits_soft_cap: Optional `tanh(logits / cap) * cap` soft-cap bound.
    mask_value: Fill value for masked-out attention logits.
    softmax_dtype: Precision for `softmax` computation (`None` keeps
      `logits.dtype`).

  Returns:
    Attention output tensor of shape `[B, T, N, V]` with dtype `value.dtype`.
  """
  b, t, qh, d = query.shape
  s = key.shape[1]
  kh = key.shape[2]
  vd = value.shape[3]
  if qh % kh != 0:
    raise ValueError(
        f"Query heads ({qh}) must be divisible by KV heads ({kh})."
    )

  if qh == kh:
    logits = jnp.einsum("BTNH,BSNH->BTNS", query, key)
  else:
    logits = jnp.einsum(
        "BTKGH,BSKH->BTKGS",
        query.reshape((b, t, kh, qh // kh, d)),
        key,
    ).reshape((b, t, qh, s))
  if scale is None:
    scale = d**-0.5
  if scale != 1.0:
    logits = logits * scale

  if logits_soft_cap is not None:
    logits = jnp.tanh(logits / logits_soft_cap) * logits_soft_cap

  if is_causal:
    causal_mask = jnp.arange(t)[:, None] >= jnp.arange(s)[None, :]
    logits = jnp.where(causal_mask[None, :, None, :], logits, mask_value)

  if local_window_size is not None:
    left_window, right_window = local_window_size
    dist = jnp.arange(t)[:, None] - jnp.arange(s)[None, :]
    win_mask = (dist <= left_window) & (-dist <= right_window)
    logits = jnp.where(win_mask[None, :, None, :], logits, mask_value)

  if attn_mask is not None:
    mask_nd = (
        jnp.swapaxes(attn_mask, 1, 2)
        if attn_mask.ndim == 4
        else jnp.expand_dims(attn_mask, -2)
    )
    logits = jnp.where(mask_nd, logits, mask_value)

  if segment_ids is not None:
    seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    logits = jnp.where(seg_mask[:, :, None, :], logits, mask_value)

  probs_in = logits if softmax_dtype is None else logits.astype(softmax_dtype)
  probs = jax.nn.softmax(probs_in, axis=-1).astype(value.dtype)
  if qh == kh:
    return jnp.einsum("BTNS,BSNV->BTNV", probs, value)
  probs_grouped = probs.reshape((b, t, kh, qh // kh, s))
  out = jnp.einsum("BTKGS,BSKV->BTKGV", probs_grouped, value)
  return out.reshape((b, t, qh, vd))
