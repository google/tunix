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

"""Qwen3-VL vision encoder implementation."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Optional

from flax import nnx
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np


def rotate_half(x: jax.Array) -> jax.Array:
  """Rotate half the hidden dims of the input"""
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate((-x2, x1), axis=-1)


@struct.dataclass
class VisionEmbeddings:
  """Container for vision tower outputs: tokens + optional deepstack features"""

  tokens: jax.Array
  deepstack: tuple[jax.Array, ...] = ()

  @classmethod
  def concatenate(
      cls, embeds: Sequence["VisionEmbeddings"]
  ) -> "VisionEmbeddings":
    if not embeds:
      return cls(tokens=jnp.zeros((0, 0), dtype=jnp.float16), deepstack=())
    tokens = jnp.concatenate([e.tokens for e in embeds], axis=0)
    base_len = len(embeds[0].deepstack)
    for e in embeds[1:]:
      if len(e.deepstack) != base_len:
        raise ValueError("All VisionEmbeddings must have same deepstack length")
    deepstack = tuple(
        jnp.concatenate([e.deepstack[i] for e in embeds], axis=0)
        for i in range(base_len)
    )
    return cls(tokens=tokens, deepstack=deepstack)

  def cast(self, dtype: jnp.dtype) -> "VisionEmbeddings":
    return VisionEmbeddings(
        tokens=self.tokens.astype(dtype),
        deepstack=tuple(f.astype(dtype) for f in self.deepstack),
    )

  def with_batch_dim(self, batch: int) -> "VisionEmbeddings":
    """Ensure batch dimension matches expected size"""
    tokens = self.tokens if self.tokens.ndim == 3 else self.tokens[None, ...]
    if tokens.shape[0] == 1 and batch > 1:
      tokens = jnp.tile(tokens, (batch, 1, 1))
    deepstack = []
    for feat in self.deepstack:
      if feat.ndim == 2:
        feat = feat[None, ...]
      if feat.shape[0] == 1 and batch > 1:
        feat = jnp.tile(feat, (batch, 1, 1))
      deepstack.append(feat)
    return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))


@struct.dataclass
class VisionGridData:
  """Pre-computed positional data derived from grid_thw.

  All fields are static (no model weights needed). Create via
  compute_grid_data() outside the JIT boundary, then pass into the
  JIT-compiled forward pass.
  """

  cos: jax.Array  # [total_patches, rotary_dim*2]   float32
  sin: jax.Array  # [total_patches, rotary_dim*2]   float32
  cu_seqlens: jax.Array  # [num_frames + 1]                int32
  pos_embed_idx: jax.Array  # [4, total_hw]  int32   bilinear corner indices
  pos_embed_weights: jax.Array  # [4, total_hw]  float32 bilinear weights
  pos_embed_gather: (
      jax.Array
  )  # [total_patches] int32  temporal-repeat + merge reorder


def compute_grid_data(grid_thw, spec: "VisionModelConfig") -> VisionGridData:
  """Compute static positional data from grid_thw.

  Call outside the JIT boundary.

  Args:
    grid_thw: image/video grid sizes, array-like of shape [N, 3] with columns
      (temporal, height, width).
    spec: VisionModelConfig describing the model geometry.

  Returns:
    VisionGridData with all fields as JAX arrays ready to be passed into JIT.
  """
  grid = np.asarray(grid_thw, dtype=np.int32)  # [N, 3]
  merge = spec.spatial_merge_size
  rotary_dim = (spec.hidden_size // spec.num_heads) // 2

  # ── rotary position ids ───
  pos_chunks = []
  for i in range(grid.shape[0]):
    t, h, w = int(grid[i, 0]), int(grid[i, 1]), int(grid[i, 2])
    hpos = np.repeat(np.arange(h)[:, None], w, axis=1)  # [h, w]
    wpos = np.repeat(np.arange(w)[None, :], h, axis=0)  # [h, w]
    hpos = (
        hpos.reshape(h // merge, merge, w // merge, merge)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    wpos = (
        wpos.reshape(h // merge, merge, w // merge, merge)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    pos = np.stack([hpos, wpos], axis=-1)  # [h*w, 2]
    pos_chunks.append(np.tile(pos, (t, 1)))  # [t*h*w, 2]

  pos_ids = (
      np.concatenate(pos_chunks, axis=0)
      if pos_chunks
      else np.zeros((0, 2), dtype=np.int32)
  )
  max_grid = int(np.max(grid[:, 1:])) if grid.size > 0 else 1
  inv_freq = 1.0 / (
      10000.0 ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
  )
  rotary_table = np.outer(
      np.arange(max_grid, dtype=np.float32), inv_freq
  )  # [max_grid, rotary_dim//2]
  rotary_emb = rotary_table[pos_ids].reshape(
      pos_ids.shape[0], -1
  )  # [total_patches, rotary_dim]
  emb = np.concatenate([rotary_emb, rotary_emb], axis=-1)
  cos = np.cos(emb).astype(np.float32)
  sin = np.sin(emb).astype(np.float32)

  # ── cu_seqlens ───
  frame_sizes = np.repeat(grid[:, 1] * grid[:, 2], grid[:, 0])
  cu_seqlens = np.concatenate([[0], np.cumsum(frame_sizes)]).astype(np.int32)

  # ── bilinear interpolation indices / weights for pos_embed ───
  total_patches = int(np.sum(grid[:, 0] * grid[:, 1] * grid[:, 2]))
  if spec.num_position_embeddings is not None:
    num_g = int(spec.num_position_embeddings**0.5)
    idx_list: list[list[np.ndarray]] = [[] for _ in range(4)]
    weight_list: list[list[np.ndarray]] = [[] for _ in range(4)]
    for i in range(grid.shape[0]):
      h, w = int(grid[i, 1]), int(grid[i, 2])
      h_f = np.linspace(0, num_g - 1, h, dtype=np.float32)
      w_f = np.linspace(0, num_g - 1, w, dtype=np.float32)
      hfl = h_f.astype(np.int32)
      wfl = w_f.astype(np.int32)
      hce = np.clip(hfl + 1, 0, num_g - 1)
      wce = np.clip(wfl + 1, 0, num_g - 1)
      dh = h_f - hfl
      dw = w_f - wfl
      bh = hfl * num_g
      bhc = hce * num_g
      for k, (ri, rw) in enumerate([
          (bh[:, None] + wfl[None], (1.0 - dh)[:, None] * (1.0 - dw)[None]),
          (bh[:, None] + wce[None], (1.0 - dh)[:, None] * dw[None]),
          (bhc[:, None] + wfl[None], dh[:, None] * (1.0 - dw)[None]),
          (bhc[:, None] + wce[None], dh[:, None] * dw[None]),
      ]):
        idx_list[k].append(ri.reshape(-1))
        weight_list[k].append(rw.reshape(-1).astype(np.float32))
    pos_embed_idx = np.stack(
        [np.concatenate(idx_list[k]) for k in range(4)], axis=0
    ).astype(np.int32)
    pos_embed_weights = np.stack(
        [np.concatenate(weight_list[k]) for k in range(4)], axis=0
    ).astype(np.float32)

    # ── gather index (temporal repeat + spatial merge reorder) ──────────────
    # Maps output position j to patch_pos_embeds source index.
    # Output layout after repeat+reshape+transpose: (t, h//m, w//m, m, m)
    # Row-major strides:  [h*w, w*m, m*m, m, 1]
    gather_idx = np.empty(total_patches, dtype=np.int32)
    offset_hw, offset_thw = 0, 0
    for i in range(grid.shape[0]):
      t, h, w = int(grid[i, 0]), int(grid[i, 1]), int(grid[i, 2])
      j = np.arange(t * h * w)
      rem = j % (h * w)
      hc = rem // (w * merge)
      rem1 = rem % (w * merge)
      wc = rem1 // (merge * merge)
      rem2 = rem1 % (merge * merge)
      mh = rem2 // merge
      mw = rem2 % merge
      source = (hc * merge + mh) * w + (wc * merge + mw)
      gather_idx[offset_thw : offset_thw + t * h * w] = offset_hw + source
      offset_hw += h * w
      offset_thw += t * h * w
  else:
    pos_embed_idx = np.zeros((4, 0), dtype=np.int32)
    pos_embed_weights = np.zeros((4, 0), dtype=np.float32)
    gather_idx = np.zeros(total_patches, dtype=np.int32)

  return VisionGridData(
      cos=jnp.asarray(cos),
      sin=jnp.asarray(sin),
      cu_seqlens=jnp.asarray(cu_seqlens),
      pos_embed_idx=jnp.asarray(pos_embed_idx),
      pos_embed_weights=jnp.asarray(pos_embed_weights),
      pos_embed_gather=jnp.asarray(gather_idx),
  )


@dataclasses.dataclass
class VisionModelConfig:
  hidden_size: int
  out_hidden_size: int
  depth: int
  num_heads: int
  intermediate_size: int
  patch_size: int
  temporal_patch_size: int
  spatial_merge_size: int
  window_size: int
  in_channels: int
  num_position_embeddings: Optional[int]
  deepstack_visual_indexes: Sequence[int]
  mrope_section: Sequence[int]
  image_pad_id: int


class VisionRotaryEmbedding(nnx.Module):

  def __init__(self, dim: int, theta: float = 10000.0):
    self.dim = dim
    self.theta = theta

  def __call__(self, seq_len: int) -> jax.Array:
    inv_freq = 1.0 / (
        self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
    )
    return jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv_freq)


class VisionPatchEmbed(nnx.Module):

  def __init__(
      self,
      embed_dim: int,
      patch_volume: int,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
      param_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.dtype = dtype
    self.proj = nnx.Linear(
        patch_volume,
        embed_dim,
        use_bias=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.proj(x.astype(self.dtype))


class VisionAttention(nnx.Module):

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
      param_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.dtype = dtype
    self.head_dim = self.hidden_size // self.num_heads
    self.scale = self.head_dim**-0.5
    self.qkv_proj = nnx.Linear(
        hidden_size,
        3 * self.hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.out_proj = nnx.Linear(
        hidden_size,
        self.hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

  def _create_attention_mask(
      self, cu_seqlens: jax.Array, seq_len: int
  ) -> jax.Array:
    """Create block-diagonal attention mask from cumulative sequence lengths.

    Args:
        cu_seqlens: Cumulative sequence lengths of shape (num_windows + 1,)
        seq_len: Total sequence length

    Returns:
        Attention bias mask of shape (1, 1, seq_len, seq_len)
    """
    positions = jnp.arange(seq_len)
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]

    # Determine which window each position belongs to
    in_segment = (positions[:, None] >= starts[None, :]) & (
        positions[:, None] < ends[None, :]
    )
    segment_ids = jnp.argmax(in_segment.astype(jnp.int32), axis=-1)

    # Create block-diagonal mask
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    attention_mask = jnp.where(
        same_segment, 0.0, jnp.finfo(self.dtype).min
    ).astype(self.dtype)

    return attention_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)

  def __call__(
      self, x: jax.Array, cos: jax.Array, sin: jax.Array, cu_seqlens: jax.Array
  ) -> jax.Array:
    qkv = self.qkv_proj(x)
    q, k, v = jnp.split(qkv, 3, axis=-1)

    seq_len = x.shape[0]
    q = q.reshape(seq_len, self.num_heads, self.head_dim)
    k = k.reshape(seq_len, self.num_heads, self.head_dim)
    v = v.reshape(seq_len, self.num_heads, self.head_dim)

    # Apply RoPE in float32 then cast back, matching PyTorch's
    # apply_rotary_pos_emb_vision: q,k = q.float(), k.float(); ...to(orig_dtype)
    cos_f = cos[:, : self.head_dim][
        :, None, :
    ]  # (seq_len, 1, head_dim) float32
    sin_f = sin[:, : self.head_dim][
        :, None, :
    ]  # (seq_len, 1, head_dim) float32
    q_f = q.astype(jnp.float32)  # (seq_len, num_heads, head_dim)
    k_f = k.astype(jnp.float32)
    q = (q_f * cos_f + rotate_half(q_f) * sin_f).astype(self.dtype)
    k = (k_f * cos_f + rotate_half(k_f) * sin_f).astype(self.dtype)

    # Transpose to (num_heads, seq_len, head_dim) for attention
    q = jnp.transpose(q, (1, 0, 2))
    k = jnp.transpose(k, (1, 0, 2))
    v = jnp.transpose(v, (1, 0, 2))

    # QK in bfloat16, matching PyTorch eager: torch.matmul(query, key.T) in query dtype.
    scores = (
        jnp.einsum("hqd,hkd->hqk", q, k) * self.scale
    )  # (num_heads, L, L) bfloat16

    # Create and apply attention mask
    attn_mask = self._create_attention_mask(cu_seqlens, seq_len)
    scores = scores + attn_mask.squeeze(0)  # Broadcast to [num_heads, L, L]

    # Softmax in float32, cast to bfloat16, then bfloat16 AV.
    # Matches PyTorch eager: F.softmax(..., dtype=float32).to(query.dtype) then matmul.
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
        self.dtype
    )
    out = jnp.einsum("hqk,hkd->hqd", weights, v)  # (num_heads, L, head_dim)

    # Transpose back to [L, num_heads, head_dim] and reshape
    out = jnp.transpose(out, (1, 0, 2)).reshape(seq_len, self.hidden_size)
    return self.out_proj(out)


class VisionMLP(nnx.Module):

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
      param_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.linear1 = nnx.Linear(
        hidden_size,
        intermediate_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.linear2 = nnx.Linear(
        intermediate_size,
        hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.linear1(x)
    x = jax.nn.gelu(x, approximate=True)
    x = self.linear2(x)
    return x


class VisionBlock(nnx.Module):

  def __init__(
      self,
      spec: VisionModelConfig,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
      param_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.norm1 = nnx.LayerNorm(
        spec.hidden_size,
        epsilon=1e-6,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.norm2 = nnx.LayerNorm(
        spec.hidden_size,
        epsilon=1e-6,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.attn = VisionAttention(
        spec.hidden_size,
        spec.num_heads,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.mlp = VisionMLP(
        spec.hidden_size,
        spec.intermediate_size,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

  def __call__(
      self, x: jax.Array, cos: jax.Array, sin: jax.Array, cu_seqlens: jax.Array
  ) -> jax.Array:
    x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens)
    x = x + self.mlp(self.norm2(x))
    return x


class VisionPatchMerger(nnx.Module):

  def __init__(
      self,
      context_dim: int,
      out_dim: int,
      spatial_merge_size: int,
      use_postshuffle_norm: bool = False,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
      param_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.context_dim = context_dim
    self.use_postshuffle_norm = use_postshuffle_norm
    self.unit = spatial_merge_size**2
    self.hidden_size = context_dim * self.unit
    norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
    self.norm = nnx.LayerNorm(
        norm_dim,
        epsilon=1e-6,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.linear_fc1 = nnx.Linear(
        self.hidden_size,
        self.hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.linear_fc2 = nnx.Linear(
        self.hidden_size,
        out_dim,
        use_bias=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.use_postshuffle_norm:
      x = x.reshape(-1, self.unit * self.context_dim)
      x = self.norm(x)
    else:
      x = self.norm(x)
      x = x.reshape(-1, self.unit * self.context_dim)
    x = jax.nn.gelu(self.linear_fc1(x), approximate=False)
    return self.linear_fc2(x)


class VisionModel(nnx.Module):

  def __init__(
      self,
      config: VisionModelConfig,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
      param_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.spec = config
    self.dtype = dtype
    patch_vol = (
        self.spec.in_channels
        * self.spec.temporal_patch_size
        * self.spec.patch_size**2
    )
    self.patch_embed = VisionPatchEmbed(
        self.spec.hidden_size,
        patch_vol,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.num_grid_per_side = None
    if self.spec.num_position_embeddings:
      self.pos_embed = nnx.Embed(
          self.spec.num_position_embeddings,
          self.spec.hidden_size,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
      self.num_grid_per_side = int(self.spec.num_position_embeddings**0.5)
    else:
      self.pos_embed = None
    rotary_dim = (self.spec.hidden_size // self.spec.num_heads) // 2
    self.rotary = VisionRotaryEmbedding(rotary_dim)
    self.blocks = nnx.List([
        VisionBlock(
            self.spec, dtype=self.dtype, param_dtype=param_dtype, rngs=rngs
        )
        for _ in range(self.spec.depth)
    ])
    self.merger = VisionPatchMerger(
        self.spec.hidden_size,
        self.spec.out_hidden_size,
        self.spec.spatial_merge_size,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    self.deepstack_visual_indexes = tuple(self.spec.deepstack_visual_indexes)
    self.deepstack_mergers = nnx.List([
        VisionPatchMerger(
            self.spec.hidden_size,
            self.spec.out_hidden_size,
            self.spec.spatial_merge_size,
            use_postshuffle_norm=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        for _ in self.deepstack_visual_indexes
    ])

  def compute_grid_data(self, grid_thw) -> VisionGridData:
    """Compute static positional data from grid_thw outside the JIT boundary.

    Delegates to the module-level :func:`compute_grid_data`. Call before JIT.

    Args:
      grid_thw: image/video grid sizes, array-like of shape [N, 3] with
        columns (temporal, height, width).

    Returns:
      VisionGridData with all static fields as JAX arrays.
    """
    return compute_grid_data(grid_thw, self.spec)

  def __call__(
      self,
      pixel_values: jax.Array,
      precomputed: VisionGridData,
  ) -> tuple[jax.Array, tuple]:
    x = self.patch_embed(pixel_values)

    # Apply learned position embeddings using precomputed static indices.
    # Three pure JAX ops — safe inside JIT.
    if self.pos_embed is not None and precomputed.pos_embed_gather.shape[0] > 0:
      raw = self.pos_embed.embedding[
          precomputed.pos_embed_idx
      ]  # [4, total_hw, hidden]
      patch_embeds = (raw * precomputed.pos_embed_weights[..., None]).sum(
          0
      )  # [total_hw, hidden]
      pos_embeds = patch_embeds[
          precomputed.pos_embed_gather
      ]  # [total_patches, hidden]
      x = x + pos_embeds.astype(x.dtype)

    cos, sin, cu_seqlens = (
        precomputed.cos,
        precomputed.sin,
        precomputed.cu_seqlens,
    )

    deepstack_feats = []
    for i, block in enumerate(self.blocks):
      x = block(x, cos, sin, cu_seqlens)
      if i in self.deepstack_visual_indexes:
        feat = self.deepstack_mergers[len(deepstack_feats)](x)
        deepstack_feats.append(feat)

    x = self.merger(x)
    return x, tuple(deepstack_feats)
