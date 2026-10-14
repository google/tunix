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

"""Gemma4 model attention."""

import functools
from functools import partial
import typing
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
import numpy as np
from tunix.models import cache_utils
from tunix.models.gemma4.config import AttentionType
from tunix.models.gemma4.config import K_MASK
from tunix.models.gemma4.config import LayerCache
from tunix.models.gemma4.config import ModelConfig
from tunix.models.gemma4.config import RematConfig
from tunix.models.gemma4.config import SplashAttentionImpl
from tunix.models.gemma4.layers import apply_rope
from tunix.models.gemma4.layers import Einsum
from tunix.models.gemma4.layers import RMSNorm
from tunix.utils.sharding_utils import shard

if typing.TYPE_CHECKING:
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash

  SplashKernel = (
      splash.SplashAttentionKernel | tokamax_splash.SplashAttentionKernel
  )

AxisSpec = str | tuple[str, ...] | None
MeshType = shd.Mesh | shd.AbstractMesh


@functools.lru_cache(maxsize=128)
def _get_local_mask(
    q_len: int, kv_len: int, window_size: int, offset: int
) -> mask_lib.LocalMask:
  """Memoized LocalMask constructor that speeds up XLA JIT compilation by caching mask closure objects across unrolled decoder layers."""
  return mask_lib.LocalMask(
      (q_len, kv_len),
      window_size=(window_size - 1, 0),
      offset=offset,
  )


@functools.lru_cache(maxsize=128)
def _get_causal_mask(
    q_len: int, kv_len: int, offset: int
) -> mask_lib.CausalMask:
  """Memoized CausalMask constructor that speeds up XLA JIT compilation by caching mask closure objects across unrolled decoder layers."""
  return mask_lib.CausalMask((q_len, kv_len), offset=offset)


def _resolve_active_mesh() -> MeshType | None:
  """Returns the mesh the caller is currently using or None."""
  mesh = pxla.thread_resources.env.physical_mesh
  if not mesh.empty:
    return mesh

  for getter_name in ('get_abstract_mesh', 'get_mesh'):
    getter = getattr(jax.sharding, getter_name, None)
    if getter is None:
      continue
    ctx_mesh = getter()
    if ctx_mesh is not None and not ctx_mesh.empty:
      return ctx_mesh

  return None


def _tokamax_splash_libs():
  """Lazily imports Tokamax's splash-attention kernel and mask modules."""
  try:
    from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash_lib
    from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask_lib
  except ImportError as e:
    raise ImportError(
        "The Tokamax splash_attention backend requires the 'tokamax'"
        ' package.Install it with `pip install tokamax` or keep the default'
        ' `splash_attention_impl=SplashAttentionImpl.JAX`'
    ) from e
  return tokamax_splash_lib, tokamax_splash_mask_lib


@functools.lru_cache(maxsize=128)
def _get_tokamax_local_mask(
    q_len: int, kv_len: int, window_size: int, offset: int
):
  """Memoized Tokamax LocalMask, mirroring `_get_local_mask`"""
  _, tokamax_mask_lib = _tokamax_splash_libs()
  return tokamax_mask_lib.LocalMask(
      (q_len, kv_len),
      window_size=(window_size - 1, 0),
      offset=offset,
  )


@functools.lru_cache(maxsize=128)
def _get_tokamax_causal_mask(q_len: int, kv_len: int, offset: int):
  """Memoized Tokamax CausalMask, mirroring `_get_causal_mask`."""
  _, tokamax_mask_lib = _tokamax_splash_libs()
  return tokamax_mask_lib.CausalMask((q_len, kv_len), offset=offset)


@functools.lru_cache(maxsize=32)
def _zeros(
    shape: tuple[int, ...], dtype: jnp.dtype, sharding: shd.NamedSharding
):
  """Compiles and caches an on-device zero allocator with explicit out_shardings."""
  return jax.jit(lambda: jnp.zeros(shape, dtype=dtype), out_shardings=sharding)


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.rope_proportion = (
        config.global_rope_proportion
        if attn_type == AttentionType.GLOBAL
        else config.local_rope_proportion
    )
    self.attn_type = attn_type
    self.rope_base_frequency = (
        config.local_base_frequency
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_base_frequency
    )
    self.rope_scale_factor = (
        config.local_scale_factor
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_scale_factor
    )

    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    if attn_type == AttentionType.GLOBAL:
      if config.num_global_kv_heads is not None:
        self.num_kv_heads = config.num_global_kv_heads
      if config.global_key_size is not None:
        self.head_dim = config.global_key_size

    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, self.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=config.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_einsum = Einsum(
        einsum_str='BTD,NDH->BTNH',
        shape=(config.num_heads, config.embed_dim, self.head_dim),
        rngs=rngs,
        sharding=config.shd_config.q_weight_ndh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    k_eq_v = (
        config.k_eq_v_global if attn_type == AttentionType.GLOBAL else False
    )
    if k_eq_v:
      self.k_einsum = Einsum(
          einsum_str='BSD,KDH->BSKH',
          shape=(
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=config.shd_config.q_weight_ndh,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    else:
      if self.num_kv_heads == 1:
        kv_sharding = (None, None, 'fsdp', None)
      else:
        kv_sharding = config.shd_config.kv_weight_cndh

      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(
              2,
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=kv_sharding,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self._query_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self._key_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def _compute_kv_projections(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      kv_shared_cache: LayerCache | None,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array | None]:
    """Computes or retrieves key/value projections."""
    kv_valid_mask = None

    if kv_shared_cache is not None:
      key_proj = kv_shared_cache['k']
      value_proj = kv_shared_cache['v']
      kv_valid_mask = kv_shared_cache.get('valid_mask', None)
    else:
      if hasattr(self, 'k_einsum'):  # case where k_eq_v is True
        key_proj = self.k_einsum(x)
        value_proj = key_proj
      else:
        key_proj, value_proj = self.kv_einsum(x)

      key_proj = shard(key_proj, self.config.shd_config.act_btnh)
      value_proj = shard(value_proj, self.config.shd_config.act_btnh)

      # Apply norms to computed KV
      value_var = jnp.mean(jnp.square(value_proj), axis=-1, keepdims=True)
      value_proj = value_proj * jax.lax.rsqrt(value_var + 1e-06)
      key_proj = self._key_norm(key_proj)
      key_proj = apply_rope(
          key_proj,
          segment_pos,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
          rope_proportion=self.rope_proportion,
      )

    return key_proj, value_proj, kv_valid_mask

  def _build_flash_mask(
      self,
      q_len: int,
      kv_len: int,
      offset: int,
  ) -> mask_lib.Mask:
    """Builds the single-head splash attention mask for one flash call.

    Uses memoized computable masks (LocalMask / CausalMask) to prevent XLA
    closure recompilations across unrolled layers and evaluate in-kernel.
    """
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      return _get_local_mask(q_len, kv_len, window_size, offset)
    return _get_causal_mask(q_len, kv_len, offset)

  def _make_block_sizes(
      self, is_rectangular: bool, q_len: int | None = None
  ) -> splash.BlockSizes:
    """Selects splash block sizes for this attention call."""
    # Choose block sizes. block_kv must divide kv_len.
    # For LOCAL_SLIDING rectangular shapes, block_kv must divide both
    # sliding_window_size and chunk_len. Use the smaller of the two.
    block_q = self.config.flash_attention_block_size
    if q_len is not None:
      block_q = min(block_q, q_len)
    if is_rectangular and self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      block_kv = min(
          self.config.flash_attention_block_size,
          window_size,
      )
    else:
      block_kv = self.config.flash_attention_block_size

    # Inner-loop tile size for the attention matmul; must divide block_kv.
    block_kv_compute = min(
        self.config.flash_attention_compute_block_size, block_kv
    )

    # Bwd holds Q + dO + attn_weights + grad accumulators; smaller Q block to
    # fit VMEM.
    block_bwd = min(self.config.flash_attention_bwd_block_size, block_q)
    use_fused = self.config.flash_attention_use_fused_bwd
    return splash.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_bwd,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_kv_compute,
        # Fused bwd kernel computes dQ+dKV in one pass; these are ignored.
        block_q_dq=None if use_fused else block_bwd,
        block_kv_dq=None if use_fused else block_kv,
        use_fused_bwd_kernel=use_fused,
    )

  def _make_sharding_specs(self, b: int, kh: int, mesh: MeshType | None):
    """Computes mesh/shard-axis specs for splash attention."""
    shd_b, shd_t, shd_n, shd_h = self.config.shd_config.act_btnh
    if mesh is None or not mesh.shape:
      shd_b = shd_t = shd_n = shd_h = None
    else:
      if shd_b is not None and (
          shd_b not in mesh.shape or b % mesh.shape[shd_b] != 0
      ):
        shd_b = None
      shd_t, shd_n, shd_h = [
          axis if axis in mesh.shape else None
          for axis in (shd_t, shd_n, shd_h)
      ]
    head_shards = (
        mesh.shape[shd_n]
        if mesh is not None and shd_n is not None and shd_n in mesh.shape
        else 1
    )
    q_seq_shards = (
        mesh.shape[shd_t]
        if mesh is not None and shd_t is not None and shd_t in mesh.shape
        else 1
    )
    shd_spec = P(shd_b, shd_n, shd_t, shd_h)
    shd_n_kv = (
        shd_n
        if mesh is not None
        and shd_n is not None
        and shd_n in mesh.shape
        and kh % mesh.shape[shd_n] == 0
        else None
    )
    unsharded_seq_kv = P(shd_b, shd_n_kv, None, shd_h)
    return (
        shd_b,
        shd_n,
        shd_t,
        shd_h,
        head_shards,
        q_seq_shards,
        shd_n_kv,
        shd_spec,
        unsharded_seq_kv,
    )

  def _make_splash_kernel(
      self,
      multi_head_mask,
      block_sizes: splash.BlockSizes,
      head_shards: int,
      q_seq_shards: int,
      mesh: MeshType | None,
      shd_n: AxisSpec,
      shd_t: AxisSpec,
      save_residuals: bool = False,
  ):
    """Builds a splash MHA kernel and its manual sharding spec."""
    kernel = splash.make_splash_mha(
        multi_head_mask,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        save_residuals=save_residuals,
    )
    kernel_spec = (
        kernel.manual_sharding_spec(shd.NamedSharding(mesh, P(shd_n, shd_t)))
        if mesh is not None
        else None
    )
    return kernel, kernel_spec

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
      is_chunked_prefill: bool = False,
      prefix_length: int = 0,
      input_mask: jaxtyping.Array | None = None,
      force_eager: bool = False,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[
          jaxtyping.Array,
          jaxtyping.Array,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
      ],
  ]:
    x = x.astype(self.config.dtype)
    seq_len = x.shape[1]
    query_proj = self.q_einsum(x)
    query_proj = shard(query_proj, self.config.shd_config.act_btnh)
    query_proj = self._query_norm(query_proj)
    query_proj = apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
        rope_proportion=self.rope_proportion,
    )

    key_proj, value_proj, kv_valid_mask = self._compute_kv_projections(
        x,
        segment_pos,
        kv_shared_cache,
    )

    prior_end_index = None
    split_prefix_k = None
    split_prefix_v = None
    if cache is not None:
      assert kv_shared_cache is None
      if seq_len > 1:  # prefill
        new_cache, prefix_res, prior_end_index = (
            cache_utils.update_cache_prefill(
                cache,
                key_proj,
                value_proj,
                seq_len,
                is_chunked_prefill=is_chunked_prefill,
                prefix_length=prefix_length,
                input_mask=input_mask,
                is_ring_buffer_read=(
                    self.config.use_sliding_window_kv_cache
                    and self.attn_type == AttentionType.LOCAL_SLIDING
                ),
                is_ring_buffer_write=self.config.use_sliding_window_kv_cache,
                use_split_attention=getattr(
                    self.config, 'use_split_attention', False
                ),
            )
        )
        key_proj = prefix_res.key
        value_proj = prefix_res.value
        kv_valid_mask = prefix_res.valid_mask
        split_prefix_k = prefix_res.split_prefix_k
        split_prefix_v = prefix_res.split_prefix_v
      else:  # decode
        key_proj, value_proj = cache_utils.write_cache_decode(
            cache,
            key_proj,
            value_proj,
            attn_mask,
            is_ring_buffer=(
                self.config.use_sliding_window_kv_cache
                and self.attn_type == AttentionType.LOCAL_SLIDING
            ),
        )
        new_cache = {
            'v': value_proj,
            'k': key_proj,
            'end_index': cache['end_index'] + seq_len,
        }
        split_prefix_k = None
        split_prefix_v = None
    else:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
      }
      if kv_shared_cache is not None:
        split_prefix_k = kv_shared_cache.get('split_prefix_k', None)
        split_prefix_v = kv_shared_cache.get('split_prefix_v', None)

    b, _, qh, _ = query_proj.shape
    _, _, kh, _ = key_proj.shape

    # Determine if we can use flash attention for this call.
    q_len = query_proj.shape[1]
    kv_len = key_proj.shape[1]
    # When split attention bypasses the concat, key_proj is suffix-only.
    # Compute total kv_len from prefix + suffix for mask/guard calculations.
    if split_prefix_k is not None:
      kv_len = split_prefix_k.shape[1] + key_proj.shape[1]
    is_rectangular = kv_len > q_len
    use_flash = (
        self.config.use_flash_attention
        and seq_len > 1
        and kv_len >= self.config.flash_attention_block_size
        and not (is_rectangular and segment_ids is not None)
        and not force_eager
    )

    if use_flash:
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      mesh = _resolve_active_mesh()

      # Offset: shifts Q positions so q[0] aligns with kv[prefix_len].
      offset = kv_len - q_len if is_rectangular else 0

      (
          shd_b,
          shd_n,
          shd_t,
          shd_h,
          head_shards,
          q_seq_shards,
          shd_n_kv,
          shd_spec,
          unsharded_seq_kv,
      ) = self._make_sharding_specs(b, kh, mesh)

      block_sizes = self._make_block_sizes(is_rectangular, q_len=q_len)

      if self.config.splash_attention_impl == SplashAttentionImpl.TOKAMAX:
        splash_attn_kernel, kernel_spec = self._make_tokamax_splash_kernel(
            q_len,
            kv_len,
            offset,
            q_seq_shards,
            mesh,
            shd_t,
        )
      else:
        mask = self._build_flash_mask(q_len, kv_len, offset)

        multi_head_mask = mask_lib.MultiHeadMask([mask for _ in range(qh)])

        splash_attn_kernel, kernel_spec = self._make_splash_kernel(
            multi_head_mask,
            block_sizes,
            head_shards,
            q_seq_shards,
            mesh,
            shd_n,
            shd_t,
        )

      # Split attention with LSE merge for chunked prefill: attend to prefix
      # and suffix KV separately and merge via log-sum-exp, eliminating the
      # jnp.concatenate data movement cost. Guards: block sizes must divide the
      # prefix and suffix lengths; segment_ids are not supported (already
      # excluded by use_flash).
      prefix_kv_len = kv_len - q_len
      can_split = (
          getattr(self.config, 'use_split_attention', False)
          and self.config.splash_attention_impl == SplashAttentionImpl.JAX
          and is_rectangular
          and segment_ids is None
          and split_prefix_k is not None
          and split_prefix_v is not None
          and prefix_kv_len % block_sizes.block_kv == 0
          and q_len % block_sizes.block_q == 0
          and q_len % block_sizes.block_kv == 0
      )
      if can_split:
        encoded, key_proj, value_proj = self._flash_attention_split(
            query_proj,
            key_proj,
            value_proj,
            split_prefix_k,
            split_prefix_v,
            q_len,
            prefix_kv_len,
            qh,
            block_sizes,
            head_shards,
            q_seq_shards,
            mesh,
            shd_b,
            shd_n,
            shd_t,
            shd_h,
            shd_n_kv,
            shd_spec,
        )

      else:
        encoded, key_proj, value_proj = self._flash_attention_single(
            query_proj,
            key_proj,
            value_proj,
            split_prefix_k,
            split_prefix_v,
            segment_ids,
            splash_attn_kernel,
            kernel_spec,
            shd_spec,
            unsharded_seq_kv,
            mesh,
            shd_b,
            shd_t,
        )
        split_prefix_k = None
        split_prefix_v = None

    else:
      if split_prefix_k is not None:
        assert split_prefix_v is not None
        key_proj = jnp.concatenate([split_prefix_k, key_proj], axis=1)
        value_proj = jnp.concatenate([split_prefix_v, value_proj], axis=1)
        split_prefix_k = None
        split_prefix_v = None
      encoded = self._eager_attention(
          query_proj,
          key_proj,
          value_proj,
          attn_mask,
          segment_pos,
          cache,
          kv_shared_cache,
          kv_valid_mask,
          prior_end_index,
          prefix_length,
          seq_len,
          is_chunked_prefill,
      )

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.config.shd_config.act_btd)
    if split_prefix_k is not None:
      assert split_prefix_v is not None
      assert split_prefix_k.ndim == 4 and key_proj.ndim == 4
      assert split_prefix_k.shape[0] == key_proj.shape[0]
      assert split_prefix_k.shape[2:] == key_proj.shape[2:]
    return (
        new_cache,
        attn_output,
        (
            key_proj,
            value_proj,
            kv_valid_mask,
            prior_end_index,
            split_prefix_k,
            split_prefix_v,
        ),
    )

  def _flash_attention_split(
      self,
      query_proj: jaxtyping.Array,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      split_prefix_k: jaxtyping.Array | None,
      split_prefix_v: jaxtyping.Array | None,
      q_len: int,
      prefix_kv_len: int,
      qh: int,
      block_sizes: splash.BlockSizes,
      head_shards: int,
      q_seq_shards: int,
      mesh: MeshType | None,
      shd_b: AxisSpec,
      shd_n: AxisSpec,
      shd_t: AxisSpec,
      shd_h: AxisSpec,
      shd_n_kv: AxisSpec,
      shd_spec: P,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array]:
    """Split-KV flash attention with LSE merge for chunked prefill."""
    # Use separately-returned prefix KV from _read_prefix_kv. No
    # concatenation occurred — prefix comes from cache (B, S, N, H),
    # suffix is the fresh projection already transposed (B, N, S, H).
    assert split_prefix_k is not None
    assert split_prefix_v is not None
    prefix_k = split_prefix_k.transpose(0, 2, 1, 3)
    prefix_v = split_prefix_v.transpose(0, 2, 1, 3)
    suffix_k = key_proj  # already transposed above
    suffix_v = value_proj

    # Build masks for prefix and suffix
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      sw = window_size
      q_ids = np.arange(q_len) + prefix_kv_len
      # Prefix mask: which prefix positions are in sliding window
      prefix_kv_ids = np.arange(prefix_kv_len)
      prefix_mask_2d = (prefix_kv_ids[None, :] > (q_ids[:, None] - sw)) & (
          prefix_kv_ids[None, :] <= q_ids[:, None]
      )
      prefix_mask = mask_lib.NumpyMask(prefix_mask_2d.astype(np.bool_))
    else:
      # GLOBAL: prefix is fully attended
      prefix_mask = mask_lib.FullMask((q_len, prefix_kv_len))

    # Suffix mask: local sliding within chunk for LOCAL, causal for GLOBAL
    suffix_mask = self._build_flash_mask(q_len, q_len, offset=0)

    prefix_multi_mask = mask_lib.MultiHeadMask([prefix_mask for _ in range(qh)])
    suffix_multi_mask = mask_lib.MultiHeadMask([suffix_mask for _ in range(qh)])

    # Build kernels with save_residuals=True for LSE merge.
    prefix_attn_kernel, prefix_kernel_spec = self._make_splash_kernel(
        prefix_multi_mask,
        block_sizes,
        head_shards,
        q_seq_shards,
        mesh,
        shd_n,
        shd_t,
        save_residuals=True,
    )
    suffix_attn_kernel, suffix_kernel_spec = self._make_splash_kernel(
        suffix_multi_mask,
        block_sizes,
        head_shards,
        q_seq_shards,
        mesh,
        shd_n,
        shd_t,
        save_residuals=True,
    )

    # Sharding specs for split KV and logsumexp
    prefix_kv_spec = P(shd_b, shd_n_kv, None, shd_h)
    suffix_kv_spec = P(shd_b, shd_n_kv, None, shd_h)
    lse_spec = P(shd_b, shd_n, shd_t)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            prefix_kernel_spec,
            shd_spec,
            prefix_kv_spec,
            prefix_kv_spec,
        ),
        out_specs=(shd_spec, (lse_spec,)),
        check_rep=False,
    )
    def sharded_prefix_attn(kernel, q_block, k_block, v_block):
      return jax.vmap(kernel)(q_block, k_block, v_block)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            suffix_kernel_spec,
            shd_spec,
            suffix_kv_spec,
            suffix_kv_spec,
        ),
        out_specs=(shd_spec, (lse_spec,)),
        check_rep=False,
    )
    def sharded_suffix_attn(kernel, q_block, k_block, v_block):
      return jax.vmap(kernel)(q_block, k_block, v_block)

    out_prefix, (lse_prefix,) = sharded_prefix_attn(
        prefix_attn_kernel,
        query_proj,
        prefix_k,
        prefix_v,
    )
    out_suffix, (lse_suffix,) = sharded_suffix_attn(
        suffix_attn_kernel,
        query_proj,
        suffix_k,
        suffix_v,
    )

    # Max-stabilized LSE merge (numerically stable softmax reweighting).
    # NaN-safe: fully-masked prefix partitions arrive as out=NaN/lse=-inf and
    # are gated to a zero contribution. See cache_utils.merge_split_attention.
    encoded = cache_utils.merge_split_attention(
        out_prefix, lse_prefix, out_suffix, lse_suffix, key_proj.dtype
    )
    # (B, N, T, H) -> (B, T, N, H)
    encoded = encoded.transpose(0, 2, 1, 3)
    # Transpose KV back for KV-sharing layers
    key_proj = key_proj.transpose(0, 2, 1, 3)
    value_proj = value_proj.transpose(0, 2, 1, 3)
    return encoded, key_proj, value_proj

  def _make_tokamax_splash_kernel(
      self,
      q_len: int,
      kv_len: int,
      offset: int,
      q_seq_shards: int,
      mesh: MeshType | None,
      shd_t: str | None,
  ):
    """Builds a Tokamax splash MHA kernel and its manual sharding spec."""
    tokamax_splash, _ = _tokamax_splash_libs()
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      tokamax_mask = _get_tokamax_local_mask(q_len, kv_len, window_size, offset)
    else:
      tokamax_mask = _get_tokamax_causal_mask(q_len, kv_len, offset=offset)

    bs = self.config.flash_attention_block_size
    splash_config = tokamax_splash.SplashConfig(
        block_q=bs,
        block_kv=bs,
        block_kv_compute=bs,
        block_q_dkv=bs,
        block_kv_dkv=bs,
        block_kv_dkv_compute=bs,
        # Enabling use_base2 can improve performance but reduce numeric precision.
        use_base2_exp=False,
    )
    splash_attn_kernel = tokamax_splash.make_splash_mha(
        tokamax_mask,
        config=splash_config,
        q_seq_shards=q_seq_shards,
    )
    kernel_spec = (
        splash_attn_kernel.manual_sharding_spec(
            shd.NamedSharding(mesh, P(shd_t))
        )
        if mesh is not None
        else None
    )
    return splash_attn_kernel, kernel_spec

  def _flash_attention_single(
      self,
      query_proj: jaxtyping.Array,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      split_prefix_k: jaxtyping.Array | None,
      split_prefix_v: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None,
      splash_attn_kernel: 'SplashKernel',
      kernel_spec: 'SplashKernel | None',
      shd_spec: P,
      unsharded_seq_kv: P,
      mesh: MeshType | None,
      shd_b: AxisSpec,
      shd_t: AxisSpec,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array]:
    """Single-kernel flash attention over concatenated (or plain) KV."""
    # Fallback: if split attention was prepared but guards failed,
    # reconstruct the concatenated KV for single-kernel path.
    if split_prefix_k is not None:
      assert split_prefix_v is not None
      key_proj = jnp.concatenate(
          [split_prefix_k.transpose(0, 2, 1, 3), key_proj], axis=2
      )
      value_proj = jnp.concatenate(
          [split_prefix_v.transpose(0, 2, 1, 3), value_proj], axis=2
      )
    # Original single-kernel attention path.
    if segment_ids is not None:
      seg_spec = P(shd_b, shd_t)
      unsharded_seg_spec = P(shd_b, None)
      if self.config.splash_attention_impl == SplashAttentionImpl.TOKAMAX:
        segment_ids_cls = _tokamax_splash_libs()[0].SegmentIds
      else:
        segment_ids_cls = splash.SegmentIds

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(
              kernel_spec,
              shd_spec,
              unsharded_seq_kv,
              unsharded_seq_kv,
              seg_spec,
              unsharded_seg_spec,
          ),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(
          kernel, q_block, k_block, v_block, q_seg_block, kv_seg_block
      ):
        seg_ids = segment_ids_cls(q=q_seg_block, kv=kv_seg_block)
        return jax.vmap(kernel)(q_block, k_block, v_block, segment_ids=seg_ids)

      qkv: jaxtyping.Array = sharded_splash_attn(
          splash_attn_kernel,
          query_proj,
          key_proj,
          value_proj,
          segment_ids,
          segment_ids,
      )
    else:

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(
              kernel_spec,
              shd_spec,
              unsharded_seq_kv,
              unsharded_seq_kv,
          ),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block):
        return jax.vmap(kernel)(q_block, k_block, v_block)

      qkv: jaxtyping.Array = sharded_splash_attn(
          splash_attn_kernel,
          query_proj,
          key_proj,
          value_proj,
      )
    encoded = qkv.transpose(0, 2, 1, 3)
    # Transpose KV back to (B, S, K, H); consumed by KV-sharing layers via
    # layers_kvs.
    key_proj = key_proj.transpose(0, 2, 1, 3)
    value_proj = value_proj.transpose(0, 2, 1, 3)
    return encoded, key_proj, value_proj

  def _eager_attention(
      self,
      query_proj: jaxtyping.Array,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      attn_mask: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      kv_shared_cache: LayerCache | None,
      kv_valid_mask: jaxtyping.Array | None = None,
      prior_end_index: jaxtyping.Array | None = None,
      prefix_length: int = 0,
      seq_len: int = 1,
      is_chunked_prefill: bool = False,
  ) -> jaxtyping.Array:
    """Eager einsum attention (non-flash path)."""
    if self.use_gqa:
      b, t, kg, h = query_proj.shape
      n_groups = kg // self.num_kv_heads
      query_reshaped = query_proj.reshape(
          (b, t, self.num_kv_heads, n_groups, h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_reshaped, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_proj, key_proj)

    kv_len = key_proj.shape[1]
    q_len = query_proj.shape[1]

    if seq_len > 1:
      if is_chunked_prefill and kv_len > q_len:
        attn_mask = cache_utils.build_dense_chunked_prefill_mask(
            attn_mask,
            q_len,
            kv_len,
            prior_end_index,
            prefix_length,
            kv_valid_mask=kv_valid_mask,
            is_ring_buffer=(
                self.config.use_sliding_window_kv_cache
                and self.attn_type == AttentionType.LOCAL_SLIDING
            ),
            sliding_window_size=self.config.sliding_window_size,
            kv_shared_cache=kv_shared_cache,
            has_own_cache=(cache is not None),
        )
      else:
        attn_mask = attn_mask[..., :kv_len]

    _skip_sliding_mask = (
        is_chunked_prefill
        and kv_len > q_len
        and self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
    )
    if self.attn_type == AttentionType.LOCAL_SLIDING and not _skip_sliding_mask:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      if segment_pos.shape[1] == 1:
        is_ring_buffer = self.config.use_sliding_window_kv_cache
        end_idx = None
        cache_len = None
        if is_ring_buffer:
          active_cache = cache if cache is not None else kv_shared_cache
          if active_cache is None:
            raise ValueError(
                'Cache or shared cache is required for local sliding attention'
                ' in decoding.'
            )
          cache_len = key_proj.shape[1]
          end_idx = active_cache['end_index']
          if cache is None:
            # In case of shared KV cache, the origin layer already updated the
            # end index. We need to subtract 1 to get the correct end index of
            # the previous token.
            end_idx = end_idx - 1
        attn_mask = cache_utils.build_sliding_window_decode_mask(
            attn_mask,
            window_size=window_size,
            is_ring_buffer=is_ring_buffer,
            end_idx=end_idx,
            cache_len=cache_len,
        )
      else:  # standard (non-chunked) prefill sliding window
        offset = kv_len - q_len
        all_ones = jnp.ones_like(attn_mask)
        sliding_mask = jnp.triu(all_ones, offset - window_size + 1) * jnp.tril(
            all_ones, offset + window_size - 1
        )
        attn_mask = sliding_mask * attn_mask

    attn = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )

    if self.use_gqa:
      b, t, kg, s = attn.shape
      n_groups = kg // self.num_kv_heads
      probs_reshaped = attn.reshape((b, t, self.num_kv_heads, n_groups, s))
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs_reshaped, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum('BTNS,BSNH->BTNH', attn, value_proj)
    return encoded

  @property
  def use_gqa(self) -> bool:
    return self.num_kv_heads != self.config.num_heads

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
      is_chunked_prefill: bool = False,
      prefix_length: int = 0,
      input_mask: jaxtyping.Array | None = None,
      force_eager: bool = False,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[
          jaxtyping.Array,
          jaxtyping.Array,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
      ],
  ]:
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument. graph_updates=False prevents TraceContextError
      # when mutating params across jax transformation trace levels.
      # Bake static args via partial to avoid ConcretizationTypeError under remat.
      # Bucket prefix_length to prevent a recompilation storm.
      active_cache = cache if cache is not None else kv_shared_cache
      bucketed_prefix = cache_utils.maybe_bucket_prefix_length(
          prefix_length,
          active_cache,
          is_chunked_prefill,
          self.config.prefix_bucket_boundaries,
      )
      block_fn = partial(
          self.block.__func__,
          is_chunked_prefill=is_chunked_prefill,
          prefix_length=bucketed_prefix,
          input_mask=input_mask,
          force_eager=force_eager,
      )
      policy = getattr(jax.checkpoint_policies, self.config.remat_policy)
      return nnx.remat(
          block_fn,
          graph_updates=False,
          policy=policy,
      )(self, x, segment_pos, cache, attn_mask, kv_shared_cache, segment_ids)
    else:
      # Bucket prefix_length for the non-remat path too (controls static slice
      # shapes which affect JAXPR identity).
      active_cache = cache if cache is not None else kv_shared_cache
      bucketed_prefix = cache_utils.maybe_bucket_prefix_length(
          prefix_length,
          active_cache,
          is_chunked_prefill,
          self.config.prefix_bucket_boundaries,
      )
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
          is_chunked_prefill=is_chunked_prefill,
          prefix_length=bucketed_prefix,
          input_mask=input_mask,
          force_eager=force_eager,
      )

  def init_cache(
      self, batch_size: int, max_seq_len: int, dtype: jnp.dtype
  ) -> LayerCache:
    cache_len = max_seq_len
    sliding_window_size = self.config.sliding_window_size
    if (
        self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
        and sliding_window_size is not None
    ):
      cache_len = min(max_seq_len, sliding_window_size)

    cache_shape = (batch_size, cache_len, self.num_kv_heads, self.head_dim)
    mesh = _resolve_active_mesh()

    shd_config = getattr(self.config, 'shd_config', None)
    act_btnh = getattr(shd_config, 'act_btnh', None) if shd_config else None

    if mesh is not None and act_btnh is not None:
      k_sharding = shd.NamedSharding(mesh, shd.PartitionSpec(*act_btnh))
      idx_sharding = shd.NamedSharding(mesh, shd.PartitionSpec(*act_btnh[:1]))

      k = _zeros(cache_shape, dtype, k_sharding)()
      v = _zeros(cache_shape, dtype, k_sharding)()
      end_index = _zeros((batch_size,), jnp.int32, idx_sharding)()
    else:
      k = jnp.zeros(cache_shape, dtype)
      v = jnp.zeros(cache_shape, dtype)
      end_index = jnp.zeros((batch_size,), jnp.int32)

    return {'k': k, 'v': v, 'end_index': end_index}
