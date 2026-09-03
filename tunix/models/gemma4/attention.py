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


def find_last_one_index(attn_mask: jnp.ndarray) -> jnp.ndarray:
  """Finds the index of the last (rightmost) '1' from attn_mask."""
  cache_len = attn_mask.shape[-1]

  # 1. check if the entire row is all zeros.
  all_zeros_mask = jnp.all(attn_mask == 0, axis=-1)

  # 2. reverse the rows in the attn_mask
  reversed_matrix = attn_mask[:, :, ::-1]

  # 3. find the fist 1 from the right.
  first_one_from_right = jnp.argmax(reversed_matrix, axis=-1)

  # 4. covert back to the original index
  last_one_index_original = cache_len - 1 - first_one_from_right

  # 5. return the final index, 0 for rows are all zeros.
  final_indices = jnp.where(
      all_zeros_mask,
      0,
      last_one_index_original,
  )

  return final_indices.squeeze(axis=-1)


def create_sliding_window_mask(
    attn_mask: jnp.ndarray,  # [B, seq_len, cache_len] seq_len=1 for decoding
    sliding_window_size: int,
) -> jnp.ndarray:
  """Helper function to create sliding window mask for local attention."""
  upper_index = find_last_one_index(attn_mask)

  # 1. compute the window start position
  window_start_pos = upper_index - sliding_window_size + 1

  # 2. create window mask
  abs_pos = jnp.arange(attn_mask.shape[-1])
  window_mask = abs_pos[None, :] >= window_start_pos[:, None]

  # 3. create causal mask
  causal_mask = abs_pos[None, :] <= upper_index[:, None]

  # 4. create final mask
  final_mask = window_mask & causal_mask
  return final_mask[:, None, :]  # [B, 1, cache_len]


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

  def _make_block_sizes(self, is_rectangular: bool) -> splash.BlockSizes:
    """Selects splash block sizes for this attention call."""
    # Choose block sizes. block_kv must divide kv_len.
    # For LOCAL_SLIDING rectangular shapes, block_kv must divide both
    # sliding_window_size and chunk_len. Use the smaller of the two.
    block_q = self.config.flash_attention_block_size
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
      shd_n: str | None,
      shd_t: str | None,
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
      force_eager: bool = False,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[
          jaxtyping.Array,
          jaxtyping.Array,
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
    if cache is not None:
      assert kv_shared_cache is None
      cache_len = cache['v'].shape[1]
      if seq_len > 1:  # prefill
        if self.config.use_sliding_window_kv_cache and seq_len > cache_len:
          valid_indices = (
              (seq_len - cache_len) + jnp.arange(cache_len)
          ) % cache_len
          new_v = value_proj[:, -cache_len:, ...]
          new_k = key_proj[:, -cache_len:, ...]
          cache_v = cache['v'].at[:, valid_indices, ...].set(new_v)
          cache_k = cache['k'].at[:, valid_indices, ...].set(new_k)
          new_cache = {
              'v': cache_v,
              'k': cache_k,
              'end_index': jnp.full(
                  (value_proj.shape[0],), seq_len, dtype=jnp.int32
              ),
          }
        else:
          slice_indices = (0, 0, 0, 0)
          cache_v = jax.lax.dynamic_update_slice(
              cache['v'], value_proj, slice_indices
          )
          cache_k = jax.lax.dynamic_update_slice(
              cache['k'], key_proj, slice_indices
          )
          new_cache = {
              'v': cache_v,
              'k': cache_k,
              'end_index': jnp.full(
                  (value_proj.shape[0],), seq_len, dtype=jnp.int32
              ),
          }
        prior_end_index = None
        split_prefix_k = None
        split_prefix_v = None
      else:  # decode
        end_index = cache['end_index'][0]
        slice_indices = (0, end_index % cache_len, 0, 0)
        value_proj = jax.lax.dynamic_update_slice(
            cache['v'], value_proj, slice_indices
        )
        key_proj = jax.lax.dynamic_update_slice(
            cache['k'], key_proj, slice_indices
        )
        new_cache = {
            'v': value_proj,
            'k': key_proj,
            'end_index': cache['end_index'] + seq_len,
        }
    else:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
      }

    b, _, qh, _ = query_proj.shape
    _, _, kh, _ = key_proj.shape

    # Determine if we can use flash attention for this call.
    q_len = query_proj.shape[1]
    kv_len = key_proj.shape[1]
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

        block_sizes = self._make_block_sizes(is_rectangular)

        splash_attn_kernel, kernel_spec = self._make_splash_kernel(
            multi_head_mask,
            block_sizes,
            head_shards,
            q_seq_shards,
            mesh,
            shd_n,
            shd_t,
        )

      encoded, key_proj, value_proj = self._flash_attention_single(
          query_proj,
          key_proj,
          value_proj,
          segment_ids,
          splash_attn_kernel,
          kernel_spec,
          shd_spec,
          unsharded_seq_kv,
          mesh,
          shd_b,
          shd_t,
      )
    else:
      encoded = self._eager_attention(
          query_proj,
          key_proj,
          value_proj,
          attn_mask,
          segment_pos,
          cache,
          kv_shared_cache,
          seq_len,
      )

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.config.shd_config.act_btd)
    return (
        new_cache,
        attn_output,
        (key_proj, value_proj, kv_valid_mask, prior_end_index),
    )

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
      seq_len: int,
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
      attn_mask = attn_mask[..., :kv_len]

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      if segment_pos.shape[1] == 1 and self.config.use_sliding_window_kv_cache:
        # for decoding with sliding window cache
        active_cache = cache if cache is not None else kv_shared_cache
        if active_cache is None:
          raise ValueError(
              'Cache or shared cache is required for local sliding attention'
              ' in decoding.'
          )
        cache_len = key_proj.shape[1]
        end_idx = active_cache['end_index']
        if cache is None:
          end_idx = end_idx - 1
        end_idx = end_idx[:, None, None]
        p = jnp.arange(cache_len)[None, None, :]

        # map physical index to logical index
        logical_indices = end_idx - ((end_idx - p) % cache_len)

        # identify uninitialized slots (before the cache fills up)
        valid_physical = logical_indices >= 0
        logical_indices = jnp.maximum(0, logical_indices)

        attn_mask = jnp.take_along_axis(attn_mask, logical_indices, axis=-1)
        attn_mask = attn_mask * valid_physical
      elif segment_pos.shape[1] == 1:
        # for decoding without sliding window cache
        sliding_mask = create_sliding_window_mask(
            attn_mask,
            sliding_window_size=window_size,
        )
        attn_mask = sliding_mask * attn_mask
      else:  # for prefill
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

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
      force_eager: bool = False,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[
          jaxtyping.Array,
          jaxtyping.Array,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
      ],
  ]:
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state,
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache,
          segment_ids,
          force_eager,
      )
    else:
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
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
