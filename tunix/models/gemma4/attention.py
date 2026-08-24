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

from functools import partial
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
from tunix.models.gemma4.config import AttentionType
from tunix.models.gemma4.config import K_MASK
from tunix.models.gemma4.config import LayerCache
from tunix.models.gemma4.config import ModelConfig
from tunix.models.gemma4.config import RematConfig
from tunix.models.gemma4.layers import apply_rope
from tunix.models.gemma4.layers import Einsum
from tunix.models.gemma4.layers import RMSNorm
from tunix.utils.sharding_utils import shard


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

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
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

    if kv_shared_cache is not None:
      assert cache is None
      key_proj = kv_shared_cache['k']
      value_proj = kv_shared_cache['v']
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

    if cache is not None:
      assert kv_shared_cache is None
      # Update cache with new kv projections
      cache_len = cache['v'].shape[1]
      if seq_len > 1:  # prefill
        if self.config.use_sliding_window_kv_cache:
          # Sliding window cache update (prefill).
          # Does not support chunked prefill.
          valid_len = min(seq_len, cache_len)
          latest_indices = jnp.arange(seq_len - valid_len, seq_len) % cache_len
          cache_v = (
              cache['v']
              .at[:, latest_indices, ...]
              .set(value_proj[:, -valid_len:, ...])
          )
          cache_k = (
              cache['k']
              .at[:, latest_indices, ...]
              .set(key_proj[:, -valid_len:, ...])
          )
        else:
          cache_v = cache['v'].at[:, :seq_len, ...].set(value_proj)
          cache_k = cache['k'].at[:, :seq_len, ...].set(key_proj)

        new_cache = {
            'v': cache_v,
            'k': cache_k,
            'end_index': cache['end_index'] + seq_len,
        }
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

    if self.config.use_flash_attention and seq_len > 1:
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      mesh = pxla.thread_resources.env.physical_mesh
      if self.attn_type == AttentionType.LOCAL_SLIDING:
        mask = mask_lib.LocalMask(
            (seq_len, seq_len),
            window_size=(self.config.sliding_window_size - 1, 0),  # pyrefly: ignore[unsupported-operation]
            offset=0,
        )
      else:
        mask = mask_lib.CausalMask((seq_len, seq_len))

      multi_head_mask = mask_lib.MultiHeadMask([mask for _ in range(qh)])

      block_sizes = splash.BlockSizes(
          block_q=self.config.flash_attention_block_size,
          block_kv=self.config.flash_attention_block_size,
          block_q_dkv=self.config.flash_attention_block_size,
          block_kv_dkv=self.config.flash_attention_block_size,
          block_kv_dkv_compute=self.config.flash_attention_block_size,
          block_q_dq=self.config.flash_attention_block_size,
          block_kv_dq=self.config.flash_attention_block_size,
      )

      shd_b, shd_t, shd_n, shd_h = self.config.shd_config.act_btnh
      if (
          mesh is not None
          and shd_b is not None
          and shd_b in mesh.shape
          and b % mesh.shape[shd_b] != 0
      ):
        shd_b = None
      head_shards = (
          mesh.shape[shd_n] if shd_n is not None and shd_n in mesh.shape else 1
      )
      q_seq_shards = (
          mesh.shape[shd_t] if shd_t is not None and shd_t in mesh.shape else 1
      )

      splash_attn_kernel = splash.make_splash_mha(
          multi_head_mask,
          block_sizes=block_sizes,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
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
      kernel_spec = splash_attn_kernel.manual_sharding_spec(
          shd.NamedSharding(mesh, P(shd_n, shd_t))
      )

      if segment_ids is not None:
        seg_spec = P(shd_b, shd_t)
        unsharded_seg_spec = P(shd_b, None)

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
          seg_ids = splash.SegmentIds(q=q_seg_block, kv=kv_seg_block)
          return jax.vmap(kernel)(
              q_block, k_block, v_block, segment_ids=seg_ids
          )

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
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

    else:
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

      if seq_len > 1:
        # Only compute attention scores for the actual sequence length.
        attn_mask = attn_mask[..., :seq_len]

      if self.attn_type == AttentionType.LOCAL_SLIDING:
        if (
            segment_pos.shape[1] == 1
            and self.config.use_sliding_window_kv_cache
        ):
          # for decoding with sliding window cache
          active_cache = cache if cache is not None else kv_shared_cache
          if active_cache is None:
            raise ValueError(
                'Cache or shared cache is required for local sliding attention'
                ' in decoding.'
            )
          cache_len = key_proj.shape[1]
          end_idx = active_cache['end_index']
          if cache is None and kv_shared_cache is not None:
            # In case of shared KV cache, the origin layer already updated the
            # end index. We need to subtract 1 to get the correct end index of
            # the previous token.
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
              sliding_window_size=self.config.sliding_window_size,  # pyrefly: ignore[bad-argument-type]
          )
          attn_mask = sliding_mask * attn_mask
        else:  # for prefill
          all_ones = jnp.ones_like(attn_mask)
          sliding_mask = jnp.triu(
              all_ones, -1 * self.config.sliding_window_size + 1  # pyrefly: ignore[unsupported-operation]
          ) * jnp.tril(
              all_ones, self.config.sliding_window_size - 1  # pyrefly: ignore[unsupported-operation]
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

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.config.shd_config.act_btd)
    return new_cache, attn_output, (key_proj, value_proj)

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.config.num_heads and self.num_kv_heads > 1

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
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
          state, x, segment_pos, cache, attn_mask, kv_shared_cache, segment_ids
      )
    else:
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
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
    k = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True,
    )
    v = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True,
    )
    end_index = shard(
        np.zeros((batch_size,), np.int32),
        self.config.shd_config.act_btnh[:1],
        eager=True,
    )
    return {'k': k, 'v': v, 'end_index': end_index}
