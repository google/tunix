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

"""Regression tests for flash-attention KV cache layout during decode."""

from __future__ import annotations

import contextlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from tunix.models.gemma4 import model as gemma4_model
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen3 import model as qwen3_model


pytestmark = pytest.mark.cpu_only


class _FakeSplashKernel:
  """CPU-safe stand-in for the TPU splash attention kernel."""

  def manual_sharding_spec(self, unused_named_sharding):
    return None

  def __call__(self, q, k, v, segment_ids=None):
    del k, v, segment_ids
    return q


def _identity_shard_map(fn=None, **unused_kwargs):
  if fn is None:
    return lambda inner_fn: inner_fn
  return fn


@contextlib.contextmanager
def _patch_flash_attention(module):
  with contextlib.ExitStack() as stack:
    stack.enter_context(
        mock.patch.object(module, 'shard_map', _identity_shard_map)
    )
    stack.enter_context(
        mock.patch.object(
            module.shd,
            'NamedSharding',
            lambda *args, **kwargs: None,
        )
    )
    stack.enter_context(
        mock.patch.object(
            module.splash,
            'make_splash_mha',
            lambda *args, **kwargs: _FakeSplashKernel(),
        )
    )
    stack.enter_context(
        mock.patch.object(
            module.splash,
            'BlockSizes',
            lambda **kwargs: object(),
        )
    )
    yield


def _build_causal_mask(
    batch_size: int,
    query_start: int,
    query_len: int,
    cache_size: int,
) -> jnp.ndarray:
  query_positions = jnp.arange(query_start, query_start + query_len)
  key_positions = jnp.arange(cache_size)
  mask = key_positions[None, :] <= query_positions[:, None]
  return jnp.broadcast_to(mask[None, :, :], (batch_size, query_len, cache_size))


def _build_positions(batch_size: int, start: int, seq_len: int) -> jnp.ndarray:
  positions = jnp.arange(start, start + seq_len, dtype=jnp.int32)
  return jnp.broadcast_to(positions[None, :], (batch_size, seq_len))


def _make_cache(num_layers: int, batch_size: int, cache_size: int, num_kv_heads: int,
                head_dim: int, dtype: jnp.dtype):
  shape = (batch_size, cache_size, num_kv_heads, head_dim)
  return {
      f'layer_{i}': {
          'k': jnp.zeros(shape, dtype=dtype),
          'v': jnp.zeros(shape, dtype=dtype),
          'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
      }
      for i in range(num_layers)
  }


def _tiny_qwen2_model():
  config = qwen2_model.ModelConfig.qwen2p5_0p5b()
  config.num_layers = 1
  config.vocab_size = 64
  config.embed_dim = 64
  config.hidden_dim = 128
  config.num_heads = 4
  config.head_dim = 16
  config.num_kv_heads = 2
  config.use_flash_attention = True
  model = qwen2_model.Qwen2(config, rngs=nnx.Rngs(0))
  return model, config, qwen2_model


def _tiny_qwen3_model():
  config = qwen3_model.ModelConfig.qwen3_0p6b()
  config.num_layers = 1
  config.vocab_size = 64
  config.embed_dim = 64
  config.hidden_dim = 128
  config.num_heads = 4
  config.head_dim = 16
  config.num_kv_heads = 2
  config.use_flash_attention = True
  model = qwen3_model.Qwen3(config, rngs=nnx.Rngs(0))
  return model, config, qwen3_model


def _tiny_gemma4_model():
  config = gemma4_model.ModelConfig.gemma4_e2b()
  config.num_layers = 1
  config.num_embed = 64
  config.embed_dim = 64
  config.hidden_dim = 128
  config.num_heads = 4
  config.head_dim = 16
  config.num_kv_heads = 2
  config.per_layer_input_dim = 0
  config.frac_shared_layers = 0.0
  config.attention_pattern = (gemma4_model.AttentionType.GLOBAL,)
  config.use_flash_attention = True
  model = gemma4_model.Gemma4(config, rngs=nnx.Rngs(0))
  return model, config, gemma4_model


def _make_model_cache(model, config, batch_size: int, cache_size: int):
  if isinstance(model, gemma4_model.Gemma4):
    return model.init_cache(batch_size, cache_size, config.dtype)

  return _make_cache(
      num_layers=config.num_layers,
      batch_size=batch_size,
      cache_size=cache_size,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
      dtype=config.dtype,
  )


class FlashAttentionCacheTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='qwen2', model_builder=_tiny_qwen2_model),
      dict(testcase_name='qwen3', model_builder=_tiny_qwen3_model),
      dict(testcase_name='gemma4', model_builder=_tiny_gemma4_model),
  )
  def test_prefill_cache_layout_survives_decode(self, model_builder):
    model, config, model_module = model_builder()
    batch_size = 2
    prefill_len = 4
    decode_len = 1
    cache_size = 8

    cache = _make_model_cache(model, config, batch_size, cache_size)
    expected_cache_shape = cache['layer_0']['k'].shape

    prefill_tokens = jnp.arange(batch_size * prefill_len, dtype=jnp.int32)
    if hasattr(config, 'vocab_size'):
      prefill_tokens = prefill_tokens.reshape(batch_size, prefill_len) % config.vocab_size
    else:
      prefill_tokens = prefill_tokens.reshape(batch_size, prefill_len) % config.num_embed
    decode_tokens = jnp.full((batch_size, decode_len), 1, dtype=jnp.int32)

    prefill_positions = _build_positions(batch_size, 0, prefill_len)
    decode_positions = _build_positions(batch_size, prefill_len, decode_len)
    prefill_mask = _build_causal_mask(batch_size, 0, prefill_len, cache_size)
    decode_mask = _build_causal_mask(
        batch_size, prefill_len, decode_len, cache_size
    )

    with _patch_flash_attention(model_module):
      _, prefill_cache = model(
          prefill_tokens,
          positions=prefill_positions,
          cache=cache,
          attention_mask=prefill_mask,
      )

    self.assertIsNotNone(prefill_cache)
    layer_cache = prefill_cache['layer_0']
    self.assertEqual(layer_cache['k'].shape, expected_cache_shape)
    self.assertEqual(layer_cache['v'].shape, expected_cache_shape)
    self.assertTrue(jnp.all(layer_cache['end_index'] == prefill_len))
    self.assertGreater(float(jnp.abs(layer_cache['k'][:, :prefill_len]).sum()), 0.0)
    self.assertGreater(float(jnp.abs(layer_cache['v'][:, :prefill_len]).sum()), 0.0)

    logits, decode_cache = model(
        decode_tokens,
        positions=decode_positions,
        cache=prefill_cache,
        attention_mask=decode_mask,
    )

    self.assertEqual(logits.shape[:2], (batch_size, decode_len))
    self.assertEqual(decode_cache['layer_0']['k'].shape, expected_cache_shape)
    self.assertEqual(decode_cache['layer_0']['v'].shape, expected_cache_shape)
    self.assertTrue(jnp.all(decode_cache['layer_0']['end_index'] == prefill_len + decode_len))


if __name__ == '__main__':
  absltest.main()
