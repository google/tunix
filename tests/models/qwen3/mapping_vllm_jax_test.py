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

"""Unit tests for Qwen3 declarative HF -> vLLM-JAX manifest."""

from __future__ import annotations

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.models import hf_mapping
from tunix.models import hf_mapping_jax
from tunix.models.qwen3 import mapping_vllm_jax


@dataclasses.dataclass
class _FakeQwen3Config:
  vocab_size: int
  hidden_size: int
  intermediate_size: int
  num_hidden_layers: int
  num_attention_heads: int
  num_key_value_heads: int | None
  head_dim: int | None
  tie_word_embeddings: bool
  attention_bias: bool


def _make_hf_weights(
    cfg: _FakeQwen3Config,
    *,
    include_lm_head: bool = True,
) -> dict[str, jnp.ndarray]:
  rng = np.random.default_rng(0)
  d_model = cfg.hidden_size
  inter = cfg.intermediate_size
  n_heads = cfg.num_attention_heads
  n_kv = (
      n_heads if cfg.num_key_value_heads is None else cfg.num_key_value_heads
  )
  head_dim = d_model // n_heads if cfg.head_dim is None else cfg.head_dim

  def _arr(shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))

  weights: dict[str, jnp.ndarray] = {
      'model.embed_tokens.weight': _arr((cfg.vocab_size, d_model)),
      'model.norm.weight': _arr((d_model,)),
  }
  for layer_idx in range(cfg.num_hidden_layers):
    p = f'model.layers.{layer_idx}'
    weights.update({
        f'{p}.input_layernorm.weight': _arr((d_model,)),
        f'{p}.post_attention_layernorm.weight': _arr((d_model,)),
        f'{p}.self_attn.q_norm.weight': _arr((head_dim,)),
        f'{p}.self_attn.k_norm.weight': _arr((head_dim,)),
        f'{p}.self_attn.q_proj.weight': _arr((n_heads * head_dim, d_model)),
        f'{p}.self_attn.k_proj.weight': _arr((n_kv * head_dim, d_model)),
        f'{p}.self_attn.v_proj.weight': _arr((n_kv * head_dim, d_model)),
        f'{p}.self_attn.o_proj.weight': _arr((d_model, n_heads * head_dim)),
        f'{p}.mlp.gate_proj.weight': _arr((inter, d_model)),
        f'{p}.mlp.up_proj.weight': _arr((inter, d_model)),
        f'{p}.mlp.down_proj.weight': _arr((d_model, inter)),
    })
  if include_lm_head:
    weights['lm_head.weight'] = _arr((cfg.vocab_size, d_model))
  return weights


class MappingVllmJaxTest(parameterized.TestCase):

  def test_protocol_compliance(self):
    cfg = _FakeQwen3Config(
        vocab_size=1003,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    self.assertIsInstance(cfg, mapping_vllm_jax.Qwen3HfConfig)

  @parameterized.named_parameters(
      dict(
          testcase_name='padded_vocab_padded_head_kv_repeat_tied',
          vocab_size=1003,
          tp_size=4,
          head_dim=32,
          expected_v_pad=1004,
          expected_h_pad=128,
          num_heads=4,
          num_kv_heads=2,
          expected_rep_q=1,
          expected_rep_kv=2,
          tie_word_embeddings=True,
          include_lm_head=True,
      ),
      dict(
          testcase_name='head_dim_64_no_pad_untied',
          vocab_size=1024,
          tp_size=2,
          head_dim=64,
          expected_v_pad=1024,
          expected_h_pad=64,
          num_heads=4,
          num_kv_heads=2,
          expected_rep_q=1,
          expected_rep_kv=1,
          tie_word_embeddings=False,
          include_lm_head=True,
      ),
      dict(
          testcase_name='tied_without_lm_head_in_src',
          vocab_size=1000,
          tp_size=1,
          head_dim=128,
          expected_v_pad=1000,
          expected_h_pad=128,
          num_heads=2,
          num_kv_heads=1,
          expected_rep_q=1,
          expected_rep_kv=1,
          tie_word_embeddings=True,
          include_lm_head=False,
      ),
  )
  def test_round_trip_invertibility_and_shapes(
      self,
      vocab_size: int,
      tp_size: int,
      head_dim: int,
      expected_v_pad: int,
      expected_h_pad: int,
      num_heads: int,
      num_kv_heads: int,
      expected_rep_q: int,
      expected_rep_kv: int,
      tie_word_embeddings: bool,
      include_lm_head: bool,
  ):
    d_model = 64
    inter = 96
    cfg = _FakeQwen3Config(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=inter,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=False,
    )
    spec = mapping_vllm_jax.build_vllm_jax_from_hf_manifest(
        cfg, tp_size=tp_size
    )
    self.assertEqual(spec.model_type, 'qwen3')
    if tie_word_embeddings:
      self.assertEqual(spec.dropped_keys, ('lm_head.weight',))
    else:
      self.assertEqual(spec.dropped_keys, ())

    hf_weights = _make_hf_weights(cfg, include_lm_head=include_lm_head)
    vllm_weights = hf_mapping_jax.from_hf_impl(spec, hf_weights)

    self.assertEqual(
        vllm_weights['model.embed_tokens.weight'].shape,
        (expected_v_pad, d_model),
    )
    self.assertEqual(
        vllm_weights['model.layers.0.self_attn.q_norm.weight'].shape,
        (expected_h_pad,),
    )
    self.assertEqual(
        vllm_weights['model.layers.0.self_attn.q_proj.weight'].shape,
        (d_model, num_heads * expected_rep_q, expected_h_pad),
    )
    self.assertEqual(
        vllm_weights['model.layers.0.self_attn.k_proj.weight'].shape,
        (d_model, num_kv_heads * expected_rep_kv, expected_h_pad),
    )
    self.assertEqual(
        vllm_weights['model.layers.0.self_attn.o_proj.weight'].shape,
        (num_heads * expected_rep_q, expected_h_pad, d_model),
    )
    self.assertEqual(
        vllm_weights['model.layers.0.mlp.gate_proj.weight'].shape,
        (d_model, inter),
    )
    self.assertEqual(
        vllm_weights['model.layers.0.mlp.down_proj.weight'].shape,
        (inter, d_model),
    )
    if tie_word_embeddings:
      self.assertNotIn('lm_head.weight', vllm_weights)
    else:
      self.assertEqual(
          vllm_weights['lm_head.weight'].shape, (d_model, expected_v_pad)
      )
      lm_head_rules = [
          r for r in spec.rules if r.target_keys == ('lm_head.weight',)
      ]
      self.assertLen(lm_head_rules, 1)
      self.assertEqual(lm_head_rules[0].sharding, (None, None))

    inv_manifest = hf_mapping.invert_manifest(spec)
    recovered = hf_mapping_jax.to_hf_impl(inv_manifest, vllm_weights)
    expected_keys = set(hf_weights.keys()) - set(spec.dropped_keys)
    self.assertEqual(set(recovered.keys()), expected_keys)
    for k in expected_keys:
      np.testing.assert_allclose(recovered[k], hf_weights[k])

  def test_indivisible_heads_and_invalid_tp_raise(self):
    cfg = _FakeQwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=None,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    with self.assertRaisesRegex(ValueError, 'divisible'):
      mapping_vllm_jax.build_vllm_jax_from_hf_manifest(cfg, tp_size=4)
    with self.assertRaisesRegex(ValueError, 'tp_size'):
      mapping_vllm_jax.build_vllm_jax_from_hf_manifest(cfg, tp_size=0)
    with self.assertRaisesRegex(ValueError, 'head_sharding_size'):
      mapping_vllm_jax.build_vllm_jax_from_hf_manifest(
          cfg, tp_size=2, head_sharding_size=-1
      )

  def test_attention_bias_rejected(self):
    cfg = _FakeQwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=None,
        tie_word_embeddings=True,
        attention_bias=True,
    )
    with self.assertRaisesRegex(ValueError, 'attention_bias=True'):
      mapping_vllm_jax.build_vllm_jax_from_hf_manifest(cfg, tp_size=2)


if __name__ == '__main__':
  absltest.main()
