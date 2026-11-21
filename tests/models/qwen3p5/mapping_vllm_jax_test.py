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

import dataclasses
import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import numpy as safe_np
from tunix.generate import mappings as mappings_lib
from tunix.generate import utils as gen_utils
from tunix.models import qwen3p5
from tunix.models.qwen3p5 import mapping_vllm_jax
from tunix.models.qwen3p5 import model as qwen3p5_model
from tunix.models.qwen3p5 import params as qwen3p5_params


def _make_hybrid_config(
    *, use_tied_embedding: bool
) -> qwen3p5_model.ModelConfig:
  return qwen3p5_model.ModelConfig(
      num_layers=2,
      vocab_size=32,
      embed_dim=16,
      hidden_dim=24,
      num_heads=4,
      head_dim=8,
      num_kv_heads=2,
      rope_theta=10_000_000,
      norm_eps=1e-6,
      use_tied_embedding=use_tied_embedding,
      use_norm_offset=True,
      attn_output_gate=True,
      partial_rotary_factor=0.25,
      layer_types=(
          qwen3p5_model.LayerType.LINEAR_ATTENTION,
          qwen3p5_model.LayerType.FULL_ATTENTION,
      ),
      linear_num_key_heads=2,
      linear_num_value_heads=4,
      linear_key_head_dim=8,
      linear_value_head_dim=8,
      linear_conv_kernel_dim=4,
      dtype=jnp.bfloat16,
      param_dtype=jnp.bfloat16,
  )


def _populate_distinct_weights(state: nnx.State) -> nnx.State:
  flat_entries = []
  for idx, (keys, param) in enumerate(nnx.to_flat_state(state)):
    arr = param.get_value()
    size = int(np.prod(arr.shape))
    vals = (
        jnp.arange(1, size + 1, dtype=jnp.float32).reshape(arr.shape) * 0.01
        + float(idx + 1)
    ).astype(arr.dtype)
    flat_entries.append((keys, nnx.Param(vals)))
  return nnx.from_flat_state(flat_entries)


def _make_dummy_vllm_target_state(
    cfg: qwen3p5_model.ModelConfig,
) -> dict[str, jax.Array]:
  d = cfg.embed_dim
  f = cfg.hidden_dim
  v = cfg.vocab_size
  q_width = 2 * cfg.num_heads * cfg.head_dim
  kv_width = cfg.num_kv_heads * cfg.head_dim
  key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
  val_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
  conv_dim = 2 * key_dim + val_dim
  nv = cfg.linear_num_value_heads
  k_conv = cfg.linear_conv_kernel_dim

  return {
      'vllm_model.model.embed_tokens.weight': jnp.zeros((v, d), jnp.bfloat16),
      'vllm_model.lm_head.weight': jnp.zeros((v, d), jnp.bfloat16),
      'vllm_model.model.norm.weight': jnp.zeros((d,), jnp.bfloat16),
      # Layer 0: Gated DeltaNet (linear attention)
      'vllm_model.model.layers.0.input_layernorm.weight': jnp.zeros(
          (d,), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.post_attention_layernorm.weight': jnp.zeros(
          (d,), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.mlp.gate_up_proj.weight': jnp.zeros(
          (d, 2 * f), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.mlp.down_proj.weight': jnp.zeros(
          (f, d), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.linear_attn.in_proj_qkvz.weight': jnp.zeros(
          (d, conv_dim + val_dim), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.linear_attn.in_proj_ba.weight': jnp.zeros(
          (d, 2 * nv), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.linear_attn.conv1d.weight': jnp.zeros(
          (conv_dim, 1, k_conv), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.linear_attn.dt_bias': jnp.zeros(
          (nv,), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.linear_attn.A_log': jnp.zeros(
          (nv,), jnp.float32
      ),
      'vllm_model.model.layers.0.linear_attn.norm.weight': jnp.zeros(
          (cfg.linear_value_head_dim,), jnp.bfloat16
      ),
      'vllm_model.model.layers.0.linear_attn.out_proj.weight': jnp.zeros(
          (val_dim, d), jnp.bfloat16
      ),
      # Layer 1: Output-Gated GQA (full attention) + rotary cache buffer
      'vllm_model.model.layers.1.input_layernorm.weight': jnp.zeros(
          (d,), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.post_attention_layernorm.weight': jnp.zeros(
          (d,), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.mlp.gate_up_proj.weight': jnp.zeros(
          (d, 2 * f), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.mlp.down_proj.weight': jnp.zeros(
          (f, d), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.self_attn.qkv_proj.weight': jnp.zeros(
          (d, q_width + 2 * kv_width), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.self_attn.o_proj.weight': jnp.zeros(
          (cfg.num_heads * cfg.head_dim, d), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.self_attn.q_norm.weight': jnp.zeros(
          (cfg.head_dim,), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.self_attn.k_norm.weight': jnp.zeros(
          (cfg.head_dim,), jnp.bfloat16
      ),
      'vllm_model.model.layers.1.self_attn.rotary_emb.cos_sin_cache': jnp.full(
          (16, 4), 42.0, dtype=jnp.float32
      ),
  }


class MappingVllmJaxTest(parameterized.TestCase):

  def _require_devices(self, min_devices: int = 4) -> None:
    num_devices = len(jax.devices())
    if num_devices < min_devices:
      self.skipTest(
          f'Requires at least {min_devices} JAX devices, got {num_devices}.'
      )

  def test_model_exposes_vllm_jax_mapping_config(self):
    cfg = _make_hybrid_config(use_tied_embedding=True)
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))

    mapping_cfg = mappings_lib.MappingConfig.build(
        model=model, backend='vllm_jax'
    )

    self.assertEqual(
        qwen3p5.BACKEND_MAPPINGS['vllm_jax'], mapping_vllm_jax.VLLM_JAX_MAPPING
    )
    self.assertEqual(
        mapping_cfg.to_hf_mappings, mapping_vllm_jax.TO_HF_MAPPINGS
    )
    self.assertEqual(
        mapping_cfg.to_hf_transpose_keys,
        mapping_vllm_jax.TO_HF_TRANSPOSE_KEYS,
    )
    self.assertIs(
        mapping_cfg.preprocess_src_state, mapping_vllm_jax.preprocess_src_state
    )

  @parameterized.named_parameters(
      dict(testcase_name='tied_embeddings', use_tied_embedding=True),
      dict(testcase_name='untied_embeddings', use_tied_embedding=False),
  )
  def test_full_hybrid_state_transfer_matches_vllm_layouts(
      self, use_tied_embedding: bool
  ):
    cfg = _make_hybrid_config(use_tied_embedding=use_tied_embedding)
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    orig_state = _populate_distinct_weights(nnx.state(model))
    dst_state = _make_dummy_vllm_target_state(cfg)
    cos_sin_ref = dst_state[
        'vllm_model.model.layers.1.self_attn.rotary_emb.cos_sin_cache'
    ]

    preprocessed = mapping_vllm_jax.preprocess_src_state(orig_state, tp_size=1)
    transferred = gen_utils.transfer_state_with_mappings(
        src_state=preprocessed,
        dst_state=dst_state,
        key_mappings=mapping_vllm_jax.TO_HF_MAPPINGS,
        transpose_keys=mapping_vllm_jax.TO_HF_TRANSPOSE_KEYS,
    )

    with self.subTest('every_non_cache_target_updated'):
      cos_sin_key = (
          'vllm_model.model.layers.1.self_attn.rotary_emb.cos_sin_cache'
      )
      for key, orig_zero in dst_state.items():
        if key == cos_sin_key:
          continue
        self.assertFalse(
            np.array_equal(transferred[key], orig_zero),
            msg=f'Target {key} remained at zero placeholder',
        )

    with self.subTest('embed_final_norm_and_lm_head'):
      np.testing.assert_array_equal(
          transferred['vllm_model.model.embed_tokens.weight'],
          orig_state['embedder']['input_embedding'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.norm.weight'],
          orig_state['final_norm']['w'].get_value(),
      )
      if use_tied_embedding:
        np.testing.assert_array_equal(
            transferred['vllm_model.lm_head.weight'],
            orig_state['embedder']['input_embedding'].get_value(),
        )
      else:
        np.testing.assert_array_equal(
            transferred['vllm_model.lm_head.weight'],
            orig_state['lm_head']['w'].get_value().T,
        )
        self.assertFalse(
            np.array_equal(
                transferred['vllm_model.lm_head.weight'],
                transferred['vllm_model.model.embed_tokens.weight'],
            )
        )

    with self.subTest('gdn_layer_0_and_mlp'):
      layer0 = orig_state['layers'][0]
      gdn = layer0['linear_attn']
      mlp0 = layer0['mlp']
      expected_qkvz = jnp.concatenate(
          [
              gdn['in_proj_qkv']['kernel'].get_value(),
              gdn['in_proj_z']['kernel'].get_value(),
          ],
          axis=-1,
      )
      expected_ba = jnp.concatenate(
          [
              gdn['in_proj_b']['kernel'].get_value(),
              gdn['in_proj_a']['kernel'].get_value(),
          ],
          axis=-1,
      )
      expected_gate_up_0 = jnp.concatenate(
          [
              mlp0['gate_proj']['kernel'].get_value(),
              mlp0['up_proj']['kernel'].get_value(),
          ],
          axis=-1,
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.input_layernorm.weight'],
          layer0['input_layernorm']['w'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred[
              'vllm_model.model.layers.0.post_attention_layernorm.weight'
          ],
          layer0['post_attention_layernorm']['w'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.mlp.gate_up_proj.weight'],
          expected_gate_up_0,
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.mlp.down_proj.weight'],
          mlp0['down_proj']['kernel'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred[
              'vllm_model.model.layers.0.linear_attn.in_proj_qkvz.weight'
          ],
          expected_qkvz,
      )
      np.testing.assert_array_equal(
          transferred[
              'vllm_model.model.layers.0.linear_attn.in_proj_ba.weight'
          ],
          expected_ba,
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.linear_attn.conv1d.weight'],
          jnp.transpose(gdn['conv1d']['kernel'].get_value(), (2, 1, 0)),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.linear_attn.dt_bias'],
          gdn['dt_bias'].get_value(),
      )
      self.assertEqual(
          transferred['vllm_model.model.layers.0.linear_attn.A_log'].dtype,
          jnp.float32,
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.linear_attn.A_log'],
          gdn['A_log'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.linear_attn.norm.weight'],
          gdn['norm']['w'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.0.linear_attn.out_proj.weight'],
          gdn['out_proj']['kernel'].get_value(),
      )

    with self.subTest('full_attn_layer_1_and_mlp'):
      layer1 = orig_state['layers'][1]
      attn = layer1['attn']
      mlp = layer1['mlp']
      expected_qkv = jnp.concatenate(
          [
              attn['q_proj']['w'].get_value().reshape(cfg.embed_dim, -1),
              attn['k_proj']['w'].get_value().reshape(cfg.embed_dim, -1),
              attn['v_proj']['w'].get_value().reshape(cfg.embed_dim, -1),
          ],
          axis=-1,
      )
      expected_gate_up = jnp.concatenate(
          [
              mlp['gate_proj']['kernel'].get_value(),
              mlp['up_proj']['kernel'].get_value(),
          ],
          axis=-1,
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.1.input_layernorm.weight'],
          layer1['input_layernorm']['w'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred[
              'vllm_model.model.layers.1.post_attention_layernorm.weight'
          ],
          layer1['post_attention_layernorm']['w'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.1.self_attn.qkv_proj.weight'],
          expected_qkv,
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.1.self_attn.o_proj.weight'],
          attn['o_proj']['w'].get_value().reshape(-1, cfg.embed_dim),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.1.self_attn.q_norm.weight'],
          attn['q_norm']['w'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.1.self_attn.k_norm.weight'],
          attn['k_norm']['w'].get_value(),
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.1.mlp.gate_up_proj.weight'],
          expected_gate_up,
      )
      np.testing.assert_array_equal(
          transferred['vllm_model.model.layers.1.mlp.down_proj.weight'],
          mlp['down_proj']['kernel'].get_value(),
      )
      self.assertIs(
          transferred[
              'vllm_model.model.layers.1.self_attn.rotary_emb.cos_sin_cache'
          ],
          cos_sin_ref,
      )

  def test_roundtrip_from_hf_safetensors_matches_vllm_tp1_semantics(self):
    cfg = dataclasses.replace(
        _make_hybrid_config(use_tied_embedding=False),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    d = cfg.embed_dim
    f = cfg.hidden_dim
    v = cfg.vocab_size
    q_width = 2 * cfg.num_heads * cfg.head_dim
    kv_width = cfg.num_kv_heads * cfg.head_dim
    key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    val_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    conv_dim = 2 * key_dim + val_dim
    nv = cfg.linear_num_value_heads
    k_conv = cfg.linear_conv_kernel_dim

    rng = np.random.default_rng(123)
    rand = lambda *shape: rng.standard_normal(shape, dtype=np.float32)

    hf = {
        'model.language_model.embed_tokens.weight': rand(v, d),
        'model.language_model.lm_head.weight': rand(v, d),
        'model.language_model.norm.weight': rand(d),
        # Layer 0: Gated DeltaNet + MLP in HF (out_features, in_features) layout
        'model.language_model.layers.0.input_layernorm.weight': rand(d),
        'model.language_model.layers.0.post_attention_layernorm.weight': rand(
            d
        ),
        'model.language_model.layers.0.mlp.gate_proj.weight': rand(f, d),
        'model.language_model.layers.0.mlp.up_proj.weight': rand(f, d),
        'model.language_model.layers.0.mlp.down_proj.weight': rand(d, f),
        'model.language_model.layers.0.linear_attn.in_proj_qkv.weight': rand(
            conv_dim, d
        ),
        'model.language_model.layers.0.linear_attn.in_proj_z.weight': rand(
            val_dim, d
        ),
        'model.language_model.layers.0.linear_attn.in_proj_b.weight': rand(
            nv, d
        ),
        'model.language_model.layers.0.linear_attn.in_proj_a.weight': rand(
            nv, d
        ),
        'model.language_model.layers.0.linear_attn.conv1d.weight': rand(
            conv_dim, 1, k_conv
        ),
        'model.language_model.layers.0.linear_attn.dt_bias': rand(nv),
        'model.language_model.layers.0.linear_attn.A_log': rand(nv),
        'model.language_model.layers.0.linear_attn.norm.weight': rand(
            cfg.linear_value_head_dim
        ),
        'model.language_model.layers.0.linear_attn.out_proj.weight': rand(
            d, val_dim
        ),
        # Layer 1: Output-gated GQA + MLP in HF layout
        'model.language_model.layers.1.input_layernorm.weight': rand(d),
        'model.language_model.layers.1.post_attention_layernorm.weight': rand(
            d
        ),
        'model.language_model.layers.1.mlp.gate_proj.weight': rand(f, d),
        'model.language_model.layers.1.mlp.up_proj.weight': rand(f, d),
        'model.language_model.layers.1.mlp.down_proj.weight': rand(d, f),
        'model.language_model.layers.1.self_attn.q_proj.weight': rand(
            q_width, d
        ),
        'model.language_model.layers.1.self_attn.k_proj.weight': rand(
            kv_width, d
        ),
        'model.language_model.layers.1.self_attn.v_proj.weight': rand(
            kv_width, d
        ),
        'model.language_model.layers.1.self_attn.o_proj.weight': rand(
            d, cfg.num_heads * cfg.head_dim
        ),
        'model.language_model.layers.1.self_attn.q_norm.weight': rand(
            cfg.head_dim
        ),
        'model.language_model.layers.1.self_attn.k_norm.weight': rand(
            cfg.head_dim
        ),
    }
    ckpt_dir = self.create_tempdir().full_path
    safe_np.save_file(hf, os.path.join(ckpt_dir, 'model.safetensors'))

    loaded_model = qwen3p5_params.create_model_from_safe_tensors(ckpt_dir, cfg)
    dst_state = {
        k: v.astype(jnp.float32)
        for k, v in _make_dummy_vllm_target_state(cfg).items()
    }
    transferred = gen_utils.transfer_state_with_mappings(
        src_state=mapping_vllm_jax.preprocess_src_state(
            nnx.state(loaded_model), tp_size=1
        ),
        dst_state=dst_state,
        key_mappings=mapping_vllm_jax.TO_HF_MAPPINGS,
        transpose_keys=mapping_vllm_jax.TO_HF_TRANSPOSE_KEYS,
    )

    p0 = 'model.language_model.layers.0'
    p1 = 'model.language_model.layers.1'
    np.testing.assert_allclose(
        transferred['vllm_model.model.embed_tokens.weight'],
        hf['model.language_model.embed_tokens.weight'],
    )
    np.testing.assert_allclose(
        transferred['vllm_model.lm_head.weight'],
        hf['model.language_model.lm_head.weight'],
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.norm.weight'],
        hf['model.language_model.norm.weight'],
    )
    np.testing.assert_allclose(
        transferred[
            'vllm_model.model.layers.0.linear_attn.in_proj_qkvz.weight'
        ],
        np.concatenate(
            [
                hf[f'{p0}.linear_attn.in_proj_qkv.weight'],
                hf[f'{p0}.linear_attn.in_proj_z.weight'],
            ],
            axis=0,
        ).T,
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.layers.0.linear_attn.in_proj_ba.weight'],
        np.concatenate(
            [
                hf[f'{p0}.linear_attn.in_proj_b.weight'],
                hf[f'{p0}.linear_attn.in_proj_a.weight'],
            ],
            axis=0,
        ).T,
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.layers.0.linear_attn.conv1d.weight'],
        hf[f'{p0}.linear_attn.conv1d.weight'],
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.layers.0.linear_attn.out_proj.weight'],
        hf[f'{p0}.linear_attn.out_proj.weight'].T,
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.layers.1.self_attn.qkv_proj.weight'],
        np.concatenate(
            [
                hf[f'{p1}.self_attn.q_proj.weight'],
                hf[f'{p1}.self_attn.k_proj.weight'],
                hf[f'{p1}.self_attn.v_proj.weight'],
            ],
            axis=0,
        ).T,
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.layers.1.self_attn.o_proj.weight'],
        hf[f'{p1}.self_attn.o_proj.weight'].T,
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.layers.1.mlp.gate_up_proj.weight'],
        np.concatenate(
            [hf[f'{p1}.mlp.gate_proj.weight'], hf[f'{p1}.mlp.up_proj.weight']],
            axis=0,
        ).T,
    )
    np.testing.assert_allclose(
        transferred['vllm_model.model.layers.1.mlp.down_proj.weight'],
        hf[f'{p1}.mlp.down_proj.weight'].T,
    )

  def test_preprocess_rejects_tp_greater_than_one(self):
    cfg = _make_hybrid_config(use_tied_embedding=True)
    state = nnx.state(qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0)))

    with self.assertRaisesRegex(
        NotImplementedError, 'tensor_parallel_size=1 only'
    ):
      mapping_vllm_jax.preprocess_src_state(state, tp_size=2)

  def test_preprocess_rejects_lora_adapter_leaves(self):
    lora_state = nnx.State({
        'layers': {
            0: {
                'mlp': {
                    'gate_proj': {
                        'kernel_lora_a': nnx.Param(jnp.ones((4, 2))),
                    }
                }
            }
        }
    })

    with self.assertRaisesRegex(NotImplementedError, 'LoRA'):
      mapping_vllm_jax.preprocess_src_state(lora_state, tp_size=1)

  def test_preprocess_rejects_incomplete_fusion_group(self):
    partial_state = nnx.State({
        'layers': {
            0: {
                'mlp': {
                    'gate_proj': {
                        'kernel': nnx.Param(jnp.ones((4, 8))),
                    }
                }
            }
        }
    })

    with self.assertRaisesRegex(ValueError, 'Incomplete Qwen3.5 fusion group'):
      mapping_vllm_jax.preprocess_src_state(partial_state, tp_size=1)

  @parameterized.named_parameters(
      dict(
          testcase_name='explicit_mesh',
          axis_type=jax.sharding.AxisType.Explicit,
      ),
      dict(testcase_name='auto_mesh', axis_type=jax.sharding.AxisType.Auto),
  )
  def test_preprocess_handles_mismatched_shardings(self, axis_type):
    self._require_devices(4)
    mesh = jax.make_mesh(
        (2, 2),
        ('fsdp', 'tp'),
        axis_types=(axis_type, axis_type),
    )
    shd_qkv = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('fsdp', None)
    )
    shd_z = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('fsdp', 'tp')
    )
    qkv = jax.device_put(
        jnp.arange(16, dtype=jnp.float32).reshape(4, 4), shd_qkv
    )
    z = jax.device_put(
        jnp.arange(100, 116, dtype=jnp.float32).reshape(4, 4), shd_z
    )
    sharded_state = nnx.State({
        'layers': {
            0: {
                'linear_attn': {
                    'in_proj_qkv': {'kernel': nnx.Param(qkv)},
                    'in_proj_z': {'kernel': nnx.Param(z)},
                }
            }
        }
    })

    fused_state = mapping_vllm_jax.preprocess_src_state(
        sharded_state, tp_size=1
    )

    fused = fused_state['layers'][0]['linear_attn']['in_proj_qkvz'][
        'kernel'
    ].get_value()
    self.assertEqual(
        fused.sharding,
        jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec('fsdp', None)
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(fused),
        np.concatenate([np.asarray(qkv), np.asarray(z)], axis=-1),
    )

  def test_preprocess_handles_non_named_first_operand(self):
    self._require_devices(4)
    mesh_1d = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))
    mesh_2x2 = jax.make_mesh((2, 2), ('fsdp', 'tp'))
    single_dev_shd = jax.sharding.SingleDeviceSharding(mesh_1d.devices.flat[0])
    named_shd = jax.sharding.NamedSharding(
        mesh_2x2, jax.sharding.PartitionSpec('fsdp', 'tp')
    )
    qkv = jax.device_put(
        jnp.arange(16, dtype=jnp.float32).reshape(4, 4), single_dev_shd
    )
    z = jax.device_put(
        jnp.arange(100, 116, dtype=jnp.float32).reshape(4, 4), named_shd
    )
    mixed_state = nnx.State({
        'layers': {
            0: {
                'linear_attn': {
                    'in_proj_qkv': {'kernel': nnx.Param(qkv)},
                    'in_proj_z': {'kernel': nnx.Param(z)},
                }
            }
        }
    })
    fused_state = mapping_vllm_jax.preprocess_src_state(mixed_state, tp_size=1)
    fused = fused_state['layers'][0]['linear_attn']['in_proj_qkvz'][
        'kernel'
    ].get_value()
    self.assertEqual(
        fused.sharding,
        jax.sharding.NamedSharding(
            mesh_2x2, jax.sharding.PartitionSpec('fsdp', None)
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(fused),
        np.concatenate([np.asarray(qkv), np.asarray(z)], axis=-1),
    )

  def test_preprocess_rejects_distinct_meshes(self):
    self._require_devices(4)
    mesh_a = jax.sharding.Mesh(np.array(jax.devices()[:2]), ('fsdp',))
    mesh_b = jax.sharding.Mesh(np.array(jax.devices()[2:4]), ('fsdp',))
    qkv_a = jax.device_put(
        jnp.ones((4, 4), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh_a, jax.sharding.PartitionSpec('fsdp')),
    )
    z_b = jax.device_put(
        jnp.ones((4, 2), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh_b, jax.sharding.PartitionSpec('fsdp')),
    )
    cross_mesh_state = nnx.State({
        'layers': {
            0: {
                'linear_attn': {
                    'in_proj_qkv': {'kernel': nnx.Param(qkv_a)},
                    'in_proj_z': {'kernel': nnx.Param(z_b)},
                }
            }
        }
    })
    with self.assertRaisesRegex(ValueError, 'distinct JAX meshes'):
      mapping_vllm_jax.preprocess_src_state(cross_mesh_state, tp_size=1)

  def test_float32_master_weights_preserve_exact_a_log_and_cast_bf16_targets(
      self,
  ):
    cfg = dataclasses.replace(
        _make_hybrid_config(use_tied_embedding=True),
        param_dtype=jnp.float32,
    )
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    orig_state = _populate_distinct_weights(nnx.state(model))
    exact_a_log = jnp.array(
        [-3.1415927, -2.7182817, -1.4142135, -0.6931472], dtype=jnp.float32
    )
    orig_state['layers'][0]['linear_attn']['A_log'].set_value(exact_a_log)
    dst_state = _make_dummy_vllm_target_state(cfg)

    preprocessed = mapping_vllm_jax.preprocess_src_state(orig_state, tp_size=1)
    transferred = gen_utils.transfer_state_with_mappings(
        src_state=preprocessed,
        dst_state=dst_state,
        key_mappings=mapping_vllm_jax.TO_HF_MAPPINGS,
        transpose_keys=mapping_vllm_jax.TO_HF_TRANSPOSE_KEYS,
    )

    self.assertEqual(
        transferred['vllm_model.model.layers.0.linear_attn.A_log'].dtype,
        jnp.float32,
    )
    np.testing.assert_array_equal(
        transferred['vllm_model.model.layers.0.linear_attn.A_log'],
        exact_a_log,
    )
    self.assertEqual(
        transferred[
            'vllm_model.model.layers.0.linear_attn.in_proj_qkvz.weight'
        ].dtype,
        jnp.bfloat16,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='explicit_trainer_mesh',
          axis_type=jax.sharding.AxisType.Explicit,
      ),
      dict(
          testcase_name='auto_trainer_mesh',
          axis_type=jax.sharding.AxisType.Auto,
      ),
  )
  def test_full_sharded_trainer_model_transfers_to_tp1_vllm_state(
      self, axis_type
  ):
    self._require_devices(4)
    cfg = _make_hybrid_config(use_tied_embedding=False)
    mesh = jax.make_mesh(
        (2, 2),
        ('fsdp', 'tp'),
        axis_types=(axis_type, axis_type),
    )
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    unsharded_state = _populate_distinct_weights(nnx.state(model))
    named_shardings = nnx.get_named_sharding(nnx.state(model), mesh)
    sharded_state = jax.tree.map(
        lambda param, shd: nnx.Param(
            jax.device_put(param.get_value(), shd.get_value())
        ),
        unsharded_state,
        named_shardings,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    dst_state = _make_dummy_vllm_target_state(cfg)

    expected = gen_utils.transfer_state_with_mappings(
        src_state=mapping_vllm_jax.preprocess_src_state(
            unsharded_state, tp_size=1
        ),
        dst_state=_make_dummy_vllm_target_state(cfg),
        key_mappings=mapping_vllm_jax.TO_HF_MAPPINGS,
        transpose_keys=mapping_vllm_jax.TO_HF_TRANSPOSE_KEYS,
    )
    actual = gen_utils.transfer_state_with_mappings(
        src_state=mapping_vllm_jax.preprocess_src_state(
            sharded_state, tp_size=1
        ),
        dst_state=dst_state,
        key_mappings=mapping_vllm_jax.TO_HF_MAPPINGS,
        transpose_keys=mapping_vllm_jax.TO_HF_TRANSPOSE_KEYS,
    )

    self.assertEqual(set(actual.keys()), set(expected.keys()))
    for key in expected:
      np.testing.assert_array_equal(
          np.asarray(actual[key]), np.asarray(expected[key]), err_msg=key
      )


if __name__ == '__main__':
  if '--xla_force_host_platform_device_count' not in os.environ.get(
      'XLA_FLAGS', ''
  ):
    jax.config.update('jax_num_cpu_devices', 4)
  absltest.main()
