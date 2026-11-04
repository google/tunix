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

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import mappings
from tunix.generate import utils
from tunix.models.qwen3p5 import mapping_vllm_jax
from tunix.models.qwen3p5 import model as model_lib


def _tiny_config() -> model_lib.ModelConfig:
  return dataclasses.replace(
      model_lib.ModelConfig.qwen3p5_0p8b(),
      num_layers=4,
      vocab_size=64,
      embed_dim=32,
      hidden_dim=48,
      num_heads=4,
      head_dim=16,
      num_kv_heads=2,
      linear_num_key_heads=2,
      linear_num_value_heads=2,
      linear_key_head_dim=8,
      linear_value_head_dim=8,
      layer_types=None,
      param_dtype=jnp.float32,
      dtype=jnp.float32,
  )


class MappingVllmJaxTest(absltest.TestCase):

  def test_backend_registered(self):
    cfg = _tiny_config()
    model = model_lib.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    mapping = mappings.MappingConfig.build(model=model)
    self.assertIs(
        mapping.preprocess_src_state, mapping_vllm_jax.preprocess_src_state
    )

  def test_transfer_to_vllm_torch_layout(self):
    cfg = _tiny_config()
    model = model_lib.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    flat = {
        '.'.join(map(str, k)): np.asarray(v.value)
        for k, v in state.flat_state()
    }
    d, h, hd, kv = cfg.embed_dim, cfg.num_heads, cfg.head_dim, cfg.num_kv_heads
    p = 'vllm_model.model'
    # Destination mimics the torchax runner state for the tiny config: layer 3
    # is full attention, layers 0-2 are Gated DeltaNet.
    q_out = h * hd * 2
    qkvz_out = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 + (
        cfg.linear_num_value_heads * cfg.linear_value_head_dim * 2
    )
    conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + (
        cfg.linear_num_value_heads * cfg.linear_value_head_dim
    )
    dst = {
        f'{p}.embed_tokens.weight': jnp.zeros((cfg.vocab_size, d)),
        'vllm_model.lm_head.weight': jnp.zeros((cfg.vocab_size, d)),
        f'{p}.norm.weight': jnp.zeros((d,)),
        f'{p}.layers.3.self_attn.qkv_proj.weight': jnp.zeros(
            (d, q_out + 2 * kv * hd)
        ),
        f'{p}.layers.3.self_attn.o_proj.weight': jnp.zeros((h * hd, d)),
        f'{p}.layers.0.linear_attn.in_proj_qkvz.weight': jnp.zeros(
            (d, qkvz_out)
        ),
        f'{p}.layers.0.linear_attn.in_proj_ba.weight': jnp.zeros(
            (d, 2 * cfg.linear_num_value_heads)
        ),
        f'{p}.layers.0.linear_attn.conv1d.weight': jnp.zeros(
            (conv_dim, 1, cfg.linear_conv_kernel_dim)
        ),
        f'{p}.layers.0.mlp.gate_up_proj.weight': jnp.zeros(
            (d, 2 * cfg.hidden_dim)
        ),
    }

    src = mapping_vllm_jax.preprocess_src_state(state, tp_size=1)
    out = utils.transfer_state_with_mappings(
        src_state=src,
        dst_state=dst,
        key_mappings=mapping_vllm_jax.TO_HF_MAPPINGS,
        transpose_keys=mapping_vllm_jax.TO_HF_TRANSPOSE_KEYS,
    )

    np.testing.assert_array_equal(
        out[f'{p}.layers.3.self_attn.qkv_proj.weight'],
        np.concatenate(
            [
                flat['layers.3.attn.q_proj.w'].reshape(d, -1),
                flat['layers.3.attn.k_proj.w'].reshape(d, -1),
                flat['layers.3.attn.v_proj.w'].reshape(d, -1),
            ],
            axis=-1,
        ),
    )
    np.testing.assert_array_equal(
        out[f'{p}.layers.3.self_attn.o_proj.weight'],
        flat['layers.3.attn.o_proj.w'].reshape(h * hd, d),
    )
    np.testing.assert_array_equal(
        out[f'{p}.layers.0.linear_attn.in_proj_qkvz.weight'],
        np.concatenate(
            [
                flat['layers.0.linear_attn.in_proj_qkv.kernel'],
                flat['layers.0.linear_attn.in_proj_z.kernel'],
            ],
            axis=-1,
        ),
    )
    np.testing.assert_array_equal(
        out[f'{p}.layers.0.linear_attn.in_proj_ba.weight'],
        np.concatenate(
            [
                flat['layers.0.linear_attn.in_proj_b.kernel'],
                flat['layers.0.linear_attn.in_proj_a.kernel'],
            ],
            axis=-1,
        ),
    )
    np.testing.assert_array_equal(
        out[f'{p}.layers.0.linear_attn.conv1d.weight'],
        flat['layers.0.linear_attn.conv1d.kernel'].transpose(2, 1, 0),
    )
    np.testing.assert_array_equal(
        out[f'{p}.layers.0.mlp.gate_up_proj.weight'],
        np.concatenate(
            [
                flat['layers.0.mlp.gate_proj.kernel'],
                flat['layers.0.mlp.up_proj.kernel'],
            ],
            axis=-1,
        ),
    )
    # Tied embeddings: lm_head mirrors the embedding.
    np.testing.assert_array_equal(
        out['vllm_model.lm_head.weight'], flat['embedder.input_embedding']
    )

  def test_rejects_tensor_parallel(self):
    cfg = _tiny_config()
    state = nnx.state(model_lib.Qwen3p5(cfg, rngs=nnx.Rngs(0)), nnx.Param)
    with self.assertRaises(NotImplementedError):
      mapping_vllm_jax.preprocess_src_state(state, tp_size=2)


if __name__ == '__main__':
  jax.config.update('jax_platforms', 'cpu')
  absltest.main()
