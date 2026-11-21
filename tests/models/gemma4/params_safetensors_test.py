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

"""Tests for Gemma 4 safetensors parameter loading helpers."""

from absl.testing import absltest
import jax.numpy as jnp
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors
from tunix.utils import torch_utils


class ParamsSafetensorsTest(absltest.TestCase):

  def test_gemma4_12b_key_mapping_uses_local_and_global_shapes(self):
    config = model_lib.ModelConfig.gemma4_12b()
    config.num_layers = 6
    config.embed_dim = 16
    config.num_heads = 2
    config.head_dim = 4
    config.num_kv_heads = 2
    config.num_global_kv_heads = 1
    config.global_key_size = 8

    key_mapping = params_safetensors._get_key_and_transform_mapping(config)

    local_key, local_transform = torch_utils.torch_key_to_jax_key(
        key_mapping, "model.language_model.layers.0.self_attn.k_proj.weight"
    )
    self.assertEqual(local_key, "tmp.layers.0.attn.k")
    self.assertEqual(local_transform[1], (16, 2, 4))

    global_key, global_transform = torch_utils.torch_key_to_jax_key(
        key_mapping, "model.language_model.layers.5.self_attn.k_proj.weight"
    )
    self.assertEqual(global_key, "tmp.layers.5.attn.k")
    self.assertEqual(global_transform[1], (16, 1, 8))

  def test_gemma4_12b_preprocess_handles_local_and_global_attention(self):
    config = model_lib.ModelConfig.gemma4_12b()
    config.num_layers = 6
    config.embed_dim = 16
    config.num_heads = 2
    config.head_dim = 4
    config.num_kv_heads = 2
    config.num_global_kv_heads = 1
    config.global_key_size = 8

    tensors = {
        "tmp.layers.0.attn.q": jnp.zeros((16, 2, 4)),
        "tmp.layers.0.attn.k": jnp.zeros((16, 2, 4)),
        "tmp.layers.0.attn.v": jnp.ones((16, 2, 4)),
        "tmp.layers.5.attn.q": jnp.zeros((16, 2, 8)),
        "tmp.layers.5.attn.k": jnp.ones((16, 1, 8)),
    }

    out = params_safetensors._make_preprocess_fn(config)(tensors)

    self.assertEqual(out["layers.0.attn.q_einsum.w"].shape, (2, 16, 4))
    self.assertEqual(out["layers.0.attn.kv_einsum.w"].shape, (2, 2, 16, 4))
    self.assertEqual(out["layers.5.attn.q_einsum.w"].shape, (2, 16, 8))
    self.assertEqual(out["layers.5.attn.k_einsum.w"].shape, (1, 16, 8))
    self.assertNotIn("layers.5.attn.kv_einsum.w", out)


if __name__ == "__main__":
  absltest.main()
