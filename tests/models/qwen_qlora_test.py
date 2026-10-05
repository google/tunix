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

"""QLoRA (nf4 weights) smoke tests for the Qwen2 and Qwen3 models."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import qwix
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen3 import model as qwen3_lib


def _tiny_model(family: str) -> nnx.Module:
  lib, config, cls = {
      'qwen2': (qwen2_lib, qwen2_lib.ModelConfig.qwen2p5_0p5b(), 'Qwen2'),
      'qwen3': (qwen3_lib, qwen3_lib.ModelConfig.qwen3_0p6b(), 'Qwen3'),
  }[family]
  config = dataclasses.replace(
      config,
      num_layers=1,
      vocab_size=128,
      embed_dim=64,
      hidden_dim=128,
      num_heads=2,
      head_dim=32,
      num_kv_heads=1,
      dtype=jnp.bfloat16,
  )
  return getattr(lib, cls)(config, rngs=nnx.Rngs(0))


class QwenQLoraTest(parameterized.TestCase):

  @parameterized.parameters('qwen2', 'qwen3')
  def test_nf4_lora_train_step(self, family: str):
    model = _tiny_model(family)
    provider = qwix.LoraProvider(
        module_path='.*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj',
        rank=4,
        alpha=2.0,
        weight_qtype='nf4',
        tile_size=32,
    )
    model_input = model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        model, provider, rngs=nnx.Rngs(params=0), **model_input
    )

    def loss_fn(m):
      logits, _ = m(**model_input)
      return jnp.mean(logits.astype(jnp.float32))

    loss, grads = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam)
    )(lora_model)

    self.assertTrue(jnp.isfinite(loss))
    grad_leaves = jax.tree.leaves(grads)
    self.assertNotEmpty(grad_leaves)
    self.assertTrue(all(jnp.all(jnp.isfinite(g)) for g in grad_leaves))


if __name__ == '__main__':
  absltest.main()
