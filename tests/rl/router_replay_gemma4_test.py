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

"""Tests for MoE router replay through `compute_per_token_logps` on Gemma 4."""

import dataclasses

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma4 import model as model_lib
from tunix.rl import common

_MOE_SHAPE = (2, 8, 2, 2)  # (batch, seq, layers, top_k)


def _tiny_moe_model() -> model_lib.Gemma4:
  cfg = dataclasses.replace(
      model_lib.ModelConfig.gemma4_26b_a4b(),
      num_layers=2,
      num_embed=128,
      embed_dim=64,
      hidden_dim=128,
      num_heads=2,
      head_dim=32,
      num_kv_heads=1,
      num_global_kv_heads=1,
      global_key_size=32,
      sliding_window_size=4,
      num_experts=4,
      num_experts_per_tok=2,
      expert_dim=32,
      use_flash_attention=False,
      attention_pattern=(
          model_lib.AttentionType.LOCAL_SLIDING,
          model_lib.AttentionType.GLOBAL,
      ),
  )
  model = model_lib.Gemma4(cfg, rngs=nnx.Rngs(0))
  leaves, treedef = jax.tree.flatten(nnx.state(model, nnx.Param))
  keys = jax.random.split(jax.random.key(1), len(leaves))
  rand = [
      0.1 * jax.random.normal(k, x.shape, x.dtype)
      for k, x in zip(keys, leaves)
  ]
  nnx.update(model, jax.tree.unflatten(treedef, rand))
  return model


class ReplayThroughGemma4Test(absltest.TestCase):

  def test_compute_per_token_logps_replays_routing(self):
    model = _tiny_moe_model()
    self.assertTrue(common.model_call_contains(model, "forced_routed_experts"))

    graphdef, state = nnx.split(model)
    prompt = jnp.array([[0, 5, 6, 7], [8, 9, 10, 11]], dtype=jnp.int32)
    completion = jnp.array([[12, 13, 1, 0], [14, 15, 16, 17]], jnp.int32)
    pad = (
        jnp.concatenate([prompt, completion], axis=1) == 0
    )[:, :, None, None]

    def logps(routed):
      return common.compute_per_token_logps(
          graphdef,
          state,
          prompt_tokens=prompt,
          completion_tokens=completion,
          pad_id=0,
          eos_id=1,
          routed_experts=routed,
      )

    r_a = jnp.where(
        pad,
        -1,
        jnp.broadcast_to(jnp.array([[0, 1], [2, 3]], jnp.int16), _MOE_SHAPE),
    )
    r_b = r_a[:, :, ::-1, :]
    logps_none = logps(None)
    logps_a = logps(r_a)
    logps_b = logps(r_b)
    self.assertFalse(np.isnan(logps_none).any())
    self.assertFalse(np.allclose(logps_a, logps_b))
    np.testing.assert_allclose(
        logps((r_a + 1).astype(jnp.uint8)), logps_a, rtol=1e-6, atol=1e-6
    )


if __name__ == "__main__":
  absltest.main()
