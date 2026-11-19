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

"""Tests for router replay and router freezing in the Gemma 4 MoE block."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma4 import config as config_lib
from tunix.models.gemma4 import moe

_EMBED_DIM = 16
_NUM_EXPERTS = 4
_TOP_K = 2


def _module(**overrides) -> moe.MoERagged:
  cfg = dataclasses.replace(
      config_lib.ModelConfig.gemma4_26b_a4b(),
      embed_dim=_EMBED_DIM,
      num_experts=_NUM_EXPERTS,
      num_experts_per_tok=_TOP_K,
      expert_dim=8,
      **overrides,
  )
  module = moe.MoERagged(cfg, rngs=nnx.Rngs(0))
  leaves, treedef = jax.tree.flatten(nnx.state(module, nnx.Param))
  keys = jax.random.split(jax.random.key(1), len(leaves))
  rand = [
      0.5 * jax.random.normal(k, x.shape, x.dtype)
      for k, x in zip(keys, leaves)
  ]
  nnx.update(module, jax.tree.unflatten(treedef, rand))
  return module


class MoERouterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('unset', [-1, -1]),
      ('partial_unset', [3, -1]),
      ('out_of_range', [_NUM_EXPERTS, 2]),
      ('negative', [-7, 2]),
  )
  def test_router_replays_valid_rows_and_falls_back_on_invalid(self, bad_row):
    module = _module()
    logits = jnp.array([[[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]]])
    own_weights, own_choices = module._router(logits)
    weights, choices = module._router(
        logits, forced_routed_experts=jnp.array([[bad_row, [1, 0]]])
    )

    np.testing.assert_array_equal(choices[:, 0], own_choices[:, 0])
    np.testing.assert_array_equal(weights[:, 0], own_weights[:, 0])
    np.testing.assert_array_equal(choices[:, 1], [[1, 0]])
    probs = jax.nn.softmax(logits[0, 1])
    expected = [
        probs[1] / (probs[1] + probs[0]),
        probs[0] / (probs[1] + probs[0]),
    ]
    np.testing.assert_allclose(
        jnp.take_along_axis(weights[0, 1], choices[0, 1], axis=-1),
        expected,
        rtol=1e-6,
    )

  def test_freeze_router_stops_only_router_param_grads(self):
    with self.assertRaisesRegex(ValueError, 'freeze_router'):
      dataclasses.replace(
          config_lib.ModelConfig.gemma4_e2b(), freeze_router=True
      )

    x = jax.random.normal(jax.random.key(2), (2, 4, _EMBED_DIM))
    r_in = jax.random.normal(jax.random.key(3), (2, 4, _EMBED_DIM))
    forced = jnp.broadcast_to(jnp.array([2, 3], jnp.int32), (2, 4, _TOP_K))
    forced = forced.at[:, -1].set(-1)

    def loss_fn(m, router_input):
      out = m(x, router_input=router_input, forced_routed_experts=forced)
      return jnp.sum(out**2)

    g_train, in_train = nnx.grad(loss_fn, argnums=(0, 1))(_module(), r_in)
    g_froz, in_froz = nnx.grad(loss_fn, argnums=(0, 1))(
        _module(freeze_router=True), r_in
    )
    flat_train = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree_util.tree_flatten_with_path(g_train)[0]
    }
    flat_froz = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree_util.tree_flatten_with_path(g_froz)[0]
    }
    for path, grad in flat_train.items():
      if 'router_logits' in path or 'router_scale' in path:
        self.assertTrue(np.any(grad != 0), path)
        np.testing.assert_array_equal(flat_froz[path], 0, err_msg=path)
      else:
        np.testing.assert_allclose(flat_froz[path], grad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(in_froz, in_train, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
