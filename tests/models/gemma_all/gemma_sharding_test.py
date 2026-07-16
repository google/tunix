# Copyright 2025 Google LLC
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

"""Divisibility guards for Gemma's default ShardingConfig.

A sharding places a mesh axis ('fsdp'/'tp') on a tensor axis; JAX requires the
tensor axis size to be divisible by that mesh axis size, otherwise it raises
`IndivisibleError` at trace time. gemma-2-2b has small structural axes
(num_kv_heads=4, num_heads=8, the k/v-pair axis C=2) that CANNOT absorb a large
fsdp; fsdp must land only on large axes (embed_dim=2304, head_dim=256,
hidden_dim=9216, vocab). These tests pin the sharding so a future edit that puts
fsdp/tp on an indivisible axis fails here instead of on TPU.
"""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.models.gemma import model as gemma_model


def _weight_shapes(cfg):
  """Maps each sharded ShardingConfig field to its weight-tensor shape.

  Shapes are read straight from the module (Attention einsums / MLP Linears /
  Embedder) for a `num_heads != num_kv_heads` (GQA) model like gemma-2-2b:
    q_einsum   (num_heads, features, head_dim)         -> q_weight_ndh
    kv_einsum  (2, num_kv_heads, features, head_dim)   -> kv_weight_cndh
    attn_vec   (num_heads, head_dim, features)         -> o_weight_nhd
    gate/up    Linear(embed, hidden) => (embed, hidden)-> ffw_weight_df
    down       Linear(hidden, embed) => (hidden, embed)-> ffw_weight_fd
    embedding  (vocab, embed)                          -> emb_vd
  """
  v, d, h = cfg.num_embed, cfg.embed_dim, cfg.head_dim
  n, k, f = cfg.num_heads, cfg.num_kv_heads, cfg.hidden_dim
  return {
      'emb_vd': (v, d),
      'q_weight_ndh': (n, d, h),
      'kv_weight_cndh': (2, k, d, h),
      'o_weight_nhd': (n, h, d),
      'ffw_weight_df': (d, f),
      'ffw_weight_fd': (f, d),
  }


def _indivisible(sharding, shapes, mesh):
  """Returns [(field, axis, axis_size, mesh_axis, mesh_size)] for bad shards."""
  bad = []
  for field, shape in shapes.items():
    spec = getattr(sharding, field)
    for axis, mesh_axis in enumerate(spec):
      if mesh_axis in mesh and shape[axis] % mesh[mesh_axis] != 0:
        bad.append((field, axis, shape[axis], mesh_axis, mesh[mesh_axis]))
  return bad


class GemmaShardingDivisibilityTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = gemma_model.ModelConfig.gemma2_2b()
    self.sharding = gemma_model.ShardingConfig.get_default_sharding()
    self.shapes = _weight_shapes(self.cfg)

  @parameterized.named_parameters(
      ('mesh_16x4', {'fsdp': 16, 'tp': 4}),   # 64 chips, recommended
      ('mesh_8x4', {'fsdp': 8, 'tp': 4}),     # 32 chips
      ('mesh_4x2', {'fsdp': 4, 'tp': 2}),     # 8 chips
      ('mesh_2x4', {'fsdp': 2, 'tp': 4}),     # config default
  )
  def test_default_sharding_is_divisible(self, mesh):
    bad = _indivisible(self.sharding, self.shapes, mesh)
    self.assertEqual(
        bad, [], f'indivisible shardings on mesh {mesh}: {bad}'
    )

  def test_kv_tp_shards_heads_not_kv_pair(self):
    # Regression guard for the kv_weight churn: tp must shard num_kv_heads
    # (axis 1, K=4), NOT the k/v-pair axis (axis 0, C=2). Putting tp on C=2
    # breaks tp=4 (2 % 4 != 0) on a (fsdp=16, tp=4) mesh.
    spec = self.sharding.kv_weight_cndh
    self.assertIsNone(spec[0], 'kv axis 0 (C=2, k/v pair) must be replicated')
    self.assertEqual(spec[1], 'tp', 'kv axis 1 (num_kv_heads) must be tp-sharded')
    self.assertEqual(spec[2], 'fsdp', 'kv axis 2 (embed_dim) must be fsdp-sharded')

  def test_fsdp_never_lands_on_small_head_axis(self):
    # fsdp on 64 chips is 8/16/32; the head-count axes (num_heads=8,
    # num_kv_heads=4, k/v-pair=2) cannot absorb it. Assert fsdp only ever lands
    # on a "large" axis for every weight.
    small = {self.cfg.num_heads, self.cfg.num_kv_heads, 2}
    for field, shape in self.shapes.items():
      spec = getattr(self.sharding, field)
      for axis, mesh_axis in enumerate(spec):
        if mesh_axis == 'fsdp':
          self.assertNotIn(
              shape[axis],
              small,
              f'{field} places fsdp on a small axis (size {shape[axis]})',
          )


if __name__ == '__main__':
  absltest.main()
