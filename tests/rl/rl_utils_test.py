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

import os
from absl.testing import absltest
import chex
import flax
from flax import nnx
import jax
from jax import sharding
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import utils
from tunix.tests import test_common as tc

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_cpus = 4
    chex.set_n_cpu_devices(self.num_cpus)
    self.device_count = jax.device_count()

  def test_get_pytree_mesh_info(self):
    mesh1 = sharding.Mesh(
        np.array(jax.devices()[: self.device_count // 2]).reshape(
            1, self.device_count // 2
        ),
        ('fsdp', 'tp'),
    )
    model1 = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
            rngs=nnx.Rngs(0),
        ),
        mesh=mesh1,
    )
    self.assertEqual(utils.get_pytree_mesh_info(nnx.state(model1)), mesh1)

    mesh2 = sharding.Mesh(
        np.array(jax.devices()[self.device_count // 2 :]).reshape(
            1, self.device_count // 2
        ),
        ('fsdp', 'tp'),
    )
    model2 = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
            rngs=nnx.Rngs(0),
        ),
        mesh=mesh2,
    )
    self.assertEqual(utils.get_pytree_mesh_info(nnx.state(model2)), mesh2)

    self.assertNotEqual(mesh1, mesh2)

    model3 = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
            rngs=nnx.Rngs(0),
        ),
    )
    self.assertIsNone(utils.get_pytree_mesh_info(nnx.state(model3)))

  def test_is_sharing_weights(self):
    m1 = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    m2 = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    m3 = nnx.clone(m1)
    self.assertIsNot(nnx.state(m1), nnx.state(m2))
    self.assertIsNot(nnx.state(m1), nnx.state(m3))
    self.assertIsNot(nnx.state(m2), nnx.state(m3))
    self.assertFalse(utils.is_sharing_weights(m1, m2))
    self.assertFalse(utils.is_sharing_weights(m2, m3))
    self.assertTrue(utils.is_sharing_weights(m1, m3))

  def test_chunk_slices_by_size(self):
    x = [0, 1, 2, 3, 4]
    y = [x[s] for s in utils.chunk_slices_by_size(stop=len(x), step=2)]
    self.assertEqual(y, [[0, 1], [2, 3], [4]])

  def test_get_batch_slice(self):
    x = {
        'a': np.array([[1], [2], [3], [4], [5], [6]]),
        'b': {'c': np.array([[7], [8], [9], [10], [11], [12]])},
    }
    y = [
        utils.get_batch_slice(x, s)
        for s in utils.chunk_slices_by_size(stop=6, step=2)
    ]
    expected = [
        {'a': np.array([[1], [2]]), 'b': {'c': np.array([[7], [8]])}},
        {'a': np.array([[3], [4]]), 'b': {'c': np.array([[9], [10]])}},
        {'a': np.array([[5], [6]]), 'b': {'c': np.array([[11], [12]])}},
    ]
    jax.tree_util.tree_map(np.testing.assert_array_equal, expected, y)

  def test_merge_micro_batches(self):
    batches = [
        {
            'a': [1, 2],
            'b': {'c': np.array([3, 4]), 'd': np.array([5])},
            'e': np.array([6, 7]),
        },
        {
            'a': [10, 11],
            'b': {'c': np.array([12, 13]), 'd': np.array([14, 15])},
            'e': np.array([16]),
        },
    ]
    merged = utils.merge_micro_batches(batches)
    self.assertEqual(merged['a'], [1, 2, 10, 11])
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        merged['b'],
        {'c': np.array([3, 4, 12, 13]), 'd': np.array([5, 14, 15])},
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, merged['e'], np.array([6, 7, 16])
    )

  def test_create_critic_model(self):
    actor_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    critic_model = utils.create_critic_model(actor_model)

    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    positions = jnp.arange(x.shape[1])
    attn_mask = common.make_causal_attn_mask(jnp.ones_like(x))
    out, _ = critic_model(x, positions, None, attn_mask)
    self.assertEqual(out.shape, (2, 3, 1))

  def test_put_params_on_memory_kind(self):
    # Test valid memory kind
    params = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
    updated_params = utils.put_params_on_memory_kind(params, 'pinned_host')
    self.assertEqual(
        jax.tree.map(lambda x: x.sharding.memory_kind, updated_params),
        {'a': 'pinned_host', 'b': 'pinned_host'},
    )

    # Test already on requested memory kind
    updated_params_2 = utils.put_params_on_memory_kind(
        updated_params, 'pinned_host'
    )
    self.assertIs(updated_params, updated_params_2)

    # Test empty tree
    empty_params = {}
    updated_empty = utils.put_params_on_memory_kind(empty_params, 'device')
    self.assertEqual(updated_empty, {})

    # Test invalid memory kind
    with self.assertRaisesRegex(ValueError, 'memory_kind must be one of'):
      utils.put_params_on_memory_kind(params, 'invalid_kind')

  def _create_mock_train_example(
      self,
      prompt_len: int,
      completion_len: int,
      pad_len: int = 0,
      cls=common.TrainExample,
      **kwargs
  ) -> common.TrainExample:
    p_ids = jnp.concatenate(
        [
            jnp.zeros((1, pad_len), dtype=jnp.int32),
            jnp.ones((1, prompt_len), dtype=jnp.int32),
        ],
        axis=1,
    )
    p_mask = jnp.concatenate(
        [
            jnp.zeros((1, pad_len), dtype=jnp.int32),
            jnp.ones((1, prompt_len), dtype=jnp.int32),
        ],
        axis=1,
    )

    c_ids = jnp.concatenate(
        [
            jnp.ones((1, completion_len), dtype=jnp.int32) * 2,
            jnp.zeros((1, pad_len), dtype=jnp.int32),
        ],
        axis=1,
    )
    c_mask = jnp.concatenate(
        [
            jnp.ones((1, completion_len), dtype=jnp.int32),
            jnp.zeros((1, pad_len), dtype=jnp.int32),
        ],
        axis=1,
    )

    base_kwargs = dict(
        prompt_ids=p_ids,
        prompt_mask=p_mask,
        completion_ids=c_ids,
        completion_mask=c_mask,
        advantages=jnp.array([1.5], dtype=jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=None,
        segment_positions=None,
    )
    base_kwargs.update(kwargs)
    return cls(**base_kwargs)

  def test_unpad_train_example(self):
    example = self._create_mock_train_example(2, 3, pad_len=2)
    unpadded = utils.unpad_train_example(example)
    self.assertLen(unpadded, 1)
    [item] = unpadded
    self.assertEqual(item['prompt_ids'].shape, (2,))
    self.assertEqual(item['completion_ids'].shape, (3,))
    self.assertFalse(item['adv_is_per_token'])

  def test_pack_sequences_skips_large(self):
    # A sequence larger than budget should be skipped.
    example1 = self._create_mock_train_example(5, 6)  # size 11
    example2 = self._create_mock_train_example(2, 3)  # size 5
    packed_iterator = utils.pack_sequences(
        iter([[example1, example2]]), max_token_budget=10
    )
    packed_batches = list(packed_iterator)
    self.assertLen(packed_batches, 1)
    self.assertEqual(packed_batches[0][0].segment_ids.shape, (1, 10))
    # Check that only example2 was packed (length 5)
    self.assertEqual(np.max(packed_batches[0][0].segment_ids), 1)

  def test_pack_sequences_with_dummy_padding(self):

    @flax.struct.dataclass(frozen=True)
    class PPOTrainExample(common.TrainExample):
      returns: jax.Array | None = None
      old_values: jax.Array | None = None
      policy_version: jax.Array | None = None

    example = self._create_mock_train_example(
        2,
        3,
        cls=PPOTrainExample,
        ref_per_token_logps=jnp.ones((1, 3)),
        returns=jnp.ones((1, 3)),
        old_values=jnp.ones((1, 3)),
        policy_version=jnp.array([1]),
    )

    # num_packs=2 with 1 example generates 1 dummy pack.
    packed_iterator = utils.pack_sequences(
        iter([[example]]), max_token_budget=10, num_packs=2
    )
    packed_batches = list(packed_iterator)
    [[pack]] = packed_batches

    with self.subTest(name='pack_counts'):
      self.assertLen(packed_batches, 1)
      self.assertLen(packed_batches[0], 1)

    with self.subTest(name='batch_size'):
      self.assertEqual(pack.prompt_ids.shape, (2, 0))
      self.assertEqual(pack.completion_ids.shape, (2, 10))

    with self.subTest(name='valid_pack_check'):
      # The arrays are shifted by prompt_len (=2).
      self.assertEqual(pack.returns[0, 2], 1.0)
      self.assertEqual(pack.old_values[0, 2], 1.0)
      self.assertEqual(pack.ref_per_token_logps[0, 2], 1.0)
      self.assertEqual(pack.policy_version[0], 1)

    with self.subTest(name='dummy_pack_check'):
      # Dummy pack check (index 1) - per-token features are zero (mask=0
      # ensures it contributes nothing to loss). policy_version is inherited
      # from the first real pack so off-policy filtering does not see a
      # sentinel for padding rows.
      self.assertEqual(pack.returns[1, 0], 0.0)
      self.assertEqual(pack.old_values[1, 0], 0.0)
      self.assertEqual(pack.ref_per_token_logps[1, 0], 0.0)
      self.assertEqual(pack.policy_version[1], 1)

    with self.subTest(name='policy_version_shape_is_per_row'):
      # policy_version is a per-row array of shape (num_packs,) where row i
      # carries the version of pack i. With num_packs=2 we expect length-2.
      self.assertEqual(pack.policy_version.shape, (2,))

  def test_pack_sequences_version_aware_flush(self):
    """Items with different policy_versions must land in separate packs."""

    @flax.struct.dataclass(frozen=True)
    class VersionedExample(common.TrainExample):
      policy_version: jax.Array | None = None

    # Two examples with different policy_versions. Both fit in the budget so
    # without version-aware flushing they would share a pack.
    example_v3 = self._create_mock_train_example(
        2, 3, cls=VersionedExample, policy_version=jnp.array([3])
    )
    example_v7 = self._create_mock_train_example(
        2, 3, cls=VersionedExample, policy_version=jnp.array([7])
    )

    packed_iterator = utils.pack_sequences(
        iter([[example_v3, example_v7]]),
        max_token_budget=20,  # plenty of room for both
        num_packs=2,
    )
    packed_batches = list(packed_iterator)
    [[pack]] = packed_batches

    with self.subTest(name='packs_are_separated'):
      # Each example becomes its own pack -> shape (2, ...) after num_packs=2.
      self.assertEqual(pack.completion_ids.shape[0], 2)

    with self.subTest(name='per_row_policy_version'):
      self.assertEqual(pack.policy_version.shape, (2,))
      self.assertEqual(int(pack.policy_version[0]), 3)
      self.assertEqual(int(pack.policy_version[1]), 7)

  def test_pack_sequences(self):
    # 3 sequences with lengths (P+C): (2+3=5), (1+2=3), (3+4=7)
    example1 = self._create_mock_train_example(2, 3)
    example2 = self._create_mock_train_example(1, 2)
    example3 = self._create_mock_train_example(3, 4)
    item_iterator = iter([[example1, example2, example3]])

    # Budget of 10. We expect item 1 (5) and item 2 (3) to fit in the first pack (8).
    # Item 3 (7) will go to the second pack (because 8+7 > 10).
    packed_iterator = utils.pack_sequences(
        item_iterator, max_token_budget=10, pad_id=0
    )

    packed_batches = list(packed_iterator)
    pack1 = packed_batches[0][0]
    # Segment IDs should be (5 ones, 3 twos, 2 padding zeros)
    expected_segments_1 = jnp.array(
        [[1] * 5 + [2] * 3 + [0] * 2], dtype=jnp.int32
    )
    # Positions should be (0..4, 0..2, 0, 0)
    expected_positions_1 = jnp.array(
        [[0, 1, 2, 3, 4, 0, 1, 2, 0, 0]], dtype=jnp.int32
    )
    # Completion mask should be 0 for prompts, 1 for completions, 0 for padding
    # Seq 1: 2 prompts (0), 3 completions (1)
    # Seq 2: 1 prompt (0), 2 completions (1)
    expected_mask_1 = jnp.array(
        [[0, 0, 1, 1, 1, 0, 1, 1, 0, 0]], dtype=jnp.int32
    )

    pack2 = packed_batches[1][0]
    expected_segments_2 = jnp.array([[1] * 7 + [0] * 3], dtype=jnp.int32)
    # Positions should be (0..6, 0, 0, 0)
    expected_positions_2 = jnp.array(
        [[0, 1, 2, 3, 4, 5, 6, 0, 0, 0]], dtype=jnp.int32
    )
    # Completion mask: 3 prompts (0), 4 completions (1), 3 padding (0)
    expected_mask_2 = jnp.array(
        [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32
    )

    with self.subTest(name='pack_counts'):
      self.assertLen(packed_batches, 2)

    with self.subTest(name='pack1_contents'):
      self.assertEqual(pack1.prompt_ids.shape, (1, 0))  # prompt_ids is empty
      self.assertEqual(pack1.completion_ids.shape, (1, 10))  # filled + padded
      self.assertEqual(pack1.segment_ids.shape, (1, 10))
      self.assertEqual(pack1.segment_positions.shape, (1, 10))
      np.testing.assert_array_equal(pack1.segment_ids, expected_segments_1)
      np.testing.assert_array_equal(
          pack1.segment_positions, expected_positions_1
      )
      np.testing.assert_array_equal(pack1.completion_mask, expected_mask_1)

    with self.subTest(name='pack2_contents'):
      self.assertEqual(pack2.prompt_ids.shape, (1, 0))
      self.assertEqual(pack2.completion_ids.shape, (1, 10))
      self.assertEqual(pack2.segment_ids.shape, (1, 10))
      self.assertEqual(pack2.segment_positions.shape, (1, 10))
      np.testing.assert_array_equal(pack2.segment_ids, expected_segments_2)
      np.testing.assert_array_equal(
          pack2.segment_positions, expected_positions_2
      )
      np.testing.assert_array_equal(pack2.completion_mask, expected_mask_2)

  def _mock_example(self, prompt_len: int, completion_len: int):
    return common.TrainExample(
        prompt_ids=jnp.ones((1, prompt_len), dtype=jnp.int32),
        prompt_mask=jnp.ones((1, prompt_len), dtype=jnp.int32),
        completion_ids=jnp.ones((1, completion_len), dtype=jnp.int32) * 2,
        completion_mask=jnp.ones((1, completion_len), dtype=jnp.int32),
        advantages=jnp.array([1.5], dtype=jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=None,
        segment_positions=None,
    )

  def test_pack_sequences_raises_on_empty_update_boundary(self):
    # An update boundary (is_update=True) reached with an empty buffer would
    # yield nothing and silently drop a gradient-accumulation update; it must
    # raise instead. An empty input batch at a `target_items_per_update`
    # boundary triggers exactly this.
    packed = utils.pack_sequences(
        iter([[]]),  # one empty item-list -> nothing buffered at the boundary
        max_token_budget=10,
        target_items_per_update=1,  # every item-list is an update boundary
    )
    with self.assertRaisesRegex(ValueError, 'update boundary'):
      list(packed)

  def test_pack_sequences_raises_on_mid_mini_batch_end(self):
    # Stream ends mid-mini-batch (1 item-list but target_items_per_update=2) ->
    # trailing flush would update on a partial mini-batch -> raise.
    packed = utils.pack_sequences(
        iter([[self._mock_example(1, 2)]]),
        max_token_budget=10,
        target_items_per_update=2,
    )
    with self.assertRaisesRegex(ValueError, 'mid-mini-batch'):
      list(packed)

  def test_pack_sequences_marks_is_update_step_at_boundary(self):
    # With target_items_per_update=1 each non-empty flush is an update boundary,
    # so every packed example is marked is_update_step=True (and no raise).
    packed = list(
        utils.pack_sequences(
            iter([[self._mock_example(1, 2)], [self._mock_example(2, 3)]]),
            max_token_budget=10,
            target_items_per_update=1,
        )
    )
    self.assertLen(packed, 2)
    for batch in packed:
      self.assertTrue(bool(np.asarray(batch[0].is_update_step)))


if __name__ == '__main__':
  absltest.main()
