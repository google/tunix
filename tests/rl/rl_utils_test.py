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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
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

  def test_pack_sequences_raises_on_oversized_sequence(self):
    # A single sequence longer than the budget cannot be packed; it must raise
    # (rather than silently drop training data).
    example1 = self._create_mock_train_example(5, 6)  # size 11 > budget 10
    example2 = self._create_mock_train_example(2, 3)  # size 5
    packed = utils.pack_sequences(
        iter([[example1, example2]]), max_token_budget=10
    )
    with self.assertRaisesRegex(ValueError, 'exceeding max_token_budget'):
      list(packed)

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

    # pack_size=2 with 1 example generates 1 dummy pack.
    packed_iterator = utils.pack_sequences(
        iter([[example]]), max_token_budget=10, pack_size=2
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

  def test_pack_sequences(self):
    # 3 sequences with lengths (P+C): item1 (2+3=5), item2 (1+2=3),
    # item3 (3+4=7). Budget 10, pack_size=1 -> each bin is its own chunk.
    # FFD sorts descending [7, 5, 3] into P = ceil(15/10) = 2 bins:
    #   bin0 (chunk0): item3 (7) then item2 (3) -> exactly fills 10
    #   bin1 (chunk1): item1 (5), padded; last chunk -> is_update_step=True
    example1 = self._create_mock_train_example(2, 3)
    example2 = self._create_mock_train_example(1, 2)
    example3 = self._create_mock_train_example(3, 4)
    packed_batches = list(
        utils.pack_sequences(
            iter([[example1, example2, example3]]),
            max_token_budget=10,
            pad_id=0,
        )
    )
    self.assertLen(packed_batches, 2)

    # chunk 0 = [item3 (seg 1, 3 prompt + 4 completion), item2 (seg 2, 1+2)].
    pack1 = packed_batches[0][0]
    with self.subTest(name='chunk0_contents'):
      self.assertEqual(pack1.prompt_ids.shape, (1, 0))  # prompt_ids is empty
      self.assertEqual(pack1.completion_ids.shape, (1, 10))  # filled, no pad
      np.testing.assert_array_equal(
          pack1.segment_ids, jnp.array([[1] * 7 + [2] * 3], dtype=jnp.int32)
      )
      np.testing.assert_array_equal(
          pack1.segment_positions,
          jnp.array([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2]], dtype=jnp.int32),
      )
      np.testing.assert_array_equal(
          pack1.completion_mask,
          jnp.array([[0, 0, 0, 1, 1, 1, 1, 0, 1, 1]], dtype=jnp.int32),
      )
      self.assertFalse(bool(np.asarray(pack1.is_update_step)[0]))

    # chunk 1 = [item1 (seg 1, 2 prompt + 3 completion)], padded to budget.
    pack2 = packed_batches[1][0]
    with self.subTest(name='chunk1_contents'):
      np.testing.assert_array_equal(
          pack2.segment_ids, jnp.array([[1] * 5 + [0] * 5], dtype=jnp.int32)
      )
      np.testing.assert_array_equal(
          pack2.segment_positions,
          jnp.array([[0, 1, 2, 3, 4, 0, 0, 0, 0, 0]], dtype=jnp.int32),
      )
      np.testing.assert_array_equal(
          pack2.completion_mask,
          jnp.array([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=jnp.int32),
      )
      self.assertTrue(bool(np.asarray(pack2.is_update_step)[0]))

  def test_pack_sequences_sets_num_segments_to_budget_plus_one(self):
    # num_segments is the static (pytree_node=False) segment-bucket upper bound.
    # It must equal budget + 1 (a pack of `budget` tokens holds at most `budget`
    # unit-length segments) and be a plain Python int (a fixed value every step
    # so the segment-aware loss compiles once, no per-step recompilation).
    budget = 10
    example1 = self._create_mock_train_example(2, 3)
    example2 = self._create_mock_train_example(1, 2)
    packed_batches = list(
        utils.pack_sequences(
            iter([[example1, example2]]),
            max_token_budget=budget,
            pad_id=0,
        )
    )
    self.assertNotEmpty(packed_batches)
    for batch in packed_batches:
      pack = batch[0]
      self.assertEqual(pack.num_segments, budget + 1)
      # Static Python int (not a traced jax.Array) -> compiles once.
      self.assertIsInstance(pack.num_segments, int)

  def test_pack_sequences_num_segments_respects_override(self):
    # An explicit max_segments_per_packed_row shrinks the buckets:
    # num_segments = override + 1 (not budget + 1).
    packed = list(
        utils.pack_sequences(
            iter([[self._create_mock_train_example(2, 3)]]),
            max_token_budget=10,
            pad_id=0,
            max_segments_per_packed_row=8,
        )
    )
    self.assertNotEmpty(packed)
    for batch in packed:
      self.assertEqual(batch[0].num_segments, 8 + 1)

  def test_pack_sequences_caps_segments_per_row(self):
    # A small max_segments_per_packed_row is RESPECTED (segments capped per row),
    # not raised on: three short sequences that would otherwise share a row are
    # spread across rows so each holds <= max_segments, no sequence is dropped,
    # and no error is raised (the segment_sum overflow can't happen).
    max_seg = 1
    examples = [self._create_mock_train_example(1, 2) for _ in range(3)]
    packed = list(
        utils.pack_sequences(
            iter([examples]),
            max_token_budget=20,
            pad_id=0,
            max_segments_per_packed_row=max_seg,
        )
    )
    self.assertNotEmpty(packed)
    total_segments = 0
    for batch in packed:
      seg = np.asarray(batch[0].segment_ids)  # [pack_size, budget], ids 1..K, 0=pad
      for row in seg:
        n = int(row.max())  # segments run 1..K per row; 0 is padding
        self.assertLessEqual(n, max_seg)  # cap respected
        total_segments += n
    self.assertEqual(total_segments, 3)  # no sequence dropped

  def test_pack_sequences_packs_multiple_rows_per_chunk(self):
    # pack_size=2: one chunk is [2, budget] and FFD distributes sequences
    # ACROSS the two rows. budget=10, seqs A=6, B=5, C=3 (one mini-batch).
    # FFD desc [6,5,3]: 6->row0, 5->row0 full so row1, 3->row0 (6+3=9).
    #   row0 = [A (seg 1), C (seg 2)], row1 = [B (seg 1)].
    example_a = self._create_mock_train_example(3, 3)  # 6
    example_b = self._create_mock_train_example(2, 3)  # 5
    example_c = self._create_mock_train_example(1, 2)  # 3
    packed = list(
        utils.pack_sequences(
            iter([[example_a, example_b, example_c]]),
            max_token_budget=10,
            pack_size=2,
        )
    )
    self.assertLen(packed, 1)  # all three fit one [2, 10] chunk
    pack = packed[0][0]
    self.assertEqual(pack.completion_ids.shape, (2, 10))
    with self.subTest(name='row0_two_segments'):
      np.testing.assert_array_equal(
          pack.segment_ids[0],
          jnp.array([1] * 6 + [2] * 3 + [0], dtype=jnp.int32),  # A then C
      )
    with self.subTest(name='row1_one_segment'):
      np.testing.assert_array_equal(
          pack.segment_ids[1],
          jnp.array([1] * 5 + [0] * 5, dtype=jnp.int32),  # B, padded
      )
    self.assertTrue(bool(np.asarray(pack.is_update_step)[0]))

  def test_pack_sequences_concatenates_per_token_features_across_seqs(self):
    # Two real sequences in one packed row: per-token features (ref logps) and
    # the scalar advantage must land on each sequence's COMPLETION tokens only
    # (0 on prompt tokens and padding), in sequence order.
    @flax.struct.dataclass(frozen=True)
    class PPOTrainExample(common.TrainExample):
      policy_version: jax.Array | None = None

    # A: prompt 2 + completion 3 (ref=2.0); B: prompt 1 + completion 2 (ref=3.0).
    example_a = self._create_mock_train_example(
        2, 3, cls=PPOTrainExample, ref_per_token_logps=jnp.full((1, 3), 2.0)
    )
    example_b = self._create_mock_train_example(
        1, 2, cls=PPOTrainExample, ref_per_token_logps=jnp.full((1, 2), 3.0)
    )
    packed = list(
        utils.pack_sequences(
            iter([[example_a, example_b]]), max_token_budget=10, pack_size=1
        )
    )
    pack = packed[0][0]  # one [1, 10] row: A (5 tok) then B (3 tok), pad 2.
    with self.subTest(name='ref_logps_on_completion_only'):
      # A: prompt(2)=0, completion(3)=2.0; B: prompt(1)=0, completion(2)=3.0.
      np.testing.assert_allclose(
          pack.ref_per_token_logps[0],
          jnp.array([0, 0, 2, 2, 2, 0, 3, 3, 0, 0], dtype=jnp.float32),
      )
    with self.subTest(name='scalar_advantage_broadcast_over_completion'):
      # Default mock advantage 1.5, broadcast over completion tokens only.
      np.testing.assert_allclose(
          pack.advantages[0],
          jnp.array(
              [0, 0, 1.5, 1.5, 1.5, 0, 1.5, 1.5, 0, 0], dtype=jnp.float32
          ),
      )

  def test_pack_sequences_is_independent_of_producer_granularity(self):
    # The mini-batch boundary is defined in SEQUENCES (sequences_per_update),
    # so how the producer chops the stream is irrelevant: the same 2 sequences
    # fed as two single-example lists or one two-example list pack identically
    # into ONE chunk (both fit a budget-10 bin) with is_update_step=True.
    def run(stream):
      packed = list(
          utils.pack_sequences(
              iter(stream), max_token_budget=10, sequences_per_update=2
          )
      )
      self.assertLen(packed, 1)
      return packed[0][0]

    fine = run([
        [self._create_mock_train_example(2, 3)],
        [self._create_mock_train_example(1, 2)],
    ])
    coarse = run([[
        self._create_mock_train_example(2, 3),
        self._create_mock_train_example(1, 2),
    ]])
    for pack in (fine, coarse):
      # Two segments in one row: 5-token seq then 3-token seq.
      np.testing.assert_array_equal(
          pack.segment_ids, jnp.array([[1] * 5 + [2] * 3 + [0] * 2], jnp.int32)
      )
      self.assertTrue(bool(np.asarray(pack.is_update_step)[0]))
    np.testing.assert_array_equal(fine.completion_ids, coarse.completion_ids)

  def test_pack_sequences_streams_chunks_before_minibatch_end(self):
    # Streaming: a chunk is emitted as soon as the buffer holds a chunk's worth
    # of tokens (pack_size*budget = 10), before the mini-batch is complete, so
    # training can overlap rollout. budget=10, pack_size=1, N=4.
    #   micro-batch 1: [6,3,5] -> 14 >= 10 tokens buffered -> emit one chunk
    #     (FFD desc [6,5,3]: 6->bin, 5->6+5>10 leftover, 3->6+3=9 -> chunk [6,3],
    #      is_update=False), leftover [5] stays buffered.
    #   micro-batch 2 (final, N reached): [4] -> buffer [5,4] -> chunk [5,4],
    #     is_update=True.
    packed = list(
        utils.pack_sequences(
            iter([
                [
                    self._create_mock_train_example(3, 3),  # 6
                    self._create_mock_train_example(1, 2),  # 3
                    self._create_mock_train_example(2, 3),  # 5
                ],
                [self._create_mock_train_example(1, 3)],  # 4
            ]),
            max_token_budget=10,
            sequences_per_update=4,
        )
    )
    self.assertLen(packed, 2)
    with self.subTest(name='mid_chunk_not_update'):
      # First chunk holds the 6- and 3-token seqs (FFD), no is_update yet.
      np.testing.assert_array_equal(
          packed[0][0].segment_ids,
          jnp.array([[1] * 6 + [2] * 3 + [0]], jnp.int32),
      )
      self.assertFalse(bool(np.asarray(packed[0][0].is_update_step)[0]))
    with self.subTest(name='final_chunk_is_update'):
      # Second chunk holds the leftover 5-token seq + the 4-token seq.
      np.testing.assert_array_equal(
          packed[1][0].segment_ids,
          jnp.array([[1] * 5 + [2] * 4 + [0]], jnp.int32),
      )
      self.assertTrue(bool(np.asarray(packed[1][0].is_update_step)[0]))

  def test_pack_sequences_emptied_buffer_still_marks_last_chunk(self):
    # If streaming drains the buffer mid-mini-batch, the last chunk must still
    # come from the final flush with is_update=True (no dropped update). N=3.
    #   micro-batch 1: [5,5] -> 10 >= 10 -> emit chunk [5,5], buffer now empty,
    #     is_update=False.
    #   micro-batch 2 (final): [7] -> chunk [7], is_update=True.
    packed = list(
        utils.pack_sequences(
            iter([
                [
                    self._create_mock_train_example(2, 3),  # 5
                    self._create_mock_train_example(2, 3),  # 5
                ],
                [self._create_mock_train_example(3, 4)],  # 7
            ]),
            max_token_budget=10,
            sequences_per_update=3,
        )
    )
    self.assertLen(packed, 2)
    self.assertFalse(bool(np.asarray(packed[0][0].is_update_step)[0]))
    self.assertTrue(bool(np.asarray(packed[1][0].is_update_step)[0]))

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
    # An update boundary reached with no sequences would yield nothing and
    # silently drop a gradient update; it must raise instead. With
    # sequences_per_update=None each input list is its own mini-batch, so an
    # empty list is an empty update boundary.
    packed = utils.pack_sequences(iter([[]]), max_token_budget=10)
    with self.assertRaisesRegex(ValueError, 'update boundary'):
      list(packed)

  def test_pack_sequences_raises_on_mid_mini_batch_end(self):
    # Stream ends mid-mini-batch (1 sequence but sequences_per_update=2) ->
    # trailing flush would update on a partial mini-batch -> raise.
    packed = utils.pack_sequences(
        iter([[self._mock_example(1, 2)]]),
        max_token_budget=10,
        sequences_per_update=2,
    )
    with self.assertRaisesRegex(ValueError, 'mid-mini-batch'):
      list(packed)

  def test_pack_sequences_raises_on_misaligned_boundary(self):
    # A mini-batch boundary falling inside an input example (2 sequences arrive
    # at once but sequences_per_update=1) is a config mismatch -> raise.
    packed = utils.pack_sequences(
        iter([[self._mock_example(1, 2), self._mock_example(2, 3)]]),
        max_token_budget=10,
        sequences_per_update=1,
    )
    with self.assertRaisesRegex(ValueError, 'boundary falls inside'):
      list(packed)

  def test_pack_sequences_marks_is_update_step_at_boundary(self):
    # sequences_per_update=1: every sequence is its own mini-batch, so every
    # packed chunk is marked is_update_step=True.
    packed = list(
        utils.pack_sequences(
            iter([[self._mock_example(1, 2)], [self._mock_example(2, 3)]]),
            max_token_budget=10,
            sequences_per_update=1,
        )
    )
    self.assertLen(packed, 2)
    for batch in packed:
      self.assertTrue(bool(np.asarray(batch[0].is_update_step)))


class ComputePackSizeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('fsdp_and_tp', {'fsdp': 4, 'tp': 2}, 4),
      ('dp_and_tp', {'dp': 2, 'tp': 4}, 2),
      ('fsdp_and_dp', {'fsdp': 4, 'dp': 2}, 8),
  )
  def test_product_of_batch_sharding_axes(self, axis_sizes, expected):
    mesh = mock.Mock(shape=axis_sizes)
    self.assertEqual(utils.compute_pack_size(mesh), expected)

  def test_no_warning_when_batch_axis_present(self):
    mesh = mock.Mock(shape={'fsdp': 4, 'tp': 2})
    with self.assertNoLogs(level='WARNING'):
      utils.compute_pack_size(mesh)

  def test_warns_and_defaults_to_one_without_batch_axis(self):
    # No 'fsdp'/'dp' axis (tp-only): pack_size falls back to 1, with a warning.
    mesh = mock.Mock(shape={'tp': 4})
    with self.assertLogs(level='WARNING') as logs:
      self.assertEqual(utils.compute_pack_size(mesh), 1)
    self.assertIn("no 'fsdp'/'dp' axis", logs.output[0])
    self.assertIn("'tp': 4", logs.output[0])


if __name__ == '__main__':
  absltest.main()
