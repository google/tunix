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

"""Tests for the type-agnostic packing core logic."""

from absl.testing import absltest
import numpy as np
from tunix.rl import packing


def _item(prompt, completion, *, mask=None, adv=0.0, per_token=None):
  completion = np.asarray(completion, dtype=np.int32)
  return packing.PackItem(
      prompt_ids=np.asarray(prompt, dtype=np.int32),
      completion_ids=completion,
      completion_mask=(
          np.ones(completion.shape[0], dtype=np.float32)
          if mask is None
          else np.asarray(mask, dtype=np.float32)
      ),
      advantages=np.full(completion.shape[0], adv, dtype=np.float32),
      per_token=per_token or {},
  )


class PackItemInvariantTest(absltest.TestCase):

  def test_rejects_completion_mask_of_wrong_legnth(self):
    with self.assertRaisesRegex(ValueError, "completion_mask must have length"):
      packing.PackItem(
          prompt_ids=np.arange(4, dtype=np.int32),
          completion_ids=np.arange(6, dtype=np.int32),
          completion_mask=np.ones(4, dtype=np.float32),
          advantages=np.zeros(6, dtype=np.float32),
      )

  def test_rejects_advantages_of_wrong_length(self):
    with self.assertRaisesRegex(ValueError, "advantages must have length"):
      packing.PackItem(
          prompt_ids=np.arange(2, dtype=np.int32),
          completion_ids=np.arange(3, dtype=np.int32),
          completion_mask=np.ones(3, dtype=np.float32),
          advantages=np.zeros(5, dtype=np.float32),
      )

  def test_rejects_2d_array(self):
    with self.assertRaisesRegex(ValueError, "must be a 1D numpy array"):
      packing.PackItem(
          prompt_ids=np.zeros((1, 2), dtype=np.int32),
          completion_ids=np.arange(3, dtype=np.int32),
          completion_mask=np.ones(3, dtype=np.float32),
          advantages=np.zeros(3, dtype=np.float32),
      )

  def test_rejects_unknown_per_token_field(self):
    with self.assertRaisesRegex(ValueError, "Unknown per-token field"):
      _item([1, 2], [3, 4], per_token={"invalid_key": np.zeros(2, np.float32)})

  def test_rejects_non_1d_per_token(self):
    with self.assertRaisesRegex(ValueError, "1D numpy array"):
      _item([1, 2], [3, 4], per_token={"returns": np.zeros((2, 1), np.float32)})

  def test_rejects_per_token_of_wrong_length(self):
    with self.assertRaisesRegex(
        ValueError, "must be a 1D numpy array or shape"
    ):
      _item([1, 2], [3, 4], per_token={"returns": np.zeros(5, np.float32)})


class PackCarriedFieldsTest(absltest.TestCase):

  def test_all_or_nothing(self):
    self.assertEqual(packing.carried_per_token_fields([_item([1], [2])]), ())
    both = [
        _item([1], [2], per_token={"returns": np.zeros(1, np.float32)}),
        _item([3], [4], per_token={"returns": np.zeros(1, np.float32)}),
    ]
    self.assertEqual(
        packing.carried_per_token_fields(both),
        ("returns",),
    )

  def test_partial_population_errors(self):
    items = [
        _item([1], [2], per_token={"returns": np.zeros(1, np.float32)}),
        _item([3], [4]),
    ]
    with self.assertRaisesRegex(ValueError, "Some but not all"):
      packing.carried_per_token_fields(items)


class PackCoreTest(absltest.TestCase):

  def test_row_layout(self):
    items = [
        _item([1, 2], [3, 4, 5], adv=1.5),
        _item([6], [7, 8], adv=0.5),
    ]
    [[row]] = packing.pack_core(items, budget=10, pack_size=1)
    np.testing.assert_array_equal(row.ids, [1, 2, 3, 4, 5, 6, 7, 8, 0, 0])
    np.testing.assert_array_equal(
        row.segment_ids, [1, 1, 1, 1, 1, 2, 2, 2, 0, 0]
    )
    np.testing.assert_array_equal(
        row.segment_positions, [0, 1, 2, 3, 4, 0, 1, 2, 0, 0]
    )
    np.testing.assert_array_equal(
        row.prompt_mask, [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        row.completion_mask, [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        row.segment_ids > 0, [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    )
    np.testing.assert_allclose(
        row.advantages, [0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.5, 0.5, 0.0, 0.0]
    )
    self.assertEqual(row.num_real_segments, 2)

  def test_reserve_non_action_mask_zeros(self):
    items = [_item([1], [2, 3, 4], mask=[1, 0, 1], adv=2.0)]
    [[row]] = packing.pack_core(items, budget=6, pack_size=1)
    np.testing.assert_array_equal(row.ids, [1, 2, 3, 4, 0, 0])
    np.testing.assert_array_equal(row.prompt_mask, [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(row.completion_mask, [0, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(row.segment_ids, [1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(row.advantages, [0, 2, 2, 2, 0, 0])

  def test_packed_chunk_with_dummy_rows(self):
    items = [_item([1], [2])]
    [rows] = packing.pack_core(items, budget=4, pack_size=3)
    self.assertLen(rows, 3)
    self.assertEqual(rows[0].num_real_segments, 1)
    for row in rows[1:]:
      self.assertEqual(row.num_real_segments, 0)
      np.testing.assert_array_equal(row.completion_mask, np.zeros(4))
      np.testing.assert_array_equal(row.prompt_mask, np.zeros(4))
      np.testing.assert_array_equal(row.segment_ids, np.zeros(4))

  def test_oversized_sequence_errors(self):
    with self.assertRaisesRegex(ValueError, "exceeding budget"):
      packing.pack_core([_item(np.arange(5), np.arange(5))], budget=8)

  def test_max_segments_bound_row_segment_count(self):
    items = [_item([i], [i]) for i in range(4)]
    chunks = packing.pack_core(
        items, budget=64, pack_size=1, max_segments_per_packed_row=2
    )
    self.assertLen(chunks, 2)
    for [row] in chunks:
      self.assertEqual(row.num_real_segments, 2)

  def test_every_item_is_packed_only_once(self):
    rng = np.random.default_rng(0)
    items = [
        _item(
            np.arange(int(rng.integers(1, 8))),
            np.arange(int(rng.integers(1, 8))),
        )
        for _ in range(37)
    ]
    chunks = packing.pack_core(items, budget=32, pack_size=2)
    total_segments = sum(r.num_real_segments for c in chunks for r in c)
    self.assertEqual(total_segments, len(items))
    total_tokens = sum(
        int((r.segment_ids > 0).sum()) for c in chunks for r in c
    )
    self.assertEqual(total_tokens, sum(i.num_tokens for i in items))

  def test_empty_items_returns_empty_list(self):
    self.assertEmpty(packing.pack_core([], budget=16))

  def test_invalid_arguments_raise(self):
    item = _item([1], [2])
    with self.assertRaisesRegex(ValueError, "Budget must be positive"):
      packing.pack_core([item], budget=0)
    with self.assertRaisesRegex(ValueError, "Pack size must be positive"):
      packing.pack_core([item], budget=10, pack_size=0)
    with self.assertRaisesRegex(
        ValueError, "Max segments per packed row must be positive"
    ):
      packing.pack_core([item], budget=10, max_segments_per_packed_row=0)

  def test_pack_bin_exceeds_budget_raises(self):
    items = [_item([1] * 5, [2] * 6)]  # num_tokens = 11 > 10
    with self.assertRaisesRegex(ValueError, "exceeds budget"):
      packing.pack_bin(items, budget=10, pad_id=0, carried=())

  def test_carried_per_token_fields_in_packed_row(self):
    items = [
        _item(
            [1, 2],
            [3, 4],
            adv=1.0,
            per_token={
                "returns": np.array([2.0, 2.5], dtype=np.float32),
                "old_values": np.array([1.0, 1.2], dtype=np.float32),
            },
        ),
        _item(
            [5],
            [6],
            adv=0.5,
            per_token={
                "returns": np.array([3.0], dtype=np.float32),
                "old_values": np.array([2.0], dtype=np.float32),
            },
        ),
    ]
    [[row]] = packing.pack_core(items, budget=8, pack_size=1)
    # Total tokens: (2+2) + (1+1) = 6 tokens, 2 padded.
    # Item 1 completion spans [2:4], Item 2 completion spans [5:6].
    np.testing.assert_allclose(
        row.per_token["returns"], [0.0, 0.0, 2.0, 2.5, 0.0, 3.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        row.per_token["old_values"], [0.0, 0.0, 1.0, 1.2, 0.0, 2.0, 0.0, 0.0]
    )

  def test_custom_pad_id(self):
    items = [_item([1], [2])]
    [[row]] = packing.pack_core(items, budget=5, pad_id=99)
    np.testing.assert_array_equal(row.ids, [1, 2, 99, 99, 99])

  def test_exact_budget_item(self):
    items = [_item([1, 2], [3, 4, 5, 6])]  # total = 6 tokens
    [[row]] = packing.pack_core(items, budget=6)
    np.testing.assert_array_equal(row.ids, [1, 2, 3, 4, 5, 6])
    self.assertTrue(np.all(row.segment_ids == 1))
    self.assertEqual(row.num_real_segments, 1)

  def test_policy_version_propagation(self):
    item = packing.PackItem(
        prompt_ids=np.array([1], dtype=np.int32),
        completion_ids=np.array([2], dtype=np.int32),
        completion_mask=np.array([1], dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
        policy_version=np.array([42]),
    )
    [[row]] = packing.pack_core([item], budget=4)
    np.testing.assert_array_equal(row.policy_version, np.array([42]))


if __name__ == "__main__":
  absltest.main()
