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
from tunix.rl.agentic.agents import agent_types


def _item(
    prompt, completion, *, mask=None, adv=0.0, per_token=None, routed=None
):
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
      routed_experts=routed,
  )


def _routed(num_tokens, value, *, num_layers=2, top_k=2):
  return np.full((num_tokens, num_layers, top_k), value, dtype=np.int16)


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

  def test_rejects_routed_experts_shorter_than_sequence_minus_one(self):
    # Routing is sequence-aligned (p + c - 1 or p + c), not completion-aligned
    # (c).
    with self.assertRaisesRegex(
        ValueError, r"\(p \+ c - 1 or p \+ c, num_layers, top_k\)"
    ):
      _item([1, 2], [3, 4], routed=_routed(2, 0))

  def test_rejects_routed_experts_exceeding_sequence_length(self):
    with self.assertRaisesRegex(
        ValueError, r"\(p \+ c - 1 or p \+ c, num_layers, top_k\)"
    ):
      _item([1, 2], [3, 4], routed=_routed(5, 0))

  def test_accepts_prefix_routed_experts(self):
    item = _item([1, 2], [3, 4], routed=_routed(3, 0))
    self.assertEqual(item.routed_experts.shape, (3, 2, 2))

  def test_rejects_routed_experts_of_wrong_rank(self):
    with self.assertRaisesRegex(ValueError, "routed_experts"):
      _item([1], [2], routed=np.zeros((2, 4), dtype=np.int16))


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

  def test_pack_chunk_rows_share_contiguous_base_buffers(self):
    items = [
        _item(
            [1, 2],
            [3, 4],
            adv=1.0,
            per_token={"returns": np.array([2.0, 3.0], dtype=np.float32)},
        ),
        _item(
            [5, 6],
            [7, 8],
            adv=0.5,
            per_token={"returns": np.array([4.0, 5.0], dtype=np.float32)},
        ),
    ]
    [rows] = packing.pack_core(items, budget=4, pack_size=2)
    self.assertIsInstance(rows, packing.PackedChunk)
    self.assertLen(rows, 2)
    for attr in (
        "ids",
        "prompt_mask",
        "completion_mask",
        "advantages",
        "segment_ids",
        "segment_positions",
    ):
      base = getattr(rows, attr)
      self.assertEqual(base.shape, (2, 4))
      self.assertIs(getattr(rows[0], attr).base, base)
      self.assertIs(getattr(rows[1], attr).base, base)
    returns_base = rows.per_token["returns"]
    self.assertEqual(returns_base.shape, (2, 4))
    self.assertIs(rows[0].per_token["returns"].base, returns_base)
    self.assertIs(rows[1].per_token["returns"].base, returns_base)

  def test_invalid_segment_alignment_boundary_raises(self):
    with self.assertRaisesRegex(
        ValueError, "segment_alignment_boundary must be positive"
    ):
      packing.pack_core(
          [_item([1], [2])], budget=10, segment_alignment_boundary=0
      )


class PackRoutedExpertsTest(absltest.TestCase):

  def test_no_routing_leaves_rows_without_routed_experts(self):
    [rows] = packing.pack_core([_item([1], [2])], budget=4, pack_size=2)
    for row in rows:
      self.assertIsNone(row.routed_experts)

  def test_routing_is_sequence_aligned_and_dummy_rows_unset(self):
    experts1 = np.arange(3 * 2 * 2, dtype=np.int16).reshape(3, 2, 2)
    experts2 = np.arange(100, 100 + 2 * 2 * 2, dtype=np.int16).reshape(2, 2, 2)
    items = [
        _item([1], [2, 3], routed=experts1),
        _item([4], [5], routed=experts2),
    ]
    [rows] = packing.pack_core(items, budget=6, pack_size=2)
    row0, row1 = rows
    self.assertEqual(row0.routed_experts.shape, (6, 2, 2))
    self.assertEqual(row0.routed_experts.dtype, np.int16)
    np.testing.assert_array_equal(row0.routed_experts[:3], experts1)
    np.testing.assert_array_equal(row0.routed_experts[3:5], experts2)
    np.testing.assert_array_equal(
        row0.routed_experts[5:], agent_types.UNSET_ROUTED_EXPERT
    )
    # The dummy row in the same chunk keeps the same structure, all unset.
    self.assertEqual(row1.routed_experts.shape, (6, 2, 2))
    np.testing.assert_array_equal(
        row1.routed_experts, agent_types.UNSET_ROUTED_EXPERT
    )
    # Rows are views into one contiguous `[n_bins, budget, L, K]` buffer.
    self.assertEqual(rows.routed_experts.shape, (2, 6, 2, 2))
    self.assertIs(row0.routed_experts.base, rows.routed_experts)
    self.assertIs(row1.routed_experts.base, rows.routed_experts)

  def test_routing_less_item_is_unset_without_dropping_chunk_routing(self):
    items = [
        _item([1], [2, 3], routed=_routed(3, 7)),
        _item([4], [5]),
    ]
    [[row]] = packing.pack_core(items, budget=6, pack_size=1)
    np.testing.assert_array_equal(row.routed_experts[:3], 7)
    np.testing.assert_array_equal(
        row.routed_experts[3:], agent_types.UNSET_ROUTED_EXPERT
    )

  def test_prefix_routed_experts_leaves_trailing_segment_tokens_unset(self):
    experts1 = np.arange(3 * 2 * 2, dtype=np.int16).reshape(3, 2, 2)
    experts2 = np.arange(100, 100 + 2 * 2 * 2, dtype=np.int16).reshape(2, 2, 2)
    items = [
        _item([1, 2], [3, 4], routed=experts1),
        _item([5], [6, 7], routed=experts2),
    ]
    [[row]] = packing.pack_core(items, budget=8, pack_size=1)
    np.testing.assert_array_equal(row.routed_experts[:3], experts1)
    np.testing.assert_array_equal(
        row.routed_experts[3:4], agent_types.UNSET_ROUTED_EXPERT
    )
    np.testing.assert_array_equal(row.routed_experts[4:6], experts2)
    np.testing.assert_array_equal(
        row.routed_experts[6:], agent_types.UNSET_ROUTED_EXPERT
    )

  def test_mismatched_routing_shapes_raise(self):
    items = [
        _item([1], [2], routed=_routed(2, 0, top_k=2)),
        _item([3], [4], routed=_routed(2, 0, top_k=4)),
    ]
    with self.assertRaisesRegex(ValueError, "disagree on routed_experts"):
      packing.pack_core(items, budget=8)

  def test_routed_experts_shape(self):
    self.assertIsNone(packing.routed_experts_shape([_item([1], [2])]))
    self.assertEqual(
        packing.routed_experts_shape(
            [_item([1], [2]), _item([3], [4], routed=_routed(2, 0, top_k=8))]
        ),
        (2, 8),
    )


class PackSegmentAlignmentTest(absltest.TestCase):

  def test_align_offset(self):
    self.assertEqual(packing.align_offset(0, 64), 0)
    self.assertEqual(packing.align_offset(1, 64), 64)
    self.assertEqual(packing.align_offset(64, 64), 64)
    self.assertEqual(packing.align_offset(65, 64), 128)
    self.assertEqual(packing.align_offset(7, 1), 7)

  def test_default_is_unaligned(self):
    self.assertEqual(packing.DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY, 1)

  def test_segments_start_on_alignment_boundary_with_padded_gaps(self):
    # FFD sorts descending by length:
    #   seg 1 (70 = 20p + 50c): [0:70],    gap [70:128]
    #   seg 2 (50 = 10p + 40c): [128:178], gap [178:192]
    #   seg 3 (20 =  5p + 15c): [192:212], trailing pad [212:256]
    pad_id = 77
    items = [
        _item([11] * 10, [12] * 40, adv=1.5, routed=_routed(50, 1)),
        _item([21] * 20, [22] * 50, adv=2.5, routed=_routed(70, 2)),
        _item([31] * 5, [32] * 15, adv=3.5, routed=_routed(20, 3)),
    ]
    [[row]] = packing.pack_core(
        items,
        budget=256,
        pack_size=1,
        pad_id=pad_id,
        segment_alignment_boundary=64,
    )
    self.assertEqual(row.num_real_segments, 3)
    for seg_id, start, length, routed_value in (
        (1, 0, 70, 2),
        (2, 128, 50, 1),
        (3, 192, 20, 3),
    ):
      idx = np.flatnonzero(row.segment_ids == seg_id)
      np.testing.assert_array_equal(idx, np.arange(start, start + length))
      np.testing.assert_array_equal(
          row.segment_positions[idx], np.arange(length)
      )
      np.testing.assert_array_equal(row.routed_experts[idx], routed_value)
    for lo, hi in ((70, 128), (178, 192), (212, 256)):
      np.testing.assert_array_equal(row.segment_ids[lo:hi], 0)
      np.testing.assert_array_equal(row.segment_positions[lo:hi], 0)
      np.testing.assert_array_equal(row.completion_mask[lo:hi], 0)
      np.testing.assert_array_equal(row.prompt_mask[lo:hi], 0)
      np.testing.assert_array_equal(row.advantages[lo:hi], 0)
      np.testing.assert_array_equal(row.ids[lo:hi], pad_id)
      np.testing.assert_array_equal(
          row.routed_experts[lo:hi], agent_types.UNSET_ROUTED_EXPERT
      )

  def test_exact_multiple_leaves_no_gap(self):
    items = [
        _item(np.ones(24), np.ones(40)),  # 64 tokens.
        _item(np.ones(10), np.ones(20)),  # 30 tokens.
    ]
    [[row]] = packing.pack_core(
        items, budget=128, pack_size=1, segment_alignment_boundary=64
    )
    np.testing.assert_array_equal(row.segment_ids[:64], 1)
    np.testing.assert_array_equal(row.segment_ids[64:94], 2)
    np.testing.assert_array_equal(row.segment_ids[94:], 0)

  def test_alignment_counts_against_budget(self):
    # 10 + 10 tokens fit a 64-token row unaligned, but aligned the second
    # segment would need [64:74], so it spills to the next chunk.
    items = [_item(np.ones(4), np.ones(6)), _item(np.ones(4), np.ones(6))]
    self.assertLen(packing.pack_core(items, budget=64, pack_size=1), 1)
    chunks = packing.pack_core(
        items, budget=64, pack_size=1, segment_alignment_boundary=64
    )
    self.assertLen(chunks, 2)
    for [row] in chunks:
      self.assertEqual(row.num_real_segments, 1)

  def test_full_budget_first_segment_needs_no_alignment(self):
    [[row]] = packing.pack_core(
        [_item(np.ones(100), np.ones(156))],
        budget=256,
        segment_alignment_boundary=64,
    )
    np.testing.assert_array_equal(row.segment_ids, 1)

  def test_every_item_is_packed_only_once_with_alignment(self):
    rng = np.random.default_rng(0)
    items = [
        _item(
            np.arange(int(rng.integers(1, 40))),
            np.arange(int(rng.integers(1, 40))),
        )
        for _ in range(37)
    ]
    chunks = packing.pack_core(
        items, budget=256, pack_size=2, segment_alignment_boundary=16
    )
    self.assertEqual(
        sum(r.num_real_segments for c in chunks for r in c), len(items)
    )
    self.assertEqual(
        sum(int((r.segment_ids > 0).sum()) for c in chunks for r in c),
        sum(i.num_tokens for i in items),
    )
    for c in chunks:
      for r in c:
        starts = np.flatnonzero(r.segment_positions == 0)
        starts = starts[r.segment_ids[starts] > 0]
        np.testing.assert_array_equal(starts % 16, 0)

  def test_pack_bin_rejects_bin_that_overflows_once_aligned(self):
    items = [_item(np.ones(4), np.ones(6)), _item(np.ones(4), np.ones(6))]
    with self.assertRaisesRegex(ValueError, "exceeds budget"):
      packing.pack_bin(
          items, budget=64, pad_id=0, carried=(), segment_alignment_boundary=64
      )


if __name__ == "__main__":
  absltest.main()
