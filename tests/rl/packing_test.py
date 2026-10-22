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

import dataclasses
import os
import signal
import time
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

  def test_cpp_extension_and_python_fallback_produce_identical_chunks(self):
    self.assertIsNotNone(packing._packing_ext)
    rng = np.random.default_rng(123)
    valid_items = [
        _item(
            np.arange(p_len := int(rng.integers(1, 25)), dtype=np.int32),
            np.arange(c_len := int(rng.integers(1, 35)), dtype=np.int32),
            adv=float(rng.uniform(-1.0, 1.0)),
            per_token={"returns": np.ones(c_len, dtype=np.float32) * 0.5},
            routed=_routed(p_len + c_len - 1, idx % 7),
        )
        for idx in range(24)
    ]
    cpp_chunks = packing.pack_core(
        valid_items, budget=128, pack_size=3, segment_alignment_boundary=16
    )
    orig_ext = packing._packing_ext
    try:
      packing._packing_ext = None
      py_chunks = packing.pack_core(
          valid_items, budget=128, pack_size=3, segment_alignment_boundary=16
      )
    finally:
      packing._packing_ext = orig_ext

    self.assertEqual(len(cpp_chunks), len(py_chunks))
    for c_chunk, p_chunk in zip(cpp_chunks, py_chunks):
      np.testing.assert_array_equal(c_chunk.ids, p_chunk.ids)
      np.testing.assert_array_equal(c_chunk.prompt_mask, p_chunk.prompt_mask)
      np.testing.assert_array_equal(
          c_chunk.completion_mask, p_chunk.completion_mask
      )
      np.testing.assert_allclose(c_chunk.advantages, p_chunk.advantages)
      np.testing.assert_array_equal(c_chunk.segment_ids, p_chunk.segment_ids)
      np.testing.assert_array_equal(
          c_chunk.segment_positions, p_chunk.segment_positions
      )
      np.testing.assert_allclose(
          c_chunk.per_token["returns"], p_chunk.per_token["returns"]
      )
      np.testing.assert_array_equal(
          c_chunk.routed_experts, p_chunk.routed_experts
      )
      self.assertEqual(c_chunk.num_real_segments, p_chunk.num_real_segments)


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
    with self.assertRaisesRegex(ValueError, "exceeds budget"):
      packing.pack_bin(
          items, budget=16, pad_id=0, carried=(), segment_alignment_boundary=64
      )

  def test_invalid_inputs_raise_value_error_in_cpp_extension(self):
    self.assertIsNotNone(packing._packing_ext)
    bad_mask = _item([1, 2], [3, 4, 5])
    object.__setattr__(
        bad_mask, "completion_mask", np.array([1.0], dtype=np.float32)
    )
    with self.assertRaisesRegex(ValueError, "completion_mask and advantages"):
      packing.pack_chunk([[bad_mask]], budget=8, pad_id=0, carried=())

    bad_pt = _item([1, 2], [3, 4, 5])
    bad_pt.per_token["logps"] = np.array([0.1], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "per_token array length"):
      packing.pack_chunk([[bad_pt]], budget=8, pad_id=0, carried=("logps",))

    bad_routed = _item([1, 2], [3, 4])
    object.__setattr__(
        bad_routed, "routed_experts", np.zeros((4, 2), dtype=np.int16)
    )
    with self.assertRaisesRegex(ValueError, "at least 3 dimensions"):
      packing._packing_ext.pack_chunk_fast(
          [[bad_routed]], [], 8, 0, 1, (2, 2)
      )

    with self.assertRaisesRegex(ValueError, "pack_size must be positive"):
      packing._packing_ext.pack_sequence_chunks_fast(
          [_item([1], [2])], [], 8, 0, 4, 0, 1, 0
      )

  def test_encode_and_decode_routed_experts_uint8_uint10_int16(self):
    self.assertIsNotNone(packing._packing_ext)
    rng = np.random.default_rng(123)

    # 1. Mode 1: uint8 (max expert <= 254, any K)
    raw_u8 = rng.integers(-1, 255, size=(12, 4, 6), dtype=np.int16)
    raw_u8[2] = agent_types.UNSET_ROUTED_EXPERT
    enc_u8 = packing.encode_routed_experts(raw_u8)
    self.assertEqual(enc_u8.dtype, np.uint8)
    self.assertEqual(enc_u8.shape, (12, 4, 6))
    np.testing.assert_array_equal(enc_u8[2], 0)
    dec_u8 = packing.decode_routed_experts(enc_u8)
    self.assertEqual(dec_u8.dtype, np.int16)
    np.testing.assert_array_equal(dec_u8, raw_u8)

    # 2. Mode 2: uint10 (255 <= max expert <= 1022, K % 4 == 0)
    raw_u10 = rng.integers(-1, 1023, size=(10, 4, 8), dtype=np.int16)
    raw_u10[0, 0, 0] = 511
    raw_u10[3] = agent_types.UNSET_ROUTED_EXPERT
    enc_u10 = packing.encode_routed_experts(raw_u10)
    self.assertEqual(enc_u10.dtype, np.uint8)
    self.assertEqual(enc_u10.shape, (10, 4, 2, 5))
    np.testing.assert_array_equal(enc_u10[3], 0)
    dec_u10 = packing.decode_routed_experts(enc_u10)
    self.assertEqual(dec_u10.dtype, np.int16)
    np.testing.assert_array_equal(dec_u10, raw_u10)

    # 3. Fallback: int16 (max expert >= 1023 or K % 4 != 0 when max >= 255)
    raw_i16 = rng.integers(-1, 2048, size=(6, 4, 8), dtype=np.int32)
    raw_i16[0, 0, 0] = 1500
    enc_i16 = packing.encode_routed_experts(raw_i16)
    self.assertEqual(enc_i16.dtype, np.int16)
    self.assertEqual(enc_i16.shape, (6, 4, 8))
    np.testing.assert_array_equal(
        packing.decode_routed_experts(enc_i16), raw_i16.astype(np.int16)
    )
    # 255 <= max <= 1022 with K % 4 != 0 falls back to int16 in "auto" mode and
    # raises ValueError when mode="uint10" is requested explicitly.
    for k_non_div4 in (2, 6):
      raw_k_non_div4 = rng.integers(
          -1, 512, size=(5, 3, k_non_div4), dtype=np.int16
      )
      raw_k_non_div4[0, 0, 0] = 300
      enc_auto = packing.encode_routed_experts(raw_k_non_div4, mode="auto")
      self.assertEqual(enc_auto.dtype, np.int16)
      np.testing.assert_array_equal(enc_auto, raw_k_non_div4)
      with self.assertRaisesRegex(ValueError, "top_k divisible by 4"):
        packing.encode_routed_experts(raw_k_non_div4, mode="uint10")

    # 4. K == 5 disambiguation between unbatched (T, L, 5) and batched
    # (B, T, L, 5) for both C++ and Python fallback.
    raw_k5_unbatched = rng.integers(-1, 255, size=(7, 3, 5), dtype=np.int16)
    raw_k5_batched = rng.integers(-1, 255, size=(2, 7, 3, 5), dtype=np.int16)
    enc_k5_unbatched = packing.encode_routed_experts(raw_k5_unbatched)
    enc_k5_batched = packing.encode_routed_experts(raw_k5_batched)
    self.assertEqual(enc_k5_unbatched.shape, (7, 3, 5))
    self.assertEqual(enc_k5_batched.shape, (2, 7, 3, 5))
    self.assertEqual(
        packing.routed_experts_logical_shape(enc_k5_unbatched, batched=False),
        (7, 3, 5),
    )
    self.assertEqual(
        packing.routed_experts_logical_shape(enc_k5_batched, batched=True),
        (2, 7, 3, 5),
    )
    np.testing.assert_array_equal(
        packing.decode_routed_experts(enc_k5_unbatched, batched=False),
        raw_k5_unbatched,
    )
    np.testing.assert_array_equal(
        packing.decode_routed_experts(enc_k5_batched, batched=True),
        raw_k5_batched,
    )

    # Verify Python fallback parity with C++ extension.
    orig_ext = packing._packing_ext
    try:
      packing._packing_ext = None
      py_u8 = packing.encode_routed_experts(raw_u8)
      py_u10 = packing.encode_routed_experts(raw_u10)
      np.testing.assert_array_equal(py_u8, enc_u8)
      np.testing.assert_array_equal(py_u10, enc_u10)
      np.testing.assert_array_equal(
          packing.decode_routed_experts(py_u8, batched=False), raw_u8
      )
      np.testing.assert_array_equal(
          packing.decode_routed_experts(py_u10, batched=False), raw_u10
      )
      np.testing.assert_array_equal(
          packing.decode_routed_experts(enc_k5_unbatched, batched=False),
          raw_k5_unbatched,
      )
      np.testing.assert_array_equal(
          packing.decode_routed_experts(enc_k5_batched, batched=True),
          raw_k5_batched,
      )
      for k_non_div4 in (2, 6):
        raw_k_non_div4 = rng.integers(
            -1, 512, size=(5, 3, k_non_div4), dtype=np.int16
        )
        raw_k_non_div4[0, 0, 0] = 300
        py_auto = packing.encode_routed_experts(raw_k_non_div4, mode="auto")
        self.assertEqual(py_auto.dtype, np.int16)
        np.testing.assert_array_equal(py_auto, raw_k_non_div4)
        with self.assertRaisesRegex(ValueError, "top_k divisible by 4"):
          packing.encode_routed_experts(raw_k_non_div4, mode="uint10")
    finally:
      packing._packing_ext = orig_ext

  def test_pack_chunk_preserves_uint8_uint10_and_widens_mixed_encodings(self):
    raw1_u8 = np.array(
        [[[0, 5, 10, 15]], [[20, 25, 30, 35]], [[-1, -1, -1, -1]]],
        dtype=np.int16,
    )
    raw2_u8 = np.array(
        [[[1, 2, 3, 4]], [[200, 201, 202, 254]]],
        dtype=np.int16,
    )
    item1_u8 = dataclasses.replace(
        _item([1, 2], [3]),
        routed_experts=packing.encode_routed_experts(raw1_u8),
    )
    item2_u8 = dataclasses.replace(
        _item([4], [5]), routed_experts=packing.encode_routed_experts(raw2_u8)
    )
    chunk_u8 = packing.pack_chunk(
        [[item1_u8, item2_u8]], budget=8, pad_id=0, carried=()
    )
    self.assertEqual(chunk_u8.routed_experts.dtype, np.uint8)
    self.assertEqual(chunk_u8.routed_experts.shape, (1, 8, 1, 4))
    # Unset trailing padding is 0x00 in uint8 mode.
    np.testing.assert_array_equal(chunk_u8.routed_experts[0, 5:], 0)
    decoded_u8 = packing.decode_routed_experts(
        chunk_u8.routed_experts, batched=True
    )
    np.testing.assert_array_equal(decoded_u8[0, :3], raw1_u8)
    np.testing.assert_array_equal(decoded_u8[0, 3:5], raw2_u8)
    np.testing.assert_array_equal(
        decoded_u8[0, 5:], agent_types.UNSET_ROUTED_EXPERT
    )

    # Pure uint10 chunk preserves 5D uint8 [B, T, L, K//4, 5].
    raw1_u10 = np.array(
        [[[0, 300, 511, 1022]], [[10, 20, 30, 40]], [[-1, -1, -1, -1]]],
        dtype=np.int16,
    )
    item1_u10 = dataclasses.replace(
        _item([1, 2], [3]),
        routed_experts=packing.encode_routed_experts(raw1_u10),
    )
    chunk_u10 = packing.pack_chunk(
        [[item1_u10]], budget=6, pad_id=0, carried=()
    )
    self.assertEqual(chunk_u10.routed_experts.dtype, np.uint8)
    self.assertEqual(chunk_u10.routed_experts.shape, (1, 6, 1, 1, 5))
    np.testing.assert_array_equal(chunk_u10.routed_experts[0, 2:], 0)
    decoded_u10 = packing.decode_routed_experts(
        chunk_u10.routed_experts, batched=True
    )
    np.testing.assert_array_equal(decoded_u10[0, :3], raw1_u10)
    np.testing.assert_array_equal(
        decoded_u10[0, 3:], agent_types.UNSET_ROUTED_EXPERT
    )

    # Mixed uint8 + uint10 in the same chunk widens losslessly to uint10.
    chunk_mixed = packing.pack_chunk(
        [[item1_u8, item1_u10]], budget=8, pad_id=0, carried=()
    )
    self.assertEqual(chunk_mixed.routed_experts.dtype, np.uint8)
    self.assertEqual(chunk_mixed.routed_experts.shape, (1, 8, 1, 1, 5))
    decoded_mixed = packing.decode_routed_experts(
        chunk_mixed.routed_experts, batched=True
    )
    np.testing.assert_array_equal(decoded_mixed[0, :3], raw1_u8)
    np.testing.assert_array_equal(decoded_mixed[0, 3:6], raw1_u10)
    np.testing.assert_array_equal(
        decoded_mixed[0, 6:], agent_types.UNSET_ROUTED_EXPERT
    )

  def test_encode_routed_experts_num_experts_validation_and_transcoding(self):
    raw_u8_vals = np.array(
        [[[0, 5, 10, 15]], [[20, 25, 30, 35]], [[-1, -1, -1, -1]]],
        dtype=np.int16,
    )
    orig_ext = packing._packing_ext
    for use_cpp in (True, False):
      with self.subTest(use_cpp=use_cpp):
        if use_cpp and orig_ext is None:
          continue
        try:
          packing._packing_ext = orig_ext if use_cpp else None
          # 1. Invalid num_experts values raise ValueError.
          for bad_num in (0, -1, True, False):
            with self.assertRaisesRegex(
                ValueError, "num_experts must be a positive int"
            ):
              packing.encode_routed_experts(
                  raw_u8_vals, mode="auto", num_experts=bad_num
              )

          # 2. max_val >= num_experts raises ValueError loudly.
          with self.assertRaisesRegex(
              ValueError, "routed_experts contains expert ID .* >= num_experts"
          ):
            packing.encode_routed_experts(
                raw_u8_vals, mode="auto", num_experts=35
            )

          # 3. Deterministic tier selection via num_experts=256 (0..255 > 254)
          # forces uint10 even when max_val == 35 <= 254.
          enc_u10 = packing.encode_routed_experts(
              raw_u8_vals, mode="auto", num_experts=256
          )
          self.assertEqual(enc_u10.dtype, np.uint8)
          self.assertEqual(enc_u10.shape, (3, 1, 1, 5))
          np.testing.assert_array_equal(
              packing.decode_routed_experts(enc_u10, batched=False),
              raw_u8_vals,
          )

          # 4. Pre-encoded pass-through vs transcoding when mode / num_experts
          # requests a different tier.
          enc_u8 = packing.encode_routed_experts(
              raw_u8_vals, mode="auto", num_experts=128
          )
          self.assertEqual(enc_u8.dtype, np.uint8)
          self.assertEqual(enc_u8.shape, (3, 1, 4))
          # Compatible call returns same buffer (zero-copy).
          passthru = packing.encode_routed_experts(
              enc_u8, mode="auto", num_experts=128
          )
          self.assertIs(passthru, enc_u8)
          # Requesting uint10 or num_experts=256 transcodes pre-encoded uint8 to
          # uint10.
          transcoded_u10 = packing.encode_routed_experts(
              enc_u8, mode="auto", num_experts=256
          )
          self.assertEqual(transcoded_u10.shape, (3, 1, 1, 5))
          np.testing.assert_array_equal(
              packing.decode_routed_experts(transcoded_u10, batched=False),
              raw_u8_vals,
          )
          # Requesting mode="int16" transcodes pre-encoded uint10 back to int16.
          transcoded_i16 = packing.encode_routed_experts(enc_u10, mode="int16")
          self.assertEqual(transcoded_i16.dtype, np.int16)
          np.testing.assert_array_equal(transcoded_i16, raw_u8_vals)
          # Pre-encoded input still validates max_val < num_experts.
          with self.assertRaisesRegex(
              ValueError, "routed_experts contains expert ID .* >= num_experts"
          ):
            packing.encode_routed_experts(enc_u8, mode="auto", num_experts=30)
        finally:
          packing._packing_ext = orig_ext

  def test_encode_routed_experts_batched_k5_and_routed_experts_encoding(self):
    rng = np.random.default_rng(2026)
    # Batched 4D array with K == 5: shape (B=2, T=7, L=3, K=5).
    raw_batched_k5 = rng.integers(-1, 200, size=(2, 7, 3, 5), dtype=np.int16)
    raw_batched_k5[:, -1] = -1
    # Unbatched 3D array with K == 8: shape (T=7, L=3, K=8).
    raw_unbatched_k8 = rng.integers(-1, 600, size=(7, 3, 8), dtype=np.int16)
    raw_unbatched_k8[0, 0, 0] = 511
    # Batched 4D array with K == 8: shape (B=2, T=7, L=3, K=8).
    raw_batched_k8 = rng.integers(-1, 600, size=(2, 7, 3, 8), dtype=np.int16)
    raw_batched_k8[0, 0, 0, 0] = 511

    orig_ext = packing._packing_ext
    for use_cpp in (True, False):
      with self.subTest(use_cpp=use_cpp):
        if use_cpp and orig_ext is None:
          continue
        try:
          packing._packing_ext = orig_ext if use_cpp else None
          enc_batched_k5 = packing.encode_routed_experts(
              raw_batched_k5, mode="auto", batched=True
          )
          self.assertEqual(enc_batched_k5.dtype, np.uint8)
          self.assertEqual(enc_batched_k5.shape, (2, 7, 3, 5))
          self.assertEqual(
              packing.routed_experts_encoding(enc_batched_k5, batched=True),
              "uint8",
          )
          # Re-encoding batched 4D uint8 (B, T, L, 5) with batched=True must NOT
          # misclassify it as unbatched 4D uint10 [T, L, K//4, 5].
          for mode in ("auto", "uint8"):
            re_u8 = packing.encode_routed_experts(
                enc_batched_k5, mode=mode, batched=True
            )
            self.assertEqual(re_u8.dtype, np.uint8)
            self.assertEqual(re_u8.shape, (2, 7, 3, 5))
            np.testing.assert_array_equal(
                packing.decode_routed_experts(re_u8, batched=True),
                raw_batched_k5,
            )
          re_i16 = packing.encode_routed_experts(
              enc_batched_k5, mode="int16", batched=True
          )
          self.assertEqual(re_i16.dtype, np.int16)
          self.assertEqual(re_i16.shape, (2, 7, 3, 5))
          np.testing.assert_array_equal(re_i16, raw_batched_k5)
          with self.assertRaisesRegex(ValueError, "top_k divisible by 4"):
            packing.encode_routed_experts(
                enc_batched_k5, mode="uint10", batched=True
            )

          # Verify routed_experts_encoding on unbatched vs batched uint10/int16.
          enc_unbatched_u10 = packing.encode_routed_experts(
              raw_unbatched_k8, mode="uint10", batched=False
          )
          self.assertEqual(
              packing.routed_experts_encoding(enc_unbatched_u10, batched=False),
              "uint10",
          )
          self.assertEqual(
              packing.routed_experts_encoding(raw_unbatched_k8, batched=False),
              "int16",
          )
          enc_batched_u10 = packing.encode_routed_experts(
              raw_batched_k8, mode="uint10", batched=True
          )
          self.assertEqual(enc_batched_u10.shape, (2, 7, 3, 2, 5))
          self.assertEqual(
              packing.routed_experts_encoding(enc_batched_u10, batched=True),
              "uint10",
          )
          self.assertEqual(
              packing.routed_experts_encoding(raw_batched_k8, batched=True),
              "int16",
          )
          re_batched_i16 = packing.encode_routed_experts(
              enc_batched_u10, mode="int16", batched=True
          )
          np.testing.assert_array_equal(re_batched_i16, raw_batched_k8)
        finally:
          packing._packing_ext = orig_ext


class ParallelPackingPoolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.assertIsNotNone(packing._packing_ext)
    self._orig_threads = packing._packing_ext.get_packing_threads()
    packing._packing_ext.set_packing_threads(4)

  def tearDown(self):
    if packing._packing_ext is not None:
      packing._packing_ext.set_packing_threads(self._orig_threads)
    super().tearDown()

  def test_set_packing_threads_validation(self):
    self.assertEqual(packing._packing_ext.get_packing_threads(), 4)
    self.assertEqual(
        packing._packing_ext.MIN_BYTES_PER_THREAD, 2 * 1024 * 1024
    )
    self.assertEqual(
        packing._packing_ext.MIN_PARALLEL_BYTES,
        2 * packing._packing_ext.MIN_BYTES_PER_THREAD,
    )
    max_t = packing._packing_ext.MAX_PACKING_THREADS
    for invalid_n in (0, -1, max_t + 1):
      with self.assertRaisesRegex(ValueError, "num_threads must be in"):
        packing._packing_ext.set_packing_threads(invalid_n)
    self.assertEqual(packing._packing_ext.get_packing_threads(), 4)
    self.assertIsNotNone(packing._packing_ext)

  def test_parallel_chunk_population_across_routed_encodings(self):
    rng = np.random.default_rng(2026)
    n_bins = 8
    budget = 4096
    top_k = 8
    num_layers = 16
    min_bytes_per_token = 28 + num_layers * top_k
    self.assertGreaterEqual(
        n_bins * budget * min_bytes_per_token,
        packing._packing_ext.MIN_PARALLEL_BYTES,
    )
    self.assertGreaterEqual(packing._packing_ext.MAX_PACKING_THREADS, 1)
    modes = (
        "int16",
        "uint8",
        "uint10",
        "mixed_u8_u10",
        "mixed_u8_i16",
        "mixed_u10_i16",
    )
    for mode in modes:
      with self.subTest(mode=mode):
        items: list[packing.PackItem] = []
        for idx in range(20):
          p_len = int(rng.integers(128, 384))
          c_len = int(rng.integers(512, 1024))
          seq_len = p_len + c_len
          prompt_ids = rng.integers(1, 1000, size=p_len, dtype=np.int32)
          completion_ids = rng.integers(1, 1000, size=c_len, dtype=np.int32)
          completion_mask = np.ones(c_len, dtype=np.float32)
          advantages = rng.standard_normal(c_len).astype(np.float32)
          old_logps = rng.standard_normal(c_len).astype(np.float32)

          if idx == 3:
            routed = None
          else:
            if mode == "uint8":
              item_tier = "uint8"
            elif mode == "uint10":
              item_tier = "uint10"
            elif mode == "int16":
              item_tier = "int16"
            elif mode == "mixed_u8_u10":
              item_tier = "uint8" if idx % 2 == 0 else "uint10"
            elif mode == "mixed_u8_i16":
              item_tier = "uint8" if idx % 2 == 0 else "int16"
            else:  # mixed_u10_i16
              item_tier = "uint10" if idx % 2 == 0 else "int16"

            max_exp = 200 if item_tier == "uint8" else 800
            raw_r = rng.integers(
                -1,
                max_exp,
                size=(seq_len - 1, num_layers, top_k),
                dtype=np.int16,
            )
            if item_tier == "int16":
              routed = raw_r
            elif item_tier == "uint8":
              routed = packing.encode_routed_experts(raw_r, mode="uint8")
            else:
              routed = packing.encode_routed_experts(raw_r, mode="uint10")

          items.append(
              packing.PackItem(
                  prompt_ids=prompt_ids,
                  completion_ids=completion_ids,
                  completion_mask=completion_mask,
                  advantages=advantages,
                  per_token={"old_per_token_logps": old_logps},
                  routed_experts=routed,
              )
          )

        cpp_chunks = packing.pack_core(
            items,
            budget=budget,
            pack_size=n_bins,
            pad_id=0,
            segment_alignment_boundary=64,
        )
        orig_ext = packing._packing_ext
        try:
          packing._packing_ext = None
          py_chunks = packing.pack_core(
              items,
              budget=budget,
              pack_size=n_bins,
              pad_id=0,
              segment_alignment_boundary=64,
          )
        finally:
          packing._packing_ext = orig_ext

        self.assertEqual(len(cpp_chunks), len(py_chunks))
        for cpp_c, py_c in zip(cpp_chunks, py_chunks):
          np.testing.assert_array_equal(cpp_c.ids, py_c.ids)
          np.testing.assert_array_equal(cpp_c.prompt_mask, py_c.prompt_mask)
          np.testing.assert_array_equal(
              cpp_c.completion_mask, py_c.completion_mask
          )
          np.testing.assert_array_equal(cpp_c.advantages, py_c.advantages)
          np.testing.assert_array_equal(cpp_c.segment_ids, py_c.segment_ids)
          np.testing.assert_array_equal(
              cpp_c.segment_positions, py_c.segment_positions
          )
          np.testing.assert_array_equal(
              cpp_c.per_token["old_per_token_logps"],
              py_c.per_token["old_per_token_logps"],
          )
          np.testing.assert_array_equal(
              cpp_c.routed_experts, py_c.routed_experts
          )

  def test_overlong_routed_experts_rejected_in_cpp_and_python(self):
    bad_item = _item([1, 2], [3, 4], routed=_routed(4, 1))
    object.__setattr__(bad_item, "routed_experts", _routed(5, 1))
    # 1. C++ single-row path.
    with self.assertRaisesRegex(ValueError, "exceeds segment token length"):
      packing.pack_chunk([[bad_item]], budget=8, pad_id=0, carried=())

    # 2. Pure-Python fallback path.
    orig_ext = packing._packing_ext
    try:
      packing._packing_ext = None
      with self.assertRaisesRegex(ValueError, "exceeds segment token length"):
        packing.pack_chunk([[bad_item]], budget=8, pad_id=0, carried=())
    finally:
      packing._packing_ext = orig_ext

    # 3. C++ multi-row parallel worker pool path (>= 4 * MIN_BYTES_PER_THREAD,
    # 4 threads) with the bad item in the last row so the exception crosses a
    # background worker thread via `worker_exc_` and rethrows as ValueError.
    good_parallel = _item(
        [1, 2], [3, 4], routed=np.ones((4, 32, 8), dtype=np.int16)
    )
    bad_parallel = _item(
        [1, 2], [3, 4], routed=np.ones((4, 32, 8), dtype=np.int16)
    )
    object.__setattr__(
        bad_parallel, "routed_experts", np.ones((5, 32, 8), dtype=np.int16)
    )
    parallel_bins = [
        [good_parallel],
        [good_parallel],
        [good_parallel],
        [bad_parallel],
    ]
    budget = 4096
    bytes_per_token = 24 + 32 * 8 * 2
    self.assertGreaterEqual(
        len(parallel_bins) * budget * bytes_per_token,
        4 * packing._packing_ext.MIN_BYTES_PER_THREAD,
    )
    with self.assertRaisesRegex(ValueError, "exceeds segment token length"):
      packing.pack_chunk(parallel_bins, budget=budget, pad_id=0, carried=())

  def test_fork_safety_resets_worker_pool_in_child(self):
    if packing._packing_ext.THREAD_SANITIZER:
      self.skipTest(
          "ThreadSanitizer does not support starting threads after"
          " multi-threaded fork without exec"
      )
    if not hasattr(os, "fork"):
      self.skipTest("os.fork not supported on this platform")

    rng = np.random.default_rng(999)
    n_bins = 4
    budget = 4096
    bytes_per_token = 24 + 32 * 8
    self.assertGreaterEqual(
        n_bins * budget * bytes_per_token,
        packing._packing_ext.MIN_PARALLEL_BYTES,
    )
    bins = [
        [
            _item(
                rng.integers(1, 100, size=512, dtype=np.int32),
                rng.integers(1, 100, size=1024, dtype=np.int32),
                routed=packing.encode_routed_experts(
                    rng.integers(0, 128, size=(1536, 32, 8), dtype=np.int16),
                    mode="uint8",
                ),
            )
        ]
        for _ in range(n_bins)
    ]
    # Warm up the persistent worker pool in the parent process before forking.
    parent_chunk = packing.pack_chunk(
        bins, budget=budget, pad_id=0, carried=()
    )

    pid = os.fork()
    if pid == 0:
      # Child process: must re-initialize its worker pool and produce identical
      # output without deadlocking on the parent's futexes.
      try:
        child_chunk = packing.pack_chunk(
            bins, budget=budget, pad_id=0, carried=()
        )
        ok = np.array_equal(
            child_chunk.ids, parent_chunk.ids
        ) and np.array_equal(
            child_chunk.routed_experts, parent_chunk.routed_experts
        )
        os._exit(0 if ok else 1)
      except BaseException:  # pylint: disable=broad-exception-caught
        os._exit(2)
    else:
      deadline = time.monotonic() + 30.0
      status = None
      while time.monotonic() < deadline:
        waited_pid, waited_status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
          status = waited_status
          break
        time.sleep(0.01)
      if status is None:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        self.fail("Child process timed out after fork()")
      self.assertTrue(
          os.WIFEXITED(status), f"Child did not exit cleanly: {status}"
      )
      self.assertEqual(os.WEXITSTATUS(status), 0)


if __name__ == "__main__":
  absltest.main()
