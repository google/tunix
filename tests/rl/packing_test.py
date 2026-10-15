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
from absl.testing import parameterized
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
    [[row]] = packing.pack_core(
        items, budget=10, pack_size=1, segment_align_multiple=1
    )
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
        items,
        budget=64,
        pack_size=1,
        max_segments_per_packed_row=2,
        segment_align_multiple=1,
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
    chunks = packing.pack_core(items, budget=256, pack_size=2)
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
    with self.assertRaisesRegex(
        ValueError, "segment_align_multiple must be positive"
    ):
      packing.pack_core([item], budget=10, segment_align_multiple=0)

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
    [[row]] = packing.pack_core(
        items, budget=8, pack_size=1, segment_align_multiple=1
    )
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

  def test_rollout_logps_and_overlong_carried_in_packed_row(self):
    items = [
        _item(
            [1],
            [2, 3],
            per_token={
                "rollout_per_token_logps": np.array(
                    [-0.5, -1.0], dtype=np.float32
                ),
                "overlong": np.array([0.0, 0.0], dtype=np.float32),
            },
        ),
        _item(
            [4],
            [5],
            per_token={
                "rollout_per_token_logps": np.array([-2.0], dtype=np.float32),
                "overlong": np.array([1.0], dtype=np.float32),
            },
        ),
    ]
    [[row]] = packing.pack_core(
        items, budget=6, pack_size=1, segment_align_multiple=1
    )
    # Item 1: prompt [0:1], completion [1:3]. Item 2: prompt [3:4], completion [4:5].
    np.testing.assert_allclose(
        row.per_token["rollout_per_token_logps"],
        [0.0, -0.5, -1.0, 0.0, -2.0, 0.0],
    )
    np.testing.assert_allclose(
        row.per_token["overlong"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )

  def test_routed_experts_packed_and_dummy_padded(self):
    experts1 = np.arange(3 * 2 * 2, dtype=np.int32).reshape(3, 2, 2)
    experts2 = np.arange(100, 100 + 2 * 2 * 2, dtype=np.int32).reshape(2, 2, 2)
    item1 = packing.PackItem(
        prompt_ids=np.array([1], dtype=np.int32),
        completion_ids=np.array([2, 3], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.ones(2, dtype=np.float32),
        routed_experts=experts1,
    )
    item2 = packing.PackItem(
        prompt_ids=np.array([4], dtype=np.int32),
        completion_ids=np.array([5], dtype=np.int32),
        completion_mask=np.ones(1, dtype=np.float32),
        advantages=np.ones(1, dtype=np.float32),
        routed_experts=experts2,
    )
    [rows] = packing.pack_core(
        [item1, item2], budget=6, pack_size=2, segment_align_multiple=1
    )
    self.assertLen(rows, 2)
    row0, row1 = rows
    self.assertIsNotNone(row0.routed_experts)
    self.assertEqual(row0.routed_experts.shape, (6, 2, 2))
    np.testing.assert_array_equal(row0.routed_experts[:3], experts1)
    np.testing.assert_array_equal(row0.routed_experts[3:5], experts2)
    np.testing.assert_array_equal(
        row0.routed_experts[5:],
        np.full((1, 2, 2), packing.UNSET_ROUTED_EXPERT, dtype=np.int32),
    )
    # Dummy row in the same chunk must also have routed_experts filled with -1.
    self.assertIsNotNone(row1.routed_experts)
    np.testing.assert_array_equal(
        row1.routed_experts,
        np.full((6, 2, 2), packing.UNSET_ROUTED_EXPERT, dtype=np.int32),
    )

  def test_default_segment_alignment_64_starts_at_multiples_of_64(self):
    # Default segment_align_multiple=64:
    # FFD sorts descending by length:
    #   seg 1 (len 70 = 20p + 50c): [0:70]
    #   gap 1: [70:128] -> padded (segment_ids=0, completion_mask=0, ids=pad_id)
    #   seg 2 (len 50 = 10p + 40c): [128:178]
    #   gap 2: [178:192] -> padded
    #   seg 3 (len 20 = 5p + 15c):  [192:212]
    #   trailing pad: [212:256]
    pad_id = 77
    items = [
        packing.PackItem(
            prompt_ids=np.full(10, 11, dtype=np.int32),
            completion_ids=np.full(40, 12, dtype=np.int32),
            completion_mask=np.ones(40, dtype=np.float32),
            advantages=np.full(40, 1.5, dtype=np.float32),
            routed_experts=np.full((50, 2, 2), 1, dtype=np.int32),
        ),
        packing.PackItem(
            prompt_ids=np.full(20, 21, dtype=np.int32),
            completion_ids=np.full(50, 22, dtype=np.int32),
            completion_mask=np.ones(50, dtype=np.float32),
            advantages=np.full(50, 2.5, dtype=np.float32),
            routed_experts=np.full((70, 2, 2), 2, dtype=np.int32),
        ),
        packing.PackItem(
            prompt_ids=np.full(5, 31, dtype=np.int32),
            completion_ids=np.full(15, 32, dtype=np.int32),
            completion_mask=np.ones(15, dtype=np.float32),
            advantages=np.full(15, 3.5, dtype=np.float32),
            routed_experts=np.full((20, 2, 2), 3, dtype=np.int32),
        ),
    ]
    # Uses default segment_align_multiple=64.
    [[row]] = packing.pack_core(items, budget=256, pack_size=1, pad_id=pad_id)
    self.assertEqual(row.num_real_segments, 3)

    # Every packed segment must start at an index divisible by 64.
    for seg_id, expected_start, expected_len in (
        (1, 0, 70),
        (2, 128, 50),
        (3, 192, 20),
    ):
      seg_indices = np.flatnonzero(row.segment_ids == seg_id)
      self.assertLen(seg_indices, expected_len)
      start_idx = int(seg_indices[0])
      self.assertEqual(start_idx % 64, 0)
      self.assertEqual(start_idx, expected_start)
      np.testing.assert_array_equal(
          row.segment_positions[seg_indices],
          np.arange(expected_len, dtype=np.int32),
      )

    # Verify alignment gaps [70:128], [178:192] and trailing pad [212:256]
    for gap_start, gap_end in ((70, 128), (178, 192), (212, 256)):
      np.testing.assert_array_equal(row.segment_ids[gap_start:gap_end], 0)
      np.testing.assert_array_equal(row.completion_mask[gap_start:gap_end], 0)
      np.testing.assert_array_equal(row.prompt_mask[gap_start:gap_end], 0)
      np.testing.assert_array_equal(row.ids[gap_start:gap_end], pad_id)
      np.testing.assert_array_equal(row.advantages[gap_start:gap_end], 0.0)
      np.testing.assert_array_equal(
          row.routed_experts[gap_start:gap_end],
          packing.UNSET_ROUTED_EXPERT,
      )

  def test_exact_64_multiple_segment_leaves_zero_alignment_gap(self):
    # Segment 1 has length 64 (already a multiple of 64), so segment 2 starts
    # immediately at offset 64 with no alignment gap.
    items = [
        _item(np.ones(24, dtype=np.int32), np.ones(40, dtype=np.int32)),
        _item(np.ones(10, dtype=np.int32), np.ones(20, dtype=np.int32)),
    ]
    [[row]] = packing.pack_core(items, budget=128, pack_size=1)
    self.assertEqual(row.num_real_segments, 2)
    np.testing.assert_array_equal(row.segment_ids[:64], 1)
    np.testing.assert_array_equal(row.segment_ids[64:94], 2)
    np.testing.assert_array_equal(row.segment_ids[94:], 0)

  def test_alignment_prevents_second_segment_overflow(self):
    # Budget is 64. Two items of length 10 each would fit unaligned (20 <= 64),
    # but with segment_align_multiple=64 the second segment would start at 64
    # and require 64 + 10 = 74 > 64, so it must be placed in a second row/chunk.
    items = [
        _item(np.ones(4, dtype=np.int32), np.ones(6, dtype=np.int32)),
        _item(np.ones(4, dtype=np.int32), np.ones(6, dtype=np.int32)),
    ]
    chunks = packing.pack_core(items, budget=64, pack_size=1)
    self.assertLen(chunks, 2)
    self.assertEqual(chunks[0][0].num_real_segments, 1)
    self.assertEqual(chunks[1][0].num_real_segments, 1)

  def test_single_max_length_65536_trajectory_fits_without_overflow(self):
    # A single maximum-length trajectory of 65,536 tokens starts at offset=0
    # and fits in a 65,536-token row without overflow (65536 % 64 == 0).
    item = _item(
        np.ones(16384, dtype=np.int32),
        np.ones(49152, dtype=np.int32),
    )
    [[row]] = packing.pack_core([item], budget=65536, pack_size=1)
    self.assertEqual(row.num_real_segments, 1)
    self.assertEqual(row.ids.shape, (65536,))
    self.assertTrue(np.all(row.segment_ids == 1))

  def test_zero_token_item_does_not_consume_max_segments_slot(self):
    items = [
        _item([1], [2], per_token={"returns": np.array([1.0], np.float32)}),
        _item([3], [4], per_token={"returns": np.array([2.0], np.float32)}),
        _item([], []),
    ]
    chunks = packing.pack_core(
        items,
        budget=8,
        pack_size=1,
        max_segments_per_packed_row=2,
        segment_align_multiple=1,
    )
    self.assertLen(chunks, 1)
    [[row]] = chunks
    self.assertEqual(row.num_real_segments, 2)
    np.testing.assert_array_equal(row.ids, [1, 2, 3, 4, 0, 0, 0, 0])
    np.testing.assert_array_equal(row.segment_ids, [1, 1, 2, 2, 0, 0, 0, 0])
    np.testing.assert_allclose(
        row.per_token["returns"], [0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    )

  def test_pack_chunk_returns_packed_chunk_sharing_underlying_buffers(self):
    item = packing.PackItem(
        prompt_ids=np.array([1], dtype=np.int32),
        completion_ids=np.array([2, 3], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.ones(2, dtype=np.float32),
        per_token={"returns": np.array([1.0, 2.0], dtype=np.float32)},
        routed_experts=np.ones((3, 2, 2), dtype=np.int16) * 4,
    )
    chunk = packing.pack_chunk(
        [[item], []],
        budget=8,
        pad_id=0,
        carried=("returns",),
    )
    self.assertIsInstance(chunk, packing.PackedChunk)
    self.assertLen(chunk, 2)
    self.assertEqual(chunk.ids.shape, (2, 8))
    self.assertIsNotNone(chunk.routed_experts)
    self.assertEqual(chunk.routed_experts.shape, (2, 8, 2, 2))
    row0 = chunk[0]
    self.assertTrue(np.shares_memory(row0.ids, chunk.ids))
    self.assertTrue(np.shares_memory(row0.routed_experts, chunk.routed_experts))


def _random_bins(rng, *, budget, segment_align_multiple, pack_size):
  """Random bins, filled as the assemblers fill them (`fill_one_chunk`).

  The items include empty ones and prompt-only or completion-only ones; all of
  them carry the same random subset of the per-token fields, and most of them
  carry routing (an item without it is packed as unset routing).

  Args:
    rng: The random generator.
    budget: Tokens per row.
    segment_align_multiple: The packer's segment alignment.
    pack_size: Rows in the chunk.

  Returns:
    The bins and the per-token fields to pack.
  """
  carried = tuple(f for f in packing.PER_TOKEN_FIELDS if rng.random() < 0.5)
  routed = rng.random() < 0.7
  items = []
  for _ in range(int(rng.integers(1, 10))):
    if rng.random() < 0.1:
      p = c = 0
    else:
      n = int(rng.integers(1, max(2, budget // int(rng.integers(1, 5)))))
      p = int(rng.integers(0, n + 1))
      c = n - p
    items.append(
        packing.PackItem(
            prompt_ids=rng.integers(5, 1000, p).astype(np.int32),
            completion_ids=rng.integers(5, 1000, c).astype(np.int32),
            completion_mask=(rng.random(c) < 0.8).astype(np.float32),
            advantages=rng.standard_normal(c).astype(np.float32),
            per_token={
                f: rng.standard_normal(c).astype(np.float32) for f in carried
            },
            policy_version=np.array([int(rng.integers(0, 9))]),
            routed_experts=(
                rng.integers(0, 64, (p + c, 3, 2)).astype(np.int16)
                if routed and rng.random() < 0.85
                else None
            ),
        )
    )
  bins, _ = packing.fill_one_chunk(
      items,
      pack_size=pack_size,
      budget=budget,
      max_segments=8,
      segment_align_multiple=segment_align_multiple,
  )
  return bins, carried


def _row_arrays(chunk):
  """Every `[rows, budget, ...]` array of a PackedChunk, by name."""
  arrays = {
      name: np.asarray(getattr(chunk, name))
      for name in (
          "ids",
          "prompt_mask",
          "completion_mask",
          "advantages",
          "segment_ids",
          "segment_positions",
      )
  }
  for name, values in chunk.per_token.items():
    arrays[f"per_token/{name}"] = np.asarray(values)
  if chunk.routed_experts is not None:
    arrays["routed_experts"] = np.asarray(chunk.routed_experts)
  return arrays


def _rotated_like_pad_first(end, *, pad_align):
  """The "end" chunk's rows, each rotated right by its rounded trailing padding.

  Computed from the "end" layout alone, not by the packer: this is the
  definition the "start" layout is tested against.

  Args:
    end: A chunk packed with pad_position="end".
    pad_align: The multiple the trailing padding is rounded down to.

  Returns:
    The rotated arrays by name, and the shift of each row.
  """
  arrays = _row_arrays(end)
  budget = arrays["ids"].shape[1]
  shifts = []
  for seg in arrays["segment_ids"]:
    real = np.flatnonzero(seg > 0)
    trailing = budget - 1 - int(real[-1]) if real.size else 0
    shifts.append(trailing // pad_align * pad_align)
  rotated = {
      name: np.stack([np.roll(a[r], s, axis=0) for r, s in enumerate(shifts)])
      for name, a in arrays.items()
  }
  return rotated, shifts


class PadPositionTest(parameterized.TestCase):
  """pad_position="start": each packed row's padding goes in front of it."""

  def _assert_same_chunk(self, got, want):
    got_arrays, want_arrays = _row_arrays(got), _row_arrays(want)
    self.assertEqual(sorted(got_arrays), sorted(want_arrays))
    for name, value in want_arrays.items():
      self.assertEqual(got_arrays[name].dtype, value.dtype, name)
      np.testing.assert_array_equal(got_arrays[name], value, err_msg=name)
    self.assertEqual(got.num_real_segments, want.num_real_segments)
    self.assertEqual(len(got.policy_versions), len(want.policy_versions))
    for a, b in zip(got.policy_versions, want.policy_versions):
      self.assertIs(a, b)

  def test_defaults(self):
    self.assertEqual(packing.DEFAULT_PAD_POSITION, "end")
    self.assertEqual(packing.DEFAULT_PAD_ALIGN, 4096)
    self.assertEqual(packing.PAD_POSITIONS, ("end", "start"))

  def test_start_row_layout(self):
    items = [
        _item([1, 2], [3, 4, 5], adv=1.5),
        _item([6], [7, 8], adv=0.5),
    ]
    [[row]] = packing.pack_core(
        items,
        budget=16,
        pack_size=1,
        segment_align_multiple=1,
        pad_position="start",
        pad_align=4,
    )
    # 8 tokens and 8 of padding: the row moves right by 8.
    np.testing.assert_array_equal(row.ids, [0] * 8 + [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(row.segment_ids, [0] * 8 + [1] * 5 + [2] * 3)
    np.testing.assert_array_equal(
        row.segment_positions, [0] * 8 + [0, 1, 2, 3, 4, 0, 1, 2]
    )
    np.testing.assert_array_equal(
        row.prompt_mask, [0] * 8 + [1, 1, 0, 0, 0, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        row.completion_mask, [0] * 8 + [0, 0, 1, 1, 1, 0, 1, 1]
    )
    np.testing.assert_allclose(
        row.advantages, [0] * 8 + [0, 0, 1.5, 1.5, 1.5, 0, 0.5, 0.5]
    )
    self.assertEqual(row.num_real_segments, 2)

  def test_start_rounds_the_shift_down_to_pad_align(self):
    # 7 tokens leave 9 of padding: pad_align 4 moves the row by 8, and one
    # padding token stays at the end.
    [[row]] = packing.pack_core(
        [_item([1, 2, 3], [4, 5, 6, 7])],
        budget=16,
        pack_size=1,
        segment_align_multiple=1,
        pad_position="start",
        pad_align=4,
    )
    np.testing.assert_array_equal(row.ids, [0] * 8 + [1, 2, 3, 4, 5, 6, 7, 0])
    np.testing.assert_array_equal(row.segment_ids, [0] * 8 + [1] * 7 + [0])

  def test_start_keeps_segment_starts_aligned(self):
    # With the default 64-token alignment the "end" row holds [0, 30) and
    # [64, 84) and 428 tokens of padding; pad_align 128 moves it by 384.
    items = [
        _item(np.arange(1, 11), np.arange(1, 21)),
        _item(np.arange(1, 6), np.arange(1, 16)),
    ]
    [[row]] = packing.pack_core(
        items, budget=512, pack_size=1, pad_position="start", pad_align=128
    )
    np.testing.assert_array_equal(
        np.flatnonzero(row.segment_ids == 1), np.arange(384, 414)
    )
    np.testing.assert_array_equal(
        np.flatnonzero(row.segment_ids == 2), np.arange(448, 468)
    )

  def test_start_moves_by_the_default_pad_align(self):
    # The recipe's setting: "start" with pad_align left at its default. 4096
    # and 4096.0 compare equal, and only an int can index a row, so pack one.
    [[row]] = packing.pack_core(
        [_item(np.arange(1, 11), np.arange(1, 21))],
        budget=8192,
        pack_size=1,
        pad_position="start",
    )
    # 30 tokens leave 8162 of padding: the row moves by 4096.
    np.testing.assert_array_equal(
        np.flatnonzero(row.segment_ids), np.arange(4096, 4126)
    )

  def test_start_measures_the_padding_after_the_last_real_token(self):
    # The shift is the padding after the row's last token, rounded down to
    # pad_align; the end of the row is not rounded up to the segment alignment
    # first. That only matters when the budget is not a multiple of
    # segment_align_multiple: 100 tokens of a 300-token row leave 200 of
    # padding, so the row moves by 192, where rounding its end up to 128 would
    # leave 172 and move it by 128. The empty item goes last, which is where
    # `fill_one_chunk` puts empty items.
    chunk = packing.pack_chunk(
        [[_item(np.arange(1, 41), np.arange(1, 61)), _item([], [])]],
        budget=300,
        pad_id=0,
        carried=(),
        pad_position="start",
        pad_align=64,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(chunk.segment_ids[0]), np.arange(192, 292)
    )

  def test_start_moves_routing_and_per_token_values_with_their_tokens(self):
    routing = np.arange(3 * 2 * 2, dtype=np.int16).reshape(3, 2, 2)
    item = packing.PackItem(
        prompt_ids=np.array([1], dtype=np.int32),
        completion_ids=np.array([2, 3], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.ones(2, dtype=np.float32),
        per_token={"returns": np.array([1.0, 2.0], dtype=np.float32)},
        routed_experts=routing,
    )
    row = packing.pack_bin(
        [item],
        budget=8,
        pad_id=0,
        carried=("returns",),
        segment_align_multiple=1,
        pad_position="start",
        pad_align=4,
    )
    # 3 tokens and 5 of padding: the row moves right by 4.
    np.testing.assert_array_equal(row.ids, [0, 0, 0, 0, 1, 2, 3, 0])
    np.testing.assert_allclose(
        row.per_token["returns"], [0, 0, 0, 0, 0, 1, 2, 0]
    )
    np.testing.assert_array_equal(row.routed_experts[4:7], routing)
    np.testing.assert_array_equal(
        row.routed_experts[:4], packing.UNSET_ROUTED_EXPERT
    )
    np.testing.assert_array_equal(
        row.routed_experts[7:], packing.UNSET_ROUTED_EXPERT
    )

  def test_start_is_the_end_row_rotated_by_its_rounded_padding(self):
    rng = np.random.default_rng(0)
    shifted_rows = 0
    for _ in range(200):
      sam = int(rng.choice([1, 8, 64, 128]))
      # 300 and 1000 are not multiples of every segment alignment.
      budget = int(rng.choice([256, 300, 1000, 1024, 4096, 8192]))
      pad_align = sam * int(rng.choice([1, 2, 4, 16]))
      bins, carried = _random_bins(
          rng,
          budget=budget,
          segment_align_multiple=sam,
          pack_size=int(rng.integers(1, 5)),
      )
      kwargs = dict(
          budget=budget, pad_id=3, carried=carried, segment_align_multiple=sam
      )
      end = packing.pack_chunk(bins, **kwargs)
      start = packing.pack_chunk(
          bins, pad_position="start", pad_align=pad_align, **kwargs
      )
      want, shifts = _rotated_like_pad_first(end, pad_align=pad_align)
      got = _row_arrays(start)
      self.assertEqual(sorted(got), sorted(want))
      for name, value in want.items():
        self.assertEqual(got[name].dtype, value.dtype, name)
        np.testing.assert_array_equal(got[name], value, err_msg=name)
      self.assertEqual(start.num_real_segments, end.num_real_segments)
      for a, b in zip(start.policy_versions, end.policy_versions):
        self.assertIs(a, b)
      for r, shift in enumerate(shifts):
        # Only padding moved to the front: the row's first `shift` positions.
        self.assertFalse(got["segment_ids"][r, :shift].any())
        np.testing.assert_array_equal(got["ids"][r, :shift], 3)
        # Every segment still starts on the segment alignment.
        seg_starts = np.flatnonzero(
            (got["segment_ids"][r] > 0) & (got["segment_positions"][r] == 0)
        )
        np.testing.assert_array_equal(seg_starts % sam, 0)
      shifted_rows += sum(s > 0 for s in shifts)
    # The comparison is not vacuous: many rows did move.
    self.assertGreater(shifted_rows, 100)

  def test_start_keeps_each_token_with_its_values(self):
    # Checked against the items themselves, not against the "end" layout.
    rng = np.random.default_rng(1)
    for _ in range(50):
      bins, carried = _random_bins(
          rng, budget=4096, segment_align_multiple=64, pack_size=3
      )
      chunk = packing.pack_chunk(
          bins,
          budget=4096,
          pad_id=3,
          carried=carried,
          pad_position="start",
          pad_align=512,
      )
      for r, bin_items in enumerate(bins):
        real_items = [item for item in bin_items if item.num_tokens]
        for seg, item in enumerate(real_items, start=1):
          p = item.prompt_ids.shape[0]
          where = np.flatnonzero(chunk.segment_ids[r] == seg)
          self.assertLen(where, item.num_tokens)
          self.assertEqual(where[-1] - where[0] + 1, item.num_tokens)
          np.testing.assert_array_equal(
              chunk.ids[r, where],
              np.concatenate([item.prompt_ids, item.completion_ids]),
          )
          np.testing.assert_array_equal(
              chunk.segment_positions[r, where], np.arange(item.num_tokens)
          )
          np.testing.assert_array_equal(chunk.prompt_mask[r, where[:p]], 1.0)
          np.testing.assert_array_equal(chunk.prompt_mask[r, where[p:]], 0.0)
          np.testing.assert_array_equal(
              chunk.completion_mask[r, where[p:]], item.completion_mask
          )
          np.testing.assert_array_equal(
              chunk.advantages[r, where[p:]], item.advantages
          )
          for name in carried:
            np.testing.assert_array_equal(
                chunk.per_token[name][r, where[p:]], item.per_token[name]
            )
          if item.routed_experts is not None:
            np.testing.assert_array_equal(
                chunk.routed_experts[r, where], item.routed_experts
            )

  def test_end_layout_is_the_same_whatever_pad_align(self):
    # pad_align is not read with "end": no value of it changes or rejects an
    # "end" row, also at segment alignments that do not divide 4096.
    rng = np.random.default_rng(2)
    for _ in range(100):
      sam = int(rng.choice([1, 3, 48, 64, 100]))
      budget = int(rng.choice([300, 1000, 4096]))
      bins, carried = _random_bins(
          rng,
          budget=budget,
          segment_align_multiple=sam,
          pack_size=int(rng.integers(1, 4)),
      )
      kwargs = dict(
          budget=budget, pad_id=3, carried=carried, segment_align_multiple=sam
      )
      default = packing.pack_chunk(bins, **kwargs)
      for pad_align in (packing.DEFAULT_PAD_ALIGN, 0, -1, 100):
        self._assert_same_chunk(
            packing.pack_chunk(
                bins, pad_position="end", pad_align=pad_align, **kwargs
            ),
            default,
        )

  def test_rows_that_cannot_move_are_unchanged(self):
    # A full row, a row whose trailing padding is shorter than pad_align, an
    # empty bin and a bin of empty items.
    bins = [
        [_item(np.arange(1, 5), np.arange(1, 5))],
        [_item([1], [2, 3])],
        [],
        [_item([], [])],
    ]
    kwargs = dict(budget=8, pad_id=0, carried=(), segment_align_multiple=1)
    self._assert_same_chunk(
        packing.pack_chunk(bins, pad_position="start", pad_align=8, **kwargs),
        packing.pack_chunk(bins, **kwargs),
    )

  @parameterized.named_parameters(
      ("unknown_position", "middle", 4096, 64),
      ("capitalized_position", "Start", 4096, 64),
      ("zero_align", "start", 0, 64),
      ("negative_align", "start", -64, 64),
      ("align_not_a_multiple", "start", 100, 64),
      ("align_below_the_segment_alignment", "start", 32, 64),
  )
  def test_invalid_settings_raise(self, pad_position, pad_align, sam):
    items = [_item([1, 2], [3])]
    kwargs = dict(
        pad_position=pad_position,
        pad_align=pad_align,
        segment_align_multiple=sam,
    )
    with self.assertRaisesRegex(ValueError, "pad_"):
      packing.validate_pad_position(pad_position, pad_align, sam)
    with self.assertRaisesRegex(ValueError, "pad_"):
      packing.pack_core(items, budget=8192, **kwargs)
    with self.assertRaisesRegex(ValueError, "pad_"):
      packing.pack_core([], budget=8192, **kwargs)
    with self.assertRaisesRegex(ValueError, "pad_"):
      packing.pack_chunk([items], budget=8192, pad_id=0, carried=(), **kwargs)
    with self.assertRaisesRegex(ValueError, "pad_"):
      packing.pack_bin(items, budget=8192, pad_id=0, carried=(), **kwargs)

  def test_start_accepts_any_multiple_of_the_segment_alignment(self):
    for pad_align, sam in ((64, 64), (4096, 64), (100, 1), (3, 3)):
      packing.validate_pad_position("start", pad_align, sam)


if __name__ == "__main__":
  absltest.main()

