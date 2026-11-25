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

"""Tests for the host-side delta encoder extension."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.delta_sync import _delta_encoder_ext
from tunix.experimental.delta_sync import synthetic_deltas

_PAD = _delta_encoder_ext.PAD_OFFSET


def _random_delta(
    seed: int, rows: int, cols: int, change_rate: float
) -> tuple[np.ndarray, np.ndarray]:
  return synthetic_deltas.generate(
      seed, (rows, cols), block_rows=16, change_rate=change_rate, block_sigma=0
  )


def _reference_encode(
    old: np.ndarray, new: np.ndarray, *, block_rows: int, capacity: int
) -> tuple[np.ndarray, np.ndarray]:
  """Plain NumPy model of the payload layout."""
  rows, cols = old.shape
  changed = old != new
  pair_changed = changed[0::2] | changed[1::2]
  pairs_per_block = block_rows // 2
  offsets = np.full((rows // block_rows, capacity), _PAD, np.int32)
  values = np.zeros_like(offsets)
  for block in range(rows // block_rows):
    first_pair = block * pairs_per_block
    pair, col = np.nonzero(
        pair_changed[first_pair : first_pair + pairs_per_block]
    )
    even = new[2 * (first_pair + pair), col].astype(np.uint32)
    odd = new[2 * (first_pair + pair) + 1, col].astype(np.uint32)
    offsets[block, : pair.size] = pair * cols + col
    values[block, : pair.size] = ((odd << 16) | even).view(np.int32)
  return offsets, values


def _apply(
    bits: np.ndarray, offsets: np.ndarray, values: np.ndarray
) -> np.ndarray:
  """Patches a copy of `bits` the way the rollout side does."""
  rows, cols = bits.shape
  block_rows = rows // offsets.shape[0]
  blocks, slots = np.nonzero(offsets != _PAD)
  word_offsets = offsets[blocks, slots]
  words = values[blocks, slots].view(np.uint32)
  even_rows = blocks * block_rows + 2 * (word_offsets // cols)
  word_cols = word_offsets % cols
  out = bits.copy()
  out[even_rows, word_cols] = (words & 0xFFFF).astype(np.uint16)
  out[even_rows + 1, word_cols] = (words >> 16).astype(np.uint16)
  return out


def _encode(
    old: np.ndarray,
    new: np.ndarray,
    *,
    block_rows: int,
    capacity: int,
    num_threads: int = 2,
    force_scalar: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
  """Encodes one tensor into buffers pre-filled with a sentinel."""
  shape = (old.shape[0] // block_rows, capacity)
  offsets = np.full(shape, 12345, np.int32)
  values = np.full(shape, 12345, np.int32)
  changed_words = _delta_encoder_ext.encode_block_patches(
      [old],
      [new],
      block_rows=[block_rows],
      offsets=[offsets],
      values=[values],
      num_threads=num_threads,
      force_scalar=force_scalar,
  )
  return offsets, values, changed_words


def _read_only(array: np.ndarray) -> np.ndarray:
  array.flags.writeable = False
  return array


_KERNELS = (("simd", False), ("scalar", True))


class EncodeBlockPatchesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_KERNELS)
  def test_a_row_pair_is_one_word_carrying_both_new_halves(self, force_scalar):
    old = np.arange(32 * 128, dtype=np.uint16).reshape(32, 128)
    new = old.copy()
    new[2, 5] = 0xAAAA  # Block 0, pair 1: only the even row.
    new[6, 127], new[7, 127] = 0x1111, 0x2222  # Block 0, pair 3: both rows.
    new[17, 0] = 0xBBBB  # Block 1, pair 0: only the odd row.

    offsets, values, changed_words = _encode(
        old, new, block_rows=16, capacity=128, force_scalar=force_scalar
    )

    self.assertEqual(changed_words, 3)
    np.testing.assert_array_equal(
        offsets[0, :3], [1 * 128 + 5, 3 * 128 + 127, _PAD]
    )
    # The unchanged odd half (old value 3 * 128 + 5 = 389) rides along.
    np.testing.assert_array_equal(
        values[0, :3].view(np.uint32), [389 << 16 | 0xAAAA, 0x22221111, 0]
    )
    np.testing.assert_array_equal(offsets[1, :2], [0, _PAD])
    self.assertEqual(values[1, 0].view(np.uint32), 0xBBBB << 16 | 16 * 128)

  @parameterized.named_parameters(
      ("simd_sparse", False, 256, 128, 16, 0.02),
      ("scalar_sparse", True, 256, 128, 16, 0.02),
      ("simd_wide_blocks", False, 512, 1024, 32, 0.02),
      ("simd_dense", False, 64, 2048, 16, 0.5),
      ("scalar_dense", True, 64, 2048, 16, 0.5),
  )
  def test_payload_matches_the_reference_layout(
      self, force_scalar, rows, cols, block_rows, change_rate
  ):
    old, new = _random_delta(0, rows, cols, change_rate)
    capacity = block_rows // 2 * cols
    want_offsets, want_values = _reference_encode(
        old, new, block_rows=block_rows, capacity=capacity
    )

    offsets, values, changed_words = _encode(
        old,
        new,
        block_rows=block_rows,
        capacity=capacity,
        num_threads=3,
        force_scalar=force_scalar,
    )

    np.testing.assert_array_equal(offsets, want_offsets)
    np.testing.assert_array_equal(values, want_values)
    self.assertEqual(changed_words, np.sum(want_offsets != _PAD))
    np.testing.assert_array_equal(_apply(old, offsets, values), new)

  @parameterized.named_parameters(*_KERNELS)
  def test_blocks_are_sorted_unique_in_range_and_padded_last(
      self, force_scalar
  ):
    old, new = _random_delta(1, 256, 256, change_rate=0.05)

    offsets, values, changed_words = _encode(
        old, new, block_rows=16, capacity=512, force_scalar=force_scalar
    )

    used = offsets != _PAD
    self.assertEqual(changed_words, used.sum())
    for block_offsets, block_values, block_used in zip(offsets, values, used):
      count = block_used.sum()
      # Used slots come first, strictly increasing, inside the block.
      self.assertTrue(block_used[:count].all())
      self.assertTrue((np.diff(block_offsets[:count]) > 0).all())
      self.assertGreaterEqual(block_offsets[:count].min(initial=0), 0)
      self.assertLess(block_offsets[:count].max(initial=0), 8 * 256)
      np.testing.assert_array_equal(block_values[count:], 0)

  @parameterized.named_parameters(*_KERNELS)
  def test_unchanged_weights_give_an_all_padding_payload(self, force_scalar):
    old, _ = _random_delta(2, 64, 128, change_rate=0.0)

    offsets, values, changed_words = _encode(
        old, old.copy(), block_rows=16, capacity=128, force_scalar=force_scalar
    )

    self.assertEqual(changed_words, 0)
    np.testing.assert_array_equal(offsets, _PAD)
    np.testing.assert_array_equal(values, 0)

  @parameterized.named_parameters(*_KERNELS)
  def test_a_block_can_fill_its_capacity_exactly(self, force_scalar):
    old = np.zeros((16, 128), np.uint16)
    new = old.copy()
    new[0] = 1  # 128 changed words: every column of row pair 0.

    offsets, values, changed_words = _encode(
        old, new, block_rows=16, capacity=128, force_scalar=force_scalar
    )

    self.assertEqual(changed_words, 128)
    np.testing.assert_array_equal(offsets[0], np.arange(128))
    np.testing.assert_array_equal(_apply(old, offsets, values), new)

  @parameterized.named_parameters(*_KERNELS)
  def test_encodes_tensors_of_different_shapes_in_one_call(self, force_scalar):
    pairs = [
        _random_delta(3, 64, 2048, 0.02),
        _random_delta(4, 128, 1024, 0.02),
    ]
    block_rows = [16, 32]
    offsets = [np.empty((4, 1024), np.int32), np.empty((4, 2048), np.int32)]
    values = [np.empty_like(o) for o in offsets]

    changed_words = _delta_encoder_ext.encode_block_patches(
        [old for old, _ in pairs],
        [new for _, new in pairs],
        block_rows=block_rows,
        offsets=offsets,
        values=values,
        num_threads=3,
        force_scalar=force_scalar,
    )

    self.assertEqual(changed_words, sum((o != _PAD).sum() for o in offsets))
    for (old, new), got_offsets, got_values in zip(pairs, offsets, values):
      np.testing.assert_array_equal(_apply(old, got_offsets, got_values), new)

  def test_no_tensors_is_not_an_error(self):
    changed_words = _delta_encoder_ext.encode_block_patches(
        [], [], block_rows=[], offsets=[], values=[], num_threads=4
    )

    self.assertEqual(changed_words, 0)


class KernelParityTest(parameterized.TestCase):
  """The vector kernel against the portable one.

  On a CPU without AVX2 both sides run the portable kernel, which makes these
  tests vacuous there; `_delta_encoder_ext.simd_kernel()` tells which ran.
  """

  # Dense rates exercise every 8-lane change mask of the vector kernel.
  @parameterized.parameters(0.05, 0.5, 0.95)
  def test_simd_and_scalar_agree(self, change_rate):
    old, new = _random_delta(5, 1024, 2048, change_rate)
    capacity = 8 * 2048

    simd = _encode(old, new, block_rows=16, capacity=capacity, num_threads=3)
    scalar = _encode(
        old, new, block_rows=16, capacity=capacity, force_scalar=True
    )
    simd_fullest = _delta_encoder_ext.fullest_block_changes(
        [old], [new], block_rows=[16], num_threads=3
    )
    scalar_fullest = _delta_encoder_ext.fullest_block_changes(
        [old], [new], block_rows=[16], num_threads=1, force_scalar=True
    )

    np.testing.assert_array_equal(simd[0], scalar[0])
    np.testing.assert_array_equal(simd[1], scalar[1])
    self.assertEqual(simd[2], scalar[2])
    self.assertEqual(simd_fullest, scalar_fullest)
    self.assertEqual(simd_fullest, [(simd[0] != _PAD).sum(axis=1).max()])


class FullestBlockChangesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_KERNELS)
  def test_reports_the_fullest_block_of_each_tensor(self, force_scalar):
    old = np.zeros((32, 128), np.uint16)
    new = old.copy()
    new[0, :3] = 1  # Block 0: three words.
    new[16, 127] = new[17, 127] = 1  # Block 1: one word, both halves.

    fullest = _delta_encoder_ext.fullest_block_changes(
        [old, old],
        [new, old.copy()],
        block_rows=[16, 16],
        num_threads=2,
        force_scalar=force_scalar,
    )

    self.assertEqual(fullest, [3, 0])

  def test_unsupported_layout_raises_value_error(self):
    bits = np.zeros((64, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, "block_rows=8"):
      _delta_encoder_ext.fullest_block_changes(
          [bits], [bits], block_rows=[8], num_threads=1
      )


class OverflowTest(parameterized.TestCase):

  @parameterized.named_parameters(*_KERNELS)
  def test_overfull_block_raises_overflow_error(self, force_scalar):
    old = np.zeros((32, 128), np.uint16)
    new = old.copy()
    new[16] = 1  # 128 changed words in block 1 ...
    new[18, 0] = 1  # ... and one more.

    with self.assertRaisesRegex(
        OverflowError, "tensor 0: block 1 has 129 changed words"
    ):
      _encode(old, new, block_rows=16, capacity=128, force_scalar=force_scalar)

  @parameterized.parameters(1, 2, 8)
  def test_reports_the_first_overfull_block_at_any_thread_count(
      self, num_threads
  ):
    # Each tensor is its own chunk of work. Four full rows are two row pairs,
    # 256 words, in a block that holds 128.
    old = [np.zeros((64, 128), np.uint16) for _ in range(6)]
    new = [bits.copy() for bits in old]
    new[2][16:20] = 1  # Tensor 2, block 1.
    new[2][48:52] = 1  # Tensor 2, block 3.
    new[4][0:4] = 1  # Tensor 4, block 0.
    offsets = [np.empty((4, 128), np.int32) for _ in old]
    values = [np.empty_like(o) for o in offsets]

    # Repeated, because which thread reaches which tensor first varies.
    for _ in range(20):
      with self.assertRaisesRegex(
          OverflowError, "tensor 2: block 1 has 256 changed words"
      ):
        _delta_encoder_ext.encode_block_patches(
            old,
            new,
            block_rows=[16] * 6,
            offsets=offsets,
            values=values,
            num_threads=num_threads,
        )


class ValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("block_not_whole_tiles", (64, 128), 8, 128, 1, "block_rows=8"),
      ("zero_block_rows", (64, 128), 0, 128, 1, "block_rows=0"),
      ("negative_block_rows", (64, 128), -16, 128, 1, "block_rows=-16"),
      ("rows_not_whole_blocks", (48, 128), 32, 128, 1, "rows=48"),
      ("cols_too_narrow", (16, 64), 16, 128, 1, "cols=64"),
      ("cols_not_power_of_two", (16, 192), 16, 128, 1, "cols=192"),
      ("capacity_not_aligned", (16, 128), 16, 100, 1, "capacity=100"),
      ("no_threads", (16, 128), 16, 128, 0, "num_threads=0"),
  )
  def test_unsupported_layout_raises_value_error(
      self, shape, block_rows, capacity, num_threads, error
  ):
    bits = np.zeros(shape, np.uint16)
    patches = np.empty((max(1, shape[0] // max(1, block_rows)), capacity))

    with self.assertRaisesRegex(ValueError, error):
      _delta_encoder_ext.encode_block_patches(
          [bits],
          [bits.copy()],
          block_rows=[block_rows],
          offsets=[patches.astype(np.int32)],
          values=[patches.astype(np.int32)],
          num_threads=num_threads,
      )

  def test_empty_capacity_raises_value_error(self):
    bits = np.zeros((16, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, "capacity=0"):
      _encode(bits, bits.copy(), block_rows=16, capacity=0)

  def test_reshaped_new_bits_raise_value_error(self):
    old = np.zeros((64, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, r"new_bits\[0\] has shape"):
      _delta_encoder_ext.fullest_block_changes(
          [old],
          [np.zeros((32, 256), np.uint16)],
          block_rows=[16],
          num_threads=1,
      )

  def test_values_shaped_unlike_offsets_raise_value_error(self):
    bits = np.zeros((64, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, r"values\[0\] has shape"):
      _delta_encoder_ext.encode_block_patches(
          [bits],
          [bits.copy()],
          block_rows=[16],
          offsets=[np.empty((4, 256), np.int32)],
          values=[np.empty((8, 128), np.int32)],
          num_threads=1,
      )

  def test_patches_for_the_wrong_block_count_raise_value_error(self):
    bits = np.zeros((64, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, "has 2 rows but .* 4 blocks"):
      _delta_encoder_ext.encode_block_patches(
          [bits],
          [bits.copy()],
          block_rows=[16],
          offsets=[np.empty((2, 128), np.int32)],
          values=[np.empty((2, 128), np.int32)],
          num_threads=1,
      )

  def test_one_block_rows_for_two_tensors_raises_value_error(self):
    bits = np.zeros((16, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, "2 old_bits, 2 new_bits and 1"):
      _delta_encoder_ext.fullest_block_changes(
          [bits, bits], [bits, bits], block_rows=[16], num_threads=1
      )

  def test_one_output_for_two_tensors_raises_value_error(self):
    bits = np.zeros((16, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, "2 tensors but 1 offsets"):
      _delta_encoder_ext.encode_block_patches(
          [bits, bits],
          [bits, bits],
          block_rows=[16, 16],
          offsets=[np.empty((1, 128), np.int32)],
          values=[np.empty((1, 128), np.int32)],
          num_threads=1,
      )

  def test_shared_output_array_raises_value_error(self):
    bits = np.zeros((16, 128), np.uint16)
    buffer = np.empty((1, 128), np.int32)

    with self.assertRaisesRegex(ValueError, "must not share memory"):
      _delta_encoder_ext.encode_block_patches(
          [bits],
          [bits.copy()],
          block_rows=[16],
          offsets=[buffer],
          values=[buffer],
          num_threads=1,
      )

  def test_partly_overlapping_outputs_raise_value_error(self):
    bits = np.zeros((16, 128), np.uint16)
    memory = np.empty(192, np.int32)

    with self.assertRaisesRegex(ValueError, "must not share memory"):
      _delta_encoder_ext.encode_block_patches(
          [bits],
          [bits.copy()],
          block_rows=[16],
          offsets=[memory[:128].reshape(1, 128)],
          values=[memory[64:].reshape(1, 128)],
          num_threads=1,
      )

  def test_output_shared_between_tensors_raises_value_error(self):
    bits = np.zeros((16, 128), np.uint16)
    shared = np.empty((1, 128), np.int32)

    with self.assertRaisesRegex(ValueError, "must not share memory"):
      _delta_encoder_ext.encode_block_patches(
          [bits, bits],
          [bits.copy(), bits.copy()],
          block_rows=[16, 16],
          offsets=[shared, np.empty((1, 128), np.int32)],
          values=[np.empty((1, 128), np.int32), shared],
          num_threads=1,
      )

  def test_weights_may_share_memory_with_each_other(self):
    bits = np.zeros((16, 128), np.uint16)

    offsets, _, changed_words = _encode(bits, bits, block_rows=16, capacity=128)

    self.assertEqual(changed_words, 0)
    np.testing.assert_array_equal(offsets, _PAD)

  def test_output_inside_the_weights_raises_value_error(self):
    memory = np.zeros(16 * 128 + 256, np.uint16)
    bits = memory[: 16 * 128].reshape(16, 128)
    # 128 int32 words over the last 256 uint16 weights.
    inside = memory[15 * 128 : 17 * 128].view(np.int32).reshape(1, 128)

    with self.assertRaisesRegex(ValueError, "share memory with the weights"):
      _delta_encoder_ext.encode_block_patches(
          [bits],
          [bits.copy()],
          block_rows=[16],
          offsets=[inside],
          values=[np.empty((1, 128), np.int32)],
          num_threads=1,
      )

  @parameterized.named_parameters(
      ("float_weights", np.zeros((16, 128), np.float32)),
      ("int16_weights", np.zeros((16, 128), np.int16)),
      ("flat_weights", np.zeros((16 * 128,), np.uint16)),
      ("strided_weights", np.zeros((16, 256), np.uint16)[:, ::2]),
      ("fortran_order_weights", np.asfortranarray(np.zeros((16, 128), "u2"))),
      ("byte_swapped_weights", np.zeros((16, 128), ">u2")),
      ("list_weights", [[0] * 128] * 16),
  )
  def test_weights_that_would_need_a_copy_raise_type_error(self, bits):
    good = np.zeros((16, 128), np.uint16)

    with self.subTest("old_bits"):
      with self.assertRaises(TypeError):
        _delta_encoder_ext.fullest_block_changes(
            [bits], [good], block_rows=[16], num_threads=1
        )
    with self.subTest("new_bits"):
      with self.assertRaises(TypeError):
        _delta_encoder_ext.fullest_block_changes(
            [good], [bits], block_rows=[16], num_threads=1
        )

  @parameterized.named_parameters(
      ("int64_output", np.zeros((1, 128), np.int64)),
      ("strided_output", np.zeros((1, 256), np.int32)[:, ::2]),
      ("fortran_order_output", np.asfortranarray(np.zeros((2, 128), "i4"))),
      ("byte_swapped_output", np.zeros((1, 128), ">i4")),
      ("read_only_output", _read_only(np.zeros((1, 128), np.int32))),
  )
  def test_outputs_that_cannot_be_filled_in_place_raise_type_error(
      self, output
  ):
    bits = np.zeros((16, 128), np.uint16)
    good = np.empty((1, 128), np.int32)

    with self.subTest("offsets"):
      with self.assertRaises(TypeError):
        _delta_encoder_ext.encode_block_patches(
            [bits],
            [bits.copy()],
            block_rows=[16],
            offsets=[output],
            values=[good],
            num_threads=1,
        )
    with self.subTest("values"):
      with self.assertRaises(TypeError):
        _delta_encoder_ext.encode_block_patches(
            [bits],
            [bits.copy()],
            block_rows=[16],
            offsets=[good],
            values=[output],
            num_threads=1,
        )

  def test_misaligned_weights_are_rejected(self):
    # uint16 data starting at an odd address.
    misaligned = np.frombuffer(
        bytearray(2 * 16 * 128 + 1), np.uint16, offset=1
    ).reshape(16, 128)
    good = np.zeros((16, 128), np.uint16)

    with self.assertRaises((TypeError, ValueError)):
      _delta_encoder_ext.fullest_block_changes(
          [misaligned], [good], block_rows=[16], num_threads=1
      )

  def test_a_bare_array_instead_of_a_list_raises_type_error(self):
    bits = np.zeros((16, 128), np.uint16)

    with self.assertRaises(TypeError):
      _delta_encoder_ext.fullest_block_changes(
          bits, bits, block_rows=[16], num_threads=1
      )


if __name__ == "__main__":
  absltest.main()
