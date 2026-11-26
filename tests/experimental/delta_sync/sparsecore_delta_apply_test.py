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

"""Tests for the SparseCore row-pair delta apply."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np
from tunix.experimental.delta_sync import sparsecore_delta_apply as sda

# Set to make the tests that need a TPU fail, rather than skip, without one.
# An environment variable, not a flag: test runners that load several test
# files into one process would define a flag twice.
_REQUIRE_TPU_ENV = "TUNIX_REQUIRE_TPU"

# bf16 bit patterns that a value-based copy could alter: signed zeros,
# subnormals, infinities, and quiet and signaling NaNs with distinct payloads.
_SPECIAL_BITS = np.array(
    [
        0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80,
        0x7FC1, 0x7FC2, 0xFFC1, 0x7F81, 0x7FFF, 0xFFFF,
    ],
    np.uint16,
)  # pyformat: disable


def _random_delta(
    seed: int, rows: int, cols: int, change_rate: float
) -> tuple[np.ndarray, np.ndarray]:
  """bf16 bits of N(0, 1) weights, and a copy with some of them nudged."""
  rng = np.random.default_rng(seed)
  weights = rng.standard_normal((rows, cols), dtype=np.float32)
  old = (weights.view(np.uint32) >> 16).astype(np.uint16)
  # Flipping low mantissa bits always changes a value and keeps it finite.
  nudges = rng.integers(
      1, 0x7F, size=(rows, cols), dtype=np.uint16, endpoint=True
  )
  new = np.where(rng.random((rows, cols)) < change_rate, old ^ nudges, old)
  return old, new


def _apply_reference(
    bits: np.ndarray, offsets: np.ndarray, values: np.ndarray
) -> np.ndarray:
  """NumPy model of the kernel: one 32-bit row-pair store per used slot."""
  rows, cols = bits.shape
  block_rows = rows // offsets.shape[0]
  blocks, slots = np.nonzero(offsets != sda.PAD_OFFSET)
  word_offsets = offsets[blocks, slots]
  words = values[blocks, slots].view(np.uint32)
  even_rows = blocks * block_rows + 2 * (word_offsets // cols)
  word_cols = word_offsets % cols
  out = bits.copy()
  out[even_rows, word_cols] = (words & 0xFFFF).astype(np.uint16)
  out[even_rows + 1, word_cols] = (words >> 16).astype(np.uint16)
  return out


class EncodeBlockPatchesTest(parameterized.TestCase):

  @parameterized.parameters((256, 128, 16), (256, 256, 32))
  def test_patches_rebuild_the_new_bits(self, rows, cols, block_rows):
    old, new = _random_delta(0, rows, cols, change_rate=0.02)

    offsets, values = sda.encode_block_patches(
        old, new, block_rows=block_rows, capacity=256
    )

    np.testing.assert_array_equal(_apply_reference(old, offsets, values), new)

  def test_reapplying_a_patch_changes_nothing(self):
    old, new = _random_delta(1, 64, 128, change_rate=0.02)

    offsets, values = sda.encode_block_patches(
        old, new, block_rows=16, capacity=128
    )

    np.testing.assert_array_equal(_apply_reference(new, offsets, values), new)

  def test_a_row_pair_is_one_word_carrying_both_new_halves(self):
    old = np.zeros((16, 128), np.uint16)
    new = old.copy()
    new[4, 7], new[5, 7] = 0x3F80, 0xBF80  # 1.0 and -1.0 in one row pair.
    new[9, 0] = 0x4000  # Only the odd row of the pair (8, 9).

    offsets, values = sda.encode_block_patches(
        old, new, block_rows=16, capacity=128
    )

    np.testing.assert_array_equal(offsets[0, :2], [2 * 128 + 7, 4 * 128])
    np.testing.assert_array_equal(
        values[0, :2].view(np.uint32), [0xBF803F80, 0x40000000]
    )
    np.testing.assert_array_equal(offsets[0, 2:], sda.PAD_OFFSET)
    np.testing.assert_array_equal(values[0, 2:], 0)

  def test_untouched_blocks_are_all_padding(self):
    old = np.ones((64, 128), np.uint16)
    new = old.copy()
    new[40, 3] = 7  # Block 2, local row 8: pair 4.

    offsets, _ = sda.encode_block_patches(old, new, block_rows=16, capacity=128)

    np.testing.assert_array_equal(offsets[[0, 1, 3]], sda.PAD_OFFSET)
    np.testing.assert_array_equal(offsets[2, :2], [4 * 128 + 3, sda.PAD_OFFSET])

  def test_unchanged_weights_give_an_all_padding_patch(self):
    old, _ = _random_delta(3, 64, 128, change_rate=0.0)

    offsets, values = sda.encode_block_patches(
        old, old, block_rows=16, capacity=128
    )

    np.testing.assert_array_equal(offsets, sda.PAD_OFFSET)
    np.testing.assert_array_equal(values, 0)

  def test_a_block_can_fill_its_capacity_exactly(self):
    old = np.zeros((16, 128), np.uint16)
    new = old.copy()
    new[0] = 1  # 128 changed words: every column of row pair 0.

    offsets, values = sda.encode_block_patches(
        old, new, block_rows=16, capacity=128
    )

    np.testing.assert_array_equal(offsets[0], np.arange(128))
    np.testing.assert_array_equal(_apply_reference(old, offsets, values), new)

  def test_overfull_block_is_an_error(self):
    old = np.zeros((16, 128), np.uint16)

    with self.assertRaisesRegex(ValueError, "1024 changed words"):
      sda.encode_block_patches(old, old + 1, block_rows=16, capacity=128)

  @parameterized.named_parameters(
      ("rows_not_whole_blocks", (40, 128), 16, 128, "rows=40"),
      ("block_not_whole_tiles", (64, 128), 8, 128, "block_rows=8"),
      ("cols_not_power_of_two", (32, 384), 16, 128, "cols=384"),
      ("capacity_not_aligned", (32, 128), 16, 100, "capacity=100"),
  )
  def test_unsupported_layout_is_an_error(
      self, shape, block_rows, capacity, error
  ):
    bits = np.zeros(shape, np.uint16)

    with self.assertRaisesRegex(ValueError, error):
      sda.encode_block_patches(
          bits, bits, block_rows=block_rows, capacity=capacity
      )

  def test_mismatched_shapes_are_an_error(self):
    with self.assertRaisesRegex(ValueError, "same shape"):
      sda.encode_block_patches(
          np.zeros((16, 128), np.uint16),
          np.zeros((32, 128), np.uint16),
          block_rows=16,
          capacity=128,
      )

  def test_values_instead_of_bit_patterns_is_an_error(self):
    weights = np.zeros((16, 128), np.float32)

    with self.assertRaisesRegex(ValueError, "uint16"):
      sda.encode_block_patches(weights, weights, block_rows=16, capacity=128)


class ApplyBlockPatchesValidationTest(parameterized.TestCase):
  """Checks that run while tracing, before any TPU is needed."""

  @parameterized.named_parameters(
      (
          "float32_weights",
          {"weights": np.zeros((64, 128), np.float32)},
          "bf16 matrix",
      ),
      (
          "batched_weights",
          {"weights": np.zeros((1, 64, 128), jnp.bfloat16)},
          "bf16 matrix",
      ),
      (
          "int16_offsets",
          {"offsets": np.zeros((4, 128), np.int16)},
          "int32 offsets",
      ),
      (
          "mismatched_payload",
          {"values": np.zeros((4, 256), np.int32)},
          "of the same",
      ),
      (
          "rows_not_whole_blocks",
          {
              "offsets": np.zeros((3, 128), np.int32),
              "values": np.zeros((3, 128), np.int32),
          },
          "do not evenly split",
      ),
      (
          "capacity_not_aligned",
          {
              "offsets": np.zeros((4, 100), np.int32),
              "values": np.zeros((4, 100), np.int32),
          },
          "capacity=100",
      ),
      (
          "int16_values",
          {"values": np.zeros((4, 128), np.int16)},
          "int32 offsets",
      ),
      (
          "no_blocks",
          {
              "offsets": np.zeros((0, 128), np.int32),
              "values": np.zeros((0, 128), np.int32),
          },
          "do not evenly split",
      ),
      (
          "block_not_whole_tiles",
          {
              "offsets": np.zeros((8, 128), np.int32),
              "values": np.zeros((8, 128), np.int32),
          },
          "block_rows=8",
      ),
      (
          "cols_too_narrow",
          {"weights": np.zeros((64, 64), jnp.bfloat16)},
          "cols=64",
      ),
      (
          "cols_not_power_of_two",
          {"weights": np.zeros((64, 192), jnp.bfloat16)},
          "cols=192",
      ),
  )
  def test_bad_input_is_an_error(self, overrides, error):
    inputs = {
        "weights": np.zeros((64, 128), jnp.bfloat16),
        "offsets": np.zeros((4, 128), np.int32),
        "values": np.zeros((4, 128), np.int32),
    } | overrides

    with self.assertRaisesRegex(ValueError, error):
      sda.apply_block_patches(
          inputs["weights"], offsets=inputs["offsets"], values=inputs["values"]
      )


class SparseCoreKernelTest(parameterized.TestCase):
  """Runs the kernel; needs a TPU with SparseCores."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      if _REQUIRE_TPU_ENV in os.environ:
        self.fail(
            f"{_REQUIRE_TPU_ENV} is set but the default backend is not a TPU"
        )
      self.skipTest("needs a TPU with SparseCores")
    info = plsc.get_sparse_core_info()
    self.num_subcores = info.num_cores * info.num_subcores
    self.vmem_bytes = info.vmem_capacity_bytes

  @parameterized.parameters((1024, 32), (2048, 16))
  def test_kernel_patches_in_place_and_reapplying_is_a_no_op(
      self, cols, block_rows
  ):
    # Four blocks per subcore: two steps of the double-buffered loop.
    rows = 4 * self.num_subcores * block_rows
    old, new = _random_delta(2, rows, cols, change_rate=0.02)
    new[: 3 * block_rows] = old[: 3 * block_rows]  # Blocks with no changes.
    # The first and the last word of every other block.
    first = np.arange(3 * block_rows, rows, block_rows)
    last = first + block_rows - 1
    new[first, 0] = old[first, 0] ^ 1
    new[last, -1] = old[last, -1] ^ 1
    offsets, values = sda.encode_block_patches(
        old, new, block_rows=block_rows, capacity=1024
    )
    offsets, values = jnp.asarray(offsets), jnp.asarray(values)
    weights = lax.bitcast_convert_type(jnp.asarray(old), jnp.bfloat16)
    buffer = weights.unsafe_buffer_pointer()

    patched = sda.apply_block_patches(weights, offsets=offsets, values=values)
    patched_bits = np.asarray(lax.bitcast_convert_type(patched, jnp.uint16))
    repatched = sda.apply_block_patches(patched, offsets=offsets, values=values)
    repatched_bits = np.asarray(lax.bitcast_convert_type(repatched, jnp.uint16))

    self.assertTrue(weights.is_deleted())
    self.assertEqual(repatched.unsafe_buffer_pointer(), buffer)
    np.testing.assert_array_equal(patched_bits, new)
    np.testing.assert_array_equal(repatched_bits, new)

  def test_kernel_writes_special_bit_patterns_unchanged(self):
    block_rows, cols, capacity = 16, 128, 256
    # Two blocks per subcore: a single step of the double-buffered loop.
    rows = 2 * self.num_subcores * block_rows
    rng = np.random.default_rng(4)
    old = rng.choice(_SPECIAL_BITS, size=(rows, cols))
    new = old.copy()
    changed = rng.random((rows, cols)) < 0.05
    new[changed] = rng.choice(_SPECIAL_BITS, size=changed.sum())
    # Block 0 changes exactly `capacity` words: two whole row pairs.
    new[:block_rows] = old[:block_rows]
    new[0] = old[0] ^ 0x8000
    new[2] = old[2] ^ 0x8000
    # Block 1 does not change at all.
    new[block_rows : 2 * block_rows] = old[block_rows : 2 * block_rows]
    offsets, values = sda.encode_block_patches(
        old, new, block_rows=block_rows, capacity=capacity
    )
    # The bits are reinterpreted on the host. A bf16 -> uint16 bitcast on the
    # TPU quiets NaN payloads and flushes subnormals to zero, which would hide
    # what the kernel wrote.
    weights = jnp.asarray(old.view(jnp.bfloat16))

    patched = sda.apply_block_patches(
        weights, offsets=jnp.asarray(offsets), values=jnp.asarray(values)
    )
    patched_bits = np.asarray(patched).view(np.uint16)

    self.assertTrue((offsets[0] != sda.PAD_OFFSET).all())
    self.assertTrue((offsets[1] == sda.PAD_OFFSET).all())
    np.testing.assert_array_equal(patched_bits, new)

  def test_blocks_that_do_not_pair_up_over_subcores_are_an_error(self):
    weights = jnp.zeros((self.num_subcores * 16, 128), jnp.bfloat16)
    payload = jnp.zeros((self.num_subcores, 128), jnp.int32)

    with self.assertRaisesRegex(ValueError, "do not split into pairs"):
      sda.apply_block_patches(weights, offsets=payload, values=payload)

  def test_block_too_big_for_vmem_is_an_error(self):
    cols = self.vmem_bytes // 32  # Two 16-row bf16 blocks need 2x the VMEM.
    weights = jnp.zeros((2 * self.num_subcores * 16, cols), jnp.bfloat16)
    payload = jnp.zeros((2 * self.num_subcores, 128), jnp.int32)

    with self.assertRaisesRegex(ValueError, "VMEM"):
      sda.apply_block_patches(weights, offsets=payload, values=payload)


if __name__ == "__main__":
  absltest.main()
