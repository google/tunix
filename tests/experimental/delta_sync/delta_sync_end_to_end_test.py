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

"""Tests the host-side encoder together with the SparseCore apply kernel."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np
from tunix.experimental.delta_sync import _delta_encoder_ext
from tunix.experimental.delta_sync import sparsecore_delta_apply as sda
from tunix.experimental.delta_sync import synthetic_deltas

# Set to make the tests that need a TPU fail, rather than skip, without one.
# An environment variable, not a flag: test runners that load several test
# files into one process would define a flag twice.
_REQUIRE_TPU_ENV = "TUNIX_REQUIRE_TPU"


def _random_delta(
    seed: int, rows: int, cols: int, change_rate: float
) -> tuple[np.ndarray, np.ndarray]:
  return synthetic_deltas.generate(
      seed, (rows, cols), block_rows=16, change_rate=change_rate, block_sigma=0
  )


def _encode(
    old: np.ndarray, new: np.ndarray, *, block_rows: int, capacity: int
) -> tuple[np.ndarray, np.ndarray]:
  """Encodes one tensor with the C++ encoder."""
  shape = (old.shape[0] // block_rows, capacity)
  offsets = np.empty(shape, np.int32)
  values = np.empty(shape, np.int32)
  _delta_encoder_ext.encode_block_patches(
      [old],
      [new],
      block_rows=[block_rows],
      offsets=[offsets],
      values=[values],
      num_threads=2,
  )
  return offsets, values


class EncoderMatchesKernelReferenceTest(parameterized.TestCase):

  def test_constants_match_the_apply_kernel(self):
    self.assertEqual(_delta_encoder_ext.PAD_OFFSET, sda.PAD_OFFSET)
    self.assertEqual(_delta_encoder_ext.CAPACITY_ALIGN, sda.CAPACITY_ALIGN)

  @parameterized.parameters(
      (256, 128, 16, 0.02), (512, 1024, 32, 0.02), (64, 2048, 16, 0.5)
  )
  def test_payload_matches_the_kernel_reference_encoder(
      self, rows, cols, block_rows, change_rate
  ):
    old, new = _random_delta(0, rows, cols, change_rate)
    capacity = block_rows // 2 * cols
    want_offsets, want_values = sda.encode_block_patches(
        old, new, block_rows=block_rows, capacity=capacity
    )

    offsets, values = _encode(
        old, new, block_rows=block_rows, capacity=capacity
    )

    np.testing.assert_array_equal(offsets, want_offsets)
    np.testing.assert_array_equal(values, want_values)


class EncodeThenApplyTest(parameterized.TestCase):
  """Encodes on the host and applies with the kernel; needs a TPU."""

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

  @parameterized.parameters((1024, 32, 0.02), (2048, 16, 0.02), (1024, 32, 0.0))
  def test_applying_the_encoded_payload_gives_the_new_weights(
      self, cols, block_rows, change_rate
  ):
    rows = 4 * self.num_subcores * block_rows
    old, new = _random_delta(3, rows, cols, change_rate)
    fullest = _delta_encoder_ext.fullest_block_changes(
        [old], [new], block_rows=[block_rows], num_threads=2
    )
    capacity = synthetic_deltas.capacities_by_shape([(rows, cols)], fullest)[0]
    offsets, values = _encode(
        old, new, block_rows=block_rows, capacity=capacity
    )
    weights = jnp.asarray(old.view(jnp.bfloat16))

    patched = sda.apply_block_patches(
        weights, offsets=jnp.asarray(offsets), values=jnp.asarray(values)
    )

    np.testing.assert_array_equal(np.asarray(patched).view(np.uint16), new)


if __name__ == "__main__":
  absltest.main()
