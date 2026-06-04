# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Gemma 4 image processor port (deterministic parts only)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from tunix.models.gemma4 import image_processing as ip
from tunix.models.gemma4 import vision_real


class ImageProcessingTest(absltest.TestCase):

  def test_target_size_divisible_and_within_budget(self):
    th, tw = ip.get_aspect_ratio_preserving_size(
        height=480, width=640, patch_size=16, max_patches=2520, pooling_kernel_size=3
    )
    self.assertEqual(th % 48, 0)
    self.assertEqual(tw % 48, 0)
    self.assertLessEqual((th // 16) * (tw // 16), 2520)

  def test_patchify_shape_and_order(self):
    # 2x3 patch grid, patch_size=2, single channel ramp to check ordering.
    patch_size = 2
    img = np.arange(1 * 4 * 6).reshape(1, 4, 6).astype(np.float32)  # (C,H,W)
    patches = ip.convert_image_to_patches(img, patch_size)
    self.assertEqual(patches.shape, (2 * 3, patch_size * patch_size * 1))
    # First patch is the top-left 2x2 block: rows [0,1] cols [0,1].
    np.testing.assert_array_equal(patches[0], np.array([0, 1, 6, 7]))

  def test_position_ids_are_xy_and_match_patch_order(self):
    pos = ip.build_position_ids(patch_height=2, patch_width=3)
    self.assertEqual(pos.shape, (6, 2))
    # Row-major over (row, col); stored as (x=col, y=row).
    np.testing.assert_array_equal(
        pos, np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    )

  def test_padding_uses_minus_one_for_positions(self):
    patches = np.ones((4, 12), dtype=np.float32)
    positions = np.zeros((4, 2), dtype=np.int64)
    patches, positions = ip.pad_along_first_dim(patches, positions, target_length=6)
    self.assertEqual(patches.shape, (6, 12))
    np.testing.assert_array_equal(patches[4:], 0)
    np.testing.assert_array_equal(positions[4:], -1)

  def test_end_to_end_into_vision_stack(self):
    # A 48x48 image -> 3x3 patches (9 patches) -> pool 3x3 -> 1 soft token.
    proc = ip.Gemma4ImageProcessor(patch_size=16, pooling_kernel_size=3, max_soft_tokens=32)
    img = (np.random.rand(3, 48, 48) * 255).astype(np.uint8)
    px, pos, num_soft = proc([img], do_resize=False)

    self.assertEqual(px.shape, (1, proc.max_patches, 3 * 16**2))
    self.assertEqual(pos.shape, (1, proc.max_patches, 2))
    self.assertEqual(num_soft, [1])

    cfg = vision_real.Gemma4VisionConfig(num_hidden_layers=1)
    stack = vision_real.Gemma4VisionStack(cfg, text_hidden_size=1536, rngs=nnx.Rngs(0))
    soft, mask = stack(jnp.asarray(px), jnp.asarray(pos))
    # max_patches=288 -> /9 = 32 pooled tokens; mask marks the 1 real soft token.
    self.assertEqual(soft.shape, (1, proc.max_patches // 9, 1536))
    self.assertEqual(int(mask.sum()), 1)


if __name__ == "__main__":
  absltest.main()
