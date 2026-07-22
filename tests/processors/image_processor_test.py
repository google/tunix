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

import dataclasses
import os
import shutil
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tunix.models.gemma4 import vision
from tunix.processors import image_processor


@dataclasses.dataclass(slots=True, kw_only=True)
class DummyConfig:

  image_height: int = 32
  image_width: int = 32
  image_channels: int = 3
  image_mean: tuple[float, ...] = (127.5, 127.5, 127.5)
  image_std: tuple[float, ...] = (127.5, 127.5, 127.5)


class ImageProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.height = 32
    self.width = 32
    self.channels = 3
    config = DummyConfig(
        image_height=self.height,
        image_width=self.width,
        image_channels=self.channels,
    )
    self.processor = image_processor.ImageProcessor(config)

  def _create_dummy_image_file(self, filename='test_image.png'):
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # TODO(abheesht17): Use self.create_tempdir(). It was failing on GitHub CI,
    # but revisit.
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir))

    temp_file = os.path.join(temp_dir, filename)
    img.save(temp_file)
    return temp_file

  def test_process_none_image(self):
    processed_image = self.processor.preprocess_image(None)
    np.testing.assert_array_equal(
        processed_image, np.zeros((self.height, self.width, 3))
    )

  def test_path_input(self):
    img_path = self._create_dummy_image_file()
    processed_image = self.processor.preprocess_image(img_path)
    np.testing.assert_allclose(
        processed_image, -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_array_input(self):
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    processed_image = self.processor.preprocess_image(img_array)
    np.testing.assert_allclose(
        processed_image, -1.0 * np.ones((self.height, self.width, 3))
    )

  @parameterized.product(
      input_type=['array', 'path'],
      is_dim_0=[True, False],
  )
  def test_single_image(self, input_type, is_dim_0):
    if input_type == 'array':
      images = np.zeros((100, 100, 3), dtype=np.uint8)
    elif input_type == 'path':
      images = self._create_dummy_image_file()
    else:
      raise ValueError(f'Invalid input_type: {input_type}')

    if not is_dim_0:
      images = [images]

    processed_images = self.processor(images=images)
    self.assertLen(processed_images, 1)
    self.assertLen(processed_images[0], 1)
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_multiple_images_dim_1(self):
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    images = [img1, img2]
    processed_images = self.processor(images=images)  # pyrefly: ignore[bad-argument-type]
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )

  @parameterized.named_parameters(
      {'testcase_name': 'all_dim_1', 'input_type': 'all_dim_1'},
      {'testcase_name': 'mixed', 'input_type': 'mixed'},
  )
  def test_padding(self, input_type):
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)

    if input_type == 'all_dim_1':
      images = [[img1], [img1, img2]]
    else:
      images = [img1, [img1, img2]]

    processed_images = self.processor(images=images)  # pyrefly: ignore[bad-argument-type]
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    # Padded image should be zeros
    np.testing.assert_allclose(
        processed_images[0][1], np.zeros((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][1], -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_mixed_inputs(self):
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = self._create_dummy_image_file()
    images = [img1, [img1, img2]]
    processed_images = self.processor(images=images)  # pyrefly: ignore[bad-argument-type]
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[0][1], np.zeros((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][1], -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_call_with_none_in_batch(self):
    images = [None, [np.zeros((100, 100, 3), dtype=np.uint8)]]
    processed_images = self.processor(images=images)  # pyrefly: ignore[bad-argument-type]
    np.testing.assert_allclose(
        processed_images[0][0], np.zeros((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_preprocess_and_patchify_parity(self):
    np.random.seed(0)
    raw_images = [
        np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
        np.random.randint(0, 256, (64, 96, 3), dtype=np.uint8),
    ]

    # 1. Run local implementation
    patches, positions, counts = image_processor.preprocess_and_patchify(
        raw_images,
        patch_size=4,
        max_soft_tokens=10,
        pooling_kernel_size=3,
    )

    # Check outputs against expected pre-calculated values (same as upstream)
    self.assertEqual(patches.shape, (2, 90, 48))
    np.testing.assert_allclose(
        np.mean(patches), 0.37482523918151855, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.std(patches), 0.22608762979507446, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.sum(patches), 3238.490234375, atol=1e-3, rtol=1e-3
    )

    self.assertEqual(positions.shape, (2, 90, 2))
    # Assert specific position coordinates
    np.testing.assert_array_equal(
        positions[0, :5], [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    )
    np.testing.assert_array_equal(positions[0, 81:], [[-1, -1]] * 9)
    np.testing.assert_array_equal(
        positions[1, :5], [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    )
    np.testing.assert_array_equal(positions[1, 54:], [[-1, -1]] * 36)

    self.assertEqual(counts, [9, 6])

  def test_add_variable_extra_tokens_for_images_parity(self):
    tokens = np.array([
        [1, 2, 258880, 4, 5],
        [10, 258880, 20, 258880, 0],
    ])
    soft_token_counts = [3, 2, 5]

    # Run local implementation
    tunix_expanded = image_processor.add_variable_extra_tokens_for_images(
        tokens,
        soft_token_counts=soft_token_counts,
    )

    # Check outputs against expected pre-calculated values (same as upstream)
    expected = np.array(
        [
            [1, 2, 108, 255999, -2, -2, -2, 258882, 108, 4, 5, 0, 0, 0, 0, 0],
            [
                10,
                108,
                255999,
                -2,
                -2,
                -2,
                258882,
                108,
                20,
                108,
                255999,
                -2,
                -2,
                258882,
                108,
                0,
            ],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(tunix_expanded, expected)

  def test_factorized_posemb(self):
    batch_size, seq_len, dim = 2, 4, 6
    pos_emb_size = 5
    posemb = jnp.arange(pos_emb_size * 2 * dim, dtype=jnp.float32).reshape(
        (pos_emb_size, 2, dim)
    )
    positions_xy = jnp.array([
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 4], [6, 0], [-1, -1], [-10, -10]],
    ])

    # Manual calculation for the first element [0, 0]
    # x=0, y=0
    pe_x_00 = posemb[0, 0, :]
    pe_y_00 = posemb[0, 1, :]
    expected_00 = pe_x_00 + pe_y_00

    # Manual calculation for the second element [1, 1]
    # x=1, y=1
    pe_x_11 = posemb[1, 0, :]
    pe_y_11 = posemb[1, 1, :]
    expected_11 = pe_x_11 + pe_y_11

    result = vision.factorized_posemb(posemb, positions_xy)

    self.assertEqual(result.shape, (batch_size, seq_len, dim))
    np.testing.assert_allclose(result[0, 0, :], expected_00)
    np.testing.assert_allclose(result[0, 1, :], expected_11)
    # Check for zeroed padding values - second item, third element [-1, -1]
    self.assertEqual(jnp.sum(result[1, 2, :]), 0)
    # Check NaN for OOB positive value - second item, second element [6, 0]
    self.assertTrue(jnp.all(jnp.isnan(result[1, 1, :])))
    # Check NaN for OOB negative value - second item, last element [-10, -10]
    self.assertTrue(jnp.all(jnp.isnan(result[1, -1, :])))

  def test_patchify(self):
    images = jnp.arange(2 * 32 * 32 * 3, dtype=jnp.float32).reshape(
        (2, 32, 32, 3)
    )
    patch_size = 16
    patches, positions_xy = image_processor.patchify(images, patch_size)

    self.assertEqual(patches.shape, (2, 4, 16 * 16 * 3))
    self.assertEqual(positions_xy.shape, (2, 4, 2))

    # Expected positions: (x, y)
    # (0,0), (1,0), (0,1), (1,1)
    expected_positions = jnp.array([
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        [[0, 0], [1, 0], [0, 1], [1, 1]],
    ])
    np.testing.assert_array_equal(positions_xy, expected_positions)

    # Check patch content for the first patch [0, 0]
    expected_patch_00 = images[0, :16, :16, :].reshape(-1)
    np.testing.assert_array_equal(patches[0, 0, :], expected_patch_00)

    # Check patch content for the second patch [1, 0]
    expected_patch_10 = images[0, :16, 16:, :].reshape(-1)
    np.testing.assert_array_equal(patches[0, 1, :], expected_patch_10)

  def test_process_gemma4_inputs_batch(self):
    class _DummyVisionConfig:
      patch_size = 16
      num_mm_tokens_per_image = 10
      pooling_kernel_size = 3

    class _DummyVisionEncoder:
      config = _DummyVisionConfig()

    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    img3 = np.zeros((200, 200, 3), dtype=np.uint8)

    images = [[img1, img2], [img3]]
    tokens = [
        np.array([1, 2, 258880, 258880, 4, 5]),
        np.array([10, 258880, 20]),
    ]
    vision_encoder = _DummyVisionEncoder()

    processed_images, new_tokens = image_processor.process_gemma4_inputs(
        images=images,
        tokens=tokens,
        vision_encoder=vision_encoder,
        pad_id=0,
    )

    self.assertEqual(processed_images.patches.shape, (2, 180, 768))
    self.assertEqual(processed_images.positions_xy.shape, (2, 180, 2))

    self.assertLen(processed_images.soft_token_counts, 2)
    self.assertEqual(processed_images.soft_token_counts[0], (9, 9))
    self.assertEqual(processed_images.soft_token_counts[1], (9,))

    self.assertLen(new_tokens, 2)
    expected_tokens_0 = np.array(
        [1, 2]
        + [108, 255999]
        + [-2] * 9
        + [258882, 108]
        + [108, 255999]
        + [-2] * 9
        + [258882, 108]
        + [4, 5]
    )
    expected_tokens_1 = np.array(
        [10] + [108, 255999] + [-2] * 9 + [258882, 108] + [20]
    )
    np.testing.assert_array_equal(new_tokens[0], expected_tokens_0)
    np.testing.assert_array_equal(new_tokens[1], expected_tokens_1)


if __name__ == '__main__':
  absltest.main()
