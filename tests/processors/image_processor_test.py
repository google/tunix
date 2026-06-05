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
import numpy as np
from PIL import Image
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
    processed_images = self.processor(images=images)
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

    processed_images = self.processor(images=images)
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
    processed_images = self.processor(images=images)
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
    processed_images = self.processor(images=images)
    np.testing.assert_allclose(
        processed_images[0][0], np.zeros((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )


if __name__ == '__main__':
  absltest.main()
