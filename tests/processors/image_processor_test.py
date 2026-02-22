import dataclasses
import os
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
    try:
      temp_path = self.create_tempdir().full_path
    except Exception:
      temp_path = tempfile.TemporaryDirectory().name
    temp_file = os.path.join(temp_path, filename)
    img.save(temp_file)
    return temp_file

  def test_process_none_image(self):
    processed_image = self.processor.preprocess_image(None)
    self.assertEqual(
        processed_image.shape, (self.height, self.width, self.channels)
    )
    np.testing.assert_array_equal(processed_image, np.zeros((32, 32, 3)))

  def test_path_input(self):
    img_path = self._create_dummy_image_file()
    processed_image = self.processor.preprocess_image(img_path)
    self.assertEqual(
        processed_image.shape, (self.height, self.width, self.channels)
    )
    np.testing.assert_allclose(processed_image, -1.0 * np.ones((32, 32, 3)))

  def test_array_input(self):
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    processed_image = self.processor.preprocess_image(img_array)
    self.assertEqual(
        processed_image.shape, (self.height, self.width, self.channels)
    )
    np.testing.assert_allclose(processed_image, -1.0 * np.ones((32, 32, 3)))

  @parameterized.named_parameters(
      dict(testcase_name='array', input_type='array'),
      dict(testcase_name='path', input_type='path'),
  )
  def test_call_one_image(self, input_type):
    if input_type == 'array':
      images = [np.zeros((100, 100, 3), dtype=np.uint8)]
    elif input_type == 'path':
      images = [self._create_dummy_image_file()]

    processed_images = self.processor(images=images)  # pylint: disable=undefined-variable
    self.assertLen(processed_images, 1)
    self.assertLen(processed_images[0], 1)
    self.assertEqual(
        processed_images[0][0].shape, (self.height, self.width, self.channels)  # pytype: disable=attribute-error
    )
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((32, 32, 3))
    )

  def test_padding(self):
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    images = [[img1], [img1, img2]]
    processed_images = self.processor(images=images)
    self.assertLen(processed_images, 2)
    self.assertLen(processed_images[0], 2)  # Padded to 2
    self.assertLen(processed_images[1], 2)
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((32, 32, 3))
    )
    # Padded image should be zeros
    np.testing.assert_allclose(processed_images[0][1], np.zeros((32, 32, 3)))
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((32, 32, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][1], -1.0 * np.ones((32, 32, 3))
    )

  def test_call_with_none_in_batch(self):
    images = [None, [np.zeros((100, 100, 3), dtype=np.uint8)]]
    processed_images = self.processor(images=images)
    self.assertLen(processed_images, 2)
    self.assertLen(processed_images[0], 1)
    self.assertLen(processed_images[1], 1)
    np.testing.assert_allclose(processed_images[0][0], np.zeros((32, 32, 3)))
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((32, 32, 3))
    )


if __name__ == '__main__':
  absltest.main()
