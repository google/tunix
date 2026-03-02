"""Image processing for VLMs."""

from typing import Any
import numpy as np
from PIL import Image


class ImageProcessor:
  """Vision-language processor.

  This class takes in a batch of images (or image paths) and processes them for
  vision encoders.
  """

  def __init__(self, config: Any):
    self._height = config.image_height
    self._width = config.image_width
    self._channels = config.image_channels
    self._mean = config.image_mean
    self._std = config.image_std

    self.config = config

  def __call__(
      self,
      images: (
          str
          | np.ndarray
          | list[str | np.ndarray | list[str | np.ndarray] | None]
      ),
  ) -> list[list[np.ndarray]]:
    """Pre-process images.
    
    Takes in a list (or list of lists of) images (or image paths), resizes
    normalises, clips, and pads the images (to maximum number of images in the
    batch).

    Args:
      images: The images to pre-process. Can be a string/array, in which case
        a batch of one image is assumed. Can be a list of strings/arrays, in
        which case a batch of images is assumed, with each element having one
        images. Or it can be a list of lists of strings/arrays, in which case
        each element in the batch has a variable number of images.

    Returns:
      Returns the processed images.
    """

    if not isinstance(images, list):
      images = [[images]]

    max_num_images = self._compute_max_num_images(images)

    processed_images = []
    for batch in images:
      if batch is None:
        processed_images.append(
            [
                np.zeros(
                    (self._height, self._width, self._channels),
                    dtype=np.float32,
                )
            ]
            * max_num_images
        )
        continue
      elif not isinstance(batch, list):
        new_batch = [batch]
      else:
        new_batch = batch

      processed_batch = []
      for img in new_batch:
        processed_image = self.preprocess_image(img)
        processed_batch.append(processed_image)

      # Pad the batch to have the same number of images as the maximum.
      processed_batch.extend(
          [
              np.zeros(
                  (self._height, self._width, self._channels), dtype=np.float32
              )
          ]
          * (max_num_images - len(batch))
      )
      processed_images.append(processed_batch)

    return processed_images

  def preprocess_image(
      self,
      image: np.ndarray | str | None,
  ) -> np.ndarray:
    """Pre-process image.

    Performs a bi-linear resize and normalizes the image.

    Args:
      image: The image to pre-process. If string, it should be the path to the
        image. Otherwise, it should be a 3D array.

    Returns:
      The pre-processed image.
    """
    if image is None:
      return np.zeros(
          (self._height, self._width, self._channels), dtype=np.float32
      )
    elif isinstance(image, str):
      image = Image.open(image)
    elif isinstance(image, np.ndarray):
      image = Image.fromarray(image)

    # Resize the image.
    image = image.resize(
        (self._width, self._height),  # Weird gotcha: PIL expects width first.
        resample=Image.Resampling.BILINEAR,
    )

    # Normalise and clip the image.
    image = np.array(image, dtype=np.float32)
    image = self._normalize_image(image)
    image = np.clip(image, -1, 1)
    return image

  def _normalize_image(
      self,
      image: np.ndarray,
  ) -> np.ndarray:
    """Normalize the image: `(x - mu) / sigma`.

    Args:
      image: The image to normalize.

    Returns:
      The normalized image.
    """
    image -= np.asarray(self._mean)
    image /= np.asarray(self._std)
    return image

  def _compute_max_num_images(self, lst):
    """Compute the maximum number of images in the batch."""
    max_num_images = 0
    for batch in lst:
      if batch is None:
        continue
      elif not isinstance(batch, list):
        max_num_images = max(max_num_images, 1)
      else:
        max_num_images = max(max_num_images, len(batch))
    return max_num_images
