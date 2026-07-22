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

"""Image processing for VLMs."""

from typing import Any
import numpy as np
from PIL import Image
from tunix.models.gemma4.model import PreprocessedVisionInput


class ImageProcessor:
  """Vision-language processor.

  This class takes in a batch of images (or image paths) and processes them for
  vision encoders.

  Attributes:
    config: The configuration object containing parameters for image processing,
      such as image height, width, channels, mean, and standard deviation.
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
      images: The images to pre-process. Can be a string/array, in which case a
        batch of one image is assumed. Can be a list of strings/arrays, in which
        case a len(images) is the batch size, with each batch having one image.
        Or it can be a list of lists of strings/arrays, in which case each
        element in the batch has a variable number of images.

    Returns:
      Returns the processed images.
    """

    # For unbatched input.
    if not isinstance(images, list):
      images = [[images]]

    max_num_images = _compute_max_num_images(images)

    processed_images = []
    for batch in images:
      if batch is None:
        processed_images.append([
            np.zeros(
                (self._height, self._width, self._channels),
                dtype=np.float32,
            )
            for _ in range(max_num_images)
        ])
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
      processed_batch.extend([
          np.zeros(
              (self._height, self._width, self._channels), dtype=np.float32
          )
          for _ in range(max_num_images - len(new_batch))
      ])
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


def _compute_max_num_images(lst):
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


# --- Gemma4 Multimodal Image Processing Utilities ---

import math
import jax
import jax.numpy as jnp

POSITIONS_PAD_VALUE = -1


def get_target_dimensions(
    height: int,
    width: int,
    patch_size: int = 16,
    max_patches: int = 10080,
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
  """Calculates target height and width preserving aspect ratio."""
  total_px = height * width
  target_px = max_patches * (patch_size**2)
  if total_px == 0:
    return pooling_kernel_size * patch_size, pooling_kernel_size * patch_size

  factor = math.sqrt(target_px / total_px)
  ideal_height = factor * height
  ideal_width = factor * width
  side_mult = pooling_kernel_size * patch_size

  target_height = int(math.floor(ideal_height / side_mult)) * side_mult
  target_width = int(math.floor(ideal_width / side_mult)) * side_mult

  if target_height == 0 and target_width == 0:
    target_height = side_mult
    target_width = side_mult
  elif target_height == 0:
    target_height = side_mult
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    target_width = min(
        max(1, int(math.floor(width / height))) * side_mult,
        max_side_length,
    )
  elif target_width == 0:
    target_width = side_mult
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    target_height = min(
        max(1, int(math.floor(height / width))) * side_mult,
        max_side_length,
    )

  return int(target_height), int(target_width)


def aspect_ratio_preserving_resize(
    image: np.ndarray,
    patch_size: int = 16,
    max_patches: int = 10080,
    pooling_kernel_size: int = 3,
) -> np.ndarray:
  """Resizes an image while preserving its aspect ratio."""
  height, width = image.shape[:2]
  target_height, target_width = get_target_dimensions(
      height,
      width,
      patch_size=patch_size,
      max_patches=max_patches,
      pooling_kernel_size=pooling_kernel_size,
  )

  if target_height == height and target_width == width:
    return image

  pil_image = Image.fromarray(image)
  pil_image = pil_image.resize(
      (target_width, target_height), resample=Image.BICUBIC
  )
  return np.array(pil_image)


def _to_rgb_uint8(image: np.ndarray | Image.Image) -> np.ndarray:
  if isinstance(image, Image.Image):
    image = image.convert("RGB")
    return np.array(image)

  if image.ndim == 2:
    image = np.stack([image] * 3, axis=-1)
  elif image.shape[-1] == 4:
    image = image[..., :3]
  elif image.shape[-1] == 1:
    image = np.concatenate([image] * 3, axis=-1)
  return image


def preprocess_single_image(
    image: np.ndarray | Image.Image,
    patch_size: int = 16,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> jnp.ndarray:
  """Preprocesses a single image for the VisionEncoder."""
  image = _to_rgb_uint8(image)
  max_patches = max_soft_tokens * pooling_kernel_size**2

  image = aspect_ratio_preserving_resize(
      image,
      patch_size=patch_size,
      max_patches=max_patches,
      pooling_kernel_size=pooling_kernel_size,
  )

  image = image.astype(np.float32) / 255.0
  return jnp.array(image)


def num_soft_tokens_for_image(
    image: jnp.ndarray,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
) -> int:
  h, w = image.shape[:2]
  num_patches = (h // patch_size) * (w // patch_size)
  return int(num_patches // (pooling_kernel_size**2))


def preprocess_images_list(
    images: list[np.ndarray | Image.Image],
    patch_size: int = 16,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> tuple[list[jnp.ndarray], list[int]]:
  """Preprocess a list of variable size images."""
  processed = []
  soft_token_counts = []
  for img in images:
    p = preprocess_single_image(
        img,
        patch_size=patch_size,
        max_soft_tokens=max_soft_tokens,
        pooling_kernel_size=pooling_kernel_size,
    )
    processed.append(p)
    soft_token_counts.append(
        num_soft_tokens_for_image(p, patch_size, pooling_kernel_size)
    )
  return processed, soft_token_counts


def patchify(
    images: jnp.ndarray, patch_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Patchifies images using numpy reshape/transpose instead of einops rearrange."""
  *b, H_val, W_val, C_val = images.shape
  h = H_val // patch_size
  w = W_val // patch_size

  reshaped = images.reshape(*b, h, patch_size, w, patch_size, C_val)
  b_dims = len(b)
  axes_order = list(range(b_dims)) + [
      b_dims,
      b_dims + 2,
      b_dims + 1,
      b_dims + 3,
      b_dims + 4,
  ]
  transposed = reshaped.transpose(axes_order)
  patches = transposed.reshape(*b, h * w, patch_size * patch_size * C_val)

  # Meshgrid XY coordinates
  y = jnp.arange(h)
  x = jnp.arange(w)
  grid_x, grid_y = jnp.meshgrid(x, y)
  positions_xy = jnp.stack([grid_x, grid_y], axis=-1)
  positions_xy = positions_xy.reshape(h * w, 2)
  broadcast_shape = tuple(b) + (h * w, 2)
  positions_xy = jnp.broadcast_to(positions_xy, broadcast_shape)

  return patches, positions_xy


def patchify_and_pad(
    images: list[jnp.ndarray],
    patch_size: int = 16,
    max_patches: int | None = None,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, list[int]]:
  """Patchifies variable size images and pads to max common patches length."""
  if max_patches is None:
    max_patches = max_soft_tokens * pooling_kernel_size**2

  all_patches = []
  all_positions = []
  num_real_patches_per_image = []

  for image in images:
    img_patches, img_positions = patchify(image[None], patch_size)
    img_patches = img_patches[0]
    img_positions = img_positions[0]

    num_real = img_patches.shape[0]
    num_real_patches_per_image.append(num_real)

    num_padding = max_patches - num_real
    if num_padding > 0:
      pad_patches = jnp.zeros(
          (num_padding, img_patches.shape[-1]), dtype=img_patches.dtype
      )
      img_patches = jnp.concatenate([img_patches, pad_patches], axis=0)

      pad_positions = jnp.full(
          (num_padding, 2), POSITIONS_PAD_VALUE, dtype=img_positions.dtype
      )
      img_positions = jnp.concatenate([img_positions, pad_positions], axis=0)

    all_patches.append(img_patches)
    all_positions.append(img_positions)

  patches = jnp.stack(all_patches, axis=0)
  positions_xy = jnp.stack(all_positions, axis=0)

  return patches, positions_xy, num_real_patches_per_image


def preprocess_and_patchify(
    images: list[np.ndarray | Image.Image],
    patch_size: int = 16,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, list[int]]:
  """Preprocess images and patchify+pad them."""
  processed, soft_token_counts = preprocess_images_list(
      images,
      patch_size=patch_size,
      max_soft_tokens=max_soft_tokens,
      pooling_kernel_size=pooling_kernel_size,
  )

  patches, positions_xy, _ = patchify_and_pad(
      processed,
      patch_size=patch_size,
      max_soft_tokens=max_soft_tokens,
      pooling_kernel_size=pooling_kernel_size,
  )

  return patches, positions_xy, soft_token_counts


def add_variable_extra_tokens_for_images(
    tokens: np.ndarray,
    *,
    soft_token_counts: list[int] | tuple[tuple[int, ...], ...],
    placeholder_token: int = 258880,
    start_token: int = 255999,
    end_token: int = 258882,
) -> np.ndarray:
  """Expand placeholder token with a variable number of placeholders."""
  double_new_line_token = 108
  soft_token_placeholder = -2  # img soft tokens

  batch_size = tokens.shape[0]
  results = []
  for b in range(batch_size):
    row = tokens[b].tolist()
    expanded = []
    image_idx = 0

    if len(soft_token_counts) > 0 and isinstance(soft_token_counts[0], int):
      counts = soft_token_counts
    else:
      counts = soft_token_counts[b] if b < len(soft_token_counts) else ()

    for token in row:
      if token == placeholder_token and image_idx < len(counts):  # pyrefly: ignore[bad-argument-type]
        count = counts[image_idx]  # pyrefly: ignore[bad-index]
        expanded.append(double_new_line_token)
        expanded.append(start_token)
        expanded.extend([soft_token_placeholder] * count)  # pyrefly: ignore[unsupported-operation]
        expanded.append(end_token)
        expanded.append(double_new_line_token)
        image_idx += 1
      else:
        expanded.append(token)
    results.append(expanded)

  max_len = max(len(r) for r in results)
  padded = np.zeros((batch_size, max_len), dtype=np.int32)
  for b, row in enumerate(results):
    padded[b, : len(row)] = row

  return padded


def process_gemma4_inputs(
    images: Any,
    tokens: list[np.ndarray],
    vision_encoder: Any,
    pad_id: int,
) -> tuple[Any, list[np.ndarray]]:
  """Processes images and tokens for Gemma4 multimodal models."""

  if not isinstance(images, list):
    images = [[images]]
  elif len(images) > 0 and not isinstance(images[0], list):
    images = [[img] for img in images]

  max_n_images = max((len(batch) for batch in images), default=0)

  batch_patches = []
  batch_positions = []
  all_soft_token_counts = []

  max_patches_per_image = 0

  for batch in images:
    if not batch:
      batch_patches.append(None)
      batch_positions.append(None)
      all_soft_token_counts.append(())
      continue

    patches, positions_xy, soft_token_counts = preprocess_and_patchify(
        batch,
        patch_size=vision_encoder.config.patch_size,
        max_soft_tokens=vision_encoder.config.num_mm_tokens_per_image,
        pooling_kernel_size=vision_encoder.config.pooling_kernel_size,
    )
    batch_patches.append(patches)
    batch_positions.append(positions_xy)
    all_soft_token_counts.append(tuple(soft_token_counts))
    max_patches_per_image = max(max_patches_per_image, patches.shape[1])

  if max_patches_per_image == 0:
    max_patches_per_image = (
        vision_encoder.config.num_mm_tokens_per_image
        * vision_encoder.config.pooling_kernel_size**2
    )

  final_patches = []
  final_positions = []
  patch_dim = 3 * (vision_encoder.config.patch_size**2)

  for b_idx in range(len(images)):
    if batch_patches[b_idx] is not None:
      p = batch_patches[b_idx]
      xy = batch_positions[b_idx]
      assert p is not None
      assert xy is not None
      patch_dim = p.shape[-1]
    else:
      p = jnp.zeros((0, max_patches_per_image, patch_dim), dtype=jnp.float32)
      xy = jnp.full(
          (0, max_patches_per_image, 2), POSITIONS_PAD_VALUE, dtype=jnp.int32
      )

    n_pad = max_n_images - p.shape[0]
    if n_pad > 0:
      pad_p = jnp.zeros(
          (n_pad, max_patches_per_image, patch_dim), dtype=p.dtype
      )
      pad_xy = jnp.full(
          (n_pad, max_patches_per_image, 2), POSITIONS_PAD_VALUE, dtype=xy.dtype
      )
      p = jnp.concatenate([p, pad_p], axis=0)
      xy = jnp.concatenate([xy, pad_xy], axis=0)

    p = jnp.reshape(p, (max_n_images * max_patches_per_image, patch_dim))
    xy = jnp.reshape(xy, (max_n_images * max_patches_per_image, 2))

    final_patches.append(p)
    final_positions.append(xy)

  if final_patches:
    patches = jnp.stack(final_patches, axis=0)
    positions_xy = jnp.stack(final_positions, axis=0)
  else:
    batches = len(images)
    patches = jnp.zeros(
        (batches, max_n_images * max_patches_per_image, patch_dim),
        dtype=jnp.float32,
    )
    positions_xy = jnp.full(
        (batches, max_n_images * max_patches_per_image, 2),
        POSITIONS_PAD_VALUE,
        dtype=jnp.int32,
    )

  processed_images = PreprocessedVisionInput(
      patches=patches,
      positions_xy=positions_xy,
      soft_token_counts=tuple(all_soft_token_counts),
  )

  if all_soft_token_counts:
    max_len = max(len(t) for t in tokens)
    padded_tokens = np.array([
        np.pad(x, (0, max_len - len(x)), constant_values=pad_id) for x in tokens
    ])
    expanded_tokens = add_variable_extra_tokens_for_images(
        padded_tokens,
        soft_token_counts=tuple(all_soft_token_counts),
    )
    tokens = [
        np.array([tid for tid in row if tid != pad_id])
        for row in expanded_tokens.tolist()
    ]

  return processed_images, tokens
