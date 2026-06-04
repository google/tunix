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

"""NumPy port of HF `Gemma4ImageProcessor` (patchify + 2D position ids).

Produces the two arrays the real vision tower consumes:
  * ``pixel_values``        [B, max_patches, 3*patch^2]  (flattened patches)
  * ``pixel_position_ids``  [B, max_patches, 2]          ((x, y); padding = -1)

PARITY CAVEAT: the deterministic parts here (target-size computation,
patchification order, position-id grid, padding) are faithful to HF. The
*resize* step in HF uses torchvision's antialiased bilinear, which is hard to
reproduce bit-exactly in PIL/NumPy. For numeric parity validation (Stage 3),
feed BOTH stacks the SAME ``pixel_values``/``pixel_position_ids`` (e.g. produced
by HF's processor) so vision-tower parity is isolated from resize differences.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

# HF supports a fixed set of soft-token budgets.
_SUPPORTED_SOFT_TOKENS = (32, 64, 256, 280)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
  """Largest (h, w) that (1) yields <= max_patches patches and (2) is divisible
  by ``pooling_kernel_size * patch_size``. Faithful to HF."""
  total_px = height * width
  target_px = max_patches * (patch_size**2)
  factor = math.sqrt(target_px / total_px)
  ideal_height = factor * height
  ideal_width = factor * width
  side_mult = pooling_kernel_size * patch_size

  target_height = int(math.floor(ideal_height / side_mult)) * side_mult
  target_width = int(math.floor(ideal_width / side_mult)) * side_mult

  if target_height == 0 and target_width == 0:
    raise ValueError(
        "Resizing to 0x0; height/width must be divisible by "
        f"pooling_kernel_size*patch_size={side_mult}."
    )
  max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
  if target_height == 0:
    target_height = side_mult
    target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
  elif target_width == 0:
    target_width = side_mult
    target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
  return target_height, target_width


def convert_image_to_patches(image_chw: np.ndarray, patch_size: int) -> np.ndarray:
  """(C, H, W) -> (num_patches, patch_size*patch_size*C), channel-last in patch.

  Mirrors HF/SigLIP2 `convert_image_to_patches`:
  reshape (C, nH, pH, nW, pW) -> permute (nH, nW, pH, pW, C) -> (nH*nW, -1).
  """
  c, h, w = image_chw.shape
  nh, nw = h // patch_size, w // patch_size
  x = image_chw.reshape(c, nh, patch_size, nw, patch_size)
  x = np.transpose(x, (1, 3, 2, 4, 0))  # (nH, nW, pH, pW, C)
  return x.reshape(nh * nw, -1)


def build_position_ids(patch_height: int, patch_width: int) -> np.ndarray:
  """(x, y) position per patch, row-major over (nH, nW) to match patchify order."""
  xs, ys = np.meshgrid(
      np.arange(patch_width), np.arange(patch_height), indexing="xy"
  )
  return np.stack([xs, ys], axis=-1).reshape(patch_height * patch_width, 2)


def pad_along_first_dim(
    patches: np.ndarray, positions: np.ndarray, target_length: int
) -> tuple[np.ndarray, np.ndarray]:
  """Pad patches with 0 and positions with -1 up to ``target_length``."""
  cur = patches.shape[0]
  if target_length > cur:
    pad = target_length - cur
    patches = np.pad(patches, ((0, pad), (0, 0)), constant_values=0)
    positions = np.pad(positions, ((0, pad), (0, 0)), constant_values=-1)
  return patches, positions


@dataclasses.dataclass
class Gemma4ImageProcessor:
  """Minimal inference image processor for the real Gemma 4 vision tower."""

  patch_size: int = 16
  pooling_kernel_size: int = 3
  max_soft_tokens: int = 280
  rescale_factor: float = 1.0 / 255.0

  def __post_init__(self):
    if self.max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
      raise ValueError(
          f"max_soft_tokens must be one of {_SUPPORTED_SOFT_TOKENS}, "
          f"got {self.max_soft_tokens}."
      )

  @property
  def max_patches(self) -> int:
    return self.max_soft_tokens * self.pooling_kernel_size**2

  def _resize(self, image_chw: np.ndarray) -> np.ndarray:
    """Aspect-ratio-preserving resize via PIL bilinear+antialias.

    NOTE: not bit-exact with torchvision (see module docstring). Skipped if the
    image already has target dimensions.
    """
    c, h, w = image_chw.shape
    th, tw = get_aspect_ratio_preserving_size(
        h, w, self.patch_size, self.max_patches, self.pooling_kernel_size
    )
    if (th, tw) == (h, w):
      return image_chw
    try:
      from PIL import Image  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise ImportError(
          "Pillow is required to resize images in Gemma4ImageProcessor; "
          "install pillow or pass pre-resized CHW arrays."
      ) from e
    hwc = np.transpose(image_chw, (1, 2, 0))
    pil = Image.fromarray(hwc.astype(np.uint8)) if hwc.dtype != np.uint8 else \
        Image.fromarray(hwc)
    pil = pil.resize((tw, th), resample=Image.BILINEAR)
    return np.transpose(np.asarray(pil), (2, 0, 1))

  def __call__(
      self, images, do_resize: bool = True
  ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """images: list of (C, H, W) uint8/float arrays. Returns
    (pixel_values [B, max_patches, 3*p^2], pixel_position_ids [B, max_patches, 2],
     num_soft_tokens_per_image)."""
    pixel_values, position_ids, num_soft = [], [], []
    for image in images:
      image = np.asarray(image)
      if do_resize:
        image = self._resize(image)
      image = image.astype(np.float32) * self.rescale_factor  # -> [0, 1]
      patch_h = image.shape[-2] // self.patch_size
      patch_w = image.shape[-1] // self.patch_size
      patches = convert_image_to_patches(image, self.patch_size)
      num_soft.append(patches.shape[0] // self.pooling_kernel_size**2)
      positions = build_position_ids(patch_h, patch_w)
      patches, positions = pad_along_first_dim(patches, positions, self.max_patches)
      pixel_values.append(patches)
      position_ids.append(positions)
    return (
        np.stack(pixel_values, axis=0),
        np.stack(position_ids, axis=0),
        num_soft,
    )
