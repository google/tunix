# tunix/models/siglip/preprocess.py
from __future__ import annotations
from typing import Iterable
import jax.numpy as jnp
import jax.image as jimg

# Defaults compatible with CLIP / many SigLIP configs; override if needed.
_CLIP_MEAN = jnp.array([0.48145466, 0.4578275, 0.40821073], dtype=jnp.float32)
_CLIP_STD  = jnp.array([0.26862954, 0.26130258, 0.27577711], dtype=jnp.float32)

def preprocess(
    images_uint8: jnp.ndarray,
    image_size: int,
    *,
    mean: Iterable[float] = _CLIP_MEAN,
    std: Iterable[float] = _CLIP_STD,
) -> jnp.ndarray:
    """Resize + normalize images for SigLIP.

    Args:
      images_uint8: [B,H,W,3] or [H,W,3], dtype uint8.
      image_size: output resolution (image_size x image_size).
      mean/std: per-channel normalization arrays.

    Returns:
      float32 array [B, image_size, image_size, 3]
    """
    x = images_uint8
    if x.dtype != jnp.uint8:
        raise ValueError(f"Expected uint8 images, got {x.dtype}")

    # Add batch if needed.
    if x.ndim == 3:
        x = x[None, ...]  # [1,H,W,3]
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected [B,H,W,3], got shape {x.shape}")

    b, h, w, c = x.shape
    # Resize to target square (simple bilinear; if you prefer center-crop+resize, do that here)
    x = jimg.resize(x, (b, image_size, image_size, c), method="bilinear", antialias=True)

    # [0,1] -> normalize
    x = x.astype(jnp.float32) / 255.0
    mean = jnp.asarray(mean, dtype=jnp.float32).reshape((1,1,1,3))
    std  = jnp.asarray(std,  dtype=jnp.float32).reshape((1,1,1,3))
    x = (x - mean) / std
    return x
