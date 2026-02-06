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

"""Merge text and vision embeddings."""

import einops
import jax
import jax.numpy as jnp
import jaxtyping


def merge_embeddings(
    *,
    text_embeddings: jaxtyping.ArrayLike,  # (B, L, D)
    vision_embeddings: jaxtyping.ArrayLike,  # (B, N, P, D)
    mask: jaxtyping.ArrayLike,  # (B, L)
) -> jaxtyping.ArrayLike:  # (B, L, D)
  """Merge the text and vision embeddings."""
  return jax.vmap(_merge_embeddings_inner, in_axes=(0, 0, 0))(
      text_embeddings, vision_embeddings, mask
  )


def _merge_embeddings_inner(
    text_embeddings: jaxtyping.ArrayLike,  # (L, D)
    vision_embeddings: jaxtyping.ArrayLike,  # (N, P, D)
    mask: jaxtyping.ArrayLike,  # (L)
) -> jaxtyping.ArrayLike:  # (L, D)
  """`merge_embeddings` without batch dimension."""

  vision_embeddings = einops.rearrange(
      vision_embeddings,
      'num_images num_toks_per_image d -> (num_images num_toks_per_image) d',
  )

  # len(vision_embeddings) == max_num_images * num_tokens_per_image
  target_pos = jnp.nonzero(mask, size=len(vision_embeddings))

  # Save and restore the first position overwritten if there's no MM tokens.
  first_pos = text_embeddings[0]

  merged = text_embeddings.at[target_pos, :].set(vision_embeddings)

  merged = merged.at[0].set(first_pos)

  return merged
