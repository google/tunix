"""Tokens manipulation utils. Forked from google-deepmind/gemma."""

import jax
import jax.numpy as jnp
import jaxtyping


def add_extra_tokens_for_images(
    tokens: jaxtyping.ArrayLike,  # (B, L)
    *,
    max_num_images: int,
    num_tokens_per_image: int,
    start_of_image_token: int,
    end_of_image_token: int,
    soft_token_placeholder: int,
    double_new_line_token: int,
) -> jaxtyping.ArrayLike:  # (B, L + (max_num_images * (num_tokens_per_image + 3)))
  r"""Add the extra image tokens to the text tokens.

  If the model has images, we expand each `<start_of_image>` token by the image
  placeholder tokens.

  Example:

  ```python
  input = [..., x, <start_of_image>, y, ...]
  output = [
      ..., x, \n\n, <start_of_image>, SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, ..., SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, <end_of_image>, \n\n, y, ...
  ]
  ```

  The `\n\n` tokens are added to match how the model was trained.

  Args:
    tokens: The text tokens.
    max_num_images: The maximum number of images in the batch.
    num_tokens_per_image: The number of soft tokens per image.

  Returns:
    The text tokens with the extra image tokens.
  """

  # New tokens which will be inserted for each image.
  mm_tokens = [
      double_new_line_token,
      start_of_image_token,
      *[soft_token_placeholder] * num_tokens_per_image,
      end_of_image_token,
      double_new_line_token,
  ]

  return insert_sequence(
      at=start_of_image_token,
      sequence=mm_tokens,
      tokens=tokens,
      max_num_images=max_num_images,
  )


def insert_sequence(
    tokens: jaxtyping.ArrayLike,  # (B, L)
    *,
    at: int,
    sequence: jaxtyping.ArrayLike,  # (L)
    max_num_images: int,
) -> jaxtyping.ArrayLike:  # (B, L)
  """Insert a sequence of tokens at a given position."""
  _, length = tokens.shape

  mm_tokens = jnp.array(sequence, dtype=jnp.int32)

  # `-1` because `<start_of_image>` is already present in the input tokens.
  offset_by = len(mm_tokens) - 1

  # Maximum length, if all images are present.
  length_with_mm = length + max_num_images * offset_by

  mm_start = tokens == at

  # Get the text tokens correctly placed at their final position.
  # The `<start_of_image>` are removed and expanded to leave space for the MM
  # tokens.
  # tokens = [..., x, <start_of_image>, y, ...]
  # new_text_tokens = [..., x, 0, 0, 0, ..., 0, 0, 0, y, ...]
  new_text_tokens = _get_new_text_tokens(
      mm_start=mm_start,
      text_tokens=tokens,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
      max_num_images=max_num_images,
  )

  # Get the mm tokens placeholders, correctly placed at their final position.
  # new_mm_tokens = [
  #     ..., 0, 0, \n\n, <start_of_image>, ..., <end_of_image>, \n\n, 0, 0, ...
  # ]
  new_mm_tokens = _get_new_mm_tokens(
      mm_start=mm_start,
      mm_tokens_to_insert=mm_tokens,
      max_num_images=max_num_images,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
  )

  # Merge the text and MM tokens.
  return new_text_tokens + new_mm_tokens


def _get_new_text_tokens(
    *,
    mm_start: jaxtyping.ArrayLike,  # (B, L)
    text_tokens: jaxtyping.ArrayLike,  # (B, L)
    offset_by: int,
    length_with_mm: int,
    max_num_images: int,
) -> jaxtyping.ArrayLike:  # (B, max_num_images, num_tokens_per_image + 4)
  # Jax vmap does not support positional arguments, so need the
  # _get_new_text_tokens_inner indirection.
  return jax.vmap(_get_new_text_tokens_inner, in_axes=(0, 0, None, None, None))(
      mm_start, text_tokens, offset_by, length_with_mm, max_num_images
  )


def _get_new_text_tokens_inner(
    mm_start: jaxtyping.ArrayLike,  # (L)
    text_tokens: jaxtyping.ArrayLike,  # (L)
    offset_by: int,
    length_with_mm: int,
    max_num_images: int,
) -> jaxtyping.ArrayLike:  # (L)
  """`_get_new_text_tokens_positions` without batch dimension."""

  # Empty buffer in which text and MM tokens will be inserted.
  tokens_with_mm = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  # Shift the original tokens, so that the new soft tokens can be inserted.
  new_text_tokens_pos = _get_new_text_tokens_positions(
      offset_on=mm_start,
      offset_by=offset_by,
      max_num_images=max_num_images,
  )

  tokens_with_mm = tokens_with_mm.at[new_text_tokens_pos].set(text_tokens)

  # Remove the `<start_of_image>` tokens (will be added afterwards when
  # merging with `_get_new_mm_tokens`).
  first_mm_pos = tokens_with_mm[0]
  new_start_mm_pos = new_text_tokens_pos * mm_start
  tokens_with_mm = tokens_with_mm.at[new_start_mm_pos].set(0)
  tokens_with_mm = tokens_with_mm.at[0].set(first_mm_pos)

  return tokens_with_mm


def _get_new_text_tokens_positions(
    *,
    offset_on: jaxtyping.ArrayLike,  # (L)
    offset_by: int,
    max_num_images: int,
) -> jaxtyping.ArrayLike:  # (L)
  """Create the positions of the new tokens.

  Input: `[x, x, x, offset_on, x, x, offset_on, x]`
  Output: `[0, 1, 2, 3, 4+Offset, 5+Offset, 6+Offset, 7+Offset^2]`

  Args:
    offset_on: The token to offset on.
    offset_by: The number of tokens to offset by.
    max_num_images: The maximum number of images in the batch.

  Returns:
    The new positions of the tokens.
  """
  offset = jnp.cumsum(offset_on, axis=-1) * offset_by
  new_positions = jnp.arange(offset_on.shape[-1]) + offset
  # Do not shift the `<start_of_image>` token, it will be overwritten by the MM
  # tokens.
  new_positions -= offset_by * offset_on

  shift = (max_num_images - jnp.sum(offset_on, axis=-1)) * offset_by
  new_positions += shift
  return new_positions


def _get_new_mm_tokens(
    *,
    mm_start: jaxtyping.ArrayLike,  # (B, L)
    mm_tokens_to_insert: jaxtyping.ArrayLike,  # (num_tokens_per_image + 4)
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> jaxtyping.ArrayLike:  # (B, max_num_images, num_tokens_per_image + 4)
  # Jax vmap does not support positional argiments, so need the
  # _get_new_mm_tokens_inner indirection.
  return jax.vmap(
      _get_new_mm_tokens_inner, in_axes=(0, None, None, None, None)
  )(mm_start, mm_tokens_to_insert, max_num_images, offset_by, length_with_mm)


def _get_new_mm_tokens_inner(
    mm_start: jaxtyping.ArrayLike,  # (L)
    mm_tokens_to_insert: jaxtyping.ArrayLike,  # (num_tokens_per_image + 4)
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> jaxtyping.ArrayLike:  # (max_num_images, num_tokens_per_image + 4)
  """`_get_new_mm_tokens` without batch dimension."""
  # Empty buffer row, which will be merged with the final tokens.
  row = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  ones = jnp.ones((len(mm_tokens_to_insert),), dtype=jnp.int32)

  (offset,) = jnp.nonzero(mm_start, size=max_num_images)

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  mask = offset != 0
  mask = jnp.einsum('...x,y->xy', mask, ones)

  # After the mask is created, offset each individual images
  offset += jnp.arange(len(offset)) * offset_by

  shift = (max_num_images - jnp.sum(mm_start)) * offset_by
  offset += shift

  new_positions = jnp.einsum('x,y->xy', offset, ones)
  new_positions += jnp.arange(len(mm_tokens_to_insert))

  new_positions = new_positions * mask

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  row = row.at[new_positions].set(mm_tokens_to_insert)
  row = row.at[0].set(0)
  return row
