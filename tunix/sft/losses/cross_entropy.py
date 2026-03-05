# Copyright 2025 Google LLC
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

"""Standard cross-entropy loss for next-token prediction."""

from typing import Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax


def per_token_cross_entropy(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> Tuple[ArrayLike, jax.Array]:
  """Computes per-token cross-entropy for next-token prediction.

  This is the shared building block for all SFT loss functions. It
  runs the forward pass, shifts targets by one position, and returns
  the raw per-token losses together with the target mask.

  Args:
    model: The model to compute logits from.
    input_tokens: Input token IDs, shape ``(batch, seq_len)``.
    input_mask: Binary mask for valid tokens, shape ``(batch, seq_len)``.
    positions: Position indices, shape ``(seq_len,)`` or
      ``(batch, seq_len)``.
    attention_mask: Attention mask, shape matching model expectations.

  Returns:
    A tuple ``(per_token_xent, target_mask)`` where
    ``per_token_xent`` has shape ``(batch, seq_len - 1)`` and
    ``target_mask`` has the same shape.
  """
  logits, _ = model(input_tokens, positions, None, attention_mask)

  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Per-token cross-entropy: -log p(y_t).
  xent = optax.softmax_cross_entropy_with_integer_labels(
      logits, target_tokens
  )

  return xent, target_mask


def aggregate_loss(
    per_token_loss: ArrayLike,
    mask: jax.Array,
) -> ArrayLike:
  """Masks and averages per-token losses over valid tokens.

  Args:
    per_token_loss: Per-token loss values, shape ``(batch, seq_len)``.
    mask: Binary mask for valid tokens, same shape.

  Returns:
    Scalar mean loss over valid tokens.
  """
  norm_factor = 1 / (jnp.sum(mask) + 1e-8)
  return jnp.sum(per_token_loss * mask) * norm_factor


def cross_entropy_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  """Standard next-token prediction loss (negative log-likelihood).

  Computes per-token cross-entropy between the model's predictions and
  the shifted target tokens, masks out padding, and returns the mean
  loss over valid tokens.

  Args:
    model: The model to compute logits from.
    input_tokens: Input token IDs, shape ``(batch, seq_len)``.
    input_mask: Binary mask for valid tokens, shape ``(batch, seq_len)``.
    positions: Position indices, shape ``(seq_len,)`` or
      ``(batch, seq_len)``.
    attention_mask: Attention mask, shape matching model expectations.

  Returns:
    Scalar NLL loss.
  """
  xent, target_mask = per_token_cross_entropy(
      model, input_tokens, input_mask, positions, attention_mask
  )
  return aggregate_loss(xent, target_mask)
