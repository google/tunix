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

"""Dynamic Fine-Tuning (DFT) loss.

Reference:
  *On the Generalization of Supervised Fine-Tuning: A Reinforcement
  Learning Perspective with Reward Rectification*
  (https://arxiv.org/abs/2508.05629)
"""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tunix.sft.losses.cross_entropy import aggregate_loss
from tunix.sft.losses.cross_entropy import per_token_cross_entropy


def dft_rescale(per_token_loss: ArrayLike) -> ArrayLike:
  """Applies Dynamic Fine-Tuning (DFT) rescaling to per-token losses.

  DFT rescales each token's cross-entropy loss by the model's own
  predicted probability for that token (with ``stop_gradient``),
  down-weighting tokens the model is already uncertain about:

  .. math::

      \\mathcal{L}^{\\text{DFT}}_t
        = \\operatorname{sg}\\bigl(p(y_t)\\bigr)\\,
          \\bigl(-\\log p(y_t)\\bigr)

  Since :math:`p(y_t) = \\exp(-\\text{xent})`, this is equivalent to::

      stop_gradient(exp(-xent)) * xent

  Args:
    per_token_loss: Per-token cross-entropy losses, shape ``(...,)``.

  Returns:
    DFT-rescaled per-token losses with the same shape.
  """
  return jax.lax.stop_gradient(jnp.exp(-per_token_loss)) * per_token_loss


def dft_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  """Next-token prediction loss with DFT rescaling.

  Computes standard per-token cross-entropy (same as
  :func:`~tunix.sft.losses.cross_entropy.cross_entropy_loss_fn`),
  applies :func:`dft_rescale`, then masks and aggregates.  Drop-in
  replacement::

      trainer.with_loss_fn(dft_loss_fn)

  Args:
    model: The model to compute logits from.
    input_tokens: Input token IDs, shape ``(batch, seq_len)``.
    input_mask: Binary mask for valid tokens, shape ``(batch, seq_len)``.
    positions: Position indices, shape ``(seq_len,)`` or
      ``(batch, seq_len)``.
    attention_mask: Attention mask, shape matching model expectations.

  Returns:
    Scalar DFT-rescaled NLL loss.
  """
  xent, target_mask = per_token_cross_entropy(
      model, input_tokens, input_mask, positions, attention_mask
  )
  return aggregate_loss(dft_rescale(xent), target_mask)
