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

"""Policy scoring for prepared, target-aligned diffusion rollouts."""

import math
import numbers

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.diffusion import interfaces as diffusion_interfaces
from tunix.diffusion import types as diffusion_types


def _validate_temperature(temperature: float) -> None:
  if (
      isinstance(temperature, bool)
      or not isinstance(temperature, numbers.Real)
      or not math.isfinite(float(temperature))
      or temperature <= 0
  ):
    raise ValueError(
        "temperature must be a finite positive real scalar; received"
        f" {temperature}"
    )


def compute_diffusion_per_token_logps(
    model: nnx.Module,
    batch: diffusion_types.DiffusionTokenBatch,
    logits_fn: diffusion_interfaces.DiffusionLogitsFn,
    *,
    temperature: float = 1.0,
    stop_gradient: bool = True,
    return_entropy: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Scores a prepared diffusion rollout without an autoregressive shift.

  The caller owns model-specific rollout and batch construction. ``batch``
  must describe the exact target-aligned canvas used to score the completion.
  Positions with zero ``loss_weights`` are inactive: their logits and target
  IDs are sanitized before probability operations, then their returned scores
  are zeroed. This permits padding to carry sentinel IDs or non-finite logits
  without contaminating active scores or gradients.

  Args:
    model: Policy model to score.
    batch: Prepared target-aligned model inputs, token IDs, and active weights.
    logits_fn: Model-specific callable returning ``[batch, length, vocab]``
      target-aligned logits.
    temperature: Positive sampling temperature used by the rollout policy.
    stop_gradient: Whether returned scores should be detached from the policy.
    return_entropy: Whether to also return per-token categorical entropy.

  Returns:
    Per-token log probabilities with the same shape as ``batch.target_ids``.
    When ``return_entropy`` is true, also returns entropy with the same shape.
    Inactive positions are zero in every returned array.
  """

  _validate_temperature(temperature)
  logits = diffusion_interfaces.compute_diffusion_logits(
      model, batch, logits_fn
  )

  active = jnp.asarray(batch.loss_weights) != 0
  logits = jnp.asarray(logits, dtype=jnp.float32)
  logits = jnp.where(active[..., None], logits, 0.0)
  targets = jnp.asarray(batch.target_ids)
  targets = jnp.where(active, targets, 0)

  log_probs = jax.nn.log_softmax(logits / temperature, axis=-1)
  per_token_logps = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[
      ..., 0
  ]
  per_token_logps = jnp.where(active, per_token_logps, 0.0)

  if return_entropy:
    probs = jnp.exp(log_probs)
    entropy_terms = jnp.where(probs > 0, probs * log_probs, 0.0)
    entropy = -jnp.sum(entropy_terms, axis=-1)
    entropy = jnp.where(active, entropy, 0.0)

  if stop_gradient:
    per_token_logps = jax.lax.stop_gradient(per_token_logps)
    if return_entropy:
      entropy = jax.lax.stop_gradient(entropy)

  if return_entropy:
    return per_token_logps, entropy
  return per_token_logps
