# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for PPO Trainer."""

import functools

import jax
import jax.numpy as jnp
from tunix.rl import function_registry

registry = function_registry.default_registry


@registry.register("advantage_estimator", "gae")
@jax.jit
def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    completion_mask: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
  """Compute advantages using Generalized Advantage Estimation (GAE).

  Computing GAE is a two-step process:

  First, compute the temporal difference (TF), `δ_t`, for each timestep `t`:

  ```
  δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
  ```

  Then, compute the GAE advantage, `A_t`, by summing the discounted TD
  residuals. It is calculated recursively, starting from the last timestep:

  ```
  A_t = δ_t + (γ * λ) * A_{t+1}
  ```

  where:

  - `A_t` is the GAE advantage at timestep `t`.
  - `δ_t` is the temporal difference at timestep `t`.
  - `γ` is the discount factor.
  - `λ` is the GAE lambda parameter.
  - `V(s_t)` is the value function at timestep `t`.
  - `r_t` is the reward at timestep `t`.

  Args:
    rewards: A 2D array of rewards for each step in the rollout.
    values: A 2D array of value estimates from the critic for each step.
    completion_mask: A 2D mask, which is 0 for padding tokens.
    gamma: The discount factor, `γ`.
    gae_lambda: The GAE lambda parameter, `λ`.

  Returns:
    A tuple of two 2D arrays - advantages and returns for each step.
  """
  batch_size = values.shape[0]

  def gae_step(state_t_plus_1, xs):
    # Unpack state and inputs.
    gae_t_plus_1, next_values = state_t_plus_1
    rewards_t, values_t, mask_t = xs

    # Compute Temporal Difference (TD).
    delta = rewards_t + gamma * next_values - values_t
    # Compute GAE for this time step.
    gae_t = delta + gamma * gae_lambda * gae_t_plus_1

    # Skip values on non-completion tokens.
    next_values = values_t * mask_t + (1 - mask_t) * next_values
    gae_t = gae_t * mask_t + (1 - mask_t) * gae_t_plus_1

    # New state to carry over comprises `gae_t` and `next_values`. Output for
    # this step is `gae_t`.
    return (gae_t, next_values), gae_t

  _, advantages_transposed = jax.lax.scan(
      gae_step,
      init=(jnp.zeros((batch_size,)), jnp.zeros((batch_size,))),
      xs=(
          jnp.transpose(jnp.array(rewards)),
          jnp.transpose(jnp.array(values)),
          jnp.transpose(jnp.array(completion_mask)),
      ),
      reverse=True,
  )
  advantages = jnp.transpose(advantages_transposed)
  returns = advantages + values

  # Normalise advantages.
  advantages = masked_whiten(advantages, completion_mask)
  return advantages, returns


@jax.jit
def masked_whiten(
    x: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
  """Normalize the input array."""
  x_mean = masked_mean(x, completion_mask)
  x_var = masked_var(
      x,
      completion_mask,
      x_mean,
  )
  x = (x - x_mean) * jax.lax.rsqrt(x_var + 1e-8)
  return x


@functools.partial(jax.jit, static_argnames=('axis',))
def masked_mean(
    x: jax.Array, mask: jax.Array, axis: int | None = None
) -> jax.Array:
  """Compute the mean of a masked array."""
  cast_mask = mask.astype(x.dtype)
  return jnp.sum(x * cast_mask, axis=axis) / (
      jnp.sum(cast_mask, axis=axis) + 1e-8
  )


@jax.jit
def masked_var(
    x: jax.Array,
    mask: jax.Array,
    mean: jax.Array | None = None,
) -> jax.Array:
  """Compute the variance of a masked array."""
  cast_mask = mask.astype(x.dtype)
  if mean is None:
    mean = masked_mean(x, cast_mask)

  variance = masked_mean(jnp.square(x - mean), cast_mask)

  mask_sum = cast_mask.sum()
  bessel_corr = mask_sum / (mask_sum - 1)
  return variance * bessel_corr


def compute_entropy_from_logits(logits: jax.Array) -> jax.Array:
  """Computes the entropy of a distribution given its logits.

  Args:
    logits: Logits as returned by the model. Of shape `[batch_size, seq_len,
      emb_dim]`.

  Returns:
    A JAX array of shape `[batch_size, seq_len]`, containing the entropy values.
  """
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  probs = jnp.exp(log_probs)
  return -jnp.sum(probs * log_probs, axis=-1)
