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


@jax.jit
def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
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
    gamma: The discount factor, `γ`.
    gae_lambda: The GAE lambda parameter, `λ`.

  Returns:
    A tuple of two 2D arrays - advantages and returns for each step.
  """
  batch_size = values.shape[0]

  next_values = jnp.concatenate(
      (values[..., 1:], jnp.zeros((batch_size, 1))), axis=1
  )

  # Compute Temporal Difference (TD).
  deltas = rewards + gamma * next_values - values

  def gae_step(gae_t_plus_1, delta_t):
    # `A_t = delta_t + (gamma * lambda) * A_{t+1}`.
    gae_t = delta_t + gamma * gae_lambda * gae_t_plus_1

    # New state to carry over is `gae_t`. Output for this step is also `gae_t`.
    return gae_t, gae_t

  _, advantages_transposed = jax.lax.scan(
      gae_step,
      init=jnp.zeros((batch_size,)),
      xs=jnp.transpose(jnp.array(deltas)),
      reverse=True,
  )
  advantages = jnp.transpose(advantages_transposed)
  returns = advantages + values
  return advantages, returns


@jax.jit
def normalize_advantages(
    advantages: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
  """Normalize advantages."""
  advantages_mean = masked_mean(advantages, completion_mask)
  advantages_var = masked_var(
      advantages,
      completion_mask,
      advantages_mean,
  )
  advantages = (advantages - advantages_mean) * jax.lax.rsqrt(
      advantages_var + 1e-8
  )
  # Zero out advantages corresponding to padding tokens.
  advantages = jnp.where(
      completion_mask,
      advantages,
      jnp.array(0).astype(advantages.dtype),
  )
  return advantages


@functools.partial(jax.jit, static_argnames=('axis',))
def masked_mean(
    x: jax.Array, mask: jax.Array, axis: int | None = None
) -> jax.Array:
  """Compute the mean of a masked array."""
  cast_mask = mask.astype(x.dtype)
  return jnp.sum(x * cast_mask, axis=axis) / jnp.sum(cast_mask, axis=axis)


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
