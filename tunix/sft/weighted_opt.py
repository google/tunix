
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

# TODO: Do not perform optimizer every step, use jax.lax.cond
"""Weighted Multi-Step Accumulation Optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Union
import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import numerics

class WeightedMultiStepsState(NamedTuple):
  mini_step: jax.Array
  gradient_step: jax.Array
  inner_opt_state: Any
  acc_grads: Any
  acc_tokens: jax.Array  # Accumulated token count

class WeightedMultiSteps:
  """Accumulates gradients weighted by token counts for True Average."""

  def __init__(
      self,
      opt: base.GradientTransformation,
      every_k_schedule: Union[int, Callable[[jax.Array], jax.Array]],
  ):
    self._opt = opt
    if isinstance(every_k_schedule, int):
      self._every_k_schedule = lambda step: every_k_schedule
    else:
      self._every_k_schedule = every_k_schedule

  def init(self, params: Any) -> WeightedMultiStepsState:
    return WeightedMultiStepsState(
        mini_step=jnp.zeros([], dtype=jnp.int32),
        gradient_step=jnp.zeros([], dtype=jnp.int32),
        inner_opt_state=self._opt.init(params),
        acc_grads=jax.tree_util.tree_map(jnp.zeros_like, params),
        acc_tokens=jnp.zeros([], dtype=jnp.float32),
    )

  def update(
      self,
      updates: Any, 
      state: WeightedMultiStepsState,
      params: Optional[Any] = None,
      **extra_args,
  ) -> tuple[Any, WeightedMultiStepsState]:
    
    # We expect token_count to be passed via kwargs
    if 'token_count' not in extra_args:
        raise ValueError("WeightedMultiSteps requires 'token_count' in extra_args (kwargs)")
    
    token_count = extra_args['token_count']
    grads = updates # Updates are just the gradients now
    
    k_steps = self._every_k_schedule(state.gradient_step)
    
    # 1. Update Accumulators
    # We accumulate (grads * token_count) to recover the sum of gradients/losses
    def accumulate_grad(acc, g):
        return acc + g * token_count

    new_acc_grads = jax.tree_util.tree_map(accumulate_grad, state.acc_grads, grads)
    new_acc_tokens = state.acc_tokens + token_count

    # 2. Check if we should emit (update inner optimizer)
    is_last_step = state.mini_step == (k_steps - 1)
    
    # 3. Calculate "True Average" Gradients
    # averaged_grads = sum(g*t) / sum(t)
    def compute_avg(acc_g):
        return acc_g / jnp.maximum(new_acc_tokens, 1e-9)

    avg_grads = jax.tree_util.tree_map(compute_avg, new_acc_grads)
    
    # 4. Compute Inner Optimizer Update
    final_updates, new_inner_state = self._opt.update(
        avg_grads, state.inner_opt_state, params=params, **extra_args
    )

    # 5. Masking / State Update Logic
    
    # Helper for conditional state update
    def where_tree(cond, true_tree, false_tree):
        return jax.tree_util.tree_map(
            lambda t, f: jnp.where(cond, t, f), true_tree, false_tree
        )

    # Next state calculation
    next_mini_step = (state.mini_step + 1) % k_steps
    next_gradient_step = state.gradient_step + jnp.where(is_last_step, 1, 0)
    
    # Reset accumulators if we just emitted
    next_acc_grads = where_tree(
        is_last_step, 
        jax.tree_util.tree_map(jnp.zeros_like, new_acc_grads), # Reset to 0
        new_acc_grads # Keep accumulated
    )
    next_acc_tokens = jnp.where(is_last_step, 0.0, new_acc_tokens)

    # Inner opt state
    next_inner_state = where_tree(
        is_last_step,
        new_inner_state,
        state.inner_opt_state
    )

    # Model updates output
    total_updates = where_tree(
        is_last_step,
        final_updates,
        jax.tree_util.tree_map(jnp.zeros_like, final_updates)
    )

    new_state = WeightedMultiStepsState(
        mini_step=next_mini_step,
        gradient_step=next_gradient_step,
        inner_opt_state=next_inner_state,
        acc_grads=next_acc_grads,
        acc_tokens=next_acc_tokens
    )

    return total_updates, new_state
