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

# Mostly forked from https://github.com/google-deepmind/gemma/blob/2a892cd9462618ce8bf8f0b5830466498858beb4/gemma/gm/nn/gemma4/_moe.py

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models import gpu_kernels as kernels_lib


def _replay_choices(
    router_choices: jax.Array,
    forced_routed_experts: jax.Array,
    num_experts: int,
) -> jax.Array:
  """Replaces the router's top-k with the replayed experts, token by token.

  A token is replayed only when all of its top-k slots hold an expert id in
  `[0, num_experts)`. Any other id, in particular `-1`
  (`tunix.rl.common.UNSET_ROUTED_EXPERT`, which marks padding and the final
  token the rollout never routed), leaves the whole token to the router.
  Unlike MaxText's Gemma 4 `get_topk`, which zeroes the combine weights of
  invalid slots, falling back to the router's top-k avoids producing an
  all-zero MoE output if an unmasked non-terminal token has unset routing.

  Args:
    router_choices: `[..., k]` experts picked by the router's top-k.
    forced_routed_experts: `[..., k]` experts captured at rollout.
    num_experts: Number of experts.

  Returns:
    `[..., k]` experts: the replayed ones where a token's row is complete, the
    router's own elsewhere.
  """
  forced = forced_routed_experts.astype(router_choices.dtype)
  is_replayed = jnp.all(
      (forced >= 0) & (forced < num_experts), axis=-1, keepdims=True
  )
  return jnp.where(is_replayed, forced, router_choices)


def _expert_dispatch(
    x: jax.Array,
    expert_choices: jax.Array,
    expert_weights: jax.Array,
    num_experts: int,
):
  """Sorts tokens by expert for each expert choice."""
  num_groups, group_size, k = expert_choices.shape
  x = x.reshape((-1, x.shape[-1]))
  batch_size = num_groups * group_size

  expert_choices_flat = expert_choices.ravel()
  xs_order = expert_choices_flat.argsort()
  xs_reverse_argsort = xs_order.argsort()
  xs_indices = jnp.repeat(jnp.arange(batch_size), k)[xs_order]
  sorted_xs = x[xs_indices, :]
  expert_choices_oh = jax.nn.one_hot(
      expert_choices, num_classes=num_experts, dtype=jnp.int32
  )
  xs_tokens_per_expert = jnp.sum(expert_choices_oh, axis=(0, 1, 2))
  return (
      xs_tokens_per_expert,
      sorted_xs,
      xs_reverse_argsort,
      expert_weights,
  )


def _expert_collect(
    sorted_xs: jax.Array,
    xs_reverse_argsort: jax.Array,
    xs_combine_weights: jax.Array,
) -> jax.Array:
  """Unshuffles tokens back to their original token order and reshapes."""
  num_groups, group_size, k = xs_combine_weights.shape
  xs = sorted_xs[xs_reverse_argsort]
  xs_reshaped = xs.reshape((num_groups, group_size, k, -1))
  return xs_reshaped


class MoERagged(nnx.Module):
  """Mixture of Experts using ragged_dot."""

  def __init__(
      self,
      config,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.features = config.embed_dim
    self.hidden_dim = config.expert_dim
    self.num_experts = config.num_experts
    self.num_experts_per_datapoint = config.num_experts_per_tok

    self.router_logits = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(), (self.features, self.num_experts)
        )
    )
    self.gating_einsum = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(), (self.num_experts, 2, self.hidden_dim, self.features)
        ),
        sharding=config.shd_config.exp_weight_edf,
    )
    self.linear = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(), (self.num_experts, self.hidden_dim, self.features)
        ),
        sharding=config.shd_config.exp_weight_efd,
    )
    self.per_expert_scale = nnx.Param(
        jnp.ones((self.num_experts,), dtype=config.param_dtype)
    )
    self.router_scale = nnx.Param(
        jnp.ones((self.features,), dtype=config.param_dtype)
    )

  def _router(
      self,
      router_logits: jax.Array,
      forced_routed_experts: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    router_logits = router_logits.astype(jnp.float32)
    _, choices = jax.lax.top_k(
        router_logits,
        k=self.num_experts_per_datapoint,
    )
    if forced_routed_experts is not None:
      choices = _replay_choices(
          choices, forced_routed_experts, self.num_experts
      )
    # Softmax over only the chosen experts' logits is mathematically identical
    # to renormalizing full-expert probabilities, while avoiding float32
    # underflow when replayed experts lie >88 logits below the router's top-1.
    chosen_logits = jnp.take_along_axis(router_logits, choices, axis=-1)
    weights = jax.nn.softmax(chosen_logits, axis=-1)
    return weights, choices

  def _run_ffw_and_routing(
      self,
      x: jax.Array,
      expert_choices: jax.Array,
      expert_weights: jax.Array,
  ) -> jax.Array:
    (
        xs_tokens_per_expert,
        sorted_xs,
        xs_reverse_argsort,
        xs_combine_weights,
    ) = _expert_dispatch(x, expert_choices, expert_weights, self.num_experts)

    w_gate = self.gating_einsum.value
    w_gate = jnp.transpose(w_gate, (0, 3, 1, 2))
    w_gate = w_gate.reshape(
        self.num_experts, self.features, 2 * self.hidden_dim
    )

    ragged_dot_fn = (
        kernels_lib.gpu_ragged_dot
        if jax.default_backend() == 'gpu'
        else jax.lax.ragged_dot
    )
    gate_out = ragged_dot_fn(
        sorted_xs,
        w_gate.astype(self.config.dtype),
        group_sizes=xs_tokens_per_expert,
    )
    gate_out = gate_out.reshape(gate_out.shape[0], 2, self.hidden_dim)
    x1 = gate_out[:, 0, :]
    x2 = gate_out[:, 1, :]
    activation = nnx.gelu(x1) * x2
    expert_outputs = ragged_dot_fn(
        activation,
        self.linear.value.astype(self.config.dtype),
        group_sizes=xs_tokens_per_expert,
    )

    expert_indices = jnp.repeat(
        jnp.arange(self.num_experts),
        xs_tokens_per_expert,
        total_repeat_length=expert_outputs.shape[0],
    )
    per_expert = self.per_expert_scale.value.astype(expert_outputs.dtype)
    expert_outputs = expert_outputs * per_expert[expert_indices, None]

    out = _expert_collect(
        expert_outputs,
        xs_reverse_argsort=xs_reverse_argsort,
        xs_combine_weights=xs_combine_weights,
    )

    out = jnp.einsum(
        'blkd,blk->bld',
        out,
        xs_combine_weights,
        preferred_element_type=out.dtype,
    )
    return out

  def block(
      self,
      x: jax.Array,
      router_input: jax.Array | None = None,
      forced_routed_experts: jax.Array | None = None,
  ) -> jax.Array:
    """Routes `x` to its top-k experts and combines their outputs.

    Args:
      x: `[B, T, D]` expert inputs.
      router_input: `[B, T, D]` router inputs; defaults to `x`.
      forced_routed_experts: Optional `[B, T, k]` experts to replay instead of
        the router's top-k; see `_replay_choices` for how unset ids are
        handled.

    Returns:
      `[B, T, D]` combined expert outputs.
    """
    if router_input is None:
      router_input = x
    var = jnp.mean(jnp.square(router_input.astype(jnp.float32)), axis=-1, keepdims=True)
    router_input = router_input * jax.lax.rsqrt(var + 1e-06).astype(router_input.dtype)

    router_scale = self.router_scale.value
    router_kernel = self.router_logits.value
    if self.config.freeze_router:
      router_scale = jax.lax.stop_gradient(router_scale)
      router_kernel = jax.lax.stop_gradient(router_kernel)

    root_size = jax.lax.rsqrt(
        jnp.array(self.features, dtype=router_input.dtype)
    )
    router_input = (
        router_input
        * root_size
        * router_scale.astype(router_input.dtype)
    )
    logits = jnp.einsum(
        'gsd,de->gse',
        router_input,
        router_kernel.astype(router_input.dtype),
    )
    weights, choices = self._router(logits, forced_routed_experts)
    out = self._run_ffw_and_routing(x, choices, weights)
    return out

  def __call__(
      self,
      x: jax.Array,
      router_input: jax.Array | None = None,
      forced_routed_experts: jax.Array | None = None,
  ) -> jax.Array:
    remat_config = self.config.remat_config
    if remat_config is not None and str(remat_config).endswith('BLOCK'):
      return nnx.remat(self.block.__func__, graph_updates=False)(
          self, x, router_input, forced_routed_experts
      )
    else:
      return self.block(
          x,
          router_input=router_input,
          forced_routed_experts=forced_routed_experts,
      )
