# Copyright 2026 The Tunix Authors.
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

"""Model runner for token generation.

`ModelRunner` owns the transformer weights and runs the forward pass and token
sampling on device.
"""

import dataclasses
import functools
import operator
from typing import Any

from flax import nnx
from flax import struct
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.rl import reshard
from tunix.rl import utils as rl_utils


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelRunnerConfig:
  """Configuration for the model runner."""

  # The maximum `top_k` value allowed in `SamplingParams`.
  # <= 0 to set no limit.
  max_top_k: int
  # The device mesh the transformer runs on.
  mesh: jax.sharding.Mesh
  # Whether or not return logprobs.
  return_logprobs: bool = False
  # Whether or not to return logits.
  return_logits: bool = False
  # The number of model forward passes to execute per engine step.
  num_scheduler_steps: int = 1
  # The engine's single sampling seed. Each step's key is derived from it, and
  # every row samples independently under that key.
  seed: int = 0

  def __post_init__(self):
    if self.num_scheduler_steps < 1:
      raise ValueError(
          f'num_scheduler_steps must be positive, got {self.num_scheduler_steps}.'
      )


@struct.dataclass
class SamplingMetadata:
  """The sampling settings of each request in a batch.

  Each array has one entry per request in the batch. Rows past the scheduled
  requests are padding, and sample greedily.
  """

  # Zero samples greedily.
  temperature: jax.Array | np.ndarray  # `[num_rows]` float32
  # Samples only from the smallest set of most likely tokens whose probability
  # reaches this value.
  top_p: jax.Array | np.ndarray        # `[num_rows]` float32, in (0, 1].
  # Samples only from the `top_k` most likely tokens. <= 0 means no limit.
  top_k: jax.Array | np.ndarray        # `[num_rows]` int32, in [1, max_top_k].

  # TODO(yatlas): Remove sampling_params. It is being deprecated.
  @classmethod
  def from_sampling_params(
      cls,
      sampling_params: tuple[sampler_lib.SamplingParams, ...],
      num_rows: int,
      max_top_k: int,
  ) -> 'SamplingMetadata':
    """Packs the scheduled requests' sampling settings into host arrays.

    Args:
      sampling_params: The sampling settings of each scheduled request, in row
        order.
      num_rows: The row capacity of the batch.
      max_top_k: The maximum `top_k` value allowed.

    Returns:
      The metadata, padded out to `num_rows` rows.
    """
    if len(sampling_params) > num_rows:
      raise ValueError(
          f'Got {len(sampling_params)} sampling params for {num_rows} rows.'
      )

    temperature = np.zeros((num_rows,), dtype=np.float32)
    top_p = np.ones((num_rows,), dtype=np.float32)
    top_k = np.ones((num_rows,), dtype=np.int32)
    for row, params in enumerate(sampling_params):
      if (
          max_top_k > 0
          and params.top_k is not None
          and params.top_k > max_top_k
      ):
        raise ValueError(
            f'top_k={params.top_k} exceeds the engine max_top_k={max_top_k}.'
        )
      temperature[row] = params.temperature

      # No `top_p` keeps every token.
      top_p[row] = 1.0 if params.top_p is None else params.top_p
      top_k[row] = max_top_k if params.top_k is None else params.top_k

    return cls(temperature=temperature, top_p=top_p, top_k=top_k)


def sample(
    logits: jax.Array,
    key: jax.Array,
    metadata: SamplingMetadata,
    max_top_k: int,
    return_logprobs: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Samples one token per row, under each row's own sampling settings.

  A row with zero temperature takes its most likely token. Every other row
  samples from its `top_k` most likely tokens, cut down further to the smallest
  prefix whose probability reaches its `top_p`, at its own temperature.

  Args:
    logits: The `[num_rows, vocab_size]` next-token logits.
    key: The step's PRNG key, derived from the engine seed. Every row samples
      independently under it.
    metadata: Each row's sampling settings.
    max_top_k: How many candidate tokens to draw for every row. Each row's
      `top_k` must not exceed it. -1 draws every token in the vocabulary.
    return_logprobs: Whether to return the sampled tokens' logprobs.

  Returns:
    A tuple of
    - `next_tokens`: The `[num_rows]` sampled tokens.
    - `logprobs`: The `[num_rows]` logprobs, or None if `return_logprobs` is
      False.
  """
  # For most current use cases of Tunix, k does not vary across rows. If this
  # changes, this function should be updated to sample individually per row so
  # that performance is not dependent on max_top_k.

  is_greedy = metadata.temperature <= 0.0
  temperature = jnp.where(is_greedy, 1.0, metadata.temperature)

  vocab_size = logits.shape[-1]
  k = vocab_size if max_top_k <= 0 else min(max_top_k, vocab_size)

  candidate_logits, candidate_tokens = jax.lax.top_k(logits, k)
  scaled_logits = candidate_logits.astype(jnp.float32) / temperature[:, None]

  ranks = jnp.arange(k)
  top_k = jnp.where(metadata.top_k <= 0, k, metadata.top_k)
  scaled_logits = jnp.where(
      ranks[None, :] < top_k[:, None], scaled_logits, -jnp.inf
  )

  # Keep each token whose predecessors have not yet reached `top_p`, which
  # always keeps the most likely token.
  probs = jax.nn.softmax(scaled_logits, axis=-1)
  probs_before = jnp.cumsum(probs, axis=-1) - probs
  scaled_logits = jnp.where(
      probs_before > metadata.top_p[:, None], -jnp.inf, scaled_logits
  )

  sampled = jax.random.categorical(key, scaled_logits, axis=-1)
  sampled_tokens = jnp.take_along_axis(
      candidate_tokens, sampled[:, None], axis=-1
  )[:, 0]
  # The candidates are sorted, so the first one is the most likely token.
  next_tokens = jnp.where(is_greedy, candidate_tokens[:, 0], sampled_tokens)

  if not return_logprobs:
    return next_tokens, None

  # Upscale logits before computing logprobs to preserve precision.
  scaled_full_logits = logits.astype(jnp.float32) / temperature[:, None]

  logp = jax.nn.log_softmax(scaled_full_logits, axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_tokens[:, None], axis=-1)
  return next_tokens, logp_sampled[:, 0]


class ModelRunner:
  """Runs the transformer forward pass and samples tokens."""

  def __init__(
      self,
      transformer: nnx.Module,
      config: ModelRunnerConfig,
  ):
    """Initializes the runner.

    Args:
      transformer: An instance of the transformer to run.
      config: The runner's configuration.
    """
    self._transformer_graphdef: graph.GraphDef[nnx.Module] = nnx.graphdef(
        transformer
    )
    self._transformer_state: statelib.State = nnx.variables(transformer)
    self._flattened_transformer_state: list[nnx.Variable] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    self._config = config

  @property
  def config(self) -> ModelRunnerConfig:
    return self._config

  def model_def_and_state(
      self,
  ) -> tuple[graph.GraphDef[nnx.Module], list[nnx.Variable]]:
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> statelib.State:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: statelib.State) -> None:
    def get_all_param_types(tree):
      param_types = set()
      jax.tree_util.tree_map(
          lambda x: param_types.add(type(x)),
          tree,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      return param_types

    def check_tree_structure(tree1, tree2):
      if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(
          tree2
      ):
        raise ValueError(
            'New state must have the same structure as the old state.'
            f' {jax.tree_util.tree_structure(tree1)} vs'
            f' {jax.tree_util.tree_structure(tree2)}'
        )

      def check_shape_dtype_sharding(x, y):

        def equivalent_sharding(x, y):
          # Lift the condition on memory_kind due to offloading.
          # Besides it seems jax.jit might change some shardings of the params
          # to equivalent representation so here we check if the specs are
          # equivalent instead of checking the identity.
          if isinstance(
              x.sharding, jax.sharding.SingleDeviceSharding
          ) and isinstance(y.sharding, jax.sharding.SingleDeviceSharding):
            return x.sharding.device_set == y.sharding.device_set
          if not (
              isinstance(x.sharding, jax.sharding.NamedSharding)
              and isinstance(y.sharding, jax.sharding.NamedSharding)
          ):
            return False
          if x.sharding.mesh != y.sharding.mesh:
            return False
          mesh = x.sharding.mesh
          diff_spec = list(set(x.sharding.spec) - set(y.sharding.spec))
          for spec in diff_spec:
            if spec and mesh.shape[spec] != 1:
              return False
          return True

        return (
            jnp.shape(x) == jnp.shape(y)
            and x.dtype == y.dtype
            and equivalent_sharding(x, y)
        )

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2)
          )
      ):
        raise ValueError(
            'New state must have the same shape, dtype and sharding as the old'
            f' state. {tree1} vs {tree2}'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      # Full state replacement.
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      # LoRA state replacement.
      if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
        raise ValueError(
            'Only LoRAParam is supported. Received invalid `param_types`: '
            f'{param_types}'
        )
      original_lora_params = statelib.filter_state(
          self._transformer_state, nnx.LoRAParam
      )
      check_tree_structure(original_lora_params, state)
      base_state = statelib.filter_state(
          self._transformer_state, filterlib.Not(nnx.LoRAParam)
      )
      self._transformer_state = statelib.merge_state(base_state, state)

    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Updates model parameters in the sampler."""
    if filter_types is not None:
      dst_params = nnx.state(self.transformer, filter_types)
      dst_leaves = jax.tree.leaves(dst_params)
      if dst_leaves and all(
          isinstance(x, jax.Array)
          and isinstance(x.sharding, jax.sharding.NamedSharding)
          for x in dst_leaves
      ):
        resharded_params = reshard.reshard_pytree(updated_weights, dst_params)
      else:
        resharded_params = updated_weights
    else:
      resharded_params = updated_weights
    flat_new_params, _ = rl_utils.to_flat_dict(resharded_params)
    # TODO(linchai): Cast on rollout devices when from lower precision to
    # higher precision.
    new_params_precision = jax.tree.leaves(flat_new_params)[0].dtype
    rollout_precision = jax.tree.leaves(self.transformer_state)[0].dtype
    if new_params_precision != rollout_precision:
      flat_new_params = jax.tree.map(
          lambda x: x.astype(rollout_precision), flat_new_params
      )
    flat_old_params, tree_def = rl_utils.to_flat_dict(
        self.transformer_state
    )
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(self._transformer_graphdef, merged_params)  # pyrefly: ignore[no-matching-overload]
    self.transformer_state = nnx.variables(new_model, nnx.Param)

  # TODO(yatlas): dtype should have one source of truth.
  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype
