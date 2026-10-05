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
from typing import Any, Optional, Tuple, TypeVar

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
from tunix.models import paged_attention
from tunix.rl import reshard
from tunix.rl import utils as rl_utils


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelRunnerConfig:
  """Configuration for the model runner."""

  # The maximum `top_k` value allowed in `SamplingParams`.
  # -1 to set no limit.
  max_top_k: int
  # The device mesh the transformer runs on.
  mesh: jax.sharding.Mesh
  # Whether or not return logprobs.
  return_logprobs: bool = False
  # Whether or not to return logits.
  return_logits: bool = False
  # The number of model forward passes to execute per engine step.
  num_scheduler_steps: int = 1
  # Seeds the random seeds drawn for requests that do not set their own.
  seed: int = 0

  def __post_init__(self):
    if self.max_top_k != -1 and self.max_top_k < 1:
      raise ValueError(
          f'max_top_k must be positive or -1, got {self.max_top_k}.'
      )

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
  # Samples only from the `top_k` most likely tokens. -1 means no limit.
  top_k: jax.Array | np.ndarray        # `[num_rows]` int32, in [1, max_top_k].
  #  The seed of each row's key.
  seed: jax.Array | np.ndarray         # `[num_rows]` uint32.

  @classmethod
  def from_sampling_params(
      cls,
      sampling_params: tuple[sampler_lib.SamplingParams, ...],
      num_rows: int,
      max_top_k: int,
      rng: np.random.Generator,
  ) -> 'SamplingMetadata':
    """Packs the scheduled requests' sampling settings into ragged arrays.

    Args:
      sampling_params: The sampling settings of each scheduled request, in row
        order.
      num_rows: The row capacity of the batch.
      max_top_k: The maximum `top_k` value allowed.
      rng: Draws a random seed for each request that does not set one.

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
    seed = np.zeros((num_rows,), dtype=np.uint32)
    for row, params in enumerate(sampling_params):
      if (
          max_top_k != -1
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

      # A request without a seed samples randomly. It draws a new
      # seed every step.
      if params.seed is None:
        seed[row] = rng.integers(
            np.iinfo(np.uint32).max, dtype=np.uint32, endpoint=True
        )
      else:
        seed[row] = params.seed

    return cls(temperature=temperature, top_p=top_p, top_k=top_k, seed=seed)


def sample(
    logits: jax.Array,
    keys: jax.Array,
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
    keys: The `[num_rows]` PRNG key of each row.
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
  next_token_logits = logits.astype(jnp.float32)

  is_greedy = metadata.temperature <= 0.0
  temperature = jnp.where(is_greedy, 1.0, metadata.temperature)
  scaled_logits = next_token_logits / temperature[:, None]

  vocab_size = scaled_logits.shape[-1]
  k = vocab_size if max_top_k == -1 else min(max_top_k, vocab_size)
  candidate_logits, candidate_tokens = jax.lax.top_k(scaled_logits, k)

  ranks = jnp.arange(k)
  top_k = jnp.where(metadata.top_k == -1, k, metadata.top_k)
  candidate_logits = jnp.where(
      ranks[None, :] < top_k[:, None], candidate_logits, -jnp.inf
  )

  # Keep each token whose predecessors have not yet reached `top_p`, which
  # always keeps the most likely token.
  probs = jax.nn.softmax(candidate_logits, axis=-1)
  probs_before = jnp.cumsum(probs, axis=-1) - probs
  candidate_logits = jnp.where(
      probs_before > metadata.top_p[:, None], -jnp.inf, candidate_logits
  )

  sampled = jax.vmap(jax.random.categorical)(keys, candidate_logits)
  sampled_tokens = jnp.take_along_axis(
      candidate_tokens, sampled[:, None], axis=-1
  )[:, 0]
  # The candidates are sorted, so the first one is the most likely token.
  next_tokens = jnp.where(is_greedy, candidate_tokens[:, 0], sampled_tokens)

  if not return_logprobs:
    return next_tokens, None
  logp = jax.nn.log_softmax(scaled_logits, axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_tokens[:, None], axis=-1)
  return next_tokens, logp_sampled[:, 0]


Cache = dict[str, jax.Array]
_PyTree = TypeVar('_PyTree')


@struct.dataclass
class RaggedArray:
  """A packed array of variable-length rows."""

  # The packed elements of each row.
  data: jax.Array  # `[total_capacity, *element_shape]`
  # The length of each row.
  lens: jax.Array  # `[num_rows]` int32

  @property
  def row_idxs(self) -> jax.Array:
    """The `[total_capacity]` indices of each element's row."""
    # Example:
    # data: [a, b, c, d, e, f, PAD, PAD]
    # lens: [2, 1, 3]
    # row_idxs: [0, 0, 1, 2, 2, 2, 2, 2]

    return jnp.repeat(
        jnp.arange(len(self.lens)),
        self.lens,
        total_repeat_length=self.data.shape[0],
    )

  @property
  def intra_offsets(self) -> jax.Array:
    """The `[total_capacity]` offsets of each element within its row."""
    # Example:
    # data: [a, b, c, d, e, f, PAD, PAD]
    # lens: [2, 1, 3]
    # intra_offsets: [0, 1, 0, 1, 2, 3, 4, 5]

    total_capacity = self.data.shape[0]
    positions = jnp.arange(total_capacity)
    starts = jnp.zeros_like(self.lens)
    if len(self.lens) > 0:
      starts = starts.at[1:].set(jnp.cumsum(self.lens)[:-1])
    return positions - starts[self.row_idxs]


@functools.partial(jax.jit, static_argnames=['num_steps'])
def _stack_steps(
    first_step: jax.Array,
    decode_steps: jax.Array | None,
    num_steps: int,
) -> jax.Array:
  """Packs per-step outputs into a `[num_rows, num_steps, ...]` device array.

  Args:
    first_step: The `[num_rows, ...]` output of the scheduled batch's step.
    decode_steps: The `[num_scheduler_steps - 1, num_rows, ...]` output of the
      decode loop, or None when the loop did not run.
    num_steps: The step capacity to pad out to.

  Returns:
    The outputs indexed by `[row, step, ...]`.
  """
  if decode_steps is not None:
    return jnp.concatenate(
        [first_step[:, None], jnp.swapaxes(decode_steps, 0, 1)], axis=1
    )
  if num_steps == 1:
    return first_step[:, None]
  out = jnp.zeros(
      (first_step.shape[0], num_steps, *first_step.shape[1:]),
      dtype=first_step.dtype,
  )
  return out.at[:, 0].set(first_step)


def _verify_metadata(metadata: paged_attention.RPAMetadata, num_tokens: int) -> None:
  """Checks that `metadata` describes a well-formed batch of `num_tokens`.

  Args:
    metadata: The execution metadata to check.
    num_tokens: The capacity of the ragged token buffer.

  Raises:
    ValueError: If the metadata is malformed.
  """
  kv_lens = np.asarray(metadata.kv_lens)
  query_lens = np.asarray(metadata.query_lens)
  distribution = np.asarray(metadata.distribution)
  if kv_lens.ndim != 1 or kv_lens.shape != query_lens.shape:
    raise ValueError(
        'kv_lens and query_lens must be 1-D with one entry per row, got shapes'
        f' {kv_lens.shape} and {query_lens.shape}.'
    )
  num_rows = query_lens.shape[0]
  for name, pages in metadata.page_indices.items():
    if np.ndim(pages) != 2 or np.shape(pages)[0] != num_rows:
      raise ValueError(
          f'page_indices[{name!r}] must be [{num_rows}, max_pages_per_seq],'
          f' got shape {np.shape(pages)}.'
      )
  if distribution.shape != (3,):
    raise ValueError(
        f'distribution must have shape (3,), got {distribution.shape}.'
    )
  num_decodes, num_chunked_end, num_seqs = (int(d) for d in distribution)
  if not 0 <= num_decodes <= num_chunked_end <= num_seqs <= num_rows:
    raise ValueError(
        f'distribution {distribution.tolist()} must be non-decreasing and'
        f' within [0, {num_rows}].'
    )
  scheduled = query_lens[:num_seqs]
  if np.any(scheduled < 1):
    raise ValueError(
        f'Every scheduled row must run a token, got query_lens {scheduled}.'
    )
  if np.any(query_lens[:num_decodes] != 1):
    raise ValueError(
        'Every decode row must run exactly one token, got query_lens'
        f' {query_lens[:num_decodes]}.'
    )
  if np.any(query_lens > kv_lens):
    raise ValueError(
        f'Query_lens {query_lens} must not exceed kv_lens {kv_lens}.'
    )
  if np.any(query_lens[num_seqs:] != 0) or np.any(kv_lens[num_seqs:] != 0):
    raise ValueError('Padding rows must have zero query_lens and kv_lens.')
  if int(query_lens.sum()) > num_tokens:
    raise ValueError(
        f'The rows run {int(query_lens.sum())} tokens, more than the'
        f' {num_tokens} in the token buffer.'
    )


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
      config: Settings shared by every request in the engine. The config is
        fixed for the runner's lifetime: it is baked into the compiled step
        functions, so changing it requires a new runner.
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

    # Draws the seeds of requests that do not set their own.
    self._rng = np.random.default_rng(config.seed)

    self._compiled_model_step_fn = jax.jit(
        self._model_step_fn, donate_argnames=['cache']
    )
    self._compiled_decode_loop_fn = jax.jit(
        self._decode_loop_fn,
        donate_argnames=['cache'],
        static_argnames=['num_steps'],
    )

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
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates model parameters in the sampler."""
    if filter_types is not None:
      dst_params = nnx.state(self.transformer, filter_types)
      try:
        resharded_params = reshard.reshard_pytree(updated_weights, dst_params)
      except (AttributeError, ValueError):
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

  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def _model_step_fn(
      self,
      params: list[nnx.Variable],
      cache: Cache,
      tokens: jax.Array,
      metadata: paged_attention.RPAMetadata,
      sampling_metadata: SamplingMetadata,
  ) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache]:
    """Runs the model over a mixed prefill + decode batch and samples the next tokens.

    Args:
      params: The flattened transformer state to run with.
      cache: The physical KV cache pages to read and write.
      tokens: The ragged array of tokens for the batch.
      metadata: Ragged execution metadata for the attention kernel.
      sampling_metadata: Each row's sampling settings.

    Returns:
      A tuple of 
      - next_tokens: `[num_rows]` array of sampled tokens.
      - logits: `[num_rows, vocab_size]` array of logits, or None.
      - logprobs: `[num_rows]` array of logprobs, or None.
      - updated_cache: The updated KV cache pages.
    """
    config = self._config

    transformer = nnx.merge(self._transformer_graphdef, params)

    # Example:
    # query_lens: [2, 3, 1]
    # start_positions: [0, 4, 3]
    # positions: [0, 1, 0, 1, 2, 0] + [0, 0, 4, 4, 4, 3]
    ragged = RaggedArray(data=tokens, lens=jnp.asarray(metadata.query_lens))
    start_positions = metadata.kv_lens - metadata.query_lens
    positions = ragged.intra_offsets + start_positions[ragged.row_idxs]

    logits, updated_cache = transformer(
        tokens,
        positions,
        cache=cache,
        metadata=metadata,
        mesh=config.mesh,
    )

    # TODO(yatlas): Pass `last_token_idxs` to the model, so it can gather the
    # hidden states before the unembedding rather than computing logits for
    # every token.
    last_token_idxs = jnp.maximum(0, jnp.cumsum(metadata.query_lens) - 1)
    logits = logits[last_token_idxs]

    # Each row samples the token at position `kv_lens`, right after the KV it
    # holds.
    keys = jax.vmap(
        lambda seed, position: jax.random.fold_in(
            jax.random.PRNGKey(seed), position
        )
    )(sampling_metadata.seed, metadata.kv_lens)

    next_tokens, logp = sample(
        logits,
        keys,
        sampling_metadata,
        max_top_k=config.max_top_k,
        return_logprobs=config.return_logprobs,
    )

    res_logits = logits if config.return_logits else None
    return next_tokens, res_logits, logp, updated_cache

  def _decode_loop_fn(
      self,
      params: list[nnx.Variable],
      cache: Cache,
      next_tokens: jax.Array,
      metadata: paged_attention.RPAMetadata,
      sampling_metadata: SamplingMetadata,
      num_steps: int,
  ) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache]:
    """Decodes `num_steps` tokens from the batch `metadata` just ran.

    Args:
      params: The flattened transformer state to run with.
      cache: The physical KV cache pages to read and write.
      next_tokens: The tokens sampled by the step that just ran, indexed by the
        row of the scheduled batch.
      metadata: The metadata of the step that just ran.
      sampling_metadata: The sampling settings of the step that just ran,
        indexed by the row of the scheduled batch.
      num_steps: How many decode steps to run.

    Returns:
      A tuple of
      - tokens: `[num_steps, num_rows]` array of sampled tokens.
      - logits: `[num_steps, num_rows, vocab_size]` array of logits, or None.
      - logprobs: `[num_steps, num_rows]` array of logprobs, or None.
      - updated cache: The updated KV cache pages.
    """
    num_rows = metadata.query_lens.shape[0]
    distribution = metadata.distribution

    # Rows are ordered `[decodes, chunked prefills, full prefills]`. A chunked
    # prefill still has prompt left to process, so its sampled token is
    # discarded and it leaves the batch for the decode loop.
    rows = jnp.arange(num_rows)
    is_chunked_prefill = (rows >= distribution[0]) & (rows < distribution[1])
    is_surviving = (rows < distribution[2]) & ~is_chunked_prefill
    num_decodes = distribution[0]
    num_full_prefills = distribution[2] - distribution[1]
    num_surviving = num_decodes + num_full_prefills
    # Move the surviving rows to the front, keeping their relative order.
    order = jnp.argsort(~is_surviving, stable=True)

    query_lens = jnp.where(rows < num_surviving, 1, 0)
    kv_lens = (
        jnp.where(rows < num_surviving, jnp.asarray(metadata.kv_lens)[order], 0)
        + query_lens
    )
    decode_metadata = paged_attention.RPAMetadata(
        page_indices={
            cache_name: jnp.asarray(idxs)[order]
            for cache_name, idxs in metadata.page_indices.items()
        },
        kv_lens=kv_lens,
        query_lens=query_lens,
        distribution=jnp.full_like(distribution, num_surviving),
        chunk_prefill_size=metadata.chunk_prefill_size,
    )

    # Each surviving request's sampling settings move along with its row. The
    # rows past `num_surviving` become padding.
    def compact(values, padding):
      return jnp.where(
          rows < num_surviving, jnp.asarray(values)[order], padding
      )

    decode_sampling_metadata = SamplingMetadata(
        temperature=compact(sampling_metadata.temperature, 0.0),
        top_p=compact(sampling_metadata.top_p, 1.0),
        top_k=compact(sampling_metadata.top_k, 1),
        seed=compact(sampling_metadata.seed, 0),
    )

    # Each surviving sequence feeds the token it just sampled back in, packed
    # into the leading rows to match the compacted metadata.
    decode_tokens = next_tokens[order]
    inverse_order = jnp.argsort(order)

    def decode_step(carry, _):
      cache, tokens, kv_lens = carry
      step_metadata = dataclasses.replace(decode_metadata, kv_lens=kv_lens)
      next_tokens, logits, logp, cache = self._model_step_fn(
          params,
          cache,
          tokens,
          step_metadata,
          decode_sampling_metadata,
      )
      carry = (cache, next_tokens, kv_lens + query_lens)
      return carry, (next_tokens, logits, logp)

    carry, outputs = jax.lax.scan(
        decode_step,
        (cache, decode_tokens, decode_metadata.kv_lens),
        xs=None,
        length=num_steps,
    )
    cache, _, _ = carry
    tokens, logits, logp = outputs
    # Send each output row back to the scheduled row it came from.
    return (
        tokens[:, inverse_order],
        logits[:, inverse_order] if logits is not None else None,
        logp[:, inverse_order] if logp is not None else None,
        cache,
    )

  def _to_device(self, tree: _PyTree) -> _PyTree:
    """Moves host arrays onto the device, replicated across the mesh."""
    return jax.device_put(
        tree,
        jax.sharding.NamedSharding(
            self._config.mesh, jax.sharding.PartitionSpec()
        ),
    )

  def execute_step(
      self,
      cache: Cache,
      tokens: jax.Array | np.ndarray,
      metadata: paged_attention.RPAMetadata,
      sampling_params: tuple[sampler_lib.SamplingParams, ...],
  ) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache]:
    """Runs `num_scheduler_steps` model forward passes on a batch of requests,
    and samples tokens.

    Args:
      cache: The physical KV cache pages to read and write.
      tokens: The ragged array of tokens for the batch.
      metadata: Ragged execution metadata for the attention kernel.
      sampling_params: The sampling settings of each scheduled request, in row
        order.

    Returns:
      A tuple of:
        - tokens: `[num_rows, num_scheduler_steps]` array of sampled tokens.
        - logits: `[num_rows, num_scheduler_steps, vocab_size]` array of
          logits, or None.
        - logprobs: `[num_rows, num_scheduler_steps]` array of logprobs, or
          None.
        - updated cache: The updated KV cache pages.

    Raises:
      ValueError: If the metadata is malformed, or `sampling_params` does
        not hold one entry per scheduled request.
    """
    config = self._config
    _verify_metadata(metadata, tokens.shape[0])
    num_rows = metadata.query_lens.shape[0]
    num_scheduled = int(metadata.distribution[2])
    if len(sampling_params) != num_scheduled:
      raise ValueError(
          f'Got {len(sampling_params)} sampling params for {num_scheduled}'
          ' scheduled requests.'
      )

    device_metadata = self._to_device(metadata)
    sampling_metadata = self._to_device(
        SamplingMetadata.from_sampling_params(
            sampling_params, num_rows, self._config.max_top_k, self._rng
        )
    )

    next_tokens, logits, logp, cache = self._compiled_model_step_fn(
        self._flattened_transformer_state,
        cache,
        tokens,
        device_metadata,
        sampling_metadata,
    )

    # Chunked prefills drop out, but the rest of the batch carries on.
    distribution = metadata.distribution

    # distribution = [i, j, k]
    # i: num decodes
    # j: i + num chunked prefills
    # k: j + num full prefills
    # num_surviving: num decodes + num full prefills = i + k - j
    num_surviving = distribution[0] + distribution[2] - distribution[1]

    num_steps = config.num_scheduler_steps
    num_decode_steps = num_steps - 1 if num_surviving else 0

    decode_tokens = decode_logits = decode_logp = None
    if num_decode_steps:
      decode_tokens, decode_logits, decode_logp, cache = (
          self._compiled_decode_loop_fn(
              self._flattened_transformer_state,
              cache,
              next_tokens,
              device_metadata,
              sampling_metadata,
              num_steps=num_decode_steps,
          )
      )

    out_tokens = _stack_steps(next_tokens, decode_tokens, num_steps)
    out_logits = (
        _stack_steps(logits, decode_logits, num_steps)
        if logits is not None
        else None
    )
    out_logp = (
        _stack_steps(logp, decode_logp, num_steps) if logp is not None else None
    )
    return out_tokens, out_logits, out_logp, cache
