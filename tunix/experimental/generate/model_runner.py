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

"""Model runner for continuous batching.

`ModelRunner` is the execution engine that runs the transformer forward pass
and handles token sampling on device for rollout generation. It is invoked for
every engine step.
"""

import dataclasses
import functools
import inspect
import operator
from typing import Any, Optional, Tuple

from flax import nnx
from flax import struct
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tunix.rl import reshard
from tunix.rl import utils as rl_utils

Cache = dict[str, jax.Array]
# The metadata arrays are built on the host and used on the device, so the row
# helpers have to accept either.
ArrayLike = np.ndarray | jnp.ndarray


@struct.dataclass
class RPAMetadata:
  """Encapsulates execution metadata arrays for the Ragged Page Attention kernel."""

  page_indices: dict[str, np.ndarray | jnp.ndarray]
  kv_lens: np.ndarray | jnp.ndarray
  query_lens: np.ndarray | jnp.ndarray
  distribution: np.ndarray | jnp.ndarray
  mesh: jax.sharding.Mesh | None = None
  chunk_prefill_size: int | None = struct.field(default=None, pytree_node=False)


@struct.dataclass
class RaggedArray:
  """Encapsulates a ragged array."""

  data: jnp.ndarray
  lens: jnp.ndarray

  @property
  def row_idxs(self) -> jnp.ndarray:
    return jnp.repeat(
        jnp.arange(len(self.lens)),
        self.lens,
        total_repeat_length=self.data.shape[0],
    )

  @property
  def intra_offsets(self) -> jnp.ndarray:
    num_tokens = self.data.shape[0]
    positions = jnp.arange(num_tokens)
    starts = jnp.zeros_like(self.lens)
    if len(self.lens) > 0:
      starts = starts.at[1:].set(jnp.cumsum(self.lens)[:-1])
    return positions - starts[self.row_idxs]


GREEDY_SAMPLING_MODE = 'greedy'
TOP_P_SAMPLING_MODE = 'top_p'
BEAM_SEARCH_SAMPLING_MODE = 'beam_search'
SUPPORTED_SAMPLING_MODES = (
    GREEDY_SAMPLING_MODE,
    TOP_P_SAMPLING_MODE,
    BEAM_SEARCH_SAMPLING_MODE,
)

# Keys recognized in `ModelRunnerConfig.sampling_parameters`.
TEMPERATURE = 'temperature'
TOP_P = 'top_p'
TOP_K = 'top_k'
BEAM_SIZE = 'beam_size'

# The sampling parameters each mode accepts. Parameters are validated against
# this map so that a knob silently ignored by the active mode is an error
# rather than a surprise at generation time.
_SAMPLING_PARAMETERS_BY_MODE = {
    GREEDY_SAMPLING_MODE: frozenset(),
    TOP_P_SAMPLING_MODE: frozenset({TEMPERATURE, TOP_P, TOP_K}),
    BEAM_SEARCH_SAMPLING_MODE: frozenset({BEAM_SIZE}),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelRunnerConfig:
  """Sampling parameters shared by every request in an engine."""

  # The sampling mode to use, one of "greedy", "top_p", or "beam_search".
  sampling_mode: str
  # For top_p, it contains "top_p", "top_k", and temperature.
  # For beam search, it contains "beam_size".
  sampling_parameters: dict[str, float | int] = dataclasses.field(
      default_factory=dict
  )
  # Forbidden token ids.
  forbidden_token_ids: tuple[int, ...] | None = None
  # Whether or not return logprobs.
  return_logprobs: bool = False
  # Whether or not to return logits.
  return_logits: bool = False
  # Whether to return the prompt as part of the output sample. Consumed when
  # the engine assembles request output, not by the runner itself.
  echo: bool = False
  # The number of decode steps to execute per sampling step.
  num_decode_steps: int = 1
  mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    if self.sampling_mode not in SUPPORTED_SAMPLING_MODES:
      raise ValueError(
          f'Unsupported sampling mode: {self.sampling_mode}. Supported modes '
          f'are {list(SUPPORTED_SAMPLING_MODES)}.'
      )

    if self.sampling_mode == BEAM_SEARCH_SAMPLING_MODE:
      raise NotImplementedError(
          f"sampling_mode='{BEAM_SEARCH_SAMPLING_MODE}' is reserved but not "
          'implemented yet.'
      )

    accepted = _SAMPLING_PARAMETERS_BY_MODE[self.sampling_mode]
    unexpected = set(self.sampling_parameters) - accepted
    if unexpected:
      raise ValueError(
          f"sampling_mode='{self.sampling_mode}' ignores "
          f'{sorted(unexpected)}. Accepted parameters: {sorted(accepted)}.'
      )

    if self.num_decode_steps < 1:
      raise ValueError(
          f'num_decode_steps must be positive, got {self.num_decode_steps}.'
      )

    # `sample_top_p` divides the logits by the temperature.
    if self.sampling_mode == TOP_P_SAMPLING_MODE and self.temperature <= 0.0:
      raise ValueError(
          f"sampling_mode='{TOP_P_SAMPLING_MODE}' divides the logits by "
          'temperature, which must be greater than 0; got '
          f'temperature={self.temperature}. Use '
          f"sampling_mode='{GREEDY_SAMPLING_MODE}' instead."
      )

  @property
  def temperature(self) -> float:
    return float(self.sampling_parameters.get(TEMPERATURE, 0.0))

  @property
  def top_p(self) -> float | None:
    top_p = self.sampling_parameters.get(TOP_P)
    return None if top_p is None else float(top_p)

  @property
  def top_k(self) -> int | None:
    top_k = self.sampling_parameters.get(TOP_K)
    return None if top_k is None else int(top_k)

  @property
  def beam_size(self) -> int | None:
    beam_size = self.sampling_parameters.get(BEAM_SIZE)
    return None if beam_size is None else int(beam_size)


def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    return_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Sample a token using top-p sampling."""
  if top_p is None:
    top_p = 1.0
  next_token_logits = logits[:, -1].astype(jnp.float32) / temperature

  _no_topk = top_k is None or top_k <= 0
  if top_p >= 1.0 and _no_topk:
    next_token = jax.random.categorical(key, logits=next_token_logits)
    if not return_logprobs:
      return next_token, None
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
    return next_token, logp_sampled

  if top_k is not None and top_k > 0:
    k = min(top_k, next_token_logits.shape[-1])
  else:
    k = next_token_logits.shape[-1]
  logits_sorted, indices = jax.lax.top_k(next_token_logits, k=k)  # pytype: disable=bad-argument-type

  probs_sorted = jax.nn.softmax(logits_sorted, axis=-1)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > top_p
  logits_sorted = jnp.where(mask, -jnp.inf, logits_sorted)

  next_token_idx = jax.random.categorical(key, logits=logits_sorted)
  next_token = jnp.take_along_axis(indices, next_token_idx[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)

  if return_logprobs:
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  else:
    logp_sampled = None

  return next_token, logp_sampled


def sample_best(
    logits: jnp.ndarray, return_logprobs: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  next_token = jnp.argmax(logits[:, -1], axis=-1)
  if not return_logprobs:
    return next_token, None
  logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
  logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  return next_token, logp_sampled


def _num_surviving_rows(distribution: ArrayLike) -> ArrayLike:
  """Returns how many rows take another decode step.

  Every decode and every full prefill carries on, and the chunked prefills
  between them drop out, so the count follows from the row layout alone.

  Args:
    distribution: The `(num_decodes, num_chunked_end, num_seqs)` row layout of
      the step that just ran.

  Returns:
    The number of surviving rows.
  """
  return distribution[0] + distribution[2] - distribution[1]


def _surviving_row_mask(
    distribution: ArrayLike, num_rows: int
) -> jnp.ndarray:
  """Returns a mask over the rows that can take another decode step.

  Rows are ordered `[decodes, chunked prefills, full prefills]`. A chunked
  prefill has only been partially processed, so the token sampled for it is
  meaningless and the scheduler reserved no KV slots for it beyond this
  chunk; it therefore drops out of the batch. Everything else carries on.

  Args:
    distribution: The `(num_decodes, num_chunked_end, num_seqs)` row layout of
      the step that just ran.
    num_rows: The row capacity of the batch, which never varies.

  Returns:
    A `[num_rows]` boolean mask that is true for the surviving rows.
  """
  rows = jnp.arange(num_rows)
  is_chunked_prefill = (rows >= distribution[0]) & (rows < distribution[1])
  is_scheduled = rows < distribution[2]
  return is_scheduled & ~is_chunked_prefill


def _compaction_order(surviving: jnp.ndarray) -> jnp.ndarray:
  """Returns the permutation that moves the surviving rows to the front.

  The attention kernel reads `distribution` as three contiguous row ranges, so
  a row can only be dropped from the batch by moving it past the last scheduled
  row. Sorting on the negated mask is stable, so the surviving rows keep their
  relative order and the dropped rows trail behind them.

  Args:
    surviving: The `[num_rows]` mask of surviving rows.

  Returns:
    A `[num_rows]` permutation of the row indices.
  """
  return jnp.argsort(~surviving, stable=True)


def _mask_rows(values: ArrayLike, num_rows: ArrayLike) -> jnp.ndarray:
  """Zeroes every row of `values` at or past `num_rows`."""
  rows = jnp.arange(values.shape[0]).reshape((-1,) + (1,) * (values.ndim - 1))
  return jnp.where(rows < num_rows, values, 0)


def _compact_rows(
    values: ArrayLike, order: jnp.ndarray, num_surviving: ArrayLike
) -> jnp.ndarray:
  """Reorders the rows of `values`, zeroing the rows past `num_surviving`."""
  return _mask_rows(values[order], num_surviving)


def _advance_kv_lens(
    kv_lens: ArrayLike, query_lens: ArrayLike
) -> jnp.ndarray:
  """Grows each sequence's KV length by the queries the next step adds.

  The padding rows have no query to consume, which pins their KV length at
  zero for as long as the batch runs.

  Args:
    kv_lens: The `[num_rows]` KV lengths of the step that just ran.
    query_lens: The `[num_rows]` query lengths of the step being prepared.

  Returns:
    The KV lengths of the step being prepared.
  """
  return kv_lens + query_lens


def _decode_metadata(metadata: RPAMetadata, order: jnp.ndarray) -> RPAMetadata:
  """Rewrites the metadata for the decode step that follows `metadata`.

  The surviving sequences are compacted into the leading rows, each processing
  exactly the single token it just sampled, which turns any batch into a pure
  decode batch. Every later step then repeats only the `kv_lens` update, so
  this rewrite runs once per engine step rather than once per decode step.

  Args:
    metadata: The metadata of the step that just ran.
    order: The compaction permutation, as returned by `_compaction_order`.

  Returns:
    The metadata for the next step, keeping the row capacity unchanged.
  """
  num_surviving = _num_surviving_rows(metadata.distribution)
  query_lens = _mask_rows(jnp.ones_like(metadata.query_lens), num_surviving)
  return RPAMetadata(
      page_indices={
          cache_name: _compact_rows(idxs, order, num_surviving)
          for cache_name, idxs in metadata.page_indices.items()
      },
      kv_lens=_advance_kv_lens(
          _compact_rows(metadata.kv_lens, order, num_surviving), query_lens
      ),
      query_lens=query_lens,
      # The survivors are all decodes now, so they fill the decode range and
      # leave the chunked prefill and mixed ranges empty.
      distribution=jnp.full_like(metadata.distribution, num_surviving),
      mesh=metadata.mesh,
      chunk_prefill_size=metadata.chunk_prefill_size,
  )


def _scatter_rows(
    values: ArrayLike, order: jnp.ndarray, num_active: ArrayLike
) -> jnp.ndarray:
  """Sends each row of `values` back to the scheduled row it came from.

  Args:
    values: The `[num_rows, ...]` per-row output of the step that just ran.
    order: The permutation that produced the step's row layout, which maps a
      row back onto the row of the scheduled batch it came from.
    num_active: How many leading rows took part in the step. The rest are
      padding and carry whatever the model happened to emit.

  Returns:
    A `[num_rows, ...]` buffer holding the active rows, indexed by the
    scheduled batch's row order and zero everywhere else.
  """
  return jnp.zeros_like(values).at[order].set(_mask_rows(values, num_active))


def _stack_steps(
    first_step: jnp.ndarray, decode_steps: jnp.ndarray | None, num_steps: int
) -> jnp.ndarray:
  """Stacks per-step outputs into a `[num_rows, num_steps, ...]` array.

  Args:
    first_step: The `[num_rows, ...]` output of the scheduled batch's step.
    decode_steps: The `[num_decode_steps - 1, num_rows, ...]` output of the
      decode loop, or None when the loop did not run.
    num_steps: The step capacity to pad out to.

  Returns:
    The outputs indexed by `[row, step, ...]`.
  """
  steps = first_step[jnp.newaxis]
  if decode_steps is not None:
    steps = jnp.concatenate([steps, decode_steps], axis=0)
  padding = num_steps - steps.shape[0]
  if padding:
    steps = jnp.pad(steps, [(0, padding)] + [(0, 0)] * (steps.ndim - 1))
  return jnp.swapaxes(steps, 0, 1)


class ModelRunner:
  """Runs the transformer forward pass and samples tokens for an engine step."""

  def __init__(
      self,
      transformer: nnx.Module,
      config: ModelRunnerConfig,
      seed: int = 0,
  ):
    """Initializes the runner.

    Args:
      transformer: An instance of the transformer to run.
      config: Sampling parameters shared by every request in the engine. The
        config is fixed for the runner's lifetime: it is baked into the
        compiled step functions, so changing it requires a new runner.
      seed: The seed for sampled decoding.
    """
    self._config = config
    self._mesh = config.mesh

    self._transformer_graphdef = nnx.graphdef(transformer)
    self._transformer_state = nnx.variables(transformer)
    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )
    self._supports_mesh = (
        'mesh' in inspect.signature(transformer.__call__).parameters
    )

    self._seed = seed

    self._step_count = jnp.array(0)

    self._compiled_model_step_fn = jax.jit(
        self._model_step_fn,
        donate_argnames=['cache'],
    )
    self._compiled_decode_loop_fn = jax.jit(
        self._decode_loop_fn,
        donate_argnames=['cache'],
        static_argnames=['num_steps'],
    )

  @property
  def config(self) -> ModelRunnerConfig:
    return self._config

  def model_def_and_state(self) -> tuple[graph.GraphDef[Any], list[Any]]:
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> statelib.State:
    """The model's variables.

    Read only: the runner owns the state it executes, so new weights are
    merged in through `update_params`.
    """
    return self._transformer_state

  def _set_transformer_state(self, state: statelib.State) -> None:
    """Replaces the state the runner executes, after validating it matches.

    Args:
      state: Either the model's full parameters, or its LoRA parameters alone.
        Either way the structure, shapes, dtypes and sharding must match the
        state being replaced.

    Raises:
      ValueError: If `state` does not match the state being replaced.
    """
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
          if not hasattr(x, 'sharding') or not hasattr(y, 'sharding'):
            return True
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

          def get_axes(spec):
            axes = set()
            for s in spec:
              if isinstance(s, (tuple, list)):
                axes.update(s)
              elif s is not None:
                axes.add(s)
            return axes

          diff_spec = get_axes(x.sharding.spec) ^ get_axes(y.sharding.spec)
          for spec in diff_spec:
            if spec and mesh.shape.get(spec, 1) != 1:
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
            ' state.'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
        raise ValueError(
            'Only LoRAParam is supported. Received invalid `param_types`:'
            f' {param_types}'
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

  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and hasattr(
        self.transformer.config, 'dtype'
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

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
    self._set_transformer_state(nnx.variables(new_model, nnx.Param))

  def _sample_tokens(
      self,
      logits: jnp.ndarray,
      step_count: jax.Array,
  ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Samples the next token per sequence under the configured mode."""
    config = self._config
    if config.sampling_mode == GREEDY_SAMPLING_MODE:
      return sample_best(
          logits,
          return_logprobs=config.return_logprobs,
      )
    if config.sampling_mode == TOP_P_SAMPLING_MODE:
      key = jax.random.fold_in(jax.random.PRNGKey(self._seed), step_count)
      return sample_top_p(
          logits,
          key,
          temperature=config.temperature,
          top_p=config.top_p,
          top_k=config.top_k,
          return_logprobs=config.return_logprobs,
      )
    raise ValueError(f'Unsupported sampling mode: {config.sampling_mode}')

  def _model_step_fn(
      self,
      params: statelib.State,
      step_count: jax.Array,
      cache: Cache,
      tokens: jnp.ndarray,
      metadata: RPAMetadata,
  ) -> tuple[
      jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, Cache, jax.Array
  ]:
    """Runs the model over a mixed prefill + decode batch and samples a token.

    Args:
      params: The transformer parameters to run with.
      step_count: How many steps have been sampled so far, which seeds sampled
        decoding.
      cache: The physical KV cache pages to read and write.
      tokens: The packed tokens of every sequence in the batch.
      metadata: Ragged execution metadata for the attention kernel.

    Returns:
      A tuple of (next tokens, logits, logprobs, updated cache, step count),
      where logits and logprobs are None unless the config asks for them.
    """
    config = self._config

    transformer = nnx.merge(self._transformer_graphdef, params)
    kwargs = {}
    decode_only_last_token = (
        self._supports_decode_only_last_token and not config.return_logits
    )
    if self._supports_decode_only_last_token:
      kwargs['decode_only_last_token'] = decode_only_last_token
    if self._supports_mesh:
      kwargs['mesh'] = self._mesh

    ragged = RaggedArray(
        data=tokens,
        lens=metadata.query_lens,  # pytype: disable=bad-argument-type
    )
    start_positions = metadata.kv_lens - metadata.query_lens
    global_positions = ragged.intra_offsets + start_positions[ragged.row_idxs]

    logits, updated_cache = transformer(
        tokens,
        global_positions,
        cache=cache,
        metadata=metadata,
        **kwargs,
    )
    if not decode_only_last_token:
      # Only the last token of each sequence predicts the next one.
      last_token_idxs = jnp.maximum(0, jnp.cumsum(metadata.query_lens) - 1)
      logits = logits[last_token_idxs]
    if logits.ndim == 2:
      logits = jnp.expand_dims(logits, axis=1)

    if config.forbidden_token_ids:
      logits = logits.at[:, :, config.forbidden_token_ids].set(-jnp.inf)

    next_tokens, logp = self._sample_tokens(logits, step_count)

    res_logits = logits if config.return_logits else None
    res_logp = logp if config.return_logprobs else None
    return next_tokens, res_logits, res_logp, updated_cache, step_count + 1

  def _decode_loop_fn(
      self,
      params: statelib.State,
      step_count: jax.Array,
      cache: Cache,
      next_tokens: jnp.ndarray,
      metadata: RPAMetadata,
      num_steps: int,
  ) -> tuple[
      jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, Cache, jax.Array
  ]:
    """Decodes `num_steps` tokens from the batch `metadata` just ran.

    The scheduled batch is compacted onto its surviving rows once, up front,
    which leaves a pure decode batch whose only per-step change is
    `kv_lens += query_lens`. The loop therefore carries nothing but the cache,
    the sampled tokens, the KV lengths and the step count, and every row buffer
    keeps the batch's row capacity.

    Args:
      params: The transformer parameters to run with.
      step_count: How many steps have been sampled so far.
      cache: The physical KV cache pages to read and write.
      next_tokens: The tokens sampled by the step that just ran, indexed by the
        row of the scheduled batch.
      metadata: The metadata of the step that just ran.
      num_steps: How many decode steps to run.

    Returns:
      A tuple of (tokens, logits, logprobs, updated cache, step count), where
      the outputs are stacked as `[num_steps, num_rows, ...]` and indexed by
      the sequence's row in the scheduled batch. Logits and logprobs are None
      unless the config asks for them.
    """
    config = self._config
    num_rows = metadata.query_lens.shape[0]

    num_surviving = _num_surviving_rows(metadata.distribution)
    order = _compaction_order(
        _surviving_row_mask(metadata.distribution, num_rows)
    )
    decode_metadata = _decode_metadata(metadata, order)

    # Each surviving sequence feeds the token it just sampled back in, packed
    # into the leading rows to match the compacted metadata.
    decode_tokens = _compact_rows(next_tokens, order, num_surviving)

    def decode_step(carry, _):
      cache, tokens, kv_lens, step_count = carry
      step_metadata = decode_metadata.replace(kv_lens=kv_lens)
      next_tokens, logits, logp, cache, step_count = self._model_step_fn(
          params, step_count, cache, tokens, step_metadata
      )
      carry = (
          cache,
          _mask_rows(next_tokens, num_surviving),
          _advance_kv_lens(kv_lens, decode_metadata.query_lens),
          step_count,
      )
      outputs = (
          _scatter_rows(next_tokens, order, num_surviving),
          _scatter_rows(logits[:, 0], order, num_surviving)
          if config.return_logits
          else None,
          _scatter_rows(logp, order, num_surviving)
          if config.return_logprobs
          else None,
      )
      return carry, outputs

    carry, outputs = jax.lax.scan(
        decode_step,
        (cache, decode_tokens, decode_metadata.kv_lens, step_count),
        xs=None,
        length=num_steps,
    )
    cache, _, _, step_count = carry
    tokens, logits, logp = outputs
    return tokens, logits, logp, cache, step_count

  def _to_device(self, metadata: RPAMetadata) -> RPAMetadata:
    """Moves the host metadata onto the device, once per engine step.

    Args:
      metadata: Ragged execution metadata held in host arrays.

    Returns:
      The same metadata, replicated across the mesh the way the attention
      kernel's `shard_map` expects to receive it.
    """
    if self._mesh is not None and not self._mesh.empty:
      sharding = jax.sharding.NamedSharding(
          self._mesh, jax.sharding.PartitionSpec()
      )
      to_device = lambda x: jax.device_put(x, sharding)
    else:
      to_device = jnp.asarray

    return RPAMetadata(
        page_indices={
            cache_name: to_device(idxs)
            for cache_name, idxs in metadata.page_indices.items()
        },
        kv_lens=to_device(metadata.kv_lens),
        query_lens=to_device(metadata.query_lens),
        distribution=to_device(metadata.distribution),
        mesh=metadata.mesh,
        chunk_prefill_size=metadata.chunk_prefill_size,
    )

  def execute_step(
      self,
      cache: Cache,
      tokens: jnp.ndarray | np.ndarray,
      metadata: RPAMetadata,
  ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Cache]:
    """Executes one engine step, generating `num_decode_steps` tokens at most.

    The scheduled batch runs on its own, because the engine packs it into a
    token buffer sized to the tokens it actually scheduled. Every step after
    that samples exactly one token per row, so the rest of the engine step runs
    as a compiled loop over a `[max_num_seqs]` token buffer.

    The metadata moves onto the device before the first step and is rewritten
    there, so nothing in the loop reads device data back and the host never
    blocks on the accelerator.

    Args:
      cache: The physical KV cache pages to read and write.
      tokens: The packed tokens scheduled for this step.
      metadata: Ragged execution metadata for the attention kernel. Its arrays
        must be host arrays; they are transferred here.

    Returns:
      A tuple of (tokens, logits, logprobs, updated cache), where tokens is
      `[max_num_seqs, num_decode_steps]`, logits is
      `[max_num_seqs, num_decode_steps, vocab_size]` and logprobs is
      `[max_num_seqs, num_decode_steps]`. Rows are indexed by the sequence's
      position in the scheduled batch, and are zero for steps a sequence did
      not take part in. Logits and logprobs are None unless the config asks
      for them.
    """
    config = self._config
    num_scheduled = metadata.distribution[2]
    device_metadata = self._to_device(metadata)

    next_tokens, logits, logp, cache, self._step_count = (
        self._compiled_model_step_fn(
            self._flattened_transformer_state,
            self._step_count,
            cache,
            tokens,
            device_metadata,
        )
    )
    first_tokens = _mask_rows(next_tokens, num_scheduled)
    first_logits = (
        _mask_rows(logits[:, 0], num_scheduled) if logits is not None else None
    )
    first_logp = (
        _mask_rows(logp, num_scheduled) if logp is not None else None
    )

    # Only the chunked prefills drop out, so the batch either survives its
    # first step or empties out entirely. Either way the host knows how many
    # decode steps are worth running without inspecting any device data.
    num_surviving = _num_surviving_rows(metadata.distribution)
    num_decode_steps = config.num_decode_steps - 1 if num_surviving else 0

    decode_tokens = decode_logits = decode_logp = None
    if num_decode_steps:
      decode_tokens, decode_logits, decode_logp, cache, self._step_count = (
          self._compiled_decode_loop_fn(
              self._flattened_transformer_state,
              self._step_count,
              cache,
              next_tokens,
              device_metadata,
              num_steps=num_decode_steps,
          )
      )

    num_steps = config.num_decode_steps
    tokens_cpu = jax.device_get(
        _stack_steps(first_tokens, decode_tokens, num_steps)
    )
    logits_cpu = (
        jax.device_get(_stack_steps(first_logits, decode_logits, num_steps))
        if config.return_logits
        else None
    )
    logp_cpu = (
        jax.device_get(_stack_steps(first_logp, decode_logp, num_steps))
        if config.return_logprobs
        else None
    )

    return tokens_cpu, logits_cpu, logp_cpu, cache

