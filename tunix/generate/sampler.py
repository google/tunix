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

"""Vanilla sampler for LLM generation."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any, Optional
import warnings

from absl import logging
import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tunix.generate import base_sampler
from tunix.generate import utils
import tunix.generate.beam_search as beam_search_lib
import tunix.generate.tokenizer_adapter as tok_adapter

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, dict[str, jaxtyping.Array]]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None  # [B, L, V]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: Sequence[int] | None

  # Random seed for sampling.
  seed: jax.Array

  # The sampling mode to use, one of "greedy", "top_p" or "beam_search"
  sampling_mode: str = flax.struct.field(pytree_node=False)

  # Number of input tokens with padding.
  num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)

  # Tempurature for top_p sampling.
  temperature: float = flax.struct.field(pytree_node=False)

  # Sampling parameters.
  # For top_p, it contains "top_p" and "top_k".
  # For beam search, it contains "beam_size"
  sampling_parameters: dict[str, float | int] = flax.struct.field(
      pytree_node=False
  )

  # Only present when sampling_mode is "beam_search".
  beam_search_sampling_state: (
      beam_search_lib._BeamSearchSamplingState | None
  ) = None


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


def _sample_top_p(
    probs: jnp.ndarray, p: float, key: jax.Array, k: Optional[int] = None
) -> jnp.ndarray:
  """Sample a token using top-p sampling."""
  k = probs.shape[-1] if k is None else k
  probs_sorted, indices = jax.lax.top_k(probs, k=k)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > p
  probs_sorted = jnp.where(mask, 0.0, probs_sorted)
  probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

  next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

  next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)
  return next_token


def sample_top_p(
    logits, key, temperature: float, top_p: float, top_k: Optional[int]
):
  probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
  next_token = _sample_top_p(probs, top_p, key, top_k)
  return next_token


def sample_best(logits):
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  return next_token


def _init_cache(
    n_layers: int,
    cache_size: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> Cache:
  """Create KV cache for the transformer.

  Args:
    n_layers: The number of attention layers.
    cache_size: The size of the cache.
    batch_size: The batch size.
    num_kv_heads: The number of KV attention heads.
    head_dim: The dimension of the KV attention head.
    dtype: The data type of the cache.

  Returns:
    The KV cache for one attention block.
  """

  shape = (batch_size, cache_size, num_kv_heads, head_dim)
  k = jnp.zeros(shape, dtype=dtype)
  v = jnp.zeros(shape, dtype=dtype)
  end_index = jnp.zeros((batch_size,), dtype=jnp.int32)
  # Jax array is immutable, so updates to each layer creates new arrays.
  return {
      f'layer_{i}': {'k': k, 'v': v, 'end_index': end_index}
      for i in range(n_layers)
  }


class Sampler(base_sampler.BaseSampler):
  """Sampler for transformer model."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: CacheConfig,
  ):
    """Initializes the sampler.

    Args:
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      cache_config: configuration for the KV cache.
    """
    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.cache_config = cache_config
    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)
    self._transformer_state: list[statelib.State] = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    # we separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time
    self._compiled_decode_fn = jax.jit(self._decode_fn)
    self._compiled_prefill_fn = jax.jit(self._prefill_fn)

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
            and jnp.dtype(x) == jnp.dtype(y)
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

  @property
  def dtype(self) -> jnp.dtype:
    return self._flattened_transformer_state[0].dtype

  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      include_logits: bool,
      forbidden_token_ids: Sequence[int] | None,
      temperature: float,
      top_p: Optional[float],
      top_k: Optional[int],
      seed: jax.Array,
      beam_size: Optional[int],
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    batch_size = all_input_ids.shape[0]
    num_input_tokens = all_input_ids.shape[1]

    token_buffer = jnp.full(
        (
            batch_size,
            total_sampling_steps,
        ),
        self.tokenizer.pad_id(),
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    token_buffer = token_buffer.at[:, :num_input_tokens].set(all_input_ids)
    input_mask = input_mask.at[:, :num_input_tokens].set(
        all_input_ids != self.tokenizer.pad_id()
    )
    positions = utils.build_positions_from_mask(input_mask)

    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    if hasattr(self.transformer, 'init_cache'):
      cache = self.transformer.init_cache(
          batch_size, self.cache_config.cache_size, self.dtype
      )
    else:
      warnings.warn(
          'Using deprecated _init_cache in Tunix sampler. Models are now'
          ' required to have their own init_cache attribute.',
          DeprecationWarning,
      )
      cache = _init_cache(
          n_layers=self.cache_config.num_layers,
          cache_size=self.cache_config.cache_size,
          batch_size=batch_size,
          num_kv_heads=self.cache_config.num_kv_heads,
          head_dim=self.cache_config.head_dim,
          dtype=self.dtype,
      )

    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, total_sampling_steps, self.transformer.num_embed),
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None
    sampling_parameters = {}
    sampling_mode = [None]

    if beam_size is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'beam_search')
      sampling_parameters['beam_size'] = beam_size

    if top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')
      sampling_parameters['top_p'] = top_p
      sampling_parameters['top_k'] = top_k

    if sampling_mode[0] is None:
      sampling_mode[0] = 'greedy'

    logging.debug('Using sampling mode: %s', sampling_mode[0])

    return _SamplingState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        sampling_parameters=sampling_parameters,
        seed=seed,
        sampling_mode=sampling_mode[0],
        beam_search_sampling_state=None,
    )

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    input_ids = np.array(
        self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
    )
    return input_ids

  def _sample(
      self,
      logits: jnp.ndarray,
      eos: jax.Array,
      cache: dict[str, dict[str, jaxtyping.Array]],
      sampler_state: _SamplingState,
  ) -> _SamplingState:
    """Samples a token from the logits."""

    logits = logits[:, -1][:, None, :]  # B, 1, V
    decoding_step = sampler_state.decoding_step
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    logits_buffer = sampler_state.logits_buffer
    beam_search_state = sampler_state.beam_search_sampling_state
    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    if sampler_state.sampling_mode == 'beam_search':
      beam_search_state, updated_args = beam_search_lib.beam_search_step(
          logits=logits,
          done=done,
          token_buffer=token_buffer,
          cache=cache,
          logits_buffer=logits_buffer,
          state=beam_search_state,
          pad_token_id=eos[0],
          decoding_step=decoding_step,
      )
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      logits_buffer = updated_args['logits_buffer']
    else:
      if sampler_state.sampling_mode == 'greedy':
        next_token_candidate = sample_best(logits)
      elif sampler_state.sampling_mode == 'top_p':
        key = jax.random.fold_in(sampler_state.seed, decoding_step)
        next_token_candidate = sample_top_p(
            logits,
            key,
            sampler_state.temperature,
            sampler_state.sampling_parameters['top_p'],
            sampler_state.sampling_parameters['top_k'],
        )
      else:
        raise ValueError(
            'Unsupported sampling mode: %s' % sampler_state.sampling_mode
        )
      token_buffer = token_buffer.at[:, decoding_step + 1].set(
          next_token_candidate
      )

    done = done | jnp.isin(token_buffer[:, decoding_step + 1], eos)
    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_state,
    )

  def _prefill_fn(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs prefill."""
    batch_size = sampler_state.token_buffer.shape[0]

    tokens = jax.lax.dynamic_slice(
        sampler_state.token_buffer,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )
    step_positions = jax.lax.dynamic_slice(
        sampler_state.positions,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )

    input_mask = tokens != self.tokenizer.pad_id()
    attention_mask = utils.make_causal_attn_mask(
        input_mask, self.cache_config.cache_size
    )

    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        tokens,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    positions = sampler_state.positions
    beam_search_sampling_state = None
    if sampler_state.logits_buffer is not None:
      logits_buffer = jax.lax.dynamic_update_slice(
          sampler_state.logits_buffer,
          logits.astype(sampler_state.logits_buffer.dtype),
          (0, 1, 0),
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    if sampler_state.sampling_mode == 'beam_search':
      # init beam state in prefill instead of init as one minor optimization
      # to avoid running unnecessary prefill for
      # duplicated input prompt per beam.
      sampling_state, updated_args = beam_search_lib.init_batched_beam_state(
          logits=logits,
          input_token_buffer=sampler_state.token_buffer,
          initial_cache=cache,
          done=sampler_state.done,
          positions=sampler_state.positions,
          logits_buffer=sampler_state.logits_buffer,
          beam_size=int(sampler_state.sampling_parameters['beam_size']),
      )
      beam_search_sampling_state = sampling_state
      logits = updated_args['logits']
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      positions = updated_args['positions']
      logits_buffer = updated_args['logits_buffer']

    updated_sampling_state = _SamplingState(
        decoding_step=sampler_state.decoding_step,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_sampling_state,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampler_state=updated_sampling_state,
    )
    return updated_sampler_state

  def _decode_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal generating function (to be jitted)."""

    def sample_with_params(sampler_state: _SamplingState):
      return self._sample_step(params, sampler_state)

    def cond_fn(sampler_state: _SamplingState):
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    return jax.lax.while_loop(cond_fn, sample_with_params, sampling_state)

  def _sample_step(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs a single sampling step."""
    batch_size = sampler_state.token_buffer.shape[0]
    decoding_step = sampler_state.decoding_step

    last_token = sampler_state.token_buffer[:, decoding_step]
    last_token = last_token.reshape((batch_size, 1))
    step_positions = jnp.expand_dims(
        sampler_state.positions[:, decoding_step], -1
    )

    input_mask = sampler_state.token_buffer == self.tokenizer.pad_id()
    attention_mask = utils.compute_attention_masks(
        decoding_step, self.cache_config.cache_size, input_mask
    )

    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        last_token,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampler_state=sampler_state,
    )

    if updated_sampler_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = updated_sampler_state.logits_buffer.at[
          :, decoding_step + 1
      ].set(next_logits)
    else:
      logits_buffer = None

    updated_sampler_state = dataclasses.replace(
        updated_sampler_state,
        logits_buffer=logits_buffer,
    )
    return updated_sampler_state

  def __call__(
      self,
      input_strings: str | Sequence[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      eos_tokens: Sequence[int] | None = None,
      forbidden_tokens: Sequence[str] | None = None,
      temperature: float = 0.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      beam_size: Optional[int] = None,
      seed: int | None = None,
      pad_output: bool = False,
  ) -> base_sampler.SamplerOutput:
    """Samples a completion of the input string.

    If top_p is provided, the sampling mode will be top_p.
    If beam_size is provided, the sampling mode will be beam_search.
    If None of them are provided, the sampling mode will be greedy.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      max_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      max_prompt_length: maximum length of the prompt. Specify to avoid
        recompilation on different prompt lengths.
      echo: whgether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      eos_tokens: end of sequence tokens to stop generation. If None, the
        tokenizer's eos_id will be used.
      forbidden_tokens: list of tokens that are forbidden to be generated. Each
        token must map to a single token id in the vocab.
      temperature: temperature for sampling.
      top_p: top-p sampling threshold.
      top_k: top-k sampling threshold.
      beam_size: beam size for beam search.
      seed: random seed for sampling.
      pad_output: whether to pad the output to maximum length. If this set as
        True, the output len will be max_generation_steps if echo is False,
        otherwise it will be max_generation_steps + max_prompt_length. The
        padding now only supports right padding. Can modify to support left
        padding if needed.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    self.eos_ids = jnp.array(eos_tokens or [self.tokenizer.eos_id()])
    input_strings = (
        [input_strings] if isinstance(input_strings, str) else input_strings
    )

    forbidden_token_ids = None
    if forbidden_tokens is not None:
      forbidden_token_ids = []
      for token in forbidden_tokens:
        token_id = self.tokenizer.encode(token)
        if len(token_id) != 1:
          raise ValueError(
              'Forbidden tokens must map to single token ids in the vocab.'
          )
        forbidden_token_ids.extend(token_id)
      forbidden_token_ids = tuple(forbidden_token_ids)

    tokens = [self.tokenize(x) for x in input_strings]
    max_tokens_length = max(len(x) for x in tokens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)

    all_input_ids = jnp.array([
        utils.pad_to_length(
            x,
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in tokens
    ])

    total_sampling_steps = max_prompt_length + max_generation_steps
    if total_sampling_steps > self.cache_config.cache_size:
      raise ValueError(
          f'Total sampling steps {total_sampling_steps} must be less than the'
          f' cache size {self.cache_config.cache_size}.'
      )

    if seed is None:
      seed = jax.random.PRNGKey(0)
    elif isinstance(seed, int):
      seed = jax.random.PRNGKey(seed)
    sampling_state = self.init_sample_state(
        all_input_ids,
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        beam_size=beam_size,
    )
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state, sampling_state
    )

    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )
    token_buffers = sampling_state.token_buffer
    logits_buffers = sampling_state.logits_buffer

    if sampling_state.sampling_mode == 'beam_search':
      updated_args = beam_search_lib.finalize_beam_search_state(
          sampling_state.beam_search_sampling_state,
          sampling_state.token_buffer,
          sampling_state.logits_buffer,
      )
      token_buffers = updated_args['token_buffer']
      logits_buffers = updated_args['logits_buffer']
      # delete the sampling state in case the further referece
      # if need more internal states, they should be updated by
      # finalize_beam_search_state
      del sampling_state
    if pad_output:
      max_len = total_sampling_steps if echo else max_generation_steps
      lengths, out_tokens, out_logits = utils.padded_fill_tokens_and_logits(
          token_buffers,
          logits_buffers,
          return_logits,
          echo,
          self.tokenizer.pad_id(),
          self.eos_ids,
          max_prompt_length,
          max_len,
      )
      out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)
      decoded_outputs = [
          self.tokenizer.decode(tokens[:length].tolist())
          for tokens, length in zip(out_tokens, lengths)
      ]
    else:
      out_tokens = []
      out_logits = []
      lengths = []
      for i, token_buffer in enumerate(token_buffers):
        start_idx = (
            utils.find_first_non_pad_idx(token_buffer, self.tokenizer.pad_id())
            if echo
            else max_prompt_length
        )
        end_idx = (
            utils.find_first_eos_idx(
                token_buffer[max_prompt_length:], self.eos_ids
            )
            + max_prompt_length
        )
        out_tokens.append(token_buffer[start_idx:end_idx])
        if return_logits:
          out_logits.append(logits_buffers[i][start_idx:end_idx])
        lengths.append(end_idx - start_idx)

      decoded_outputs = [
          self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
      ]

    result = base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else [],
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=None,
    )
    return result
