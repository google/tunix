# Copyright 2026 Google LLC
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
import functools
import inspect
import operator
from typing import Any, Optional, Tuple
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
from tunix.common import configs
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.rollout.raiden_weight_sync_mixin import (
    RaidenDestinationWeightSyncMixin,
)
from tunix.experimental.weight_sync import weight_sync
from tunix.generate import beam_search as beam_search_lib
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import utils as generate_utils
from tunix.processors import image_processor as image_processor_lib
from tunix.rl import reshard
from tunix.rl import utils as rl_utils

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32  # pyrefly: ignore[not-a-type]

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Boolean mask indicating valid tokens (True) vs left-padding (False).
  input_mask: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: Cache

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None  # [B, L, V]

  # Fixed-size buffer for accumulating the output logprobs.
  logprobs_buffer: jnp.ndarray | None  # [B, L]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: tuple[int, ...] | None

  # Random seed for sampling.
  seed: jax.Array

  # The sampling mode to use, one of "greedy", "top_p" or "beam_search"
  sampling_mode: str = flax.struct.field(pytree_node=False)

  # Number of input tokens with padding.
  num_input_tokens: int = flax.struct.field(pytree_node=False)

  # Only present when sampling_mode is "beam_search".
  beam_search_sampling_state: (
      beam_search_lib._BeamSearchSamplingState | None
  ) = None


def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    return_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Sample a token using top-p sampling."""
  # Upcast to float32 for numerical stability of softmax and subsequent cumsum.
  next_token_logits = logits[:, -1].astype(jnp.float32) / temperature

  # top_k=0 or None both mean "no top-k filtering" — use full vocabulary.
  _no_topk = top_k is None or top_k <= 0
  # Skip softmax and sorting if top_p is 1.0 and top_k is full vocab.
  if top_p >= 1.0 and _no_topk:
    next_token = jax.random.categorical(key, logits=next_token_logits)
    if not return_logprobs:
      return next_token, None
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
    return next_token, logp_sampled

  k = next_token_logits.shape[-1] if _no_topk else top_k
  logits_sorted, indices = jax.lax.top_k(next_token_logits, k=k)  # pyrefly: ignore[bad-argument-type]

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
  """Greedy argmax token selection."""
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  if not return_logprobs:
    return next_token, None
  logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
  logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  return next_token, logp_sampled


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
  # Jax array is immutable, so updates to each layer creates new arrays.
  return {
      f"layer_{i}": {
          "k": jnp.zeros(shape, dtype=dtype),
          "v": jnp.zeros(shape, dtype=dtype),
          "end_index": jnp.zeros((batch_size,), dtype=jnp.int32),
      }
      for i in range(n_layers)
  }


class VanillaSampler(
    RaidenDestinationWeightSyncMixin, base_sampler_lib.Sampler
):
  """Sampler for transformer model."""

  def __init__(
      self,
      server_id: str,
      config: configs.RolloutConfig,
      transformer: nnx.Module,
      tokenizer: Any,
      image_processor: image_processor_lib.ImageProcessor | None = None,
      raiden_sync_delegate: Any = None,
  ):
    """Initializes the sampler.

    Args:
      server_id: Unique identifier for this sampler server slice.
      config: Rollout configuration for this worker.
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      image_processor: The image processor.
      raiden_sync_delegate: Optional Raiden weight sync delegate.
    """
    if config.kv_cache_size <= 0:
      raise ValueError(
          f"VanillaSampler [{server_id}] requires positive"
          f" config.kv_cache_size, got {config.kv_cache_size}."
      )
    self.server_id = server_id
    self.config: configs.RolloutConfig = config
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.tokenizer: tok_adapter.TokenizerAdapter = tokenizer
    self.image_processor = image_processor
    self.cache_size: int = self.config.kv_cache_size
    self.raiden_sync_delegate = raiden_sync_delegate
    self.weight_sync_mode = self.config.weight_sync_mode
    self.enable_raiden = (
        self.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
    )

    if self.enable_raiden and self.raiden_sync_delegate is None:
      from tunix.experimental.weight_sync import raiden_weight_sync_delegate  # pylint: disable=g-import-not-at-top

      self.raiden_sync_delegate = (
          raiden_weight_sync_delegate.RaidenWeightSyncDelegate(
              server_id=self.server_id,
              partial_rollout=self.config.partial_rollout,
          )
      )

    if not self.enable_raiden and self.raiden_sync_delegate:
      logging.warning(
          "VanillaSampler [%s] raiden_sync_delegate is set but"
          " enable_raiden is False.",
          self.server_id,
      )

    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)  # pyrefly: ignore[bad-assignment]
    self._transformer_state: statelib.State = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    self._compiled_decode_fn: Any = None
    self._compiled_prefill_fn: Any = None
    self._supports_decode_only_last_token: bool = (
        "decode_only_last_token"
        in inspect.signature(transformer.__call__).parameters
    )
    self.eos_ids: jax.Array = jnp.array(
        self.config.eos_tokens or [self.tokenizer.eos_id()]
    )
    if self.config.seed is None:
      self._rng: jax.Array = jax.random.PRNGKey(0)
    elif isinstance(self.config.seed, int):
      self._rng = jax.random.PRNGKey(self.config.seed)
    else:
      self._rng = self.config.seed

  def initialize(self) -> None:
    """Compiles the prefill and decode functions."""
    # We separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time.
    #
    # We donate the sampling_state (argnum 1) containing the KV cache arrays.
    # JAX arrays are immutable, so updating the cache at each decoding step
    # would normally force JAX to allocate a new memory buffer and copy old
    # contents. Since the KV cache memory footprint scales with batch size and
    # prompt+decoding length (reaching gigabytes), this continuous reallocation
    # and copying triggers massive memory overhead and OOMs. Donating the input
    # state allows the XLA compiler to reuse the memory buffer in-place,
    # completely avoiding allocation/copy overhead.
    self._compiled_decode_fn = jax.jit(self._decode_fn, donate_argnums=(1,))
    self._compiled_prefill_fn = jax.jit(
        self._prefill_fn,
        donate_argnums=(1,),
        static_argnames=("echo",),
    )

  def model_def_and_state(self) -> tuple[graph.NodeDef, list[statelib.State]]:
    """Returns the transformer graphdef and flattened state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(  # pyrefly: ignore[no-matching-overload]
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
            "New state must have the same structure as the old state."
            f" {jax.tree_util.tree_structure(tree1)} vs"
            f" {jax.tree_util.tree_structure(tree2)}"
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
            "New state must have the same shape, dtype and sharding as the old"
            f" state. {tree1} vs {tree2}"
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
            "Only LoRAParam is supported. Received invalid `param_types`: "
            f"{param_types}"
        )
      original_lora_params = statelib.filter_state(
          self._transformer_state, nnx.LoRAParam
      )
      check_tree_structure(original_lora_params, state)
      base_state = statelib.filter_state(
          self._transformer_state, filterlib.Not(nnx.LoRAParam)
      )
      self._transformer_state = statelib.merge_state(base_state, state)

    self.refresh_state_leaves()

  def refresh_state_leaves(self) -> None:
    """Recomputes cached state leaves after in-place weight updates."""
    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def delete_cache(self) -> None:
    """No-op for VanillaSampler (KV cache is allocated per sample call)."""

  def reset_prefix_cache(self) -> None:
    """No-op for VanillaSampler (no persistent prefix cache)."""

  def reinitialize_cache(self) -> None:
    """Refreshes cached state leaves after weight sync."""
    self.refresh_state_leaves()

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates model parameters in the sampler."""
    if filter_types is not None:
      dst_params = nnx.state(self.transformer, filter_types)
      if any(
          isinstance(x.sharding, jax.sharding.NamedSharding)
          for x in jax.tree.leaves(dst_params)
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
    flat_old_params, tree_def = rl_utils.to_flat_dict(self.transformer_state)
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(self._transformer_graphdef, merged_params)  # pyrefly: ignore[no-matching-overload]
    self.transformer_state = nnx.variables(new_model, nnx.Param)

  @property
  def dtype(self) -> jnp.dtype:
    return self._flattened_transformer_state[0].dtype

  # --- Lifecycle & Topology ---
  async def start(self, **kwargs) -> str | None | Any:
    """Starts the sampling engine or local loop."""
    del kwargs
    return True

  async def stop(self, **kwargs) -> str | None | Any:
    del kwargs
    return True

  async def pause(self, **kwargs) -> str | None | Any:
    """Pauses inference processing on this worker slice."""
    del kwargs
    return True

  async def resume(self, **kwargs) -> str | None | Any:
    """Resumes inference processing on this worker slice."""
    del kwargs
    return True

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    del kwargs
    return None

  # --- Internal JAX Sampling Primitives ---
  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      forbidden_token_ids: tuple[int, ...] | None,
      prompt_lengths: jax.Array | None = None,
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    batch_size = all_input_ids.shape[0]
    num_input_tokens = all_input_ids.shape[1]

    token_buffer = jnp.full(
        (batch_size, total_sampling_steps),
        self.tokenizer.pad_id(),
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    token_buffer = token_buffer.at[:, :num_input_tokens].set(all_input_ids)
    if prompt_lengths is not None:
      prompt_mask = jnp.arange(num_input_tokens)[None, :] >= (
          num_input_tokens - prompt_lengths[:, None]
      )
    else:
      prompt_mask = all_input_ids != self.tokenizer.pad_id()
    input_mask = input_mask.at[:, :num_input_tokens].set(prompt_mask)
    positions = generate_utils.build_positions_from_mask(input_mask)

    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    if hasattr(self.transformer, "init_cache"):
      cache = self.transformer.init_cache(
          batch_size, self.cache_size, self.dtype
      )
    else:
      warnings.warn(
          "Using deprecated _init_cache in Tunix sampler. Models are now"
          " required to have their own init_cache attribute.",
          DeprecationWarning,
      )
      model_cfg = self.transformer.config  # pyrefly: ignore[missing-attribute]
      cache = _init_cache(
          n_layers=model_cfg.num_layers,
          cache_size=self.cache_size,
          batch_size=batch_size,
          num_kv_heads=model_cfg.num_kv_heads,
          head_dim=model_cfg.head_dim,
          dtype=self.dtype,
      )

    if self.config.return_logits:
      logits_buffer = jnp.zeros(
          (batch_size, total_sampling_steps, self.transformer.num_embed),  # pyrefly: ignore[missing-attribute]
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None

    if self.config.return_logprobs:
      logprobs_buffer = jnp.zeros(
          (batch_size, total_sampling_steps),
          dtype=jnp.float32,
      )
    else:
      logprobs_buffer = None

    if self.config.beam_size is not None:
      sampling_mode = "beam_search"
    elif self.config.temperature <= 0.0 or self.config.top_p is None:
      sampling_mode = "greedy"
    else:
      sampling_mode = "top_p"

    logging.debug("Using sampling mode: %s", sampling_mode)

    self._rng, seed = jax.random.split(self._rng)

    return _SamplingState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=int(num_input_tokens),
        token_buffer=token_buffer,
        input_mask=input_mask,
        positions=positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        seed=seed,
        sampling_mode=sampling_mode,
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
      cache: Cache,
      sampler_state: _SamplingState,
  ) -> _SamplingState:
    """Samples a token from the logits."""
    logits = logits[:, -1][:, None, :]  # B, 1, V
    decoding_step = sampler_state.decoding_step
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    logits_buffer = sampler_state.logits_buffer
    logprobs_buffer = sampler_state.logprobs_buffer
    beam_search_state = sampler_state.beam_search_sampling_state
    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    if sampler_state.sampling_mode == "beam_search":
      beam_search_state, updated_args = beam_search_lib.beam_search_step(
          logits=logits,
          done=done,
          token_buffer=token_buffer,
          cache=cache,
          logits_buffer=logits_buffer,
          state=beam_search_state,  # pyrefly: ignore[bad-argument-type]
          pad_token_id=eos[0],
          decoding_step=decoding_step,
          logprobs_buffer=logprobs_buffer,
      )
      cache = updated_args["cache"]
      token_buffer = updated_args["token_buffer"]
      done = updated_args["done"]
      logits_buffer = updated_args["logits_buffer"]
      logprobs_buffer = updated_args["logprobs_buffer"]
    else:
      if sampler_state.sampling_mode == "greedy":
        next_token_candidate, logp = sample_best(
            logits, return_logprobs=(logprobs_buffer is not None)
        )
      else:
        key = jax.random.fold_in(sampler_state.seed, decoding_step)
        assert self.config.top_p is not None
        next_token_candidate, logp = sample_top_p(
            logits,
            key,
            self.config.temperature,
            self.config.top_p,
            self.config.top_k,
            return_logprobs=(logprobs_buffer is not None),
        )
      token_buffer = token_buffer.at[:, decoding_step + 1].set(
          next_token_candidate
      )
      if logprobs_buffer is not None:
        logprobs_buffer = logprobs_buffer.at[:, decoding_step + 1].set(logp)

    done = done | jnp.isin(token_buffer[:, decoding_step + 1], eos)
    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        input_mask=sampler_state.input_mask,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_state,
    )

  def _prefill_fn(
      self,
      params: statelib.State,
      sampler_state: _SamplingState,
      images: jnp.ndarray | None = None,
      audios: Any = None,
      echo: bool = True,
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

    input_mask = jax.lax.dynamic_slice(
        sampler_state.input_mask,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )

    if hasattr(self.transformer, "get_attention_mask"):
      attention_mask = self.transformer.get_attention_mask(
          tokens, inputs_mask=input_mask
      )
      seq_len = attention_mask.shape[-1]
      padding = self.cache_size - seq_len
      attention_mask = jnp.pad(
          attention_mask,
          (*((0, 0) for _ in range(attention_mask.ndim - 1)), (0, padding)),
      )
    else:
      attention_mask = generate_utils.make_causal_attn_mask(
          input_mask, self.cache_size
      )

    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
    kwargs = {}
    if images is not None:
      kwargs["images"] = images
    if audios is not None:
      kwargs["audios"] = audios
    decode_only_last_token = self._supports_decode_only_last_token and not echo
    if decode_only_last_token:
      kwargs["decode_only_last_token"] = True
    logits, cache = transformer(
        tokens,
        step_positions,
        sampler_state.cache,
        attention_mask,
        **kwargs,
    )
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    positions = sampler_state.positions
    state_input_mask = sampler_state.input_mask
    beam_search_sampling_state = None
    if sampler_state.logits_buffer is not None:
      start_idx = (
          sampler_state.num_input_tokens if decode_only_last_token else 1
      )
      logits_buffer = jax.lax.dynamic_update_slice(
          sampler_state.logits_buffer,
          logits.astype(sampler_state.logits_buffer.dtype),
          (0, start_idx, 0),
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    if sampler_state.sampling_mode == "beam_search":
      # init beam state in prefill instead of init as one minor optimization
      # to avoid running unnecessary prefill for
      # duplicated input prompt per beam.
      assert self.config.beam_size is not None
      beam_size = int(self.config.beam_size)
      sampling_state, updated_args = beam_search_lib.init_batched_beam_state(
          logits=logits,
          input_token_buffer=sampler_state.token_buffer,
          initial_cache=cache,
          done=sampler_state.done,
          positions=sampler_state.positions,
          logits_buffer=sampler_state.logits_buffer,
          beam_size=beam_size,
      )
      beam_search_sampling_state = sampling_state
      logits = updated_args["logits"]
      cache = updated_args["cache"]
      token_buffer = updated_args["token_buffer"]
      done = updated_args["done"]
      positions = updated_args["positions"]
      logits_buffer = updated_args["logits_buffer"]
      state_input_mask = jnp.repeat(sampler_state.input_mask, beam_size, axis=0)

    updated_sampling_state = _SamplingState(
        decoding_step=sampler_state.decoding_step,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        input_mask=state_input_mask,
        positions=positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=sampler_state.logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
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
          sampler_state.decoding_step < sampler_state.total_sampling_steps - 1
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

    input_mask = jnp.logical_not(sampler_state.input_mask)
    attention_mask = generate_utils.compute_attention_masks(
        decoding_step, self.cache_size, input_mask
    )

    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
    logits, cache = transformer(
        last_token,
        positions=step_positions,
        cache=sampler_state.cache,
        attention_mask=attention_mask,
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

    return dataclasses.replace(
        updated_sampler_state,
        logits_buffer=logits_buffer,
    )

  # --- Inference ---
  async def sample(
      self,
      sampling_requests: (
          base_sampler_lib.SamplingRequest
          | Sequence[base_sampler_lib.SamplingRequest]
      ),
      **kwargs,
  ) -> (
      base_sampler_lib.SamplingResponse
      | list[base_sampler_lib.SamplingResponse]
  ):
    """Generates completions for a single SamplingRequest or a batch."""
    del kwargs
    is_sequence = not isinstance(
        sampling_requests, base_sampler_lib.SamplingRequest
    )
    requests: Sequence[base_sampler_lib.SamplingRequest] = (
        sampling_requests if is_sequence else [sampling_requests]
    )

    prompts: list[str] = []
    prompt_token_ids_batch: list[np.ndarray] = []
    has_token_prompts = False
    max_gen_steps_list: list[int] = []

    for req in requests:
      prompt = req.prompt
      if isinstance(prompt, str):
        prompts.append(prompt)
      else:
        has_token_prompts = True
        prompt_token_ids_batch.append(prompt)

      max_tokens = (
          req.sampling_params.max_tokens
          if req.sampling_params is not None
          else self.config.max_tokens_to_generate
      )
      max_gen_steps_list.append(max_tokens)

    max_generation_steps = (
        max(max_gen_steps_list)
        if max_gen_steps_list
        else self.config.max_tokens_to_generate
    )
    return_logprobs = self.config.return_logprobs
    beam_size = self.config.beam_size

    self.eos_ids = jnp.array(
        self.config.eos_tokens or [self.tokenizer.eos_id()]
    )
    tokens = generate_utils.resolve_prompt_tokens(
        None if has_token_prompts else prompts,
        prompt_token_ids_batch if has_token_prompts else None,
        self.tokenize,
        max_generation_steps=max_generation_steps,
        max_total_length=self.cache_size,
        max_length_name="cache_size",
        single_output_per_row=beam_size is None,
    )

    all_input_ids, prompt_lengths, max_prompt_length = (
        generate_utils.left_pad_prompt_tokens(
            tokens,
            None,
            self.tokenizer.pad_id(),
            max_allowed_length=(
                self.cache_size - max_generation_steps
            ),
        )
    )

    total_sampling_steps = max_prompt_length + max_generation_steps
    if total_sampling_steps > self.cache_size:
      raise ValueError(
          f"Total sampling steps {total_sampling_steps} must be less than the"
          f" cache size {self.cache_size}."
      )

    sampling_state = self.init_sample_state(
        jnp.array(all_input_ids),
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=None,
        prompt_lengths=(
            jnp.asarray(prompt_lengths, dtype=jnp.int32)
            if has_token_prompts
            else None
        ),
    )
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state,
        sampling_state,
        images=None,
        audios=None,
        echo=False,
    )
    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )
    token_buffers = sampling_state.token_buffer
    final_logprobs_buffer = sampling_state.logprobs_buffer

    if sampling_state.sampling_mode == "beam_search":
      updated_args = beam_search_lib.finalize_beam_search_state(
          sampling_state.beam_search_sampling_state,
          sampling_state.token_buffer,
          sampling_state.logits_buffer,
          sampling_state.logprobs_buffer,
      )
      token_buffers = updated_args["token_buffer"]
      final_logprobs_buffer = updated_args["logprobs_buffer"]
      # delete the sampling state in case the further referece
      # if need more internal states, they should be updated by
      # finalize_beam_search_state
      del sampling_state

    token_buffers = jax.device_get(token_buffers)
    if return_logprobs:
      final_logprobs_buffer = jax.device_get(final_logprobs_buffer)

    responses: list[base_sampler_lib.SamplingResponse] = []
    pad_id = self.tokenizer.pad_id()
    for i, req in enumerate(requests):
      token_buffer = token_buffers[i]
      start_idx = max_prompt_length
      end_idx = (
          generate_utils.np_find_first_eos_idx(
              token_buffer[max_prompt_length:], self.eos_ids
          )
          + max_prompt_length
      )
      tok_ids = np.asarray(token_buffer[start_idx:end_idx], dtype=np.int32)
      txt = self.tokenizer.decode(tok_ids.tolist())
      prompt_len = int(prompt_lengths[i])
      prompt_token_ids = generate_utils.unpad_prompt_tokens(
          all_input_ids[i],
          pad_id=pad_id,
          prompt_length=prompt_len,
      )
      # Extract logprobs for the generated tokens
      log_ps = (
          np.asarray(
              final_logprobs_buffer[i][start_idx:end_idx], dtype=np.float32
          )
          if return_logprobs and final_logprobs_buffer is not None
          else None
      )
      responses.append(
          base_sampler_lib.SamplingResponse(
              request_id=req.request_id,
              text=txt,
              prompt_token_ids=prompt_token_ids,
              token_ids=tok_ids,
              logprobs=log_ps,
              finish_reason="stop",
          )
      )

    if is_sequence:
      return responses
    return responses[0]

  # --- Weight Synchronization & Load Info ---
  async def get_transfer_status(self, req_id: str | Any, **kwargs) -> str | Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del req_id, kwargs
    return "SUCCESS"

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns best-effort local sampler load information."""
    del kwargs
    return base_sampler_lib.LoadInfo()

  # --- KV-cache Migration ---
  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: list[int],
      **kwargs,
  ) -> bool:
    """Triggers Raiden P2P KV-cache transfer across TPU slices."""
    del source_server_id, target_server_id, token_ids, kwargs
    return True
