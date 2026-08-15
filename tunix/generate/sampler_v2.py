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

"""Continuous Batching Sampler for Tunix."""

from __future__ import annotations

import dataclasses
import enum
import threading
import inspect
import time
from typing import Any, Dict, List, Sequence, Tuple, Union
import sys

import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import jaxtyping
import numpy as np
from tunix.generate import sampler as sampler_lib
from tunix.generate import page_manager as page_manager_lib 
from tunix.generate import utils


@dataclasses.dataclass
class SamplingConfig:
  max_generation_steps: int
  max_prompt_length: int
  temperature: float = 0.0
  top_p: float | None = None
  top_k: int | None = None
  beam_size: int | None = None
  seed: int | None = None
  forbidden_tokens: Tuple[int, ...] | None = None
  eos_tokens: Sequence[int] | None = None
  include_logits: bool = False
  include_logprobs: bool = True 
  pad_output: bool = False
  max_audio_length: int | None = None
  max_audio_clips: int | None = None
  attn_logits_soft_cap: float | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True)
class _SamplingState:
  """Internal sampling state for Continuous Batching."""
  # Decoding steps: ith entry contains the decoding step of the ith HBM sequence
  decoding_steps: jnp.ndarray  # i32[max_num_sequences]
  # (i, j, k) represents that sequences[0:i] are decode-only,
  # sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed.
  distribution: jnp.ndarray  # i32[3]
  # Sharded TPU HBM cache storing tokens and KV values for active sequences on TPU
  hbm_cache: page_manager_lib.PageManager
  # Is decoding done on the given sequence?
  done: jnp.ndarray  # bool[max_num_sequences]
  # Fixed-size buffer for accumulating output logits.
  logits_buffer: jnp.ndarray | None
  # Fixed-size buffer for accumulating output logprobs.
  logprobs_buffer: jnp.ndarray | None
  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: tuple[int, ...] | None
  # Uniform EOS token IDs for all requests.
  eos_token_ids: tuple[int, ...] | None
  # Random seed for sampling.
  seed: jax.Array

  # Host-side Python sequence tracking (not traced by JAX JIT)
  sampling_mode: str = flax.struct.field(
      default="greedy", pytree_node=False
  )
  max_prompt_length: int = flax.struct.field(
      default=128, pytree_node=False
  )
  max_generation_steps: int = flax.struct.field(
      default=128, pytree_node=False
  )
  temperature: float = flax.struct.field(
      default=0.0, pytree_node=False
  )
  sampling_parameters: dict[str, float | int] = flax.struct.field(
      default_factory=dict, pytree_node=False
  )
  attn_logits_soft_cap: float | None = flax.struct.field(
      default=None, pytree_node=False
  )
  beam_search_sampling_state: (
      beam_search_lib._BeamSearchSamplingState | None
  ) = None

@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Serving & execution config (decoupled from ModelConfig)."""
  # Paged memory allocation
  page_size: int = 8
  max_num_seqs: int = 32
  max_prompt_length: int = 200
  max_tokens_to_generate: int = 500
  # TODO: Replace hbm_cache_max_bytes w/ hbm_utilization 
  # like VLLM
  hbm_cache_max_bytes: int = 5 * 1024 **3 # 5 GiB


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
    logits, return_logprobs: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  if not return_logprobs:
    return next_token, None
  logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
  logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  return next_token, logp_sampled

class VanillaSampler:
  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: Any,
      image_processor: image_processor.ImageProcessor | None = None,
  ):
    """Initializes the sampler.

    Args:
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      cache_config: configuration for the KV cache.
      image_processor: The image processor.
    """

    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.cache_config = cache_config
    self.image_processor = image_processor
    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)  # pyrefly: ignore[bad-assignment]
    self._transformer_state: list[statelib.State] = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    # We separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time.
    self._compiled_decode_fn = jax.jit(
      self._decode_fn, static_argnames=("echo", "max_prompt_length")
    )
    self._compiled_prefill_fn = jax.jit(
      self._prefill_fn, static_argnames=("echo", "max_prompt_length")
    )
    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )

  def model_def_and_state(self) -> tuple[graph.NodeDef, statelib.State]:
    """Returns the transformer graphdef and state."""
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


  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def init_sample_state(
      self,
      sampling_config: SamplingConfig,
  ) -> _SamplingState:
    """Initialize sampling state with HBM, Offloaded, and Prefix cache pools."""
    max_seq_len = sampling_config.max_generation_steps + sampling_config.max_prompt_length
    
    page_size = self.cache_config.page_size
    max_num_seqs = self.cache_config.max_num_seqs
    
    num_kv_heads = self.transformer.config.num_kv_heads
    head_dim = self.transformer.config.head_dim
    num_layers = self.transformer.config.num_layers
    dtype = self.dtype

    hbm_max_bytes = self.cache_config.hbm_cache_max_bytes

    shd_config = getattr(getattr(self.transformer, "config", None), "shd_config", None)
    
    dp_size = 1
    tp_size = 1
    if shd_config is not None:
      dp_axis = shd_config.act_btd[0]
      tp_axis = shd_config.act_btnh[2]
      
      try:
        param_0 = jax.tree.leaves(self._flattened_transformer_state)[0]
        if hasattr(param_0, "sharding") and hasattr(param_0.sharding, "mesh") and param_0.sharding.mesh is not None:
          mesh = param_0.sharding.mesh
          dp_size = mesh.shape.get(dp_axis, 1) if dp_axis else 1
          tp_size = mesh.shape.get(tp_axis, 1) if tp_axis else 1
      except Exception:
        pass
    else:
      dp_axis = None
      tp_axis = None
    
    hbm_pm_config = page_manager_lib.PageManagerConfig(
        page_size=page_size,
        max_seq_len=max_seq_len,
        max_bytes=hbm_max_bytes,
        num_kv_heads=num_kv_heads,
        max_num_seqs=max_num_seqs,
        head_dim=head_dim,
        dtype=dtype,
        num_layers=num_layers,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        dp_size=dp_size,
        tp_size=tp_size,
    )
    hbm_cache = hbm_pm_config.init()
    
    eos_ids = tuple(sampling_config.eos_tokens) if sampling_config.eos_tokens is not None else None
    
    if sampling_config.include_logprobs:
      logprobs_buffer=jnp.zeros((max_num_seqs, max_seq_len))
    else:
      logprobs_buffer = None

    if sampling_config.include_logits:
      vocab_size = self.transformer.config.vocab_size
      logits_buffer=jnp.zeros((max_num_seqs, max_seq_len, vocab_size))
    else:
      logits_buffer = None

    sampling_parameters = {}
    sampling_mode = [None]

    if sampling_config.beam_size is not None:
      # Beam search is ommited from this CL for brevity 
      raise ValueError("Beam Search not yet supported")

    if sampling_config.top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')  # pyrefly: ignore[bad-argument-type]
      sampling_parameters['top_p'] = sampling_config.top_p
      sampling_parameters['top_k'] = sampling_config.top_k
    
    if sampling_mode[0] is None:
      sampling_mode[0] = 'greedy'  # pyrefly: ignore[unsupported-operation]    
    
    logging.debug('Using sampling mode: %s', sampling_mode[0])

    return _SamplingState(
        max_prompt_length=sampling_config.max_prompt_length,
        max_generation_steps=sampling_config.max_generation_steps,
        distribution=jnp.array([0, 0, 0], dtype=jnp.int32),
        hbm_cache=hbm_cache,
        done=jnp.zeros((max_num_seqs,), dtype=jnp.bool_),
        insertion_timestamps=jnp.zeros((max_num_seqs,), dtype=jnp.float32),
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        forbidden_token_ids=sampling_config.forbidden_tokens,
        eos_token_ids=eos_ids,
        seed=jax.random.PRNGKey(sampling_config.seed or 0),
        hbm_request_ids=[],
        cpu_request_ids=[],
        sampling_mode="top_p" if sampling_config.top_p else "greedy",
        temperature=sampling_config.temperature,
        sampling_parameters=sampling_parameters,
        attn_logits_soft_cap=sampling_config.attn_logits_soft_cap,
    )

  def _tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    input_ids = np.array(
        self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
    )
    return input_ids

  def _load_new_requests(
      self,
      sampling_state: _SamplingState,
      new_requests: str | Sequence[str],
  ) -> _SamplingState:
    """Load incoming requests into cache."""
    if isinstance(new_requests, str):
      new_requests = [new_requests]

    hbm_cache = sampling_state.hbm_cache
    batch_size = hbm_cache.batch_size
    max_prompt_length = sampling_state. max_prompt_length
    
    q_lens = jnp.zeros((batch_size,), dtype=jnp.int32)
    
    static_token_capacity = int(
         max_prompt_length * batch_size 
    )
    token_buff = jnp.zeros((static_token_capacity), dtype=jnp.int32) 
    
    total_tokens = 0
    for prompt_str in new_requests:
      prompt_tokens = self._tokenize(prompt_str)
      prompt_len = len(prompt_tokens)

      q_lens = q_lens.at[slot_idx].set(prompt_len)
      token_buff = token_buff.at[total_tokens: total_tokens + prompt_len].set(prompt_tokens)
      total_tokens += prompt_len

    hbm_cache = hbm_cache.allocate(q_lens=q_lens)
    hbm_cache = hbm_cache.load_prompt_tokens(token_buff, lens=q_lens, key="token_buffer")

    return dataclasses.replace(
        sampling_state,
        hbm_cache=-hbm_cache,
    )

  def _sample(
      self,
      cache: page_manager_lib.PageManager,
      logits: jnp.ndarray,
      eos: jax.Array,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Samples a token from the logits."""

    logits = logits[:, -1][:, None, :]  # B, 1, V
    done = sampling_state.done
    logits_buffer = sampling_state.logits_buffer
    logprobs_buffer = sampling_state.logprobs_buffer
    if sampling_state.forbidden_token_ids:
      logits = logits.at[:, :, sampling_state.forbidden_token_ids].set(-jnp.inf)

    if sampling_state.sampling_mode == 'beam_search':
      raise ValueError("Beam search is not yet supported")
    else:
      if sampling_state.sampling_mode == 'greedy':
        next_token_candidate, logp = sample_best(
            logits, return_logprobs=(logprobs_buffer is not None)
        )
      elif sampling_state.sampling_mode == 'top_p':
        key = jax.random.fold_in(sampling_state.seed, decoding_step)
        next_token_candidate, logp = sample_top_p(
            logits,
            key,
            sampling_state.temperature,
            sampling_state.sampling_parameters['top_p'],
            sampling_state.sampling_parameters['top_k'],  # pyrefly: ignore[bad-argument-type]
            return_logprobs=(logprobs_buffer is not None),
        )
      else:
        raise ValueError(
            'Unsupported sampling mode: %s' % sampling_state.sampling_mode
        )
    
      not_done = ~done 
      cache = cache.append_tokens(next_token_candidate, valid_mask=not_done)
      if logprobs_buffer is not None:
        logprobs_buffer = logprobs_buffer.at[:, decoding_steps].set(logp)
      if logits_buffer is not None:
        logits_buffer = logits_buffer.at[:, decoding_steps, :].set(logits)
    
    decoding_steps += not_done
    done = done | jnp.isin(next_token_candidate, eos)
    return dataclasses.replace(
        sampling_state,
        decoding_steps=decoding_steps,
        hbm_cache=cache,
        done=done,
        
    )

  def _prefill_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
      images: jnp.ndarray | None = None,
      audios: Any = None,
      echo: bool = False,
  ) -> _SamplingState:
    """Performs prefill."""
    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
      
    cache = sampling_state.hbm_cache
    batch_size = cache.batch_size
    max_prompt_length = sampling_state.max_prompt_length 
    logits_buffer = sampling_state.logits_buffer
    logprobs_buffer = sampling_state.logprobs_buffer
    soft_cap = sampling_state.soft_cap
    done = sampling_state.done

    ragged = page_manager_lib.RaggedArray(
        data=jnp.zeros((batch_size,), dtype=jnp.int32),
        lens=cache.seq_lens,
    )

    not_done = ~done 
    seq_idxs = ragged.row_idxs
    positions = cache.seq_lens - 1 

    page_cols = positions // cache.page_size
    page_offsets = positions % cache.page_size
    phys_page_ids = cache.page_indices[seq_idxs, page_cols]

    tokens = cache.pages["token_buffer"][
        phys_page_ids, page_offsets
    ]

    # Tokens must be ragged so we move 'eos' tokens to the end of the buffer 
    target_idxs = jnp.cumsum(not_done) - 1
    target_idxs = jnp.where(not_done, target_idxs, batch_size - 1)
    
    tokens = tokens.at[target_idxs].set(tokens)    
    
    distribution = jax.Array([batch_size, batch_size, batch_size])
    logits, cache = transformer(
        tokens,
        positions.reshape(-1), 
        cache=cache,
        distribution=distribution,
        seq_lens=cache.seq_lens,
        soft_cap=soft_cap,
    )

    updated_sampling_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampling_state=updated_sampling_state,
    )

    return updated_sampling_state 

  def _decode_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal generating function (to be jitted)."""

    def sample_with_params(sampling_state: _SamplingState):
      return self._sample_step(params, sampling_state)

    def cond_fn(sampling_state: _SamplingState):
      return jnp.all(
          sampling_state.hbm_decoding_steps < sampling_state.total_sampling_steps - 1
      ) & jnp.any(jnp.logical_not(sampling_state.done))

    return jax.lax.while_loop(cond_fn, sample_with_params, sampling_state)

  def _sample_step(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Performs prefill."""
    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
      
    cache = sampling_state.hbm_cache
    batch_size = cache.batch_size
    max_prompt_length = sampling_state.max_prompt_length 
    logits_buffer = sampling_state.logits_buffer
    logprobs_buffer = sampling_state.logprobs_buffer
    soft_cap = sampling_state.soft_cap

    static_token_capacity = int(
         batch_size 
    )
    ragged = page_manager_lib.RaggedArray(
        data=jnp.zeros((static_token_capacity,), dtype=jnp.int32),
        lens=,
    )
    seq_idxs = ragged.row_idxs
    positions = ragged.intra_offsets

    page_cols = positions // cache.page_size
    page_offsets = positions % cache.page_size
    phys_page_ids = cache.page_indices[seq_idxs, page_cols]

    tokens = cache.pages["token_buffer"][
        phys_page_ids, page_offsets
    ]
    
    distribution = jax.Array([0, batch_size, batch_size])
    logits, cache = transformer(
        tokens,
        positions.reshape(-1), 
        cache=cache,
        distribution=distribution,
        seq_lens=cache.seq_lens,
        soft_cap=soft_cap,
        **kwargs,
    )

    updated_sampling_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampling_state=updated_sampling_state,
    )

    return updated_sampling_state 

  def __call__(
      self,
      input_strings: str | Sequence[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      return_logprobs: bool = False,
      eos_tokens: Sequence[int] | None = None,
      forbidden_tokens: Iterable[int] | None = None,
      temperature: float = 0.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      beam_size: Optional[int] = None,
      seed: int | None = None,
      pad_output: bool = False,
      images: (
          str
          | np.ndarray
          | list[str | np.ndarray | list[str | np.ndarray] | None]
          | jnp.ndarray
          | None
      ) = None,
      audios: (
          np.ndarray | list[np.ndarray | list[np.ndarray] | None] | None
      ) = None,
      max_audio_length: int | None = None,
      max_audio_clips: int | None = None,
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
      forbidden_tokens: Optional Iterable of token IDs that are disallowed.
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
      images: input images to process. Can be a string/array, list of
        strings/arrays, or list of list of strings/arrays depending on whether
        there is one, multiple, or varying number of images per batch.
      audios: Raw audio waveforms. Can be a single array (batch_size=1), list of
        arrays (multiple samples in a batch, each sample with one clip), or a
        list of list of arrays (multiple clips for multiple samples in a batch).
        A mix of these is also allowed. E.g. `[a1, [a2, a3], []]` would mean the
        first sample has 1 audio clip (a1), the second sample has 2 audio clips
        (a2 and a3), and the third sample has 0 audio clips.
      max_audio_length: Maximum length of audio waveforms. If specified, audio
        input to the model will be padded upto this length. Specify to avoid
        recompilation on different audio lengths across calls.
      max_audio_clips: Maximum number of audio clips in a sample. If specified,
        audio input to the model will be padded upto this count. Specify to
        avoid recompilation on different number of clips across calls.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    self.eos_ids = jnp.array(eos_tokens or [self.tokenizer.eos_id()])
    input_strings = (
        [input_strings] if isinstance(input_strings, str) else input_strings
    )

    forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None

    tokens = [self.tokenize(x) for x in input_strings]

    is_gemma4 = self.transformer.__class__.__name__ == 'Gemma4'

    processed_images = None
    if is_gemma4 and images is not None:
      assert hasattr(self.transformer, 'vision_encoder')
      assert self.transformer.vision_encoder is not None
      processed_images, tokens = image_processor.process_gemma4_inputs(
          images,
          tokens,  # pyrefly: ignore[bad-argument-type]
          self.transformer.vision_encoder,
          self.tokenizer.pad_id(),
      )

    elif images is not None and self.image_processor is not None:
      processed_images = self.image_processor(images)  # pyrefly: ignore[bad-argument-type]
      processed_images = jnp.array(processed_images)

    processed_audios = None
    if audios is not None:
      if is_gemma4:
        assert hasattr(self.transformer, 'audio_encoder')
        assert self.transformer.audio_encoder is not None
        processed_audios, tokens = audio_processor.process_gemma4_inputs(
            audios=audios,  # pyrefly: ignore[bad-argument-type]
            tokens=tokens,  # pyrefly: ignore[bad-argument-type]
            audio_encoder=self.transformer.audio_encoder,
            max_audio_length=max_audio_length,
            max_audio_clips=max_audio_clips,
        )
      else:
        raise NotImplementedError('Audio support only implemented for Gemma4.')

    max_tokens_length = max(len(x) for x in tokens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)

    all_input_ids = np.array([
        utils.pad_to_length(
            x,  # pyrefly: ignore[bad-argument-type]
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
      seed = jax.random.PRNGKey(0)  # pyrefly: ignore[bad-assignment]
    elif isinstance(seed, int):
      seed = jax.random.PRNGKey(seed)  # pyrefly: ignore[bad-assignment]
    sampling_state = self.init_sample_state(
        jnp.array(all_input_ids),
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,  # pyrefly: ignore[bad-argument-type]
        beam_size=beam_size,
        include_logprobs=return_logprobs,
    )
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state,
        sampling_state,
        images=processed_images,
        audios=processed_audios,
        echo=echo,
    )

    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )
    token_buffers = sampling_state.token_buffer
    logits_buffers = sampling_state.logits_buffer

    final_logprobs_buffer = sampling_state.logprobs_buffer

    if sampling_state.sampling_mode == 'beam_search':
      updated_args = beam_search_lib.finalize_beam_search_state(
          sampling_state.beam_search_sampling_state,
          sampling_state.token_buffer,
          sampling_state.logits_buffer,
          sampling_state.logprobs_buffer,
      )
      token_buffers = updated_args['token_buffer']
      logits_buffers = updated_args['logits_buffer']
      final_logprobs_buffer = updated_args['logprobs_buffer']
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
      out_logprobs = []
      if return_logprobs:
        token_buffers = jax.device_get(token_buffers)
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
        for i in range(len(token_buffers)):
          start_idx = (
              utils.np_find_first_non_pad_idx(
                  token_buffers[i], self.tokenizer.pad_id()
              )
              if echo
              else max_prompt_length
          )
          end_idx = (
              utils.np_find_first_eos_idx(
                  token_buffers[i][max_prompt_length:], self.eos_ids
              )
              + max_prompt_length
          )
          length = end_idx - start_idx
          # Slice logprobs and pad to max_len
          sliced_logprobs = final_logprobs_buffer[i][start_idx:end_idx]
          padded_logprobs = np.pad(
              sliced_logprobs,
              (0, max_len - length),
              mode='constant',
              constant_values=0.0,
          )
          out_logprobs.append(padded_logprobs.tolist())

    else:
      out_tokens = []
      out_logits = []
      out_logprobs = []
      token_buffers = jax.device_get(token_buffers)
      if return_logprobs:
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
      if return_logits:
        logits_buffers = jax.device_get(logits_buffers)
      for i in range(len(token_buffers)):
        token_buffer = token_buffers[i]
        start_idx = (
            utils.np_find_first_non_pad_idx(
                token_buffer, self.tokenizer.pad_id()
            )
            if echo
            else max_prompt_length
        )
        end_idx = (
            utils.np_find_first_eos_idx(
                token_buffer[max_prompt_length:], self.eos_ids
            )
            + max_prompt_length
        )
        out_tokens.append(token_buffer[start_idx:end_idx])
        if return_logits:
          out_logits.append(logits_buffers[i][start_idx:end_idx])
        if return_logprobs:
          # Extract logprobs for the generated tokens
          out_logprobs.append(
              final_logprobs_buffer[i][start_idx:end_idx].tolist()
          )

      decoded_outputs = [
          self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
      ]

    result = base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else [],
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs if return_logprobs else None,
    )
    return result
