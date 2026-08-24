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
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union, Optional
import sys

import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import jaxtyping
import numpy as np
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.generate import base_sampler
from tunix.generate import cache_manager as cache_manager_lib 
from tunix.generate import utils
import logging

def _get_dtype_packing(dtype):
  n_bytes = jnp.dtype(dtype).itemsize
  return 4 // n_bytes

@dataclasses.dataclass
class SamplingConfig:
  max_generation_steps: int
  max_prompt_length: int
  batch_size: int = 128
  temperature: float = 0.0
  top_p: float | None = None
  top_k: int | None = None
  beam_size: int | None = None
  seed: int | None = None
  forbidden_tokens: Tuple[int, ...] | None = None
  eos_tokens: Sequence[int] | None = None
  include_logits: bool = False
  include_logprobs: bool = False 
  pad_output: bool = False
  max_audio_length: int | None = None
  max_audio_clips: int | None = None
  attn_logits_soft_cap: float | None = None

@dataclasses.dataclass
class RequestOutput:
  request_id: str
  text: str
  tokens: List[int]
  padded_tokens: np.ndarray
  logprobs: np.ndarray | None = None
  logits: np.ndarray | None = None

class RequestFuture:
  """Future handle returned to callers submitting async requests."""

  def __init__(self, request_id: str):
    self.request_id = request_id
    self._event = threading.Event()
    self._output: RequestOutput | None = None
    self._error: Exception | None = None

  def set_result(self, output: RequestOutput) -> None:
    self._output = output
    self._event.set()

  def set_error(self, error: Exception) -> None:
    self._error = error
    self._event.set()

  def result(self, timeout: float | None = None) -> RequestOutput:
    self._event.wait(timeout=timeout)
    if self._error is not None:
      raise self._error
    if self._output is None:
      raise TimeoutError(f"RequestFuture {self.request_id} timed out.")
    return self._output

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
  
  # Keep for backwards compatability
  cache_size: int = 0
  num_layers: int = 0
  num_kv_heads: int = 0
  head_dim: int = 0

def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    include_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Sample a token using top-p sampling."""
  # Upcast to float32 for numerical stability of softmax and subsequent cumsum.
  next_token_logits = logits[:, -1].astype(jnp.float32) / temperature

  # top_k=0 or None both mean "no top-k filtering" — use full vocabulary.
  _no_topk = top_k is None or top_k <= 0
  # Skip softmax and sorting if top_p is 1.0 and top_k is full vocab.
  if top_p >= 1.0 and _no_topk:
    next_token = jax.random.categorical(key, logits=next_token_logits)
    if not include_logprobs:
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

  if include_logprobs:
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  else:
    logp_sampled = None

  return next_token, logp_sampled


def sample_best(
    logits, include_logprobs: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  if not include_logprobs:
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
      image_processor: Any | None = None,
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
    self._transformer_graphdef: Any = nnx.graphdef(transformer)  # pyrefly: ignore[bad-assignment]
    self._transformer_state: list[Any] = nnx.variables(transformer)
    self._flattened_transformer_state: list[Any] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    self._compiled_step_fn = jax.jit(self._model_step_fn, static_argnames=("echo",))
    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )
    
  def model_def_and_state(self) -> tuple[Any, Any]:
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(  # pyrefly: ignore[no-matching-overload]
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> Any:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: Any) -> None:

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

  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def _form_cache_block_specs(self, sampling_config):
    num_kv_heads = self.transformer.config.num_kv_heads
    head_dim = self.transformer.config.head_dim
    num_layers = self.transformer.config.num_layers
    kv_dtype = self.dtype

    block_specs = []
    token_spec = cache_manager_lib.BlockSpec(
        name="tokens",
        subshape=(),
        dtype=jnp.int32,
    )
    block_specs.append(token_spec)

    if sampling_config.include_logprobs:
      logprobs_spec = cache_manager_lib.BlockSpec(
        name="logprobs",
        subshape=(),
        dtype=jnp.float32,
      )
      block_specs.append(logprobs_spec)

    if sampling_config.include_logits:
      num_embed = self.transformer.num_embed
      logits_spec = cache_manager_lib.BlockSpec(
        name="logits",
        subshape=(num_embed,),
        dtype=jnp.float32,
      )
      block_specs.append(logits_spec)
    
    kv_packing = _get_dtype_packing(kv_dtype) 
    packed_kv_head_dim = 2 * num_kv_heads // kv_packing
    for i in range(num_layers):
      layer_spec = cache_manager_lib.BlockSpec(
          name=f"layer_{i}",
          subshape=(packed_kv_head_dim, kv_packing, head_dim,),
          dtype=kv_dtype,
          logical_subsharding=("tp_axis", None, None)
      )
      block_specs.append(layer_spec)
    
    return block_specs

  def _init_cache(self, sampling_config):
    hbm_max_bytes = self.cache_config.hbm_cache_max_bytes
    page_size = self.cache_config.page_size
    max_num_seqs = sampling_config.batch_size 
    max_seq_len = sampling_config.max_prompt_length + sampling_config.max_generation_steps
    page_size = min(page_size, max_seq_len)

    block_specs = self._form_cache_block_specs(sampling_config)
    shd_config = getattr(getattr(self.transformer, "config", None), "shd_config", None)

    dp_size = 1
    tp_size = 1
    dp_axis = None
    tp_axis = None

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
    
    hbm_pm_config = cache_manager_lib.PageManagerConfig(
        logical_page_sharding="dp_axis",
        block_specs=block_specs,
        page_size=page_size,
        max_seq_len=max_seq_len,
        max_bytes=hbm_max_bytes,
        max_num_seqs=max_num_seqs,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        dp_size=dp_size,
        tp_size=tp_size,
    )
    hbm_cache = hbm_pm_config.init()

    cpu_device = jax.devices("cpu")[0] if jax.devices("cpu") else None
    cpu_pm_config = dataclasses.replace(
      hbm_pm_config,
      max_bytes=cpu_max_bytes,
      dp_axis=None,
      tp_axis=None,
      dp_size=1,
      tp_size=1,
      device=cpu_device
    )
    offloaded_cache = cpu_pm_config.init()

    return hbm_cache, offloaded_cache
    
  def init_sample_state(
      self,
      sampling_config: SamplingConfig,
  ) -> _SamplingState:
    """Initialize sampling state with HBM, Offloaded, and Prefix cache pools."""
    max_seq_len = sampling_config.max_generation_steps + sampling_config.max_prompt_length
    hbm_cache, offloaded_cache = self._init_cache(sampling_config)
    batch_size = sampling_config.batch_size
        
    sampling_parameters = {}
    sampling_mode = [None]

    if sampling_config.beam_size is not None:
      # Beam search is ommited from this CL for brevity 
      raise ValueError("Beam Search not yet supported")

    if sampling_config.top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')  # pyrefly: ignore[bad-argument-type]
      sampling_parameters['top_p'] = sampling_config.top_p
      sampling_parameters['top_k'] = sampling_config.top_k
      sampling_mode[0] = 'top_p'
    
    if sampling_mode[0] is None:   
      sampling_mode[0] = 'greedy'
    
    logging.debug('Using sampling mode: %s', sampling_mode[0])
    
    eos_ids = sampling_config.eos_tokens
    eos_ids = jnp.array(eos_ids or [self.tokenizer.eos_id()])
    
    seed = sampling_config.seed
    if seed is None:
      seed = jax.random.PRNGKey(0)  # pyrefly: ignore[bad-assignment]
    elif isinstance(seed, int):
      seed = jax.random.PRNGKey(seed)  # pyrefly: ignore[bad-assignment]

    return _SamplingState(
        max_prompt_length=sampling_config.max_prompt_length,
        max_generation_steps=sampling_config.max_generation_steps,
        distribution=jnp.array([0, 0, 0], dtype=jnp.int32),
        hbm_cache=hbm_cache,
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        decoding_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
        offloaded_decoding_steps=jnp.zeros((max_num_seqs,), dtype=jnp.int32),
        include_logits=sampling_config.include_logits,
        include_logprobs=sampling_config.include_logprobs,
        forbidden_token_ids=sampling_config.forbidden_tokens,
        eos_token_ids=eos_ids,
        hbm_request_ids=[],
        cpu_request_ids=[],
        seed=seed,
        sampling_mode=sampling_mode[0],
        temperature=sampling_config.temperature,
        sampling_parameters=sampling_parameters,
        attn_logits_soft_cap=sampling_config.attn_logits_soft_cap,
    )

  def cancel_request(
      self,
      sampling_state: _SamplingState,
      request_id: str,
  ) -> _SamplingState:
    """Cancels a request, and release slots."""
    sampling_state, updated_offloaded, removed = self._remove_request_from_pool(
        sampling_state, request_id, sampling_state.cpu_request_ids, sampling_state.offloaded_cache
    )
    if removed:
      return dataclasses.replace(sampling_state, offloaded_cache=updated_offloaded)

    sampling_state, updated_hbm, removed = self._remove_request_from_pool(
        sampling_state, request_id, sampling_state.hbm_request_ids, sampling_state.hbm_cache
    )
    if removed:
      slot = len(sampling_state.hbm_request_ids)
      sampling_state = dataclasses.replace(
          sampling_state,
          hbm_cache=updated_hbm,
          done=sampling_state.done.at[slot].set(True),
      )
      return self._compact_batch(sampling_state)

    return sampling_state

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    
    if hasattr(self.tokenizer, 'bos_id') and callable(self.tokenizer.bos_id):
      bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
      if hasattr(self.tokenizer, 'dedup_bos_ids'):
        input_ids = np.array(
            self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
        )
      else:
        input_ids = np.array(bos_tok + input_ids, dtype=np.int32)
    else:
      input_ids = np.array(input_ids, dtype=np.int32)
    return input_ids

  def _model_step_fn(
      self,
      params: statelib.State,
      images: jnp.ndarray | None = None,
      audios: Any = None,
      echo: bool = False,
      soft_cap: float | None = None,
      **kwargs,
  ) -> Tuple[jnp.ndarray, cache_manager_lib.PageManager]:
    """JIT-compiled forward pass invoking Gemma with ragged paged attention and explicit soft_cap."""
    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
    
    # TODO: kwargs should only be used for prefill seqs
      
    kwargs = {}
    if images is not None:
      kwargs['images'] = images
    if audios is not None:
      kwargs['audios'] = audios
    decode_only_last_token = self._supports_decode_only_last_token and not echo
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True
    

    transformer = nnx.merge(self._transformer_graphdef, params)

    cache = sampling_state.hbm_cache
    batch_size = cache.batch_size
    max_prompt_length = sampling_state.max_prompt_length
    include_logits = sampling_state.include_logits
    soft_cap = sampling_state.attn_logits_soft_cap

    is_decode = (decoding_steps > 0) & (cache.seq_lens > 0)
    active_seq_lens = jnp.where(
        is_decode,
        1,
        cache.seq_lens,
    )

    token_start_idxs = jnp.where(
        is_decode,
        cache.seq_lens - 1,
        0,
    )

    # TODO: Replace static_token_capacity with max_tokens.
    # We should priortize decode sequences and fill in
    # remaining space with chunked prefill.
    max_seqs = cache.batch_size
    static_token_capacity = int(
         max_prompt_length * max_seqs
    )

    ragged = cache_manager_lib.RaggedArray(
        data=jnp.zeros((static_token_capacity,), dtype=jnp.int32),
        lens=active_seq_lens,
    )
    seq_idxs = ragged.row_idxs
    intra_offsets = ragged.intra_offsets

    abs_positions = token_start_idxs[seq_idxs] + intra_offsets
    page_cols = abs_positions // cache.page_size
    page_offsets = abs_positions % cache.page_size
    phys_page_ids = cache.page_indices[seq_idxs, page_cols]

    tokens = cache.pages["tokens"][
        phys_page_ids, page_offsets
    ]

    logits, cache = transformer(
        tokens,
        abs_positions.reshape(-1),
        cache=cache,
        distribution=distribution,
        seq_lens=cache.seq_lens,
        soft_cap=soft_cap,
        **kwargs,
    )

    last_token_idxs = jnp.cumsum(active_seq_lens) - 1
    last_token_logits = logits[last_token_idxs]
    last_token_logits = jnp.expand_dims(last_token_logits, axis=1)

    has_new_logits = ~sampling_state.done
    updated_sampling_state = self._sample(
        logits=last_token_logits,
        cache=cache,
        eos=jnp.array(sampling_state.eos_token_ids),
        sampling_state=sampling_state,
    )

    if not include_logits:
      return updated_sampling_state

    cache = updated_sampling_state.hbm_cache
    n_decode = distribution[0]
  
    # Record decode logits
    input_token_idxs = cache.seq_lens - 2 # Ignore new token
    decode_logits = last_token_logits[:, 0, :]
    is_decode_seq = jnp.arange(batch_size) < n_decode

    cache = cache.insert_values(
        decode_logits,
        idxs=input_token_idxs,
        valid_mask=has_new_logits & is_decode_seq,
        block_id="logits"
    )
    
    # Record prefill logits
    prompt_lens = cache.seq_lens - 1 # Ignore new token
    is_prefill_logit = jnp.arange(static_token_capacity) >= n_decode
    cache = cache.load_values(
      logits,
      lens=prompt_lens,
      valid_mask=is_prefill_logit,
      block_id="logits"
    )
    
    return dataclasses.replace(
        updated_sampling_state,
        hbm_cache=cache
    )


  def _sample_step(
      self,
      sampling_state: _SamplingState,
      requests: Sequence[dict[str, Any]] = (),
  ) -> dict[str, RequestOutput]:
    """Complete a single sampling step, and return completions."""
    sampling_state = self._queue_new_requests(sampling_state, requests)
    hbm_available, sampling_state = self._make_room_for_allocation(sampling_state)
    sampling_state = self._drain_pending_queue(hbm_available, sampling_state)

    num_tpu = len(sampling_state.hbm_request_ids)
    if num_tpu == 0:
      return sampling_state, {}

    cache = sampling_state.hbm_cache

    logits, updated_hbm_cache = self._compiled_step_fn(
        sampling_state.max_prompt_length,
        params=self._flattened_transformer_state,
        cache=sampling_state.hbm_cache,
        decoding_steps=sampling_state.decoding_steps,
        distribution=sampling_state.distribution,
        soft_cap=sampling_state.attn_logits_soft_cap,
    )

    key, subkey = jax.random.split(sampling_state.seed)

    next_tokens, log_probs = sampler_lib.sample_top_p(
        logits=logits,
        key=subkey,
        temperature=sampling_state.temperature,
        top_p=sampling_state.sampling_parameters['top_p'],
        top_k=sampling_state.sampling_parameters['top_k'],  # pyrefly: ignore[bad-argument-type]
        return_logprobs=True,
    )

    seq_idxs = jnp.arange(updated_hbm_cache.batch_size)
    valid_mask = seq_idxs  < num_tpu
    updated_hbm_cache = updated_hbm_cache.append_tokens(next_tokens, valid_mask)

    updated_logits = None
    updated_logprobs = None

    decoding_steps = sampling_state.decoding_steps
    if sampling_state.logits_buffer is not None:
      updated_logits = sampling_state.logits_buffer.at[seq_idxs, decoding_steps, :].set(logits)
    if sampling_state.logprobs_buffer is not None:
      updated_logprobs = sampling_state.logprobs_buffer.at[seq_idxs, decoding_steps].set(log_probs)

    sampling_state = dataclasses.replace(
        sampling_state,
        hbm_cache=updated_hbm_cache,
        logits_buffer=updated_logits,
        logprobs_buffer=updated_logprobs,
        seed=key,
        decoding_steps=decoding_steps + 1,
    )

    sampling_state, completed_outputs = self._release_completed(sampling_state, next_tokens)
    return sampling_state, completed_outputs

  def __call__(
      self,
      sampling_state: _SamplingState,
      requests: Sequence[dict[str, Any]] = (),
  ) -> dict[str, RequestOutput]:
    """Forward call to _sample_step."""
    return self._sample_step(sampling_state, requests)

  def _sample(
      self,
      cache: cache_manager_lib.PageManager,
      logits: jnp.ndarray,
      eos: jax.Array,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Samples a token from the logits."""
    done = sampling_state.done
    # logprobs_buffer = sampling_state.logprobs_buffer
    if sampling_state.forbidden_token_ids:
      logits = logits.at[:, :, sampling_state.forbidden_token_ids].set(-jnp.inf)
    
    if sampling_state.sampling_mode == 'beam_search':
      raise ValueError("Beam search is not yet supported")
    else:
      if sampling_state.sampling_mode == 'greedy':
        next_token_candidate, logp = sample_best(
            logits, include_logprobs=(sampling_state.include_logprobs)
        )
      elif sampling_state.sampling_mode == 'top_p':
        key = jax.random.fold_in(sampling_state.seed, sampling_state.global_decoding_step)
        next_token_candidate, logp = sample_top_p(
            logits=logits,
            key=key,
            temperature=sampling_state.temperature,
            top_p=sampling_state.sampling_parameters['top_p'],
            top_k=sampling_state.sampling_parameters['top_k'],  # pyrefly: ignore[bad-argument-type]
            include_logprobs=(sampling_state.include_logprobs),
        )
      else:
        raise ValueError(
            'Unsupported sampling mode: %s' % sampling_state.sampling_mode
        )
      
      is_new_logp = ~done 
      done = done | jnp.isin(next_token_candidate, eos)
      not_done = ~done 
    
      cache = cache.allocate(q_lens=not_done)
      cache = cache.insert_values(next_token_candidate, valid_mask=not_done, block_id="tokens")
      
      if sampling_state.include_logprobs:
        cache = cache.insert_values(logp, valid_mask=is_new_logp, block_id="logprobs")

      decoding_steps = sampling_state.decoding_steps + not_done
    trunc = decoding_steps >= sampling_state.max_generation_steps 
    done = done | trunc

    return dataclasses.replace(
        sampling_state,
        decoding_steps=decoding_steps,
        global_decoding_step=sampling_state.global_decoding_step+1,
        hbm_cache=cache,
        done=done,
    )

  def _prefill_fn(
      self,
      params: Any,
      sampling_state: _SamplingState,
      images: jnp.ndarray | None = None,
      audios: Any = None,
      echo: bool = True,
  ) -> _SamplingState:
    """Performs prefill."""
    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]

    kwargs = {}
    if images is not None:
      kwargs['images'] = images
    if audios is not None:
      kwargs['audios'] = audios
    decode_only_last_token = self._supports_decode_only_last_token and not echo
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True
      
    cache = sampling_state.hbm_cache
    batch_size = cache.batch_size
    max_prompt_length = sampling_state.max_prompt_length 
    include_logits = sampling_state.include_logits
    include_logprobs = sampling_state.include_logprobs
    soft_cap = sampling_state.attn_logits_soft_cap
    done = sampling_state.done
    
    static_token_capacity = batch_size * max_prompt_length
    ragged = cache_manager_lib.RaggedArray(
        data=jnp.zeros((static_token_capacity,), dtype=jnp.int32),
        lens=cache.seq_lens,
    )

    seq_idxs = ragged.row_idxs
    positions = ragged.intra_offsets

    page_cols = positions // cache.page_size
    page_offsets = positions % cache.page_size
    phys_page_ids = cache.page_indices[seq_idxs, page_cols]

    tokens = cache.pages["tokens"][
        phys_page_ids, page_offsets
    ]
    
    distribution = jnp.array([0, 0, batch_size])

    logits, cache = transformer(
        tokens,
        positions.reshape(-1), 
        cache=cache,
        distribution=distribution,
        seq_lens=cache.seq_lens,
        soft_cap=soft_cap,
        **kwargs,
    )
   
    last_token_idxs = jnp.cumsum(cache.seq_lens) - 1
    last_token_logits = logits[last_token_idxs]
    last_token_logits = jnp.expand_dims(last_token_logits, axis=1)
    
    updated_sampling_state = self._sample(
      logits=last_token_logits,
      cache=cache,
      eos=jnp.array(sampling_state.eos_token_ids),
      sampling_state=sampling_state,
    )

    if not include_logits:
      return updated_sampling_state 
    
    batch_idxs = jnp.arange(batch_size)
    cache = updated_sampling_state.hbm_cache
  
    if decode_only_last_token:
      last_input_idxs = cache.seq_lens - 2
      cache = cache.insert_values(
          last_token_logits[:, 0, :], 
          idxs=last_input_idxs, 
          block_id="logits"
    )
    else: 
      input_lens = cache.seq_lens - 1
      cache = cache.load_values(
          logits, 
          lens=input_lens, 
          block_id="logits"
      )
  
    return dataclasses.replace(
        updated_sampling_state,
        hbm_cache=cache
    )

  def _decode_fn(
      self,
      params: Any,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal generating function (to be jitted)."""

    def sample_with_params(sampling_state: _SamplingState):
      return self._sample_step(params, sampling_state)

    def cond_fn(sampling_state: _SamplingState):
      return jnp.any(jnp.logical_not(sampling_state.done))
      
    return jax.lax.while_loop(cond_fn, sample_with_params, sampling_state)

  def _sample_step(
      self,
      params: Any,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Performs decode step."""
    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
      
    cache = sampling_state.hbm_cache
    batch_size = cache.batch_size
    max_prompt_length = sampling_state.max_prompt_length 
    include_logits = sampling_state.include_logits
    soft_cap = sampling_state.attn_logits_soft_cap

    static_token_capacity = batch_size 
    
    not_done = ~sampling_state.done
    active_seq_lens = jnp.where(not_done, 1, 0)

    ragged = cache_manager_lib.RaggedArray(
        data=jnp.zeros((static_token_capacity,), dtype=jnp.int32),
        lens=active_seq_lens,
    )
    seq_idxs = ragged.row_idxs
    positions = (cache.seq_lens - 1)[seq_idxs]

    page_cols = positions // cache.page_size
    page_offsets = positions % cache.page_size
    phys_page_ids = cache.page_indices[seq_idxs, page_cols]

    tokens = cache.pages["tokens"][
        phys_page_ids, page_offsets
    ]

    distribution = jnp.array([batch_size, batch_size, batch_size], dtype=jnp.int32)

    logits, cache = transformer(
        tokens,
        positions, 
        cache=cache,
        distribution=distribution,
        seq_lens=active_seq_lens,
        soft_cap=soft_cap,
    )

    last_token_idxs = jnp.cumsum(active_seq_lens) - 1
    last_token_logits = logits[last_token_idxs]
    last_token_logits = jnp.expand_dims(last_token_logits, axis=1)
    
    has_new_logits = ~sampling_state.done
    updated_sampling_state = self._sample(
        logits=last_token_logits,
        cache=cache,
        eos=jnp.array(sampling_state.eos_token_ids),
        sampling_state=sampling_state,
    )
    
    if not include_logits:
      return updated_sampling_state 
    
    cache = updated_sampling_state.hbm_cache 
    not_done = ~updated_sampling_state.done

    logits_idxs = cache.seq_lens - 2
    batch_idxs = jnp.arange(batch_size)

    cache = cache.insert_values(
        last_token_logits[last_token_idxs, 0, :], 
        idxs=logits_idxs, 
        valid_mask=has_new_logits, 
        block_id="logits"
    )

    return dataclasses.replace(
        updated_sampling_state,
        hbm_cache=cache
    )


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
      images: Any = None,
      audios: Any = None,
      max_audio_length: int | None = None,
      max_audio_clips: int | None = None,
  ) -> base_sampler.SamplerOutput:
    
    self.eos_ids = jnp.array(eos_tokens or [self.tokenizer.eos_id()])
    input_strings = (
        [input_strings] if isinstance(input_strings, str) else list(input_strings)
    )

    forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None

    tokens = [self.tokenize(x) for x in input_strings]
    lens = np.array([len(x) for x in tokens], dtype=np.int32)
    
    max_tokens_length = np.max(lens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)

    all_input_ids = np.array([
        utils.pad_to_length(
            x,
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in tokens
    ])


    flat_prompt_tokens = np.concatenate(tokens)
    total_sampling_steps = max_prompt_length + max_generation_steps

    sampling_config = SamplingConfig(
        max_generation_steps=max_generation_steps,
        max_prompt_length=max_prompt_length,
        batch_size=len(input_strings),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        beam_size=beam_size,
        seed=seed,
        forbidden_tokens=forbidden_token_ids,
        eos_tokens=eos_tokens,
        include_logits=return_logits,
        include_logprobs=return_logprobs,
        pad_output=pad_output,
    )

    sampling_state = self.init_sample_state(
        sampling_config=sampling_config,
        all_input_ids=jnp.array(flat_prompt_tokens, dtype=jnp.int32),
        q_lens=jnp.array(lens, dtype=jnp.int32),
    )
    
    # TODO: Remove global_decoding_step 
    # Set global decoding step to maintain same random keys 
    # as original Vanilla Sampler (used to test that 
    # behaivor is identical) 
    sampling_state = dataclasses.replace(
        sampling_state,
        global_decoding_step=all_input_ids.shape[1] - 1
    )

    is_gemma4 = self.transformer.__class__.__name__ == 'Gemma4'

    processed_images = None
    processed_audios = None
    if is_gemma4 and images is not None:
      raise NotImplementedError('Images are not implemeted')

    if is_gemma4 and audios is not None:
      raise NotImplementedError('Audios are not implemeted')

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
    
    total_len = jnp.sum(sampling_state.hbm_cache.seq_lens)
    ragged_token_buffer = sampling_state.hbm_cache.to_array(total_len, block_id="tokens")
    
    ragged_token_buffer = jax.device_get(ragged_token_buffer)
    seq_lens_cpu = jax.device_get(sampling_state.hbm_cache.seq_lens)
    
    out_tokens = []
    decoded_outputs = []
    out_logprobs = []
    out_logits = []
    
    if return_logprobs:
      ragged_logprobs_buffer = sampling_state.hbm_cache.to_array(total_len, block_id="logprobs")
      ragged_logprobs_buffer = jax.device_get(ragged_logprobs_buffer)

    if return_logits:
      ragged_logits_buffer = sampling_state.hbm_cache.to_array(total_len, block_id="logits")
      ragged_logits_buffer = jax.device_get(ragged_logits_buffer)

    total_len_idx = 0
    batch_size = len(tokens)
    for idx in range(batch_size):
        seq_len = seq_lens_cpu[idx]
        prompt_len = lens[idx]

        offset = 0 if echo else prompt_len
        start_idx = total_len_idx + offset
        end_idx = total_len_idx + seq_len
        res_tokens = ragged_token_buffer[start_idx: end_idx]
        
        out_tokens.append(np.array(res_tokens))
        decoded_outputs.append(self.tokenizer.decode(res_tokens.tolist()))
        
        if return_logprobs:
            res_logprobs = ragged_logprobs_buffer[start_idx: end_idx] 
            out_logprobs.append(res_logprobs)

        if return_logits:
            logits_start_idx = start_idx if echo else start_idx - 1
            logits_end_idx = end_idx if echo else end_idx - 1
            res_logits = ragged_logits_buffer[logits_start_idx: logits_end_idx] 
            out_logits.append(res_logits)

        total_len_idx += seq_len

    result = base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else None,
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs if return_logprobs else None,
    )
    return result
