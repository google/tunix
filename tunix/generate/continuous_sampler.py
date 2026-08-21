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
from tunix.generate import page_manager as page_manager_lib 
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


from tunix.generate.cache_manager import CacheManager
from tunix.generate.scheduler import Scheduler, Request

class ContinuousSampler:
  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: Any,
      image_processor: Any | None = None,
      max_seq_len: int = 1000,
  ):
    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    
    self.cache_config = cache_config
    self.image_processor = image_processor
    self._transformer_graphdef: Any = nnx.graphdef(transformer) 
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

  def _model_step_fn(
      self,
      params: statelib.State,
      tokens: jnp.ndarray,
      positions: jnp.ndarray,
      cache: CacheManager,
      distribution: jnp.ndarray,
      seq_lens: jnp.ndarray,
      soft_cap: float | None = None,
      images: jnp.ndarray | None = None,
      audios: Any = None,
      echo: bool = False,
  ) -> Tuple[jnp.ndarray, CacheManager]:
    """Unified forward pass invoking transformer with explicit distribution."""
    transformer = nnx.merge(self._transformer_graphdef, params)
    
    kwargs = {}
    if images is not None:
      kwargs['images'] = images
    if audios is not None:
      kwargs['audios'] = audios
    decode_only_last_token = self._supports_decode_only_last_token and not echo
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True
      
    logits, cache = transformer(
        tokens,
        positions, 
        cache=cache,
        distribution=distribution,
        seq_lens=seq_lens,
        soft_cap=soft_cap,
        **kwargs
    )
    
    # We only care about retrieving the logits corresponding to the LAST output token!
    last_token_idxs = jnp.cumsum(seq_lens) - 1
    last_token_logits = logits[last_token_idxs]
    last_token_logits = jnp.expand_dims(last_token_logits, axis=1)
    
    # Simple sampling block natively
    next_tokens, _ = sample_best(last_token_logits)
    
    return next_tokens, cache
    
  def sample(self, all_active, prefills, cache: CacheManager):
      """One map execution of the sampler mapping python active arrays to JAX primitives."""
      num_decodes = len(all_active) - len(prefills)
      total_active = len(all_active)
      
      # Build inputs natively mapping dynamically! 
      # (0, i) is decodes, (i, j) is chunked prefills, (j, k) is mixed prefills
      # The array `distribution` maps exactly to [i, j, k]. We'll map all prefills to mixed chunk bound.
      distribution_arr = jnp.array([num_decodes, total_active, total_active], dtype=jnp.int32)
      
      # We construct tokens, positions and seq_lens matrices cleanly across decodes then prefills
      seq_lens_arr = []
      tokens_arr = []
      positions_arr = []
      
      for r in all_active:
          is_prefill = r in prefills 
          active_len = len(r.prompt_tokens) if is_prefill else 1
          
          # For decoding, the token is just the last mapped token added (or last prompt token)
          # We fetch directly using standard python tracking block
          tks = r.prompt_tokens[-active_len:]
          pos = list(range(len(r.prompt_tokens) - active_len, len(r.prompt_tokens)))
          
          tokens_arr.extend(tks)
          positions_arr.extend(pos)
          seq_lens_arr.append(active_len)
          
      tokens_np = jnp.array(tokens_arr, dtype=jnp.int32)
      positions_np = jnp.array(positions_arr, dtype=jnp.int32)
      seq_lens_np = jnp.array(seq_lens_arr, dtype=jnp.int32)
      
      # Pad arrays to static loop block capacities if necessary (Optional, nnx handles ragged lengths)
      # Then execute JIT wrapper
      next_tokens, next_cache = self._compiled_step_fn(
          self._flattened_transformer_state,
          tokens=tokens_np,
          positions=positions_np,
          cache=cache,
          distribution=distribution_arr,
          seq_lens=seq_lens_np,
          soft_cap=None,
      )
      
      next_tokens_cpu = jax.device_get(next_tokens)
      return next_tokens_cpu, next_cache
