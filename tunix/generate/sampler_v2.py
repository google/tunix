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
from tunix.generate import cache_manager as cache_manager_lib
from tunix.generate import page_manager as page_manager_lib 
from tunix.generate import utils
from tunix.generate import base_sampler


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
  hbm_cache: Any
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
  """
  Vanilla rollout wrapper that utilizes the Continuous Batch Engine natively.
  Replaces disorganized monolithic JAX loops with a clean synchronous API onto the LLMEngine.
  """
  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: Any,
      image_processor: Any | None = None,
  ):
    from tunix.generate import engine
    self.tokenizer = tokenizer
    self.cache_config = cache_config
    
    self.engine = engine.LLMEngine(
        transformer=transformer,
        tokenizer=tokenizer,
        cache_config=cache_config,
        image_processor=image_processor,
        max_seq_len=getattr(cache_config, "max_seq_len", 1000)
    )

  def _tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if hasattr(self.tokenizer, 'bos_id') and self.tokenizer.bos_id() else []
    
    if hasattr(self.tokenizer, 'dedup_bos_ids'):
        input_ids = self.tokenizer.dedup_bos_ids(bos_tok + input_ids)
    else:
        input_ids = bos_tok + input_ids
        
    return np.array(input_ids, dtype=np.int32).tolist()

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
      top_p: float | None = None,
      top_k: int | None = None,
      seed: int | None = None,
      attn_logits_soft_cap: float | None = None,
  ) -> base_sampler.SamplerOutput:
    """Run synchronous rollout by pushing requests to the Engine."""
    
    if isinstance(input_strings, str):
      input_strings = [input_strings]

    req_ids = []
    for i, prompt_str in enumerate(input_strings):
      req_id = f"vanilla_{i}_{id(self)}"
      prompt_tokens = self._tokenize(prompt_str)
      self.engine.add_request(req_id, prompt_tokens)
      req_ids.append(req_id)
      
    while self.engine.has_unfinished_requests():
      self.engine.step()
      
    # Gather results
    decoded_outputs = []
    # Logprobs not yet fully piped out from Engine to generated_tokens format for historical tracking, 
    # but we will return the generated output identically.
    out_logprobs = None 
    
    for req_id in req_ids:
      gen_tokens = self.engine.generated_tokens[req_id]
      if hasattr(self.tokenizer, "decode"):
          decoded_str = self.tokenizer.decode(gen_tokens)
      else:
          decoded_str = "".join(str(t) for t in gen_tokens)
      decoded_outputs.append(decoded_str)
      
    
    out_tokens = [self.engine.generated_tokens[req_id] for req_id in req_ids]
    result = base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=[] if not return_logits else None,
        tokens=[np.array(t, dtype=np.int32) for t in out_tokens],
        padded_prompt_tokens=[],
        logprobs=None,
    )
    return result
