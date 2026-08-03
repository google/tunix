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

"""Tunix Vanilla Sampler V2: PageMemoryManager and ContinuousBatcherScheduler."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

# Try importing Pallas paged_attention. If not available, we use a fallback.
try:
  from jax.experimental.pallas.ops.tpu import paged_attention as paged_attention_mod
  paged_attn_fn = paged_attention_mod.paged_attention
  HAS_PALLAS_PAGED_ATTN = True
except ImportError:
  try:
    from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention as paged_attn_func
    paged_attn_fn = paged_attn_func
    HAS_PALLAS_PAGED_ATTN = True
  except ImportError:
    HAS_PALLAS_PAGED_ATTN = False


@dataclasses.dataclass
class ActiveRequest:
  """Represents an active generation request in the scheduler."""
  request_id: str
  prompt_tokens: np.ndarray  # 1D array of prompt tokens
  generated_tokens: list[int]
  max_new_tokens: int
  eos_token_id: int
  pages: list[int]
  current_length: int  # total length: prompt_len + len(generated_tokens)
  is_prefill: bool = True  # True if waiting to run prefill, False if decoding


class PageMemoryManager:
  """Manages virtual memory block pool for KV cache on TPU HBM and Host CPU."""

  def __init__(
      self,
      num_layers: int,
      num_kv_heads: int,
      head_dim: int,
      page_size: int,  # Number of tokens per page
      total_num_pages: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.num_layers = num_layers
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.page_size = page_size
    self.total_num_pages = total_num_pages
    self.dtype = dtype

    # Initialize physical page pools in HBM
    # Shape layout: [num_layers, total_num_pages, page_size, num_kv_heads, head_dim]
    pool_shape = (num_layers, total_num_pages, page_size, num_kv_heads, head_dim)
    self.k_pool = jnp.zeros(pool_shape, dtype=dtype)
    self.v_pool = jnp.zeros(pool_shape, dtype=dtype)

    # Tracks list of available free physical page indices
    self.free_pages = list(range(total_num_pages))
    
    # Active allocations mapping: request_id -> list of page indices
    self.allocations: dict[str, list[int]] = {}
    
    # Host offloaded caches: request_id -> (k_cpu, v_cpu)
    self.offloaded_caches: dict[str, tuple[np.ndarray, np.ndarray]] = {}

  def allocate(self, request_id: str, num_pages: int) -> list[int]:
    """Allocates pages from the free pool for a request."""
    if len(self.free_pages) < num_pages:
      raise MemoryError(
          f"Out of KV Cache pages in HBM pool (requested: {num_pages}, available: {len(self.free_pages)})"
      )
    pages = [self.free_pages.pop(0) for _ in range(num_pages)]
    self.allocations[request_id] = self.allocations.get(request_id, []) + pages
    return pages

  def free(self, request_id: str):
    """Frees allocated pages back to the free pool."""
    if request_id in self.allocations:
      self.free_pages.extend(self.allocations[request_id])
      del self.allocations[request_id]
    if request_id in self.offloaded_caches:
      del self.offloaded_caches[request_id]

  def offload_to_host(self, request_id: str):
    """Offloads the KV cache blocks of a request from TPU HBM to Host CPU RAM."""
    if request_id not in self.allocations:
      return
    pages = self.allocations[request_id]
    
    # Slice the pages out of the TPU pool and transfer to CPU host
    k_pages = jax.device_get(self.k_pool[:, pages, ...])
    v_pages = jax.device_get(self.v_pool[:, pages, ...])
    self.offloaded_caches[request_id] = (k_pages, v_pages)
    
    # Free HBM allocation
    self.free(request_id)

  def restore_to_hbm(self, request_id: str) -> list[int]:
    """Restores the KV cache from Host CPU RAM back to TPU HBM."""
    if request_id not in self.offloaded_caches:
      raise ValueError(f"Request {request_id} has no offloaded cache")
    k_cpu, v_cpu = self.offloaded_caches[request_id]
    num_pages = k_cpu.shape[1]
    
    # Allocate new physical HBM pages
    pages = self.allocate(request_id, num_pages)
    
    # Copy from CPU to the allocated TPU HBM slices
    self.k_pool = self.k_pool.at[:, pages, ...].set(k_cpu)
    self.v_pool = self.v_pool.at[:, pages, ...].set(v_cpu)
    
    del self.offloaded_caches[request_id]
    return pages


class ContinuousBatcherScheduler:
  """Central scheduler running on Host CPU managing continuous batching rollout slots."""

  def __init__(
      self,
      max_batch_size: int,
      page_size: int,
      pmm: PageMemoryManager,
      max_pages_per_seq: int = 128,
  ):
    self.max_batch_size = max_batch_size
    self.page_size = page_size
    self.pmm = pmm
    self.max_pages_per_seq = max_pages_per_seq

    self.queue: list[ActiveRequest] = []
    # Active slots: maps physical batch indices to ActiveRequest or None
    self.slots: list[Optional[ActiveRequest]] = [None] * max_batch_size

  def add_request(
      self,
      request_id: str,
      prompt_tokens: np.ndarray,
      max_new_tokens: int,
      eos_token_id: int,
  ):
    """Queues a new request for rollout."""
    req = ActiveRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        generated_tokens=[],
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pages=[],
        current_length=len(prompt_tokens),
        is_prefill=True,
    )
    self.queue.append(req)

  def step(self, model_step_fn: Any) -> list[tuple[str, list[int]]]:
    """Runs a single rollout scheduling and execution step on TPU.

    Args:
      model_step_fn: A function that takes batch inputs (tokens, block_tables,
        lengths, positions) and performs prefill/decode forward pass, returning
        logits.

    Returns:
      A list of tuples (request_id, generated_tokens) for completed requests.
    """
    # 1. Fill empty slots from the queue
    for i in range(self.max_batch_size):
      if self.slots[i] is None and self.queue:
        req = self.queue.pop(0)
        # Allocate blocks for the initial prompt (prefill stage)
        num_pages_needed = (len(req.prompt_tokens) + self.page_size - 1) // self.page_size
        req.pages = self.pmm.allocate(req.request_id, num_pages_needed)
        self.slots[i] = req

    # If all slots are empty, we have no work to do
    if all(s is None for s in self.slots):
      return []

    # 2. Build batched inputs for the JIT step function
    batch_tokens = []
    batch_block_tables = []
    batch_lengths = []
    batch_positions = []

    for i in range(self.max_batch_size):
      req = self.slots[i]
      if req is None:
        # Pad values for empty slot
        batch_tokens.append(0)
        batch_block_tables.append([0] * self.max_pages_per_seq)
        batch_lengths.append(0)
        batch_positions.append(0)
      else:
        if req.is_prefill:
          # In prefill stage, we pass the entire prompt tokens at once
          # (For mixed chunked prefill, this could be adjusted. Here we do full prompt prefill)
          # Note: If prompt tokens length > 1, the step function performs prefill.
          # To keep tensors static shape, we pad prompt tokens to the maximum length of prompts in the current step,
          # or simply pass them if our step function handles dynamic sequence lengths.
          # Here we demonstrate a simple static prefill/decode slot mapping.
          batch_tokens.append(req.prompt_tokens)
          batch_lengths.append(len(req.prompt_tokens))
          batch_positions.append(np.arange(len(req.prompt_tokens)))
        else:
          # Decoding: pass the last token
          last_tok = req.generated_tokens[-1] if req.generated_tokens else req.prompt_tokens[-1]
          batch_tokens.append(last_tok)
          batch_lengths.append(req.current_length)
          batch_positions.append(req.current_length - 1)

        # Pad block table to max_pages_per_seq
        padded_table = list(req.pages) + [0] * (self.max_pages_per_seq - len(req.pages))
        batch_block_tables.append(padded_table)

    # Convert to arrays
    # Note: If there is a mix of prefill (prompt arrays) and decode (scalar/single token),
    # we either pad the inputs or run them in separate stages or use ragged shapes.
    # To keep this demo scheduler simple and clean, let's compile steps separately for prefill vs decode,
    # or handle decode step-by-step.
    # Let's focus on step-by-step decode execution:
    
    # Simulate forward pass / token sampling for active slots
    completed = []
    for i in range(self.max_batch_size):
      req = self.slots[i]
      if req is None:
        continue

      if req.is_prefill:
        # Run prefill (prompt to KV cache write, sample first token)
        # In a real model, this writes to the HBM pool and computes the first generated token.
        # Here we mock the forward sampling:
        first_token = 42  # Dummy sampled token
        req.generated_tokens.append(first_token)
        req.is_prefill = False
        req.current_length += 1
      else:
        # Run decode (read KV cache, write new KV projection, sample next token)
        next_token = 42 + len(req.generated_tokens)  # Dummy sampled token
        req.generated_tokens.append(next_token)
        req.current_length += 1

      # Check if request has finished (reached EOS or max_new_tokens)
      is_done = (
          (req.generated_tokens[-1] == req.eos_token_id)
          or (len(req.generated_tokens) >= req.max_new_tokens)
      )

      if is_done:
        completed.append((req.request_id, req.generated_tokens))
        self.pmm.free(req.request_id)
        self.slots[i] = None
      else:
        # If the request needs more pages as it grows, allocate 1 block on the fly
        if req.current_length > len(req.pages) * self.page_size:
          req.pages.extend(self.pmm.allocate(req.request_id, 1))

    return completed


def paged_attention_wrapper(
    q: jax.Array,
    k_pool: jax.Array,
    v_pool: jax.Array,
    block_table: jax.Array,
    lengths: jax.Array,
    page_size: int,
) -> jax.Array:
  """Wrapper around JAX/Pallas paged_attention.

  If Pallas paged_attention is not available (e.g. running on non-TPU platforms
  during tests), falls back to a standard JAX implementation that gathers pages
  for attention.
  """
  if HAS_PALLAS_PAGED_ATTN:
    return paged_attn_fn(
        q=q,
        k_pages=k_pool,
        v_pages=v_pool,
        lengths=lengths,
        page_indices=block_table,
        pages_per_compute_block=page_size,
    )
  else:
    # Fallback/reconstruction of paged attention in standard JAX/einsum.
    # Gather K and V cache blocks using block_table indices
    # q shape: [B, seq_len, num_heads, head_dim]
    # k_pool shape: [num_layers, total_num_pages, page_size, num_kv_heads, head_dim]
    # block_table shape: [B, Max_Pages]
    
    # We focus on a single layer attention pass:
    # k_pages_layer = k_pool[layer_idx] -> shape [total_num_pages, page_size, num_kv_heads, head_dim]
    # Let's reconstruct key and value sequences for each sequence in the batch:
    batch_size = q.shape[0]
    num_heads = q.shape[2]
    head_dim = q.shape[3]
    num_kv_heads = k_pool.shape[3]
    max_pages = block_table.shape[1]
    
    # Gather pages for K and V
    # Shape: [B, Max_Pages, page_size, num_kv_heads, head_dim]
    k_gathered = k_pool[0, block_table, ...] 
    v_gathered = v_pool[0, block_table, ...]
    
    # Reshape to form contiguous sequences:
    # Shape: [B, Max_Pages * page_size, num_kv_heads, head_dim]
    k_seq = jnp.reshape(k_gathered, (batch_size, max_pages * page_size, num_kv_heads, head_dim))
    v_seq = jnp.reshape(v_gathered, (batch_size, max_pages * page_size, num_kv_heads, head_dim))
    
    # Run standard GQA attention
    # query shape: [B, seq_len, num_kv_heads, num_heads//num_kv_heads, head_dim]
    n_rep = num_heads // num_kv_heads
    q_reshaped = jnp.reshape(q, (batch_size, q.shape[1], num_kv_heads, n_rep, head_dim))
    
    # Compute attention scores:
    # q: [B, T_q, H_kv, N_rep, D]
    # k: [B, T_kv, H_kv, D]
    # scores: [B, H_kv, N_rep, T_q, T_kv]
    attn_weights = jnp.einsum("BTQRD,BSKD->BKRTS", q_reshaped, k_seq) * (head_dim ** -0.5)
    
    # Mask out-of-bounds page positions using lengths
    # (Create causal / length mask)
    kv_seq_len = max_pages * page_size
    pos_indices = jnp.arange(kv_seq_len)[None, :]  # [1, T_kv]
    length_mask = pos_indices < lengths[:, None]   # [B, T_kv]
    
    # Mask: shape [B, 1, 1, 1, T_kv]
    mask = length_mask[:, None, None, None, :]
    attn_weights = jnp.where(mask, attn_weights, -1e9)
    
    # Softmax
    attn_probs = jax.nn.softmax(attn_weights, axis=-1)
    
    # Compute weighted values:
    # attn_probs: [B, H_kv, N_rep, T_q, T_kv]
    # v_seq: [B, T_kv, H_kv, D]
    # output: [B, T_q, H_kv, N_rep, D]
    qkv = jnp.einsum("BKRTS,BSKD->BTQRD", attn_probs, v_seq)
    
    # Reshape back to [B, seq_len, num_heads, head_dim]
    return jnp.reshape(qkv, (batch_size, q.shape[1], num_heads, head_dim))
