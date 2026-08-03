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
import time
from typing import Any, Dict, List, Sequence, Tuple, Union

import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import jaxtyping
import numpy as np
from tunix.generate import beam_search as beam_search_lib
from tunix.generate import sampler as sampler_lib
from tunix.generate import utils


class KVStatus(enum.IntEnum):
  """Status of KV values stored for each sequence slot."""
  NONE = 0    # Prompt tokens only; no KV layers written
  PREFIX = 1  # Shared prompt prefix KV layers loaded from prefix_cache
  FULL = 2    # Full KV layers computed and written


@dataclasses.dataclass
class SamplingConfig:
  max_num_sequences: int
  max_generation_steps: int
  max_prompt_length: int = 1024
  page_size: int = 16
  dtype: jnp.dtype = jnp.bfloat16
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
  group_size: int | None = None
  max_audio_length: int | None = None
  max_audio_clips: int | None = None
  attn_logits_soft_cap: float | None = None


@dataclasses.dataclass
class RequestOutput:
  request_id: str
  text: str
  tokens: List[int]
  logits: np.ndarray | None = None
  logprobs: np.ndarray | None = None


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


@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state for Continuous Batching."""
  # Decoding steps: ith entry contains the decoding step of the ith HBM sequence
  decoding_steps: jnp.ndarray  # i32[max_num_sequences]
  # (i, j, k) represents that sequences[0:i] are decode-only,
  # sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed.
  distribution: jnp.ndarray  # i32[3]
  # Sequence Lengths on TPU HBM
  sequence_lengths: jnp.ndarray  # i32[max_num_sequences]
  # Sharded TPU HBM cache storing tokens and KV values for active sequences on TPU
  hbm_cache: sampler_lib.PageManager
  # CPU cache storing tokens and offloaded KV values for pending/preempted sequences
  offloaded_cache: sampler_lib.PageManager
  # CPU prefix cache for GRPO group prompts (max_seq_len == max_prompt_length)
  prefix_cache: sampler_lib.PageManager
  # Is decoding done on the given sequence?
  done: jnp.ndarray  # bool[max_num_sequences]
  # Static array mapping seq_idx -> monotonic insertion timestamp for fairness ordering
  insertion_timestamps: jnp.ndarray  # f64[max_num_sequences]
  # Status of KV layers for each sequence slot: NONE (0), PREFIX (1), or FULL (2)
  kv_status: jnp.ndarray  # i32[max_num_sequences]
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
  hbm_request_ids: list[str] = flax.struct.field(
      default_factory=list, pytree_node=False
  )
  cpu_request_ids: list[str] = flax.struct.field(
      default_factory=list, pytree_node=False
  )
  group_ids: list[str | None] = flax.struct.field(
      default_factory=list, pytree_node=False
  )
  group_member_count: Dict[str, int] = flax.struct.field(
      default_factory=dict, pytree_node=False
  )
  group_prefix_slots: Dict[str, int] = flax.struct.field(
      default_factory=dict, pytree_node=False
  )

  group_size: int | None = flax.struct.field(
      default=None, pytree_node=False
  )
  sampling_mode: str = flax.struct.field(
      default="greedy", pytree_node=False
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


class VanillaSampler:
  """Continuous Batching Sampler with three-cache architecture and static metadata arrays."""

  def __init__(
      self,
      model: nnx.Module,
      tokenizer: Any,
      cache_config: sampler_lib.CacheConfig,
  ):
    self.model_module = model
    self.tokenizer = tokenizer
    self.cache_config = cache_config
    self._jit_step = jax.jit(self._model_step_fn)

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
  ) -> None:
    """Update underlying NNX model weights in-place with synchronization barrier."""
    jax.effects_barrier()
    nnx.update(self.model_module, updated_weights)
    jax.effects_barrier()

  def _init_page_manager(
      self,
      max_seqs: int,
      max_num_pages_per_seq: int,
      page_size: int,
      max_seq_len: int,
      dp_axis: str | None = None,
      tp_axis: str | None = None,
      device: Any = None,
  ) -> sampler_lib.PageManager:
    """Explicitly initializes physical page tensors for a PageManager pool, placing CPU caches on host memory."""
    blocks: dict[str, jax.Array] = {}

    token_block = jax.lax.empty(
        (self.cache_config.cache_size, page_size), dtype=jnp.int32
    )
    if dp_axis is not None:
      token_block = sampler_lib.shard(token_block, (dp_axis, None))
    if device is not None:
      token_block = jax.device_put(token_block, device)
    blocks["token_buffer"] = token_block

    layer_dtype = getattr(self.cache_config, "dtype", jnp.bfloat16)
    for i in range(self.cache_config.num_layers):
      layer_block = jax.lax.empty(
          (
              self.cache_config.cache_size,
              page_size,
              self.cache_config.num_kv_heads,
              self.cache_config.head_dim,
          ),
          dtype=layer_dtype,
      )
      if dp_axis is not None or tp_axis is not None:
        layer_block = sampler_lib.shard(layer_block, (dp_axis, None, tp_axis, None, None))
      if device is not None:
        layer_block = jax.device_put(layer_block, device)
      blocks[f"layer_{i}"] = layer_block

    page_indices = jnp.zeros((max_seqs, max_num_pages_per_seq), dtype=jnp.int32)
    available_page_indices = jnp.arange(self.cache_config.cache_size, dtype=jnp.int32)
    num_available_pages = jnp.array(self.cache_config.cache_size, dtype=jnp.int32)
    seq_lens = jnp.zeros((max_seqs,), dtype=jnp.int32)

    if device is not None:
      page_indices = jax.device_put(page_indices, device)
      available_page_indices = jax.device_put(available_page_indices, device)
      num_available_pages = jax.device_put(num_available_pages, device)
      seq_lens = jax.device_put(seq_lens, device)

    return sampler_lib.PageManager(
        pages=blocks,
        page_indices=page_indices,
        available_page_indices=available_page_indices,
        num_available_pages=num_available_pages,
        seq_lens=seq_lens,
        page_size=page_size,
        max_seq_len=max_seq_len,
    )

  def init_sample_state(
      self,
      sampling_config: SamplingConfig,
  ) -> _SamplingState:
    """Initialize sampling state with HBM, Offloaded, and Prefix cache pools."""
    max_seqs = sampling_config.max_num_sequences
    page_size = sampling_config.page_size
    max_seq_len = sampling_config.max_generation_steps + sampling_config.max_prompt_length
    max_num_pages_per_seq = sampler_lib.cdiv(max_seq_len, page_size)
    prefix_pages_per_seq = sampler_lib.cdiv(sampling_config.max_prompt_length, page_size)

    shd_config = getattr(getattr(self.model_module, "config", None), "shd_config", None)
    if shd_config is not None:
      dp_axis = shd_config.act_btd[0]
      tp_axis = shd_config.act_btnh[2]
    else:
      dp_axis = None
      tp_axis = None

    cpu_device = jax.devices("cpu")[0] if jax.devices("cpu") else None

    hbm_cache = self._init_page_manager(
        max_seqs=max_seqs,
        max_num_pages_per_seq=max_num_pages_per_seq,
        page_size=page_size,
        max_seq_len=max_seq_len,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        device=None,
    )
    offloaded_cache = self._init_page_manager(
        max_seqs=max_seqs,
        max_num_pages_per_seq=max_num_pages_per_seq,
        page_size=page_size,
        max_seq_len=max_seq_len,
        dp_axis=None,
        tp_axis=None,
        device=cpu_device,
    )
    prefix_cache = self._init_page_manager(
        max_seqs=max_seqs,
        max_num_pages_per_seq=prefix_pages_per_seq,
        page_size=page_size,
        max_seq_len=sampling_config.max_prompt_length,
        dp_axis=None,
        tp_axis=None,
        device=cpu_device,
    )

    eos_ids = tuple(sampling_config.eos_tokens) if sampling_config.eos_tokens is not None else None

    return _SamplingState(
        decoding_steps=jnp.zeros((max_seqs,), dtype=jnp.int32),
        distribution=jnp.array([0, 0, 0], dtype=jnp.int32),
        sequence_lengths=jnp.zeros((max_seqs,), dtype=jnp.int32),
        hbm_cache=hbm_cache,
        offloaded_cache=offloaded_cache,
        prefix_cache=prefix_cache,
        done=jnp.zeros((max_seqs,), dtype=jnp.bool_),
        insertion_timestamps=jnp.zeros((max_seqs,), dtype=jnp.float64),
        kv_status=jnp.zeros((max_seqs,), dtype=jnp.int32),
        logits_buffer=None,
        logprobs_buffer=None,
        forbidden_token_ids=sampling_config.forbidden_tokens,
        eos_token_ids=eos_ids,
        seed=jax.random.PRNGKey(sampling_config.seed or 0),
        hbm_request_ids=[],
        cpu_request_ids=[],
        group_ids=[None] * max_seqs,
        group_member_count={},
        group_prefix_slots={},
        group_size=sampling_config.group_size,
        sampling_mode="top_p" if sampling_config.top_p else "greedy",
        temperature=sampling_config.temperature,
        sampling_parameters={"top_p": sampling_config.top_p or 1.0},
        attn_logits_soft_cap=sampling_config.attn_logits_soft_cap,
    )

  def _release_slots(
      self,
      pm: sampler_lib.PageManager,
      should_release: jax.Array,
  ) -> sampler_lib.PageManager:
    """Unified release helper function across CPU, prefix, and TPU HBM PageManagers."""
    return pm.release(should_release)

  def _remove_request_from_pool(
      self,
      sampling_state: _SamplingState,
      request_id: str,
      request_ids: list[str],
      cache: sampler_lib.PageManager,
  ) -> tuple[_SamplingState, sampler_lib.PageManager, bool]:
    """Removes request_id from a pool, decrements group refcounts, and releases allocated pages."""
    if request_id not in request_ids:
      return sampling_state, cache, False

    slot = request_ids.index(request_id)
    request_ids.pop(slot)
    group_id = sampling_state.group_ids[slot]
    sampling_state.group_ids[slot] = None
    sampling_state = dataclasses.replace(
        sampling_state,
        kv_status=sampling_state.kv_status.at[slot].set(KVStatus.NONE),
    )

    prefix_cache = sampling_state.prefix_cache
    if group_id is not None and group_id in sampling_state.group_member_count:
      sampling_state.group_member_count[group_id] -= 1
      if sampling_state.group_member_count[group_id] <= 0:
        p_slot = sampling_state.group_prefix_slots.pop(group_id, None)
        if p_slot is not None:
          should_rel_p = jnp.zeros((int(prefix_cache.batch_size),), dtype=jnp.bool_).at[p_slot].set(True)
          prefix_cache = self._release_slots(prefix_cache, should_rel_p)

    should_release = jnp.zeros((int(cache.batch_size),), dtype=jnp.bool_).at[slot].set(True)
    updated_cache = self._release_slots(cache, should_release)
    sampling_state = dataclasses.replace(sampling_state, prefix_cache=prefix_cache)
    return sampling_state, updated_cache, True

  def cancel_request(
      self,
      sampling_state: _SamplingState,
      request_id: str,
  ) -> _SamplingState:
    """Cancels a request, decrementing group refcounts and releasing slots."""
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

  def _batch_transfer_pages(
      self,
      src_cache: sampler_lib.PageManager,
      dst_cache: sampler_lib.PageManager,
      src_slots: Sequence[int],
      dst_slots: Sequence[int],
      transfer_kv: bool = True,
  ) -> sampler_lib.PageManager:
    """Transfers pages across specified slots, optionally copying only token_buffer when transfer_kv is False."""
    if not src_slots:
      return dst_cache

    src_idxs = src_cache.page_indices[jnp.array(src_slots)].reshape(-1)
    dst_idxs = dst_cache.page_indices[jnp.array(dst_slots)].reshape(-1)

    if not transfer_kv:
      src_tensor = src_cache.pages["token_buffer"]
      dst_tensor = dst_cache.pages.get("token_buffer", jnp.zeros_like(src_tensor))
      dst_cache.pages["token_buffer"] = dst_tensor.at[dst_idxs].set(src_tensor[src_idxs])
      return dst_cache

    for key, src_tensor in src_cache.pages.items():
      dst_tensor = dst_cache.pages.get(key, jnp.zeros_like(src_tensor))
      dst_cache.pages[key] = dst_tensor.at[dst_idxs].set(src_tensor[src_idxs])
    return dst_cache

  def _queue_new_requests(
      self,
      sampling_state: _SamplingState,
      new_requests: Sequence[dict[str, Any]],
  ) -> _SamplingState:
    """Load incoming requests into offloaded_cache, initializing kv_status unconditionally to NONE."""
    if not new_requests:
      return sampling_state

    cpu_req_ids = list(sampling_state.cpu_request_ids)
    offloaded_cache = sampling_state.offloaded_cache
    num_tpu = len(sampling_state.hbm_request_ids)
    num_cpu = len(cpu_req_ids)
    max_seqs = int(offloaded_cache.batch_size)

    if num_tpu + num_cpu + len(new_requests) > max_seqs:
      raise ValueError(
          f"Cannot queue {len(new_requests)} new requests: total active "
          f"sequences ({num_tpu + num_cpu + len(new_requests)}) exceeds "
          f"max_num_sequences ({max_seqs})."
      )

    insertion_timestamps = sampling_state.insertion_timestamps
    kv_status = sampling_state.kv_status
    group_ids = list(sampling_state.group_ids)

    for req_dict in new_requests:
      req_id = req_dict["id"]
      prompt_str = req_dict["prompt"]
      prompt_tokens = self.tokenizer.encode(prompt_str)
      prompt_len = len(prompt_tokens)

      slot_idx = len(cpu_req_ids)
      cpu_req_ids.append(req_id)
      insertion_timestamps = insertion_timestamps.at[slot_idx].set(time.perf_counter())

      group_id = req_dict.get("group_id")
      group_ids[slot_idx] = group_id
      if (
          sampling_state.group_size is not None
          and group_id is not None
          and group_id not in sampling_state.group_member_count
      ):
        sampling_state.group_member_count[group_id] = sampling_state.group_size

      q_lens = jnp.zeros((max_seqs,), dtype=jnp.int32).at[slot_idx].set(prompt_len)
      offloaded_cache = offloaded_cache.allocate(q_lens=q_lens)

      tokens_arr = jnp.array(prompt_tokens, dtype=jnp.int32)
      lens_arr = jnp.zeros((max_seqs,), dtype=jnp.int32).at[slot_idx].set(prompt_len)
      offloaded_cache = offloaded_cache.load_prompt_tokens(tokens_arr, lens=lens_arr, key="token_buffer")
      kv_status = kv_status.at[slot_idx].set(KVStatus.NONE)

    return dataclasses.replace(
        sampling_state,
        offloaded_cache=offloaded_cache,
        cpu_request_ids=cpu_req_ids,
        insertion_timestamps=insertion_timestamps,
        kv_status=kv_status,
        group_ids=group_ids,
    )

  def _make_room_for_allocation(self, sampling_state: _SamplingState) -> _SamplingState:
    """Evicts newest active HBM sequence to offloaded_cache when HBM is constrained.
    
    Since eviction occurs before queue draining, any sequence in hbm_cache has already
    completed at least one sampling step and thus unconditionally transitions to KVStatus.FULL.
    """
    hbm_available = int(sampling_state.hbm_cache.num_available_pages)
    num_tpu = len(sampling_state.hbm_request_ids)
    num_cpu = len(sampling_state.cpu_request_ids)

    evict_hbm_slots: list[int] = []
    evict_cpu_slots: list[int] = []

    hbm_request_ids = list(sampling_state.hbm_request_ids)
    cpu_request_ids = list(sampling_state.cpu_request_ids)
    group_ids = list(sampling_state.group_ids)
    offloaded_cache = sampling_state.offloaded_cache
    kv_status = sampling_state.kv_status

    while hbm_available < num_tpu:
      evict_idx = num_tpu - 1 - len(evict_hbm_slots)
      req_id = hbm_request_ids[evict_idx]
      seq_len = int(sampling_state.hbm_cache.seq_lens[evict_idx])
      pages_needed = sampler_lib.cdiv(seq_len, offloaded_cache.page_size)

      if int(offloaded_cache.num_available_pages) < pages_needed:
        raise RuntimeError("CPU Swap space is too small to evict request.")

      cpu_slot = num_cpu + len(evict_cpu_slots)
      cpu_request_ids.append(req_id)
      group_ids[cpu_slot] = group_ids[evict_idx]
      kv_status = kv_status.at[cpu_slot].set(KVStatus.FULL)

      q_lens = jnp.zeros((int(offloaded_cache.batch_size),), dtype=jnp.int32).at[cpu_slot].set(seq_len)
      offloaded_cache = offloaded_cache.allocate(q_lens=q_lens)

      evict_hbm_slots.append(evict_idx)
      evict_cpu_slots.append(cpu_slot)
      hbm_available += int(pages_needed)

    if evict_hbm_slots:
      offloaded_cache = self._batch_transfer_pages(
          sampling_state.hbm_cache,
          offloaded_cache,
          evict_hbm_slots,
          evict_cpu_slots,
          transfer_kv=True,
      )
      should_release = jnp.zeros((int(sampling_state.hbm_cache.batch_size),), dtype=jnp.bool_)
      for slot in evict_hbm_slots:
        should_release = should_release.at[slot].set(True)
      updated_hbm_cache = self._release_slots(sampling_state.hbm_cache, should_release)

      for _ in evict_hbm_slots:
        hbm_request_ids.pop()

      return dataclasses.replace(
          sampling_state,
          hbm_cache=updated_hbm_cache,
          offloaded_cache=offloaded_cache,
          hbm_request_ids=hbm_request_ids,
          cpu_request_ids=cpu_request_ids,
          group_ids=group_ids,
          kv_status=kv_status,
      )

    return sampling_state

  def _drain_pending_queue(self, sampling_state: _SamplingState) -> _SamplingState:
    """Admit sequences from offloaded_cache to HBM cache with selective KV transfers based on kv_status."""
    hbm_req_ids = list(sampling_state.hbm_request_ids)
    cpu_req_ids = list(sampling_state.cpu_request_ids)

    sorted_cpu_slots = sorted(
        range(len(cpu_req_ids)),
        key=lambda s: float(sampling_state.insertion_timestamps[s]),
    )

    decode_src_slots: list[int] = []
    prefill_src_slots: list[int] = []
    mixed_src_slots: list[int] = []

    num_existing_decode = len(hbm_req_ids)
    max_seqs = int(sampling_state.hbm_cache.batch_size)
    kv_status = sampling_state.kv_status
    offloaded_cache = sampling_state.offloaded_cache

    for cpu_slot in sorted_cpu_slots:
      group_id = sampling_state.group_ids[cpu_slot]
      status = int(kv_status[cpu_slot])

      # Check again if a group prefix became available while queued in offloaded_cache
      if (
          status == KVStatus.NONE
          and group_id is not None
          and group_id in sampling_state.group_prefix_slots
      ):
        p_slot = sampling_state.group_prefix_slots[group_id]
        offloaded_cache = self._batch_transfer_pages(
            sampling_state.prefix_cache,
            offloaded_cache,
            [p_slot],
            [cpu_slot],
            transfer_kv=True,
        )
        status = KVStatus.PREFIX
        kv_status = kv_status.at[cpu_slot].set(KVStatus.PREFIX)

      if status == KVStatus.FULL:
        decode_src_slots.append(cpu_slot)
      elif status == KVStatus.PREFIX:
        mixed_src_slots.append(cpu_slot)
      else:
        prefill_src_slots.append(cpu_slot)

    i_val = num_existing_decode + len(decode_src_slots)
    j_val = i_val + len(prefill_src_slots)
    k_val = j_val + len(mixed_src_slots)

    all_src_slots = decode_src_slots + prefill_src_slots + mixed_src_slots
    all_dst_slots = list(range(num_existing_decode, k_val))

    q_lens = jnp.zeros((max_seqs,), dtype=jnp.int32)
    for idx in range(num_existing_decode):
      q_lens = q_lens.at[idx].set(1)

    sequence_lengths = sampling_state.sequence_lengths
    for src_slot, dst_slot in zip(all_src_slots, all_dst_slots):
      req_id = cpu_req_ids[src_slot]
      seq_len = int(offloaded_cache.seq_lens[src_slot])
      hbm_req_ids.append(req_id)
      q_lens = q_lens.at[dst_slot].set(seq_len)
      sequence_lengths = sequence_lengths.at[dst_slot].set(seq_len)

    updated_hbm_cache = sampling_state.hbm_cache.allocate(q_lens=q_lens)

    # Decode sequences have full KV values -> transfer_kv=True
    updated_hbm_cache = self._batch_transfer_pages(
        offloaded_cache,
        updated_hbm_cache,
        decode_src_slots,
        list(range(num_existing_decode, i_val)),
        transfer_kv=True,
    )
    # Prefill sequences have NO KV values written -> transfer_kv=False (skips empty layer tensors)
    updated_hbm_cache = self._batch_transfer_pages(
        offloaded_cache,
        updated_hbm_cache,
        prefill_src_slots,
        list(range(i_val, j_val)),
        transfer_kv=False,
    )
    # Mixed sequences have prefix KV values written -> transfer_kv=True
    updated_hbm_cache = self._batch_transfer_pages(
        offloaded_cache,
        updated_hbm_cache,
        mixed_src_slots,
        list(range(j_val, k_val)),
        transfer_kv=True,
    )

    should_release_cpu = jnp.zeros((max_seqs,), dtype=jnp.bool_)
    for slot in all_src_slots:
      should_release_cpu = should_release_cpu.at[slot].set(True)

    admitted_set = set(all_src_slots)
    cpu_req_ids = [rid for i, rid in enumerate(cpu_req_ids) if i not in admitted_set]

    dist = jnp.array([i_val, j_val, k_val], dtype=jnp.int32)
    return dataclasses.replace(
        sampling_state,
        distribution=dist,
        sequence_lengths=sequence_lengths,
        kv_status=kv_status,
        hbm_cache=updated_hbm_cache,
        offloaded_cache=self._release_slots(offloaded_cache, should_release_cpu),
        hbm_request_ids=hbm_req_ids,
        cpu_request_ids=cpu_req_ids,
    )

  def _compact_batch(self, sampling_state: _SamplingState) -> _SamplingState:
    """Compact continuing sequences into contiguous HBM slots and permute static metadata arrays."""
    num_tpu = len(sampling_state.hbm_request_ids)
    active_mask = ~sampling_state.done & (
        jnp.arange(sampling_state.hbm_cache.batch_size) < num_tpu
    )
    num_remaining = int(jnp.sum(active_mask))
    slot_perm = jnp.argsort(~active_mask)

    compacted_hbm_cache = dataclasses.replace(
        sampling_state.hbm_cache,
        page_indices=sampling_state.hbm_cache.page_indices[slot_perm],
        seq_lens=sampling_state.hbm_cache.seq_lens[slot_perm],
    )
    reordered_ids = [sampling_state.hbm_request_ids[int(i)] for i in jax.device_get(slot_perm)[:num_remaining]]
    reordered_group_ids = [sampling_state.group_ids[int(i)] for i in jax.device_get(slot_perm)]

    return dataclasses.replace(
        sampling_state,
        decoding_steps=sampling_state.decoding_steps[slot_perm],
        sequence_lengths=sampling_state.sequence_lengths[slot_perm],
        insertion_timestamps=sampling_state.insertion_timestamps[slot_perm],
        kv_status=sampling_state.kv_status[slot_perm],
        done=jnp.zeros_like(sampling_state.done),
        hbm_cache=compacted_hbm_cache,
        hbm_request_ids=reordered_ids,
        group_ids=reordered_group_ids,
    )

  def _model_step_fn(
      self,
      cache: sampler_lib.PageManager,
      last_tokens: jnp.ndarray,
      positions: jnp.ndarray,
      distribution: jnp.ndarray,
      soft_cap: float | None = None,
  ) -> Tuple[jnp.ndarray, sampler_lib.PageManager]:
    """JIT-compiled forward pass invoking Gemma with ragged paged attention and explicit soft_cap."""
    logits, new_cache = self.model_module(
        last_tokens=last_tokens,
        positions=positions,
        cache=cache,
        distribution=distribution,
        soft_cap=soft_cap,
    )
    return logits, new_cache

  def _release_completed(
      self,
      sampling_state: _SamplingState,
      next_tokens: jax.Array,
  ) -> tuple[_SamplingState, dict[str, RequestOutput]]:
    """Inspects generated tokens, appends EOS at max_seq_len - 1, formats outputs, and releases completed slots."""
    completed_outputs: dict[str, RequestOutput] = {}
    should_release_hbm = jnp.zeros((int(sampling_state.hbm_cache.batch_size),), dtype=jnp.bool_)
    should_release_prefix = jnp.zeros((int(sampling_state.prefix_cache.batch_size),), dtype=jnp.bool_)

    for idx, req_id in enumerate(sampling_state.hbm_request_ids):
      sampled_token = int(next_tokens[idx])
      eos_ids = sampling_state.eos_token_ids or (1,)

      if int(sampling_state.decoding_steps[idx]) == 0:
        group_id = sampling_state.group_ids[idx]
        if group_id is not None and group_id in sampling_state.group_member_count:
          sampling_state.group_member_count[group_id] -= 1
          if sampling_state.group_member_count[group_id] <= 0:
            p_slot = sampling_state.group_prefix_slots.pop(group_id, None)
            if p_slot is not None:
              should_release_prefix = should_release_prefix.at[p_slot].set(True)

      is_done = (
          (sampled_token in eos_ids)
          or (int(sampling_state.decoding_steps[idx]) >= sampling_state.hbm_cache.max_seq_len - 1)
      )
      if is_done:
        if sampled_token not in eos_ids:
          sampled_token = eos_ids[0]
        completed_outputs[req_id] = RequestOutput(
            request_id=req_id,
            text=self.tokenizer.decode([sampled_token]),
            tokens=[sampled_token],
        )
        should_release_hbm = should_release_hbm.at[idx].set(True)

    updated_hbm_cache = self._release_slots(sampling_state.hbm_cache, should_release_hbm)
    updated_prefix_cache = self._release_slots(sampling_state.prefix_cache, should_release_prefix)

    sampling_state = dataclasses.replace(
        sampling_state,
        hbm_cache=updated_hbm_cache,
        prefix_cache=updated_prefix_cache,
        done=should_release_hbm,
    )
    return self._compact_batch(sampling_state), completed_outputs

  def _sample_step(
      self,
      sampling_state: _SamplingState,
      requests: Sequence[dict[str, Any]] = (),
  ) -> dict[str, RequestOutput]:
    """Complete a single sampling step, store prefix KV values if needed, and return completions."""
    sampling_state = self._queue_new_requests(sampling_state, requests)
    sampling_state = self._make_room_for_allocation(sampling_state)
    sampling_state = self._drain_pending_queue(sampling_state)

    num_tpu = len(sampling_state.hbm_request_ids)
    if num_tpu == 0:
      return {}

    last_tokens = jnp.zeros((int(sampling_state.hbm_cache.batch_size), 1), dtype=jnp.int32)
    positions = jnp.expand_dims(sampling_state.decoding_steps, axis=-1)

    logits, updated_hbm_cache = self._jit_step(
        cache=sampling_state.hbm_cache,
        last_tokens=last_tokens,
        positions=positions,
        distribution=sampling_state.distribution,
        soft_cap=sampling_state.attn_logits_soft_cap,
    )

    key, subkey = jax.random.split(sampling_state.seed)
    next_tokens, _ = sampler_lib.sample_top_p(
        logits=logits,
        key=subkey,
        temperature=sampling_state.temperature,
        top_p=float(sampling_state.sampling_parameters.get("top_p", 1.0)),
        top_k=None,
    )

    # Immediately copy prefix KV layers into prefix_cache for any group member finishing prefill step 0
    prefix_cache = sampling_state.prefix_cache
    for idx, _ in enumerate(sampling_state.hbm_request_ids):
      if int(sampling_state.decoding_steps[idx]) == 0:
        group_id = sampling_state.group_ids[idx]
        if group_id is not None and group_id not in sampling_state.group_prefix_slots:
          p_slot = len(sampling_state.group_prefix_slots)
          sampling_state.group_prefix_slots[group_id] = p_slot
          prefix_cache = self._batch_transfer_pages(
              updated_hbm_cache,
              prefix_cache,
              [idx],
              [p_slot],
              transfer_kv=True,
          )

    sampling_state = dataclasses.replace(
        sampling_state,
        hbm_cache=updated_hbm_cache,
        prefix_cache=prefix_cache,
        seed=key,
        decoding_steps=sampling_state.decoding_steps + 1,
    )

    sampling_state, completed_outputs = self._release_completed(sampling_state, next_tokens)
    return completed_outputs

  def __call__(
      self,
      sampling_state: _SamplingState,
      requests: Sequence[dict[str, Any]] = (),
  ) -> dict[str, RequestOutput]:
    """Forward call to _sample_step."""
    return self._sample_step(sampling_state, requests)
