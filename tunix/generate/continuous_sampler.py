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
from tunix.generate import beam_search as beam_search_lib
from tunix.generate import sampler as sampler_lib
from tunix.generate import page_manager as page_manager_lib 
from tunix.generate import utils


@dataclasses.dataclass
class SamplingConfig:
  max_num_sequences: int
  max_generation_steps: int
  max_prompt_length: int = 200 
  num_cpu_pages: int = 4096 * 32
  num_hbm_pages: int = 256 * 32
  page_size: int = 8 
  dtype: jnp.dtype = jnp.bfloat16
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True)
class _SamplingState:
  """Internal sampling state for Continuous Batching."""
  # Decoding steps: ith entry contains the decoding step of the ith HBM sequence
  decoding_steps: jnp.ndarray  # i32[max_num_sequences]
  offloaded_decoding_steps: jnp.ndarray  # i32[max_num_sequences]
  # (i, j, k) represents that sequences[0:i] are decode-only,
  # sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed.
  distribution: jnp.ndarray  # i32[3]
  # Sharded TPU HBM cache storing tokens and KV values for active sequences on TPU
  hbm_cache: page_manager_lib.PageManager
  # CPU cache storing tokens and offloaded KV values for pending/preempted sequences
  offloaded_cache: page_manager_lib.PageManager
  # Is decoding done on the given sequence?
  done: jnp.ndarray  # bool[max_num_sequences]
  # Static array mapping seq_idx -> monotonic insertion timestamp for fairness ordering
  insertion_timestamps: jnp.ndarray  # f64[max_num_sequences]
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


class VanillaSampler:
  """Continuous Batching Sampler with three-cache architecture and static metadata arrays."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: sampler_lib.CacheConfig,
  ):
    self.tokenizer = tokenizer
    self.cache_config = cache_config

    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)  # pyrefly: ignore[bad-assignment]
    self._transformer_state: list[statelib.State] = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    
    self._compiled_step_fn = jax.jit(self._model_step_fn, static_argnames=("echo", "max_prompt_length"))
    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
  ) -> None:
    """Update underlying NNX model weights in-place with synchronization barrier."""
    pass

  def _init_page_manager(
      self,
      max_seqs: int,
      page_size: int,
      max_seq_len: int,
      num_pages: int,
      dp_axis: str | None = None,
      tp_axis: str | None = None,
      device: Any = None,
  ) -> page_manager_lib.PageManager:
    """Explicitly initializes physical page tensors for a PageManager pool, placing CPU caches on host memory."""
    blocks: dict[str, jax.Array] = {}
    max_num_pages_per_seq = int(max_seq_len / page_size)

    token_block = jax.lax.empty(
        (num_pages, page_size), dtype=jnp.int32
    )
    if dp_axis is not None:
      token_block = sampler_lib.shard(token_block, (dp_axis, None))
    if device is not None:
      token_block = jax.device_put(token_block, device)
    blocks["token_buffer"] = token_block

    # layer_dtype = getattr(self.cache_config, "dtype", jnp.bfloat16)
    for i in range(self.cache_config.num_layers):
      layer_block = jax.lax.empty(
          (
              num_pages,
              page_size,
              2 * self.cache_config.num_kv_heads // self.cache_config.kv_packing,
              self.cache_config.kv_packing,
              self.cache_config.head_dim,
          ),
          dtype=self.dtype,
      )
      if dp_axis is not None or tp_axis is not None:
        layer_block = sampler_lib.shard(layer_block, (dp_axis, None, tp_axis, None, None))
      if device is not None:
        layer_block = jax.device_put(layer_block, device)
      blocks[f"layer_{i}"] = layer_block

    page_indices = jnp.zeros((max_seqs, max_num_pages_per_seq), dtype=jnp.int32)
    available_page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    num_available_pages = jnp.array(num_pages, dtype=jnp.int32)
    seq_lens = jnp.zeros((max_seqs,), dtype=jnp.int32)

    if device is not None:
      page_indices = jax.device_put(page_indices, device)
      available_page_indices = jax.device_put(available_page_indices, device)
      num_available_pages = jax.device_put(num_available_pages, device)
      seq_lens = jax.device_put(seq_lens, device)
    

    return page_manager_lib.PageManager(
        pages=blocks,
        page_indices=page_indices,
        available_page_indices=available_page_indices,
        num_available_pages=num_available_pages,
        seq_lens=seq_lens,
        page_size=page_size,
        max_seq_len=max_seq_len,
        window_size=None,
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
      beam_size: Optional[int] = None,
  ) -> _SamplingState:
    """Initialize sampling state with HBM, Offloaded, and Prefix cache pools."""
    max_seqs = sampling_config.max_num_sequences
    max_seq_len = sampling_config.max_generation_steps + sampling_config.max_prompt_length
    
    page_size = self.cache_config.page_size
    hbm_num_pages = self.cache_config.num_hbm_pages
    cpu_num_pages = self.cache_config.num_cpu_pages


    shd_config = getattr(getattr(self.transformer, "config", None), "shd_config", None)
    if shd_config is not None:
      dp_axis = shd_config.act_btd[0]
      tp_axis = shd_config.act_btnh[2]
    else:
      dp_axis = None
      tp_axis = None

    hbm_pm_config = page_manager_lib.PageManagerConfig(
        page_size=hbm_num_pages,
        max_seq_len=max_seq_len,
        num_pages=hbm_num_pages,
        num_kv_heads=num_kv_heads
        max_num_seqs=sampling_config.max_num_sequences,
        head_dim=self.cache_config.head_dim,
        dtype=self.sampling_config.dtype,
        dp_axis=dp_axis,
        tp_axis=tp_axis
    )
    hbm_cache = hbm_pm_config.init()

    cpu_device = jax.devices("cpu")[0] if jax.devices("cpu") else None
    cpu_pm_config = dataclasses.replace(
      hbm_pm_config,
      num_pages = num_cpu_pages,
      dp_axis=None,
      tp_axis=None,
      device=cpu_device
    ) 
    cpu_cache = cpu_pm_config.init()



    hbm_cache = self._init_page_manager(
        num_pages=hbm_num_pages,
        max_seqs=max_seqs,
        page_size=page_size,
        max_seq_len=max_seq_len,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        device=None,
    )
    offloaded_cache = self._init_page_manager(
        num_pages=cpu_num_pages, 
        max_seqs=max_seqs,
        page_size=page_size,
        max_seq_len=max_seq_len,
        dp_axis=None,
        tp_axis=None,
        device=cpu_device,
    )

    eos_ids = tuple(sampling_config.eos_tokens) if sampling_config.eos_tokens is not None else None
    
    if sampling_config.include_logprobs:
      logprobs_buffer=jnp.zeros((max_seqs, max_seq_len))
    else:
      logprobs_buffer = None

    if sampling_config.include_logits:
      vocab_size = self.transformer.config.vocab_size
      logits_buffer=jnp.zeros((max_seqs, max_seq_len, vocab_size))
    else:
      logits_buffer = None

    sampling_parameters = {}
    sampling_mode = [None]

    if beam_size is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'beam_search')  # pyrefly: ignore[bad-argument-type]
      sampling_parameters['beam_size'] = beam_size

    if sampling_config.top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')  # pyrefly: ignore[bad-argument-type]
      sampling_parameters['top_p'] = sampling_config.top_p
      sampling_parameters['top_k'] = sampling_config.top_k

    return _SamplingState(
        max_prompt_length=sampling_config.max_prompt_length,
        max_generation_steps=sampling_config.max_generation_steps,
        decoding_steps=jnp.zeros((max_seqs,), dtype=jnp.int32),
        offloaded_decoding_steps=jnp.zeros((max_seqs,), dtype=jnp.int32),
        distribution=jnp.array([0, 0, 0], dtype=jnp.int32),
        hbm_cache=hbm_cache,
        offloaded_cache=offloaded_cache,
        done=jnp.zeros((max_seqs,), dtype=jnp.bool_),
        insertion_timestamps=jnp.zeros((max_seqs,), dtype=jnp.float32),
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

  def _release_slots(
      self,
      pm: page_manager_lib.PageManager,
      should_release: jax.Array,
  ) -> page_manager_lib.PageManager:
    """Unified release helper function across CPU, and TPU HBM PageManagers."""
    return pm.release(should_release)

  def _remove_request_from_pool(
      self,
      sampling_state: _SamplingState,
      request_id: str,
      request_ids: list[str],
      cache: page_manager_lib.PageManager,
  ) -> tuple[_SamplingState, page_manager_lib.PageManager, bool]:
    """Removes request_id from a pool, and releases allocated pages."""
    if request_id not in request_ids:
      return sampling_state, cache, False

    slot = request_ids.index(request_id)
    request_ids.pop(slot)
    sampling_state = dataclasses.replace(
        sampling_state,
    )

    should_release = jnp.zeros((int(cache.batch_size),), dtype=jnp.bool_).at[slot].set(True)
    updated_cache = self._release_slots(cache, should_release)
    return sampling_state, updated_cache, True

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

  def _tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    input_ids = np.array(
        self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
    )
    return input_ids

  def _queue_new_requests(
      self,
      sampling_state: _SamplingState,
      new_requests: Sequence[dict[str, Any]],
  ) -> _SamplingState:
    """Load incoming requests into offloaded_cache."""
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
    offloaded_decoding_steps = sampling_state.offloaded_decoding_steps
    for req_dict in new_requests:
      req_id = req_dict["id"]
      prompt_str = req_dict["prompt"]
      prompt_tokens = self._tokenize(prompt_str)
      prompt_len = len(prompt_tokens)

      slot_idx = len(cpu_req_ids)
      cpu_req_ids.append(req_id)
      insertion_timestamps = insertion_timestamps.at[slot_idx].set(time.perf_counter())
      offloaded_decoding_steps = offloaded_decoding_steps.at[slot_idx].set(0)

      q_lens = jnp.zeros((max_seqs,), dtype=jnp.int32).at[slot_idx].set(prompt_len)
      offloaded_cache = offloaded_cache.allocate(q_lens=q_lens)

      lens_arr = jnp.zeros((max_seqs,), dtype=jnp.int32).at[slot_idx].set(prompt_len)
      offloaded_cache = offloaded_cache.load_prompt_tokens(prompt_tokens, lens=lens_arr, key="token_buffer")

    return dataclasses.replace(
        sampling_state,
        offloaded_decoding_steps=offloaded_decoding_steps,
        offloaded_cache=offloaded_cache,
        cpu_request_ids=cpu_req_ids,
        insertion_timestamps=insertion_timestamps,
    )

  def _make_room_for_allocation(self, sampling_state: _SamplingState) -> _SamplingState:
    """Evicts newest active HBM sequence to offloaded_cache when HBM is constrained.
    """
    hbm_available = int(sampling_state.hbm_cache.num_available_pages)

    evict_hbm_slots: list[int] = []
    evict_cpu_slots: list[int] = []

    hbm_request_ids = list(sampling_state.hbm_request_ids)
    cpu_request_ids = list(sampling_state.cpu_request_ids)
    offloaded_cache = sampling_state.offloaded_cache
    
    num_tpu = len(hbm_request_ids)    
    num_cpu = len(cpu_request_ids) 
    while hbm_available < num_tpu:
      evict_idx = num_tpu - 1

      req_id = hbm_request_ids[evict_idx]
      seq_len = int(sampling_state.hbm_cache.seq_lens[evict_idx])
      pages_needed = utils.cdiv(seq_len, offloaded_cache.page_size)

      if int(offloaded_cache.num_available_pages) < pages_needed:
        raise RuntimeError("CPU Swap space is too small to evict request.")

      cpu_slot = num_cpu + len(evict_cpu_slots)
      cpu_request_ids.append(req_id)
      hbm_request_ids.pop()

      q_lens = jnp.zeros((int(offloaded_cache.batch_size),), dtype=jnp.int32).at[cpu_slot].set(seq_len)
      offloaded_cache = offloaded_cache.allocate(q_lens=q_lens)

      evict_hbm_slots.append(evict_idx)
      evict_cpu_slots.append(cpu_slot)
      
      num_tpu -= 1
      num_cpu += 1

    if evict_hbm_slots:
      evict_hbm_slots = jnp.array(evict_hbm_slots, dtype=jnp.int32)
      evict_cpu_slots = jnp.array(evict_cpu_slots, dtype=jnp.int32)
      
      offloaded_decoding_steps = sampling_state.offloaded_decoding_steps.at[evict_cpu_slots].set(sampling_state.decoding_steps[evict_hbm_slots])
      offloaded_cache = page_manager_lib.batch_copy_pages(
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
    
      return hbm_available, dataclasses.replace(
          sampling_state,
          offloaded_decoding_steps=offloaded_decoding_steps,
          hbm_cache=updated_hbm_cache,
          offloaded_cache=offloaded_cache,
          hbm_request_ids=hbm_request_ids,
          cpu_request_ids=cpu_request_ids,
      )

    return hbm_available, sampling_state

  def _drain_pending_queue(self, hbm_available: int, sampling_state: _SamplingState) -> _SamplingState:
    """Admit sequences from offloaded_cache to HBM cache."""
    # TODO:
    # 1. We can keep the cpu cache in insertion order and ditch timestamps
    # 2. Support chunked prefill?  

    hbm_req_ids = list(sampling_state.hbm_request_ids)
    cpu_req_ids = list(sampling_state.cpu_request_ids)

    sorted_cpu_slots = sorted(
        range(len(cpu_req_ids)),
        key=lambda s: float(sampling_state.insertion_timestamps[s]),
    )

    decode_src_slots: list[int] = []
    prefill_src_slots: list[int] = []
    
    is_decode = sampling_state.offloaded_decoding_steps > 0
    for cpu_slot in sorted_cpu_slots:
      seq_len = int(sampling_state.offloaded_cache.seq_lens[cpu_slot])
      pages_needed = utils.cdiv(seq_len + 1, sampling_state.hbm_cache.page_size)
      
      if pages_needed > hbm_available:
        break
      
      hbm_available -= pages_needed
      if is_decode[cpu_slot]: 
        decode_src_slots.append(cpu_slot)
      else:
        prefill_src_slots.append(cpu_slot)

    num_existing_decode = len(hbm_req_ids)
  
    max_seqs = int(sampling_state.hbm_cache.batch_size)
    offloaded_cache = sampling_state.offloaded_cache
        
    i_val = num_existing_decode + len(decode_src_slots)
    j_val = i_val + len(prefill_src_slots)
    k_val = j_val
  
    src_slots = decode_src_slots + prefill_src_slots
    dst_slots = list(range(num_existing_decode, k_val))

    q_lens = jnp.zeros((max_seqs,), dtype=jnp.int32)

    for src_slot, dst_slot in zip(src_slots, dst_slots):
      req_id = cpu_req_ids[src_slot]
      seq_len = int(offloaded_cache.seq_lens[src_slot])
      hbm_req_ids.append(req_id)
      q_lens = q_lens.at[dst_slot].set(seq_len)

    updated_hbm_cache = sampling_state.hbm_cache.allocate(q_lens=q_lens)
    
    # Decode sequences have full KV values -> transfer_kv=True
    decode_dst_slots = list(range(num_existing_decode, i_val))
    updated_hbm_cache = page_manager_lib.batch_copy_pages(
        offloaded_cache,
        updated_hbm_cache,
        decode_src_slots,
        decode_dst_slots,
        transfer_kv=True,
    )

    prefill_dst_slots = list(range(i_val, j_val))
    updated_hbm_cache = page_manager_lib.batch_copy_pages(
        offloaded_cache,
        updated_hbm_cache,
        prefill_src_slots,
        prefill_dst_slots,
        transfer_kv=False,
    )
    
    should_release_cpu = jnp.zeros((max_seqs,), dtype=jnp.bool_)
    for slot in src_slots:
      should_release_cpu = should_release_cpu.at[slot].set(True)

    admitted_set = set(src_slots)
    cpu_req_ids = [rid for i, rid in enumerate(cpu_req_ids) if i not in admitted_set]

    dist = jnp.array([i_val, i_val, k_val], dtype=jnp.int32)
    src_slots = jnp.array(src_slots, dtype=jnp.int32)
    dst_slots = jnp.array(dst_slots, dtype=jnp.int32)

    src_decoding_steps = sampling_state.offloaded_decoding_steps[src_slots]
    updated_decoding_steps = sampling_state.decoding_steps.at[dst_slots].set(src_decoding_steps)

    return dataclasses.replace(
        sampling_state,
        decoding_steps=updated_decoding_steps,
        distribution=dist,
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

    # reordered_ids = sampling_state.hbm_request_ids[slot_perm]
    reordered_ids = [sampling_state.hbm_request_ids[int(i)] for i in jax.device_get(slot_perm)[:num_remaining]]

    return dataclasses.replace(
        sampling_state,
        decoding_steps=sampling_state.decoding_steps[slot_perm],
        insertion_timestamps=sampling_state.insertion_timestamps[slot_perm],
        done=jnp.zeros_like(sampling_state.done),
        hbm_cache=compacted_hbm_cache,
        hbm_request_ids=reordered_ids,
    )

  def _model_step_fn(
      self,
      max_prompt_length: int,
      params: statelib.State,
      cache: page_manager_lib.PageManager,
      decoding_steps: jax.Array,
      distribution: jnp.ndarray,
      images: jnp.ndarray | None = None,
      audios: Any = None,
      echo: bool = False,
      soft_cap: float | None = None,
      **kwargs,
  ) -> Tuple[jnp.ndarray, page_manager_lib.PageManager]:
    """JIT-compiled forward pass invoking Gemma with ragged paged attention and explicit soft_cap."""
    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]

    kwargs = {}
    if images is not None:
      kwargs['images'] = images
    if audios is not None:
      kwargs['audios'] = audios
    decode_only_last_token = self._supports_decode_only_last_token and not echo
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True

    transformer = nnx.merge(self._transformer_graphdef, params)
    max_seqs = int(cache.batch_size)

    is_decode = (decoding_steps > 0) & (cache.seq_lens > 0)
    active_seq_lens = jnp.where(
        is_decode,
        1,
        cache.seq_lens, # Prefill
    )

    token_start_idxs = jnp.where(
        is_decode,
        cache.seq_lens - 1,
        0,
    )

    max_seqs = int(cache.batch_size)
    static_token_capacity = int(
         max_prompt_length * max_seqs
    )

    ragged = page_manager_lib.RaggedArray(
        data=jnp.zeros((static_token_capacity,), dtype=jnp.int32),
        lens=active_seq_lens,
    )
    seq_idxs = ragged.row_idxs
    intra_offsets = ragged.intra_offsets

    abs_positions = token_start_idxs[seq_idxs] + intra_offsets
    page_cols = abs_positions // cache.page_size
    page_offsets = abs_positions % cache.page_size
    phys_page_ids = cache.page_indices[seq_idxs, page_cols]

    tokens = cache.pages["token_buffer"][
        phys_page_ids, page_offsets
    ]

    logits, new_cache = transformer(
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
    
    return last_token_logits, new_cache

  def _release_completed(
      self,
      sampling_state: _SamplingState,
      next_tokens: jax.Array,
  ) -> tuple[_SamplingState, dict[str, RequestOutput]]:
    """Inspects generated tokens, appends EOS at max_seq_len - 1, formats outputs, and releases completed slots."""
    completed_outputs: dict[str, RequestOutput] = {}
    cache = sampling_state.hbm_cache

    batch_size = cache.batch_size
    max_seq_len = cache.max_seq_len

    should_release_hbm = jnp.zeros((batch_size,), dtype=jnp.bool)
    logp_buffer = sampling_state.logprobs_buffer
    logits_buffer = sampling_state.logits_buffer
    
    total_len = jnp.sum(cache.seq_lens)
    ragged_token_buffer = cache.to_array(total_len)
    
    # TODO: We should find and write the completed outputs in batched operations.
    total_len = 0
    for idx, req_id in enumerate(sampling_state.hbm_request_ids):
      sampled_token = int(next_tokens[idx])
      eos_ids = sampling_state.eos_token_ids or (1,)
      n_decoding_steps = int(sampling_state.decoding_steps[idx])
      is_done = (
          (sampled_token in eos_ids)
          or (n_decoding_steps >= sampling_state.max_generation_steps)
      )
      seq_len = int(sampling_state.hbm_cache.seq_lens[idx])

      if is_done:
        if sampled_token not in eos_ids:
          sampled_token = eos_ids[0]
      
        logps = None
        logits = None
        tokens = jax.device_get(ragged_token_buffer[total_len: total_len + cache.seq_lens[idx]])
        
        prompt_len = seq_len - n_decoding_steps 
        padded_prompt = utils.pad_to_length(
          tokens[:prompt_len],
          target_length=sampling_state.max_prompt_length,
          pad_value=self.tokenizer.pad_id(),
          left=True,
        )

        if logp_buffer is not None:
          logps = logp_buffer[idx]
        if logits_buffer is not None:
          logits = logits_buffer[idx, :n_decoding_steps]
         
        gen_tokens = tokens[prompt_len:]
        completed_outputs[req_id] = RequestOutput(
            request_id=req_id,
            text=self.tokenizer.decode(gen_tokens.tolist()),
            tokens=gen_tokens,
            logprobs=jax.device_get(logps),
            logits=jax.device_get(logits),
            padded_tokens=padded_prompt,
        )
        should_release_hbm = should_release_hbm.at[idx].set(True)
      total_len += cache.seq_lens[idx]

    updated_hbm_cache = self._release_slots(sampling_state.hbm_cache, should_release_hbm)

    sampling_state = dataclasses.replace(
        sampling_state,
        hbm_cache=updated_hbm_cache,
        done=should_release_hbm,
    )
    return self._compact_batch(sampling_state), completed_outputs

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
      return {}
    
    hbm_cache = sampling_state.hbm_cache

    max_prompt_length = sampling_state.max_prompt_length
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
