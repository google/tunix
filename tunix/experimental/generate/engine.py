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

"""Continuous batching engine.

`LLMEngine` ties the v2 sampler together: each step, the `Scheduler` picks a
batch of requests, the `KVCacheManager` maps them onto KV cache pages, and the
`ModelRunner` runs the model over the batch and samples the next tokens.
"""

import dataclasses
import itertools
from typing import Any

from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jaxtyping
import numpy as np

from tunix.common import configs as common_configs
from tunix.experimental.generate import configs
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.generate import utils
from tunix.models import paged_attention


_FINISH_REASONS: dict[request_lib.RequestStatus, request_lib.FinishReason] = {
    request_lib.RequestStatus.FINISHED_EOS: 'stop',
    request_lib.RequestStatus.FINISHED_LENGTH: 'length',
}


@dataclasses.dataclass(frozen=True)
class _InFlightStep:
  """A model runner step currently executing on the device."""

  scheduled: tuple[request_lib.RequestState, ...]
  distribution: tuple[int, int, int]
  generated_tokens: jax.Array
  logits: jax.Array | None
  logprobs: jax.Array | None


def _instantiate_components(
    transformer: nnx.Module,
    rollout_config: common_configs.RolloutConfig,
    mesh: jax.sharding.Mesh,
) -> tuple[
    model_runner_lib.ModelRunner,
    kv_cache_manager_lib.KVCacheManager,
    scheduler_lib.Scheduler,
]:
  """Builds the engine's components from `rollout_config` and `mesh`.

  Args:
    transformer: The model to sample from.
    rollout_cfg: The rollout configuration.
    mesh: The device mesh the model runs on.

  Returns:
    A tuple of (the model runner, the KV cache manager, the scheduler).
  """
  max_device_size_gib = utils.resolve_max_device_size_gib(
      rollout_config.rollout_vanilla_v2_hbm_utilization, mesh
  )
  cache_config = configs.CacheConfig(
      max_device_size_gib=max_device_size_gib,
      max_host_size_gib=rollout_config.rollout_vanilla_v2_host_size_gib,
      page_size=rollout_config.rollout_vanilla_v2_kv_cache_page_size,
      enable_prefix_caching=rollout_config.rollout_vanilla_v2_enable_prefix_caching,
      dtype=transformer.config.dtype,  # pyrefly: ignore[missing-attribute]
  )
  cache_geometries = transformer.cache_geometries()  # pyrefly: ignore[missing-attribute]
  with jax.set_mesh(mesh):
    kv_cache_manager = kv_cache_manager_lib.KVCacheManager(
        cache_config, cache_geometries
    )

  # A sliding window layer must hold the KV of every token in its window while
  # it processes a prefill chunk, so a longer chunk size is truncated to the
  # largest power of 2 within the smallest window.
  window_sizes = [
      g.window_size
      for g in cache_geometries.values()
      if g.window_size is not None
  ]
  max_chunked_prefill_length = (
      rollout_config.rollout_vanilla_v2_chunked_prefill_length
  )
  if window_sizes:
    min_window_size = min(window_sizes)
    max_chunked_prefill_length = min(
        max_chunked_prefill_length,
        1 << (min_window_size.bit_length() - 1)
    )
  scheduler_config = configs.SchedulerConfig(
      max_num_batched_tokens=rollout_config.rollout_vanilla_v2_max_num_batched_tokens,
      max_num_seqs=rollout_config.rollout_vanilla_v2_max_num_seqs,
      max_chunked_prefill_length=max_chunked_prefill_length,
      num_scheduler_steps=rollout_config.rollout_vanilla_v2_num_scheduler_steps,
  )
  scheduler = scheduler_lib.Scheduler(
      config=scheduler_config, kv_cache_manager=kv_cache_manager
  )

  model_runner_config = configs.ModelRunnerConfig(
      max_top_k=rollout_config.top_k or -1,
      mesh=mesh,
      return_logprobs=rollout_config.return_logprobs,
      return_logits=rollout_config.return_logits,
      num_scheduler_steps=rollout_config.rollout_vanilla_v2_num_scheduler_steps,
      seed=0 if rollout_config.seed is None else int(rollout_config.seed),
  )
  model_runner = model_runner_lib.ModelRunner(transformer, model_runner_config)

  return model_runner, kv_cache_manager, scheduler


class LLMEngine:
  """A continuous batching engine."""

  def __init__(
      self,
      transformer: nnx.Module,
      rollout_config: common_configs.RolloutConfig,
      mesh: jax.sharding.Mesh,
      max_model_len: int,
  ):
    """Initializes the engine.

    Args:
      transformer: The model to sample from.
      rollout_config: The engine's rollout configuration.
      mesh: The device mesh the model runs on.
      max_model_len: The maximum number of tokens a sequence may hold (prompt +
        generated).

    Raises:
      ValueError: If `max_model_len` is not positive, or if `rollout_config`
        settings are invalid.
    """
    if max_model_len <= 0:
      raise ValueError(f'max_model_len must be positive, got {max_model_len}.')

    self._model_runner, self._kv_cache_manager, self._scheduler = (
        _instantiate_components(transformer, rollout_config, mesh)
    )
    self._scheduler_config = self._scheduler.config
    self._model_runner_config = self._model_runner.config
    self._eos_token_ids = frozenset(rollout_config.eos_tokens or ())
    self._replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # The maximum number of tokens a sequence may hold (prompt + generated).
    self._max_model_len = max_model_len
    # The maximum number of pages a sequence may hold in the KV cache.
    self._max_pages_per_seq = utils.cdiv(
        max_model_len + self._scheduler_config.num_scheduler_steps,
        self._kv_cache_manager.page_size,
    )
    if self._max_pages_per_seq > self._kv_cache_manager.num_device_pages:
      raise ValueError(
          f'A sequence of max_model_len={max_model_len} with'
          f' num_scheduler_steps={self._scheduler_config.num_scheduler_steps} and'
          f' page_size={self._kv_cache_manager.page_size} needs up to'
          f' {self._max_pages_per_seq} pages, but the KV cache only has'
          f' {self._kv_cache_manager.num_device_pages} device pages.'
      )

    # Requests submitted since the last step, in submission order.
    self._queued_requests: list[request_lib.RequestState] = []
    # Requests that are neither finished nor aborted, by request id. Entries are
    # removed when aborted in `abort_request` or when finished in `step`.
    self._requests: dict[str, request_lib.RequestState] = {}
    # The step launched during the previous step's forward pass.
    self._in_flight_step: _InFlightStep | None = None

  def model_def_and_state(
      self,
  ) -> tuple[graph.GraphDef[nnx.Module], list[nnx.Variable]]:
    return self._model_runner.model_def_and_state()

  @property
  def transformer_state(self) -> statelib.State:
    """The weights the engine runs, which every step reads in place."""
    return self._model_runner.transformer_state

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._model_runner_config.mesh

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Syncs new weights into the model the engine runs.

    Every in-flight request is preempted: the KV it holds was computed with
    the previous weights, so it is discarded, and the request re-prefills its
    prompt and the tokens sampled so far with the new weights. Tokens already
    sampled are kept. The prefix cache is cleared for the same reason.

    Args:
      updated_weights: The new weights, either all of the model's parameters
        or its LoRA parameters alone.
      filter_types: The variable types to update. All of them when None.
    """
    # Only the weight update can fail, so running it first leaves the engine
    # usable if it does.
    self._model_runner.update_params(updated_weights, filter_types)
    self.reset_kv_caches()

  def reset_kv_caches(self) -> None:
    """Discards all KV, for after the weights change in place.

    Every in-flight request is preempted, and re-prefills its prompt and the
    tokens sampled so far. The prefix cache is cleared. If the device KV page
    pools were deleted, they are reallocated.
    """
    self._in_flight_step = None
    self._scheduler.preempt_all()
    self._kv_cache_manager.reset_kv_caches()
    self._kv_cache_manager.reallocate_device_pool()

  def delete_kv_caches(self) -> list[request_lib.EngineResponse]:
    """Drains any in-flight step, preempts requests, and deletes device KV buffers."""
    outputs: list[request_lib.EngineResponse] = []
    if self._in_flight_step is not None:
      in_flight = self._in_flight_step
      self._in_flight_step = None
      outputs = self._consume_in_flight_step(in_flight)
    self._scheduler.preempt_all()
    self._kv_cache_manager.delete_device_pool()
    return outputs

  def add_request(self, request: request_lib.EngineRequest) -> None:
    """Submits a request, to be scheduled from the next step on."""
    request_id = request.request_id
    if request_id in self._requests:
      raise ValueError(f'Request {request_id} is already running.')
    sampling_params = request.sampling_params
    self._check_sampling_params(request_id, sampling_params)

    prompt_token_ids = request.prompt_token_ids
    if not prompt_token_ids:
      raise ValueError(f'Request {request_id} has an empty prompt.')
    max_len = len(prompt_token_ids) + sampling_params.max_tokens
    if max_len > self._max_model_len:
      raise ValueError(
          f'Request {request_id} has a prompt of {len(prompt_token_ids)}'
          f' tokens and generates up to {sampling_params.max_tokens} tokens,'
          f' {max_len} in total, over max_model_len={self._max_model_len}.'
      )

    req = request_lib.RequestState(
        request_id, prompt_token_ids, sampling_params
    )
    self._requests[request_id] = req
    self._queued_requests.append(req)

  def _check_sampling_params(
      self, request_id: str, params: request_lib.SamplingParams
  ) -> None:
    """Raises a ValueError if the engine cannot sample with `params`."""
    if params.max_tokens < 1:
      raise ValueError(
          f'Request {request_id} has max_tokens={params.max_tokens}; it must'
          ' be positive.'
      )
    if params.temperature < 0.0:
      raise ValueError(
          f'Request {request_id} has temperature={params.temperature}; it'
          ' must be non-negative.'
      )
    if params.top_p is not None and not 0.0 < params.top_p <= 1.0:
      raise ValueError(
          f'Request {request_id} has top_p={params.top_p}; it must be in'
          ' (0, 1].'
      )
    if params.top_k is not None:
      max_top_k = self._model_runner_config.max_top_k
      if max_top_k > 0 and params.top_k > max_top_k:
        raise ValueError(
            f'Request {request_id} has top_k={params.top_k}, over'
            f' max_top_k={max_top_k}.'
        )

    unsupported = {
        'cache_salt': params.cache_salt is not None,
        'beam_size': params.beam_size is not None,
        'return_routed_experts': params.return_routed_experts,
        'routed_experts_prompt_start': params.routed_experts_prompt_start != 0,
    }
    for name, is_set in unsupported.items():
      if is_set:
        raise ValueError(
            f'Request {request_id} sets {name}, which the engine does not'
            ' support.'
        )
    # The model runner config fixes the outputs of every step, so a request
    # must ask for exactly those.
    config = self._model_runner_config
    if params.return_logprobs != config.return_logprobs:
      raise ValueError(
          f'Request {request_id} has return_logprobs={params.return_logprobs},'
          f' but the engine has return_logprobs={config.return_logprobs}.'
      )
    if params.return_logits != config.return_logits:
      raise ValueError(
          f'Request {request_id} has return_logits={params.return_logits},'
          f' but the engine has return_logits={config.return_logits}.'
      )

  def abort_request(self, request_id: str) -> None:
    """Aborts a request. Has no effect if it is unknown or already done.

    A request still in `_queued_requests` is removed immediately. A request the
    scheduler already holds keeps its KV cache pages until the next step, which
    releases them.

    Args:
      request_id: The id of the request to abort.
    """
    req = self._requests.pop(request_id, None)
    if req is None:
      return
    if req in self._queued_requests:
      self._queued_requests.remove(req)
    self._scheduler.abort_request(req)

  def has_unfinished_requests(self) -> bool:
    """Whether stepping the engine has work left to do."""
    return (
        any(not req.is_done for req in self._queued_requests)
        or self._scheduler.num_active_requests > 0
    )

  def step(self) -> list[request_lib.EngineResponse]:
    """Runs one step of the continuous batch.

    Returns:
      The outputs of the requests that finished during this step.
    """
    new_requests = [req for req in self._queued_requests if not req.is_done]
    self._queued_requests = []

    if self._in_flight_step is not None and any(
        not req.is_done for req in self._in_flight_step.scheduled
    ):
      in_flight = self._in_flight_step
      self._in_flight_step = None
    else:
      self._in_flight_step = None
      scheduled, distribution = self._scheduler.schedule_step(new_requests)
      new_requests = []
      if not scheduled:
        return []
      tokens, target_idxs, metadata = self._prepare_batch(
          scheduled, distribution
      )
      in_flight = self._execute_batch(
          scheduled, distribution, tokens, target_idxs, metadata
      )

    # Schedule the next step, load its prefill token buffer onto the device,
    # write its decode tokens from `in_flight.generated_tokens`, and launch it
    # while the current forward pass runs on device.
    next_scheduled, next_distribution = self._scheduler.schedule_step(
        new_requests
    )
    if next_scheduled:
      next_tokens, target_idxs, next_metadata = self._prepare_batch(
          next_scheduled,
          next_distribution,
          prev_scheduled=in_flight.scheduled,
      )
      self._in_flight_step = self._execute_batch(
          next_scheduled,
          next_distribution,
          next_tokens,
          target_idxs,
          next_metadata,
          prev_generated_tokens=in_flight.generated_tokens,
      )

    return self._consume_in_flight_step(in_flight)

  def _consume_in_flight_step(
      self, in_flight: _InFlightStep
  ) -> list[request_lib.EngineResponse]:
    """Blocks on `in_flight`, updates request states, and returns finished outputs."""
    generated_tokens, logits, logprobs = jax.device_get(
        (in_flight.generated_tokens, in_flight.logits, in_flight.logprobs)
    )

    # The runner pads its outputs to `max_num_seqs` rows. Drop the
    # padding.
    n = len(in_flight.scheduled)
    finished = self._update_from_output(
        in_flight.scheduled,
        in_flight.distribution,
        generated_tokens[:n],
        None if logits is None else logits[:n],
        None if logprobs is None else logprobs[:n],
    )
    self._scheduler.drop_completed()
    if self._in_flight_step is not None and all(
        req.is_done for req in self._in_flight_step.scheduled
    ):
      self._in_flight_step = None

    for req in finished:
      del self._requests[req.request_id]
    return [self._make_output(req) for req in finished]

  def _execute_batch(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      tokens: jax.Array,
      target_idxs: jax.Array,
      metadata: paged_attention.PagedAttnMetadata,
      prev_generated_tokens: jax.Array | None = None,
  ) -> _InFlightStep:
    """Writes any in-flight decode tokens on device and launches the runner."""
    if prev_generated_tokens is not None:
      tokens = utils.jit_scatter_last_column(
          tokens, prev_generated_tokens, target_idxs
      )
    generated_tokens, logits, logprobs, pages = (
        self._model_runner.execute_step(
            cache=self._kv_cache_manager.get_physical_pages(),
            tokens=tokens,
            metadata=metadata,
            sampling_params=tuple(req.sampling_params for req in scheduled),
        )
    )
    # JAX functions cannot have side effects, so the model runner returns the
    # updated pages rather than writing them in place. They must be written back
    # to the KV cache manager here.
    self._kv_cache_manager.update_device_pool(pages)
    generated_tokens.copy_to_host_async()
    if logits is not None:
      logits.copy_to_host_async()
    if logprobs is not None:
      logprobs.copy_to_host_async()
    return _InFlightStep(
        scheduled=scheduled,
        distribution=distribution,
        generated_tokens=generated_tokens,
        logits=logits,
        logprobs=logprobs,
    )

  def _update_from_output(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ) -> list[request_lib.RequestState]:
    """Updates request state based on a sampler output."""
    i, j, k = distribution
    completed_reqs = []
    for row in itertools.chain(range(i), range(j, k)):
      req = scheduled[row]
      if req.is_done:
        req.num_in_flight_tokens = 0
        req.num_stale_output_tokens = 0
        continue

      self._update_req_from_output(
          req,
          generated_tokens[row],
          logits[row] if logits is not None else None,
          logprobs[row] if logprobs is not None else None,
      )

      if req.is_done:
        completed_reqs.append(req)

    return completed_reqs

  def _update_req_from_output(
      self,
      req: request_lib.RequestState,
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ) -> None:
    """Updates request state based on a sampler output."""
    num_steps = self._scheduler_config.num_scheduler_steps
    had_stale_output = req.num_stale_output_tokens > 0
    if had_stale_output:
      # Sampled before `req` was preempted. The tokens are still appended,
      # but `_preempt` already reset the counters this step advanced.
      req.num_stale_output_tokens = max(
          0, req.num_stale_output_tokens - num_steps
      )
    else:
      req.num_in_flight_tokens = max(0, req.num_in_flight_tokens - num_steps)
    if req.is_done:
      return

    was_running = req.status == request_lib.RequestStatus.RUNNING

    # Gather the new tokens generated for this request. The model runner
    # does not check for EOS and may produce garbage tokens.
    new_tokens = []
    for new_token in generated_tokens:
      token_id = int(new_token)
      new_tokens.append(token_id)
      if token_id in self._eos_token_ids:
        req.status = request_lib.RequestStatus.FINISHED_EOS
        break

    # Truncate the new tokens if the request has reached its max length.
    n_generated = (
        len(req.token_ids) + len(new_tokens) - req.prompt_length
    )
    n_overflow = n_generated - req.sampling_params.max_tokens
    if n_overflow > 0:
      new_tokens = new_tokens[:-n_overflow]
      req.status = request_lib.RequestStatus.FINISHED_LENGTH
    elif n_overflow == 0 and not req.is_done:
      req.status = request_lib.RequestStatus.FINISHED_LENGTH

    # Add the new tokens to the request.
    num_added = len(new_tokens)
    req.token_ids.extend(new_tokens)
    if logprobs is not None:
      req.logprobs.extend(logprobs[:num_added])
    if logits is not None:
      req.logits.extend(logits[:num_added])

    if req.is_done:
      req.num_in_flight_tokens = 0
      req.num_stale_output_tokens = 0
      # A request that was preempted and re-admitted since this output was
      # sampled has computed only the tokens it re-ran, not the whole sequence.
      if was_running and not had_stale_output:
        req.num_computed_tokens = len(req.token_ids) - 1

  def _make_output(
      self, req: request_lib.RequestState
  ) -> request_lib.EngineResponse:
    """Returns the output of the finished request `req`."""
    generated = req.token_ids[req.prompt_length :]
    return request_lib.EngineResponse(
        request_id=req.request_id,
        prompt_token_ids=np.asarray(
            req.token_ids[: req.prompt_length], dtype=np.int32
        ),
        token_ids=np.asarray(generated, dtype=np.int32),
        logprobs=(
            np.asarray(req.logprobs, dtype=np.float32)
            if self._model_runner_config.return_logprobs
            else None
        ),
        logits=(
            np.asarray(req.logits)
            if self._model_runner_config.return_logits
            else None
        ),
        finish_reason=_FINISH_REASONS[req.status],
    )

  def _prepare_batch(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      prev_scheduled: tuple[request_lib.RequestState, ...] = (),
  ) -> tuple[jax.Array, jax.Array, paged_attention.PagedAttnMetadata]:
    """Packs the scheduled requests into the model runner's inputs.

    Every row buffer spans `max_num_seqs` rows and every page table
    `max_pages_per_seq` pages, so their shapes never change.

    Args:
      scheduled: The scheduled requests, ordered as decodes, chunked prefills
        and full prefills.
      distribution: The row layout returned by the scheduler.
      prev_scheduled: The requests running in the in-flight step, if any.

    Returns:
      A tuple of (the device token buffer with prefill tokens loaded and blank
      slots for in-flight decodes, the target indices mapping each row of the
      previous step to its decode slot or out of bounds, the ragged execution
      metadata).
    """
    num_rows = self._scheduler_config.max_num_seqs
    query_lens = np.zeros((num_rows,), dtype=np.int32)
    kv_lens = np.zeros((num_rows,), dtype=np.int32)
    page_indices = {
        cache_name: np.full(
            (num_rows, self._max_pages_per_seq), -1, dtype=np.int32
        )
        for cache_name in self._kv_cache_manager.cache_names
    }

    prev_row_by_id = {
        req.request_id: prev_row for prev_row, req in enumerate(prev_scheduled)
    }
    prev_to_decode_row: dict[int, int] = {}

    batch_tokens: list[int] = []
    for row, req in enumerate(scheduled):
      start = req.num_computed_tokens

      if req.is_chunked_prefill:
        end = start + self._scheduler.chunked_prefill_length
        req.num_computed_tokens = end
        req.num_in_flight_tokens = 0
        batch_tokens.extend(req.token_ids[start:end])
      else:
        end = req.num_total_tokens
        if req.num_in_flight_tokens > 0:
          batch_tokens.append(0)
          prev_to_decode_row[prev_row_by_id[req.request_id]] = row
        else:
          batch_tokens.extend(req.token_ids[start:end])
        req.num_in_flight_tokens += self._scheduler_config.num_scheduler_steps
        req.num_computed_tokens = (
            end + self._scheduler_config.num_scheduler_steps - 1
        )

      query_lens[row] = end - start
      kv_lens[row] = end
      for cache_name, idxs in self._kv_cache_manager.get_page_idxs(req).items():
        page_indices[cache_name][row, : len(idxs)] = idxs

    # Pad the tokens to a power of 2 to limit jax recompilation in the model
    # runner.
    num_tokens = min(
        utils.next_power_of_2(len(batch_tokens)),
        self._scheduler_config.max_num_batched_tokens,
    )
    tokens = np.zeros((1, num_tokens), dtype=np.int32)
    tokens[0, : len(batch_tokens)] = batch_tokens

    target_idxs = np.full((num_rows,), num_tokens, dtype=np.int32)
    for prev_row, decode_row in prev_to_decode_row.items():
      target_idxs[prev_row] = decode_row

    metadata = paged_attention.PagedAttnMetadata(
        page_indices={
            cache_name: table for cache_name, table in page_indices.items()
        },
        kv_lens=kv_lens,
        query_lens=query_lens,
        distribution=np.asarray((distribution,), dtype=np.int32),
        decode_only=bool(
            distribution[0] == distribution[2] and distribution[2] > 0
        ),
    )
    device_tokens, device_target_idxs = jax.device_put(
        (tokens, target_idxs), self._replicated_sharding
    )
    return device_tokens, device_target_idxs, metadata
