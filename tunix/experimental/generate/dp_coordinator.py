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

"""Per-rank data-parallel coordinator.

`DPCoordinator` manages a single DP rank's `Scheduler`, `KVCacheManager`,
request queueing, batch preparation, and output token updates.
"""

from collections.abc import Mapping
import itertools

import jax
import numpy as np

from tunix.common import datatypes
from tunix.experimental.generate import configs
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.generate import utils


RankBatch = tuple[tuple[request_lib.RequestState, ...], tuple[int, int, int]]


class DPCoordinator:
  """Coordinates scheduling, KV cache management, and batching for one DP rank."""

  def __init__(
      self,
      cache_config: configs.CacheConfig,
      scheduler_config: configs.SchedulerConfig,
      cache_geometries: Mapping[str, datatypes.CacheGeometry],
      mesh: jax.sharding.Mesh,
      max_model_len: int,
      eos_token_ids: frozenset[int],
  ):
    """Initializes the DP coordinator.

    Args:
      cache_config: The KV cache configuration for this rank.
      scheduler_config: The scheduler configuration for this rank.
      cache_geometries: The KV cache geometries of the model.
      mesh: The device mesh slice for this DP rank.
      max_model_len: The maximum number of tokens a sequence may hold (prompt +
        generated).
      eos_token_ids: The token IDs that stop generation.

    Raises:
      ValueError: If `max_model_len` is not positive, or if `max_model_len`
        requires more pages than fit in the rank's device KV cache.
    """
    if max_model_len <= 0:
      raise ValueError(f'max_model_len must be positive, got {max_model_len}.')

    with jax.set_mesh(mesh):
      self._kv_cache_manager = kv_cache_manager_lib.KVCacheManager(
          cache_config, cache_geometries
      )
    self._scheduler = scheduler_lib.Scheduler(
        config=scheduler_config, kv_cache_manager=self._kv_cache_manager
    )
    self._scheduler_config = self._scheduler.config
    self._eos_token_ids = eos_token_ids

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
    # removed when aborted in `abort_request` or when finished in
    # `update_from_output`.
    self._requests: dict[str, request_lib.RequestState] = {}

  def reset_kv_caches(self) -> None:
    """Discards all KV, for after the weights change in place.

    Every in-flight request is preempted, and re-prefills its prompt and the
    tokens sampled so far. The prefix cache is cleared. If the device KV page
    pools were deleted, they are reallocated.
    """
    self._scheduler.preempt_all()
    self._kv_cache_manager.reset_kv_caches()
    self._kv_cache_manager.reallocate_device_pool()

  def delete_kv_caches(self) -> None:
    """Preempts requests and deletes device KV buffers."""
    self._scheduler.preempt_all()
    self._kv_cache_manager.delete_device_pool()

  def get_physical_pages(self) -> dict[str, jax.Array]:
    """Returns this rank's physical KV cache buffers."""
    return self._kv_cache_manager.get_physical_pages()

  def update_device_pool(self, pages: dict[str, jax.Array]) -> None:
    """Updates this rank's physical KV cache buffers."""
    self._kv_cache_manager.update_device_pool(pages)

  def add_request(self, request: request_lib.EngineRequest) -> None:
    """Submits a request, to be scheduled from the next step on."""
    request_id = request.request_id
    if request_id in self._requests:
      raise ValueError(f'Request {request_id} is already running.')
    sampling_params = request.sampling_params

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
    """Whether stepping this rank has work left to do."""
    return (
        any(not req.is_done for req in self._queued_requests)
        or self._scheduler.num_active_requests > 0
    )

  def schedule_step(self) -> RankBatch:
    """Schedules the next step on this rank, draining its queued requests."""
    new_requests = [req for req in self._queued_requests if not req.is_done]
    self._queued_requests = []
    return self._scheduler.schedule_step(new_requests)

  def update_from_output(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ) -> list[request_lib.RequestState]:
    """Updates request state based on a sampler output and returns completed requests."""
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

    self._scheduler.drop_completed()
    for req in completed_reqs:
      del self._requests[req.request_id]
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

  def prepare_batch(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      prev_scheduled: tuple[request_lib.RequestState, ...] = (),
  ) -> tuple[
      list[int],
      dict[int, int],
      np.ndarray,
      np.ndarray,
      dict[str, np.ndarray],
  ]:
    """Packs the scheduled requests into this rank's tokens and ragged metadata.

    Every row buffer spans `max_num_seqs` rows and every page table
    `max_pages_per_seq` pages, so their shapes never change.

    Args:
      scheduled: The scheduled requests, ordered as decodes, chunked prefills
        and full prefills.
      prev_scheduled: The requests running in the in-flight step, if any.

    Returns:
      A tuple of (the rank's token list with prefill tokens loaded and blank
      slots for in-flight decodes, the mapping from previous-step row to decode
      slot, query_lens, kv_lens, page_indices).
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

    return (
        batch_tokens,
        prev_to_decode_row,
        query_lens,
        kv_lens,
        page_indices,
    )
