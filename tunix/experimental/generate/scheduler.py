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

"""A Scheduler for rollout requests in Tunix."""

import collections
import dataclasses
import numpy as np
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import request as request_lib


@dataclasses.dataclass(frozen=True, kw_only=True)
class SchedulerConfig:
  """Configuration for the scheduler."""

  # The maximum number of tokens that can be scheduled in a single batch.
  max_num_batch_tokens: int
  # The maximum number of sequences that can be scheduled in a single batch.
  max_seqs_per_batch: int
  # The size of a chunked prefill request in tokens. Must be a power of 2 and
  # less than or equal to max_num_batch_tokens.
  chunked_prefill_length: int
  # The number of decode steps per engine step.
  num_decode_steps: int = 1

  def __post_init__(self):
    if self.chunked_prefill_length < 0:
      raise ValueError(
          "Chunked prefill length must be non-negative. Got"
          f" {self.chunked_prefill_length}."
      )

    is_power_of_two = lambda n: n > 0 and (n & (n - 1)) == 0
    if not is_power_of_two(self.chunked_prefill_length):
      raise ValueError(
          "Chunked prefill length must be a power of 2. Got"
          f" {self.chunked_prefill_length}."
      )

    if self.chunked_prefill_length > self.max_num_batch_tokens:
      raise ValueError(
          "Chunked prefill length must be less than or equal to "
          f"max_num_batch_tokens. Got  {self.chunked_prefill_length} "
          f"and {self.max_num_batch_tokens}."
      )


class Scheduler:
  """A continuous batching scheduler.

  Each engine step schedules decodes first, then in-progress chunked prefills,
  then pending requests in arrival order, while KV cache space is available
  and the batch is within `max_seqs_per_batch` requests and
  `max_num_batch_tokens` tokens.
  """

  def __init__(
      self,
      config: SchedulerConfig,
      kv_cache_manager: kv_cache_manager_lib.KVCacheManager,
  ):
    self._config = config

    self._running_requests: collections.deque[request_lib.Request] = (
        collections.deque()
    )
    self._pending_requests: collections.deque[request_lib.Request] = (
        collections.deque()
    )
    self._scheduled_requests: tuple[request_lib.Request, ...] = ()
    self._token_budget: int = 0

    self._kv_cache_manager = kv_cache_manager

  @property
  def chunked_prefill_length(self) -> int:
    return self._config.chunked_prefill_length

  @property
  def num_active_requests(self) -> int:
    return len(self._running_requests) + len(self._pending_requests)

  def update_from_output(
      self,
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ):
    """Update the scheduler state from the output of the sampler.

    A request finishes when it samples one of its EOS tokens or reaches its
    `max_tokens_to_generate`. Aborted requests are discarded instead.
    """
    completed_reqs = []
    surviving_non_chunked_reqs = []
    surviving_chunked_reqs = []

    for i, req in enumerate(self._scheduled_requests):
      if req.is_aborted:
        self._kv_cache_manager.release_request(req)
        continue

      req_tokens = generated_tokens[i]

      if req.is_chunked_prefill:
        req.num_completed_tokens += req.num_in_flight_tokens
        req.num_in_flight_tokens = 0
        surviving_chunked_reqs.append(req)
        continue

      terminated = False
      truncated = False

      sampling_params = req.sampling_params

      valid_new_tokens = []
      for new_token in req_tokens:
        token_id = int(new_token)
        if token_id in sampling_params.eos_token_ids:
          terminated = True
          break
        valid_new_tokens.append(token_id)

      num_added = len(valid_new_tokens)
      n_generated = len(req.token_ids) + num_added - req.prompt_length

      n_overflow = n_generated - sampling_params.max_tokens_to_generate
      if n_overflow >= 0:
        # Reaching the budget exactly still finishes the request; only a
        # strict overflow has surplus tokens to drop.
        truncated = True

        if n_overflow > 0:
          valid_new_tokens = valid_new_tokens[:-n_overflow]
          num_added = len(valid_new_tokens)

      req.token_ids.extend(valid_new_tokens)

      if logprobs is not None:
        req.logprobs.extend(logprobs[i][:num_added])

      if logits is not None:
        req.logits.extend(logits[i][:num_added])

      # The request is not chunked prefill, so KVs are computed for all but the
      # last token.
      req.num_completed_tokens = len(req.token_ids) - 1
      req.num_in_flight_tokens = 0

      if terminated or truncated:
        self._kv_cache_manager.release_request(req)
        completed_reqs.append(req)
      else:
        surviving_non_chunked_reqs.append(req)

    # Move chunked prefill requests to the end of the running queue.
    # This prioritizes running decode requests over chunked prefill requests.
    self._running_requests = collections.deque(
        surviving_non_chunked_reqs + surviving_chunked_reqs
    )
    return completed_reqs

  def abort_request(self, req: request_lib.Request) -> None:
    """Aborts a request.

    The request is discarded, and its pages released, the next time the
    scheduler processes it.
    """
    req.is_aborted = True

  def _preempt(self):
    """Remove the newest request from the active batch."""
    preempted_request = self._running_requests.pop()
    preempted_request.num_in_flight_tokens = 0
    preempted_request.num_completed_tokens = 0

    self._kv_cache_manager.release_request(preempted_request)

    self._pending_requests.appendleft(preempted_request)

  def preempt_all(self) -> None:
    """Returns every running request to the pending queue.
    """
    while self._running_requests:
      self._preempt()
    self._scheduled_requests = tuple()

  def schedule_step(
      self, new_requests: list[request_lib.Request]
  ) -> tuple[tuple[request_lib.Request, ...], tuple[int, int, int]]:
    """Select the requests to sample in the next step, and fetch their pages.

    Requests are scheduled in order of arrival until the batch reaches either
    `max_num_batch_tokens` compute tokens or `max_seqs_per_batch` requests.

    Args:
      new_requests: New requests to schedule.

    Returns:
      A tuple of (scheduled_requests, distribution) where:
        - scheduled_requests: The list of scheduled requests ordered as
          [decodes, chunked_prefills, full_prefills].
        - distribution: A three tuple (i, j, k) where
          - i: The number of decode requests.
          - j: i + the number of chunked prefill requests.
          - k: j + the number of full prefill requests.
    """
    self._token_budget = self._config.max_num_batch_tokens
    self._queue_new_requests(new_requests)
    self._running_requests = self._discard_aborted(self._running_requests)
    self._pending_requests = self._discard_aborted(self._pending_requests)

    self._schedule_running_sequences()
    self._schedule_pending_sequences()

    assert len(self._running_requests) > 0 or self.num_active_requests == 0

    decodes = []
    prefills = []
    chunked = []

    for r in self._running_requests:
      if r.is_decode:
        decodes.append(r)
      elif r.is_chunked_prefill:
        chunked.append(r)
      else:
        prefills.append(r)

    i = len(decodes)
    j = i + len(chunked)
    k = j + len(prefills)

    # The RPA kernel expects [decodes, chunked, prefills] ordering.
    ordered_requests = tuple(decodes + chunked + prefills)
    self._scheduled_requests = ordered_requests
    distribution = (i, j, k)

    return ordered_requests, distribution

  def _queue_new_requests(self, new_requests: list[request_lib.Request]):
    """Insert new requests into the pending queue."""
    for req in new_requests:
      self._pending_requests.append(req)

  def _discard_aborted(
      self, queue: collections.deque[request_lib.Request]
  ) -> collections.deque[request_lib.Request]:
    """Releases the aborted requests in `queue`, and returns the rest."""
    kept = collections.deque()
    for req in queue:
      if req.is_aborted:
        self._kv_cache_manager.release_request(req)
      else:
        kept.append(req)
    return kept

  def _allocate_slots(self, req: request_lib.Request) -> bool:
    """Prepare a request for the next engine step."""
    if self._token_budget < 1:
      return False

    # The KV cache manager must advance its view of the request to the
    # current token count.
    self._kv_cache_manager.sync_request_state(req)

    if req.num_processed_tokens == 0:
      n_pages_hit, computed_pages = self._kv_cache_manager.get_computed_pages(
          req
      )
      n_tokens_hit = n_pages_hit * self._kv_cache_manager.page_size
    else:
      n_tokens_hit = 0
      computed_pages = None

    n_unprocessed_tokens = len(req.token_ids) - (
        req.num_processed_tokens + n_tokens_hit
    )

    if n_unprocessed_tokens == 0:
      raise RuntimeError(
          "Cannot schedule a request with no unprocessed tokens."
      )
    elif n_unprocessed_tokens == 1:
      req.is_decode = True
      req.is_chunked_prefill = False

      n_to_allocate = self._config.num_decode_steps
      n_to_schedule = 1
    elif (
        n_unprocessed_tokens > self._token_budget
        or n_unprocessed_tokens > self._config.chunked_prefill_length
    ):
      req.is_decode = False
      req.is_chunked_prefill = True

      if self._token_budget < self._config.chunked_prefill_length:
        return False

      n_to_allocate = self._config.chunked_prefill_length
      n_to_schedule = self._config.chunked_prefill_length
    else:
      req.is_decode = False
      req.is_chunked_prefill = False

      n_decode_steps = self._config.num_decode_steps - 1
      n_to_allocate = n_unprocessed_tokens + n_decode_steps
      n_to_schedule = n_unprocessed_tokens

    is_success = self._kv_cache_manager.allocate_slots(
        req,
        num_new_tokens=n_to_allocate,
        new_computed_pages=computed_pages,
    )

    if not is_success:
      return False

    req.num_completed_tokens += n_tokens_hit
    req.num_in_flight_tokens += n_to_schedule

    return True

  def _schedule_running_sequences(self):
    """Schedule current running sequences while device pages are available."""
    if len(self._running_requests) == 0:
      return

    n_running_admitted = 0
    while n_running_admitted < len(self._running_requests):
      if (
          self._token_budget <= 0
          or n_running_admitted >= self._config.max_seqs_per_batch
      ):
        break

      req = self._running_requests[n_running_admitted]

      is_success = self._allocate_slots(req)

      # If no tokens can be scheduled, preempt the latest running request, and
      # try again.
      if not is_success:
        self._preempt()
        continue

      self._token_budget -= req.num_in_flight_tokens
      n_running_admitted += 1

    if n_running_admitted == 0:
      raise RuntimeError(
          "No running requests could be scheduled."
      )

    # Preempt the remaining unscheduled requests if the token
    # budget is exceeded.
    while n_running_admitted < len(self._running_requests):
      self._preempt()

  def _schedule_pending_sequences(self):
    """Schedule pending sequences while device space is available."""
    while self._pending_requests:
      if (
          self._token_budget <= 0
          or len(self._running_requests) >= self._config.max_seqs_per_batch
      ):
        break

      req = self._pending_requests[0]

      # Allocate token slots for the request in the KV cache.
      is_success = self._allocate_slots(req)

      # If there was not enough space to schedule any tokens, no more requests
      # can be admitted.
      if not is_success:
        break

      req = self._pending_requests.popleft()
      self._token_budget -= req.num_in_flight_tokens

      self._running_requests.append(req)
