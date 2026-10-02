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

"""Continuous batching scheduler for Tunix rollout requests."""

import collections
from collections.abc import Sequence
import dataclasses
import itertools

import numpy as np
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import request as request_lib


_PrefixMatch = Sequence[Sequence[kv_cache_manager_lib.Page | None]]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SchedulerConfig:
  """Configuration for the scheduler."""

  # The maximum number of tokens that can be scheduled in a single batch.
  max_num_batched_tokens: int
  # The maximum number of sequences that can be scheduled in a single batch.
  max_num_seqs: int
  # The size of a chunked prefill request in tokens. Must be a power of 2 and
  # less than or equal to max_num_batched_tokens.
  chunked_prefill_length: int
  # The number of model forward passes to execute per engine step.
  num_scheduler_steps: int = 1
  # A request finishes once it samples any of these tokens.
  eos_token_ids: frozenset[int] = frozenset()

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

    if self.chunked_prefill_length > self.max_num_batched_tokens:
      raise ValueError(
          "Chunked prefill length must be less than or equal to "
          f"max_num_batched_tokens. Got  {self.chunked_prefill_length} "
          f"and {self.max_num_batched_tokens}."
      )


class Scheduler:
  """Continuous batching scheduler.

  On each engine step, the scheduler picks which requests to run, subject to
  arrival order, the token budget, and KV cache availability, and fetches the
  pages those requests need onto the device. After the step, it updates the
  state of each scheduled request.
  """

  def __init__(
      self,
      config: SchedulerConfig,
      kv_cache_manager: kv_cache_manager_lib.KVCacheManager,
  ):
    self._config = config

    # Requests currently being processed with allocated KV cache slots.
    self._running_requests: collections.deque[request_lib.RequestState] = (
        collections.deque()
    )

    # Requests that have been submitted but are waiting for KV cache slots.
    self._pending_requests: collections.deque[request_lib.RequestState] = (
        collections.deque()
    )
    self._token_budget: int = 0
    self._has_in_flight: bool = False

    self._kv_cache_manager = kv_cache_manager

  @property
  def chunked_prefill_length(self) -> int:
    return self._config.chunked_prefill_length

  @property
  def num_active_requests(self) -> int:
    return sum(
        not req.is_done
        for req in itertools.chain(
            self._running_requests, self._pending_requests
        )
    )

  def advance_in_flight(
      self, scheduled: Sequence[request_lib.RequestState]
  ) -> None:
    """Marks `scheduled` as in flight on the device before scheduling the next step.

    Advances each request's completed KV token count and records how many tokens
    the in-flight step will sample, so `schedule_step` can plan the next step
    before the device finishes.

    Args:
      scheduled: The requests whose step was just launched on the device.
    """
    self._has_in_flight = True
    for req in scheduled:
      if req.is_chunked_prefill:
        req.num_completed_tokens += req.num_tokens_scheduled
        req.num_tokens_scheduled = 0
        req.num_in_flight_tokens = 0
      else:
        req.num_completed_tokens += (
            req.num_tokens_scheduled + self._config.num_scheduler_steps - 1
        )
        req.num_tokens_scheduled = 0
        req.num_in_flight_tokens = self._config.num_scheduler_steps

  def schedule_step(
      self, new_requests: list[request_lib.RequestState]
  ) -> tuple[tuple[request_lib.RequestState, ...], tuple[int, int, int]]:
    """Selects the next step's requests and fetches their pages to the device.

    Requests are scheduled in arrival order until all requests are scheduled,
    the batch reaches `max_num_seqs` requests, the token budget is
    exhausted, or there is insufficient KV cache memory for a request.

    Args:
      new_requests: New requests to schedule.

    Returns:
      A tuple of (scheduled_requests, distribution) where:
        - scheduled_requests: The scheduled requests ordered as
          [decodes, chunked_prefills, full_prefills].
        - distribution: A three tuple (i, j, k) where
          - i: The number of decode requests.
          - j: i + the number of chunked prefill requests.
          - k: j + the number of full prefill requests.
    """
    self._token_budget = self._config.max_num_batched_tokens
    self._pending_requests.extend(new_requests)
    self._running_requests = self._drop_done(self._running_requests)
    self._pending_requests = self._drop_done(self._pending_requests)

    scheduled = self._schedule_running_sequences()
    scheduled += self._schedule_pending_sequences(
        self._config.max_num_seqs - len(scheduled)
    )

    decodes = [r for r in scheduled if r.is_decode]
    chunked = [r for r in scheduled if r.is_chunked_prefill]
    prefills = [
        r for r in scheduled if not r.is_decode and not r.is_chunked_prefill
    ]

    i = len(decodes)
    j = i + len(chunked)
    k = j + len(prefills)

    # The RPA kernel expects [decodes, chunked, prefills] ordering.
    return tuple(decodes + chunked + prefills), (i, j, k)

  def update_from_output(
      self,
      requests: Sequence[request_lib.RequestState],
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ) -> list[request_lib.RequestState]:
    """Updates request state based on a sampler output.

    A request finishes when it samples one of the configured EOS tokens or
    reaches its `max_tokens`.

    Args:
      requests: The requests to update. Row `i` of each output belongs to
        `requests[i]`.
      generated_tokens: Sampled tokens, one row per request.
      logits: Optional logits, one row per request.
      logprobs: Optional logprobs, one row per request.

    Returns:
      The requests that finished this step.
    """
    for name, rows in (
        ("generated_tokens", generated_tokens),
        ("logits", logits),
        ("logprobs", logprobs),
    ):
      if rows is not None and len(rows) != len(requests):
        raise ValueError(
            f"{name} has {len(rows)} rows for {len(requests)} requests."
        )

    in_flight_step = self._has_in_flight
    self._has_in_flight = False

    completed_reqs = []
    for i, req in enumerate(requests):
      in_flight = req.num_in_flight_tokens
      req.num_in_flight_tokens = 0
      if req.is_done:
        continue  # Aborted mid-step, or already finished.

      if in_flight_step:
        if in_flight == 0:
          continue
      else:
        if req.num_tokens_scheduled == 0:
          raise ValueError(f"Request {req.request_id} has no scheduled tokens.")
        if req.is_chunked_prefill:
          req.num_completed_tokens += req.num_tokens_scheduled
          req.num_tokens_scheduled = 0
          continue

      was_preempted = req.status == request_lib.RequestStatus.PENDING
      valid_new_tokens = []
      for new_token in generated_tokens[i]:
        token_id = int(new_token)
        valid_new_tokens.append(token_id)
        if token_id in self._config.eos_token_ids:
          req.status = request_lib.RequestStatus.FINISHED_EOS
          break

      n_generated = (
          len(req.token_ids) + len(valid_new_tokens) - req.prompt_length
      )
      n_overflow = n_generated - req.sampling_params.max_tokens
      if n_overflow > 0:
        # The limit is hit before the last token, so any EOS is dropped too.
        valid_new_tokens = valid_new_tokens[:-n_overflow]
        req.status = request_lib.RequestStatus.FINISHED_LENGTH
      elif n_overflow == 0 and not req.is_done:
        req.status = request_lib.RequestStatus.FINISHED_LENGTH

      num_added = len(valid_new_tokens)
      req.token_ids.extend(valid_new_tokens)
      if logprobs is not None:
        req.logprobs.extend(logprobs[i][:num_added])
      if logits is not None:
        req.logits.extend(logits[i][:num_added])

      if not in_flight_step or (req.is_done and not was_preempted):
        # The request is not chunked prefill, so KVs are computed for all but
        # the last token.
        req.num_completed_tokens = len(req.token_ids) - 1
        req.num_tokens_scheduled = 0

      if req.is_done:
        completed_reqs.append(req)

    return completed_reqs

  def drop_completed(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
  ) -> tuple[tuple[request_lib.RequestState, ...], tuple[int, int, int]]:
    """Drops finished or aborted requests from the queues and `scheduled`.

    Releases the KV cache pages of every dropped request.

    Args:
      scheduled: The pre-scheduled batch `[decodes, chunked, prefills]`.
      distribution: The `(i, j, k)` cumulative counts for `scheduled`.

    Returns:
      The filtered `(scheduled, distribution)` with done requests removed.
    """
    self._running_requests = self._drop_done(self._running_requests)
    self._pending_requests = self._drop_done(self._pending_requests)

    i, j, k = distribution
    decodes = [r for r in scheduled[:i] if not r.is_done]
    chunked = [r for r in scheduled[i:j] if not r.is_done]
    prefills = [r for r in scheduled[j:k] if not r.is_done]

    new_i = len(decodes)
    new_j = new_i + len(chunked)
    new_k = new_j + len(prefills)
    return tuple(decodes + chunked + prefills), (new_i, new_j, new_k)

  def abort_request(self, req: request_lib.RequestState) -> None:
    """Aborts `req`. Has no effect if `req` is already done."""
    if not req.is_done:
      req.status = request_lib.RequestStatus.ABORTED

  def preempt_all(self) -> None:
    """Returns every running request to the pending queue."""
    self._has_in_flight = False
    self._running_requests = self._drop_done(self._running_requests)
    while self._running_requests:
      self._preempt()
    for req in self._pending_requests:
      req.num_in_flight_tokens = 0

  def _drop_done(
      self, queue: collections.deque[request_lib.RequestState]
  ) -> collections.deque[request_lib.RequestState]:
    """Drops done requests from `queue`, and releases their pages."""
    kept = collections.deque()
    for req in queue:
      if req.is_done:
        self._kv_cache_manager.release_request(req)
      else:
        kept.append(req)
    return kept

  def _preempt(self) -> None:
    """Moves the most recently admitted request to the front of pending."""
    req = self._running_requests.pop()
    req.num_tokens_scheduled = 0
    req.num_completed_tokens = 0
    req.status = request_lib.RequestStatus.PENDING

    self._kv_cache_manager.release_request(req)

    self._pending_requests.appendleft(req)

  def _schedule_running_sequences(self) -> list[request_lib.RequestState]:
    """Schedules running requests.

    Determines which running requests to schedule next, based on token budget
    and KV cache availability. Also allocates KV cache slots, and fetches KV
    cache pages onto the device for each scheduled request.

    Returns:
      The scheduled running requests.
    """
    had_running = False
    scheduled = []
    idx = 0
    while (
        idx < len(self._running_requests)
        and len(scheduled) < self._config.max_num_seqs
        and self._token_budget > 0
    ):
      req = self._running_requests[idx]
      if (
          req.num_total_tokens - req.prompt_length
          >= req.sampling_params.max_tokens
      ):
        idx += 1
        continue
      had_running = True
      n_tokens_hit, computed_pages = self._get_prefix_hit(req)
      n_unprocessed = req.num_total_tokens - (
          req.num_processed_tokens + n_tokens_hit
      )
      n_tokens_to_schedule = min(
          n_unprocessed, self._config.chunked_prefill_length
      )

      if n_tokens_to_schedule > self._token_budget:
        # Since only token budget is limiting, there is no benefit to
        # preempting. Instead skip to the next request so that the token budget
        # is better utilized.
        idx += 1
        continue

      # Out of KV cache. Preempt the newest request, which may be `req`.
      if not self._allocate_slots(
          req, n_tokens_to_schedule, n_tokens_hit, computed_pages
      ):
        self._preempt()
        continue

      self._token_budget -= n_tokens_to_schedule
      scheduled.append(req)
      idx += 1

    if had_running and not scheduled:
      raise RuntimeError("No running requests could be scheduled.")
    return scheduled

  def _schedule_pending_sequences(
      self, max_new_requests: int
  ) -> list[request_lib.RequestState]:
    """Schedules pending requests.

    Determines which pending requests to schedule next, based on token budget
    and KV cache availability. Also allocates KV cache slots, and fetches KV
    cache pages onto the device for each scheduled request.

    Args:
      max_new_requests: The maximum number of requests to schedule.

    Returns:
      The scheduled pending requests.
    """
    scheduled = []
    while self._pending_requests and len(scheduled) < max_new_requests:
      req = self._pending_requests[0]
      if req.num_in_flight_tokens > 0:
        break
      n_tokens_hit, computed_pages = self._get_prefix_hit(req)
      n_unprocessed = req.num_total_tokens - (
          req.num_processed_tokens + n_tokens_hit
      )
      n_tokens_to_schedule = min(
          n_unprocessed, self._config.chunked_prefill_length
      )

      if n_tokens_to_schedule > self._token_budget:
        # Skipping to the next request would allow requests to jump position
        # in the queue. Break instead.
        break

      if not self._allocate_slots(
          req, n_tokens_to_schedule, n_tokens_hit, computed_pages
      ):
        break

      self._pending_requests.popleft()
      req.status = request_lib.RequestStatus.RUNNING
      self._running_requests.append(req)
      self._token_budget -= n_tokens_to_schedule
      scheduled.append(req)
    return scheduled

  def _get_prefix_hit(
      self, req: request_lib.RequestState
  ) -> tuple[int, _PrefixMatch | None]:
    """Syncs `req` with the KV cache, and returns its prefix-cache hit.

    Args:
      req: The request to look up a prefix-cache hit for.

    Returns:
      (n_tokens_hit, computed_pages), to be passed to `_allocate_slots`.
    """
    # The KV cache manager must advance its view of the request to the current
    # token count.
    self._kv_cache_manager.sync_request_state(req)
    if req.num_processed_tokens > 0:
      return 0, None
    n_pages_hit, computed_pages = self._kv_cache_manager.get_computed_pages(req)
    return n_pages_hit * self._kv_cache_manager.page_size, computed_pages

  def _allocate_slots(
      self,
      req: request_lib.RequestState,
      n_tokens_to_schedule: int,
      n_tokens_hit: int,
      computed_pages: _PrefixMatch | None,
  ) -> bool:
    """Allocates KV slots for `req`'s next step. Ignores the token budget.

    Request state is only updated if the allocation succeeds.

    Args:
      req: The request to allocate slots for.
      n_tokens_to_schedule: The number of tokens to compute next step.
      n_tokens_hit: The number of tokens served by the prefix cache.
      computed_pages: The prefix-cache pages, from `_get_prefix_hit`.

    Returns:
      Whether the allocation succeeded.
    """
    if n_tokens_to_schedule == 0:
      raise RuntimeError("Cannot schedule a request with no unprocessed tokens.")

    n_tokens_remaining = req.num_total_tokens - (
        req.num_processed_tokens + n_tokens_hit + n_tokens_to_schedule
    )
    is_chunked_prefill = n_tokens_remaining > 0
    is_decode = not is_chunked_prefill and n_tokens_to_schedule == 1

    if is_chunked_prefill:
      # Tokens remain to be computed after this step, so no token can be
      # decoded yet. Only the chunk itself needs slots.
      n_to_allocate = n_tokens_to_schedule
    else:
      # Decodes and full prefills sample, and reserve slots for the remaining
      # scheduler steps.
      n_to_allocate = (
          n_tokens_to_schedule + self._config.num_scheduler_steps - 1
      )

    if not self._kv_cache_manager.allocate_slots(
        req, num_new_tokens=n_to_allocate, new_computed_pages=computed_pages
    ):
      return False

    req.is_decode = is_decode
    req.is_chunked_prefill = is_chunked_prefill
    req.num_completed_tokens += n_tokens_hit
    req.num_tokens_scheduled += n_tokens_to_schedule
    return True
