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
import functools
import itertools

from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import request as request_lib


_PrefixMatch = Sequence[Sequence[kv_cache_manager_lib.Page | None]]


def _has_final_tokens_in_flight(req: request_lib.RequestState) -> bool:
  """Whether every token `req` may still generate is being sampled in flight."""
  return (
      req.num_total_tokens - req.prompt_length
      >= req.sampling_params.max_tokens
  )


def _is_skipped(req: request_lib.RequestState) -> bool:
  """Whether the admitted `req` keeps its place but sits out the next step.

  It sits out while its final tokens are in flight, or while it waits for the
  stale output of a step it was preempted from, unless it runs a chunked
  prefill, which only reads tokens that already exist.

  Args:
    req: A request admitted to the running queue this step.
  """
  return _has_final_tokens_in_flight(req) or (
      req.num_stale_output_tokens > 0 and not req.is_chunked_prefill
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SchedulerConfig:
  """Configuration for the scheduler."""

  # The maximum number of tokens that can be scheduled in a single batch.
  max_num_batched_tokens: int
  # The maximum number of sequences that can be scheduled in a single batch.
  max_num_seqs: int
  # The maximum size of a chunked prefill request in tokens.
  max_chunked_prefill_length: int
  # The number of model forward passes to execute per engine step.
  num_scheduler_steps: int = 1

  def __post_init__(self):
    if self.max_chunked_prefill_length <= 0:
      raise ValueError(
          "max_chunked_prefill_length must be positive. Got"
          f" {self.max_chunked_prefill_length}."
      )

    if self.max_chunked_prefill_length > self.max_num_batched_tokens:
      raise ValueError(
          "max_chunked_prefill_length must be less than or equal to "
          f"max_num_batched_tokens. Got {self.max_chunked_prefill_length} "
          f"and {self.max_num_batched_tokens}."
      )


class Scheduler:
  """Continuous batching scheduler.

  On each engine step, the scheduler picks which requests to run, subject to
  arrival order, the token budget, and KV cache availability, and fetches the
  pages those requests need onto the device.
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

    self._kv_cache_manager = kv_cache_manager

  @functools.cached_property
  def chunked_prefill_length(self) -> int:
    # Chunked prefill length must be a power of two.
    return 1 << (self._config.max_chunked_prefill_length.bit_length() - 1)

  @property
  def num_active_requests(self) -> int:
    return sum(
        not req.is_done
        for req in itertools.chain(
            self._running_requests, self._pending_requests
        )
    )

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
    self._pending_requests.extend(new_requests)
    self._running_requests = self._drop_done(self._running_requests)
    self._pending_requests = self._drop_done(self._pending_requests)

    self._schedule_requests()

    # Skipped requests hold their place in the running queue, but sit out
    # this step.
    runnable = [r for r in self._running_requests if not _is_skipped(r)]
    decodes = [r for r in runnable if r.is_decode]
    chunked = [r for r in runnable if r.is_chunked_prefill]
    prefills = [
        r for r in runnable if not r.is_decode and not r.is_chunked_prefill
    ]

    i = len(decodes)
    j = i + len(chunked)
    k = j + len(prefills)

    # The RPA kernel expects [decodes, chunked, prefills] ordering.
    return tuple(decodes + chunked + prefills), (i, j, k)

  def drop_completed(self) -> None:
    """Drops finished or aborted requests from the queues."""
    self._running_requests = self._drop_done(self._running_requests)
    self._pending_requests = self._drop_done(self._pending_requests)

  def abort_request(self, req: request_lib.RequestState) -> None:
    """Aborts `req`. Has no effect if `req` is already done."""
    if not req.is_done:
      req.status = request_lib.RequestStatus.ABORTED

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

  def _schedule_requests(self) -> None:
    """Schedules requests from the running and pending queues."""
    requests = list(self._running_requests) + list(self._pending_requests)
    new_running_requests = []
    self._token_budget = self._config.max_num_batched_tokens

    n_visited = 0
    # Skipped requests take no batch row, so only runnable requests count
    # toward `max_num_seqs`.
    n_runnable = 0
    while (
        n_visited < len(requests) and n_runnable < self._config.max_num_seqs
    ):
      req = requests[n_visited]
      scheduled = self._try_schedule_request(req, len(new_running_requests))
      if not scheduled:
        break

      n_visited += 1
      new_running_requests.append(req)
      if not _is_skipped(req):
        n_runnable += 1

    while len(self._running_requests) > len(new_running_requests):
      self._preempt()

    self._running_requests = collections.deque(new_running_requests)
    self._pending_requests = collections.deque(requests[n_visited:])

  def _try_schedule_request(
      self,
      req: request_lib.RequestState,
      n_admitted_requests: int,
  ) -> bool:
    """Tries to schedule `req` for the next step."""
    if _has_final_tokens_in_flight(req):
      # Every token `req` may still generate is already being sampled in
      # flight, so it finishes once that output arrives. Keep its pages, but
      # skip it rather than allocate slots for tokens it will never generate.
      self._skip(req)
      return True

    n_tokens_hit, computed_pages = self._get_prefix_hit(req)
    is_decode, is_chunked_prefill, n_tokens_to_schedule, n_to_allocate = (
        self._get_n_tokens_to_schedule(req, n_tokens_hit)
    )

    if req.num_stale_output_tokens > 0 and not is_chunked_prefill:
      # A decode or full prefill would sample the positions that the in-flight
      # output is about to fill, e.g. for a request preempted earlier in this
      # same pass. Keep its place in the queue so later requests cannot jump
      # ahead of it, but skip it until that output arrives. A chunked prefill
      # only reads tokens that already exist, so it can run.
      self._skip(req)
      return True

    if n_tokens_to_schedule > self._token_budget:
      return False

    while True:
      allocated = self._kv_cache_manager.allocate_slots(
          req,
          num_new_tokens=n_to_allocate,
          new_computed_pages=computed_pages,
      )

      if allocated:
        break

      # If there was a request scheduled in the previous step, and it arrived
      # after the current request, preempt it to free up KV cache slots.
      can_preempt = len(self._running_requests) > 1 + n_admitted_requests
      if can_preempt:
        self._preempt()
      else:
        return False

    self._token_budget -= n_tokens_to_schedule
    req.is_decode = is_decode
    req.is_chunked_prefill = is_chunked_prefill

    req.status = request_lib.RequestStatus.RUNNING
    req.num_computed_tokens += n_tokens_hit
    return True

  def _skip(self, req: request_lib.RequestState) -> None:
    """Keeps `req`'s place in the running queue, but leaves it out of the step."""
    req.is_decode = False
    req.is_chunked_prefill = False

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
    if req.num_computed_tokens > 0:
      return 0, None

    n_pages_hit, computed_pages = self._kv_cache_manager.get_computed_pages(req)
    return n_pages_hit * self._kv_cache_manager.page_size, computed_pages

  def _get_n_tokens_to_schedule(
      self,
      req: request_lib.RequestState,
      n_tokens_hit: int,
  ) -> tuple[bool, bool, int, int]:
    """Returns (is_decode, is_chunked_prefill, n_tokens_to_schedule, n_to_allocate)."""

    n_unprocessed = req.num_total_tokens - (
        req.num_computed_tokens + n_tokens_hit
    )

    if n_unprocessed == 1:
      is_decode = True
      is_chunked_prefill = False
      n_tokens_to_schedule = 1
      n_to_allocate = self._config.num_scheduler_steps
    elif n_unprocessed > self.chunked_prefill_length:
      is_decode = False
      is_chunked_prefill = True
      n_tokens_to_schedule = self.chunked_prefill_length
      n_to_allocate = self.chunked_prefill_length
    else:
      is_decode = False
      is_chunked_prefill = False
      n_tokens_to_schedule = n_unprocessed
      n_to_allocate = n_unprocessed + self._config.num_scheduler_steps - 1

    return is_decode, is_chunked_prefill, n_tokens_to_schedule, n_to_allocate

  def preempt_all(self) -> None:
    """Returns every running request to the pending queue."""
    self._running_requests = self._drop_done(self._running_requests)
    while self._running_requests:
      self._preempt()
    for req in self._pending_requests:
      req.num_in_flight_tokens = 0
      # The caller discards the in-flight step, so its output never arrives.
      req.num_stale_output_tokens = 0

  def _preempt(self) -> None:
    """Moves the most recently admitted request to the front of pending."""
    req = self._running_requests.pop()
    self._kv_cache_manager.release_request(req)

    req.num_computed_tokens = 0
    # Tokens an in-flight step is sampling are still appended when it
    # returns, so the request waits for them before it samples again.
    req.num_stale_output_tokens += req.num_in_flight_tokens
    req.num_in_flight_tokens = 0
    req.is_decode = False
    req.is_chunked_prefill = False
    req.status = request_lib.RequestStatus.PENDING

    self._pending_requests.appendleft(req)
