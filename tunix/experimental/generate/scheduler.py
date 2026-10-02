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
import dataclasses

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
  # The number of model forward passes to execute per engine step.
  num_scheduler_steps: int = 1

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

  @property
  def chunked_prefill_length(self) -> int:
    return self._config.chunked_prefill_length

  @property
  def num_active_requests(self) -> int:
    return len(self._running_requests) + len(self._pending_requests)

  def preempt_all(self) -> None:
    """Returns every running request to the pending queue."""
    while self._running_requests:
      self._preempt()

  def _preempt(self) -> None:
    """Moves the most recently admitted request to the front of pending."""
    req = self._running_requests.pop()
    req.num_tokens_scheduled = 0
    req.num_completed_tokens = 0
    req.status = request_lib.RequestStatus.PENDING

    self._kv_cache_manager.release_request(req)

    self._pending_requests.appendleft(req)
