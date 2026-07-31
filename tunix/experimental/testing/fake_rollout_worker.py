# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A deterministic rollout worker for tests, with no sampler and no model.

Answers the per-trajectory rollout contract with results derived from the
request id, so a test gets the same tokens and reward on every run and can
assert on exact values. Requests can be made to fail or to stall on demand, so
callers that pool, retry, or time out have something to exercise those paths
against.

This is a behavior-bearing stand-in, not a lifecycle mock: it reports real
policy versions across weight syncs and returns failures in band the way the
contract requires.
"""

import asyncio
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.worker import rollout_worker

WorkerState = datatypes.WorkerState

RequestOrBatch = Union[
    datatypes.RolloutRequest, Sequence[datatypes.RolloutRequest]
]


class FakeRolloutWorker(rollout_worker.RolloutWorker):
  """Deterministic in-memory rollout worker."""

  def __init__(
      self,
      worker_id: str = "fake-rollout",
      *,
      fail_prompt_ids: Sequence[str] = (),
      stall_prompt_ids: Sequence[str] = (),
  ):
    """Initializes the worker.

    Args:
      worker_id: Identifier reported to the control plane.
      fail_prompt_ids: Prompt ids answered with an in-band failure.
      stall_prompt_ids: Prompt ids that block until `release()` is called.
    """
    super().__init__(worker_id=worker_id)
    self._policy_version = 0
    self._fail = set(fail_prompt_ids)
    self._stall = set(stall_prompt_ids)
    self._released = asyncio.Event()
    self._completed: asyncio.Queue[datatypes.RolloutResponse] = asyncio.Queue()
    self.seen_prompt_ids: list[str] = []
    self.sync_calls: list[Any] = []

  # --- Control plane --------------------------------------------------------

  def initialize(self) -> datatypes.Response:
    self.state = WorkerState.INITIALIZING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    self.state = WorkerState.READY
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    return datatypes.Response()

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state, policy_version=self._policy_version
    )

  # --- Test controls --------------------------------------------------------

  def release(self) -> None:
    """Unblocks every stalled request."""
    self._released.set()

  @property
  def policy_version(self) -> int:
    return self._policy_version

  # --- Data plane -----------------------------------------------------------

  async def generate(
      self,
      requests: RequestOrBatch,
      on_complete: Optional[
          Callable[[datatypes.RolloutResponse], None]
      ] = None,
  ) -> Union[datatypes.RolloutResponse, Sequence[datatypes.RolloutResponse]]:
    """Answers one request or a batch, streaming each result as it lands."""
    is_single = isinstance(requests, datatypes.RolloutRequest)
    batch = [requests] if is_single else list(requests)

    responses = []
    for request in batch:
      self.seen_prompt_ids.append(request.prompt_id)
      if request.prompt_id in self._stall:
        await self._released.wait()
      response = self._response_for(request)
      responses.append(response)
      self._completed.put_nowait(response)
      if on_complete is not None:
        on_complete(response)

    return responses[0] if is_single else responses

  async def pop_next_completed(self) -> datatypes.RolloutResponse:
    return await self._completed.get()

  def prepare_weight_sync(self, metadata: Any) -> datatypes.Response:
    del metadata  # Nothing in flight to fence in a synchronous fake.
    return datatypes.Response()

  def sync_weights(self, metadata: Any) -> int:
    self.sync_calls.append(metadata)
    version = getattr(metadata, "policy_version", None)
    self._policy_version = (
        int(version) if version is not None else self._policy_version + 1
    )
    return self._policy_version

  # --- Deterministic results ------------------------------------------------

  def _response_for(
      self, request: datatypes.RolloutRequest
  ) -> datatypes.RolloutResponse:
    if request.prompt_id in self._fail:
      return datatypes.RolloutResponse(
          request_id=request.request_id,
          status="FAILED",
          error=datatypes.ErrorInfo(
              error_type="RuntimeError",
              message=f"generation failed for {request.prompt_id!r}",
          ),
          policy_version=self._policy_version,
      )

    seed = sum(ord(c) for c in request.request_id or request.prompt_id)
    prompt_tokens = np.array([seed % 7 + 1, seed % 5 + 1], dtype=np.int32)
    completion = np.array([seed % 3 + 1, seed % 4 + 1], dtype=np.int32)
    return datatypes.RolloutResponse(
        request_id=request.request_id,
        status="SUCCEEDED",
        prompt_tokens=prompt_tokens,
        segments=[
            datatypes.TokenSegment(
                source="assistant",
                tokens=completion,
                loss_mask=np.ones_like(completion),
                logps=np.zeros_like(completion, dtype=np.float32),
            )
        ],
        env_reward=float(seed % 5),
        policy_version=self._policy_version,
    )
