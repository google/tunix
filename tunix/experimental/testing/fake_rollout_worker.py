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

"""A deterministic in-memory RolloutWorker fake for orchestrator tests.

Implements the RolloutWorker interface as it currently stands in main (a batched
`generate` with an optional `on_complete` callback) plus the extended Worker
lifecycle/health/info, so the orchestrator can register and drive a rollout role
without a real sampler. The real RolloutWorker interface is owned by its sibling
plan; this fake deliberately tracks main and returns wire `RolloutResult`s.
"""

from typing import Callable, Sequence

import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.worker import rollout_worker


class FakeRolloutWorker(rollout_worker.RolloutWorker):
  """Deterministic in-memory RolloutWorker for tests."""

  def __init__(self, worker_id: str = "fake-rollout"):
    super().__init__(worker_id=worker_id)
    self._running = False
    self._version = 0

  def initialize(self) -> None:
    pass

  def compile(self, shape_config: datatypes.ShapeConfig) -> None:
    del shape_config

  def start(self) -> None:
    self._running = True

  def stop(self) -> None:
    self._running = False

  def health(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state="READY" if self._running else "STOPPED",
        policy_version=self._version,
    )

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  async def generate(
      self,
      requests: datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest],
      on_complete: Callable[[datatypes.RolloutResult], None] | None = None,
  ) -> datatypes.RolloutResult | Sequence[datatypes.RolloutResult]:
    single = isinstance(requests, datatypes.RolloutRequest)
    reqs = [requests] if single else list(requests)
    results = [self._golden_result(r) for r in reqs]
    if on_complete is not None:
      for result in results:
        on_complete(result)
    return results[0] if single else results

  def prepare_weight_sync(self, metadata) -> None:
    del metadata

  def sync_weights(self, metadata) -> int:
    del metadata
    self._version += 1
    return self._version

  def _golden_result(
      self, req: datatypes.RolloutRequest
  ) -> datatypes.RolloutResult:
    completion = np.array([3, 4, 5], dtype=np.int32)
    segment = datatypes.TokenSegment(
        source="assistant",
        tokens=completion,
        loss_mask=np.ones_like(completion),
    )
    return datatypes.RolloutResult(
        request_id=req.request_id,
        prompt_id=req.prompt_id,
        status="COMPLETED",
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        segments=[segment],
        env_reward=1.0,
        policy_version=self._version,
    )
