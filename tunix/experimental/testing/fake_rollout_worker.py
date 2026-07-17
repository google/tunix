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

"""A contract-conformant, deterministic fake RolloutWorker.

Implements the full ticketed cursor-read channel (generate/get_completed/ack/
cancel) plus the weight-sync fence over deterministic golden results, with no
real sampler or model. It lets the orchestrator (and the RolloutWorker contract
suite) run before the real RolloutWorker lands.
"""

from typing import Sequence

import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.worker import rollout_worker


class FakeRolloutWorker(rollout_worker.RolloutWorker):
  """Deterministic in-memory RolloutWorker for tests and as an executable spec."""

  def __init__(self, worker_id: str = "fake-rollout"):
    super().__init__(worker_id=worker_id)
    self._running = False
    self._version = 0
    self._ticket_counter = 0
    # ticket -> list of mutable [seq, TrajectoryResult] entries (un-acked).
    self._pending: dict[str, list] = {}
    # ticket -> total results ever produced (max seq), unaffected by ack/GC.
    self._max_seq: dict[str, int] = {}

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
        inflight=sum(len(entries) for entries in self._pending.values()),
        policy_version=self._version,
    )

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  async def generate(
      self, requests: Sequence[datatypes.TrajectoryRequest]
  ) -> str:
    self._ticket_counter += 1
    ticket = f"{self.worker_id}/ticket-{self._ticket_counter}"
    entries = [[i + 1, self._golden_result(req)] for i, req in enumerate(requests)]
    self._pending[ticket] = entries
    self._max_seq[ticket] = len(entries)
    return ticket

  async def get_completed(
      self,
      ticket: str,
      after_seq: int,
      max_items: int = 16,
      wait_s: float = 0.0,
  ) -> datatypes.CompletedPage:
    del wait_s  # Results are ready immediately; no long-poll needed.
    self._require_ticket(ticket)
    fresh = [
        (seq, result)
        for seq, result in self._pending[ticket]
        if seq > after_seq
    ][:max_items]
    last_seq = fresh[-1][0] if fresh else after_seq
    return datatypes.CompletedPage(
        results=[result for _, result in fresh],
        last_seq=last_seq,
        done=last_seq >= self._max_seq[ticket],
    )

  async def ack(self, ticket: str, upto_seq: int) -> None:
    self._require_ticket(ticket)
    self._pending[ticket] = [
        entry for entry in self._pending[ticket] if entry[0] > upto_seq
    ]

  async def cancel(
      self, ticket: str, request_ids: Sequence[str] | None = None
  ) -> None:
    self._require_ticket(ticket)
    targets = None if request_ids is None else set(request_ids)
    for entry in self._pending[ticket]:
      result = entry[1]
      if targets is None or result.request_id in targets:
        entry[1] = datatypes.TrajectoryResult(
            request_id=result.request_id,
            prompt_id=result.prompt_id,
            status="CANCELLED",
        )

  def prepare_weight_sync(self, meta: datatypes.WeightSyncMetadata) -> None:
    del meta  # No in-flight episodes to fence in the fake.

  def sync_weights(self, meta: datatypes.WeightSyncMetadata) -> int:
    self._version = meta.version
    return self._version

  def _require_ticket(self, ticket: str) -> None:
    if ticket not in self._pending:
      raise rollout_worker.TicketNotFound(ticket)

  def _golden_result(
      self, req: datatypes.TrajectoryRequest
  ) -> datatypes.TrajectoryResult:
    """Deterministic result derived from the request id (stable across runs)."""
    seed = sum(ord(c) for c in req.request_id) if req.request_id else 0
    prompt_tokens = np.array([seed % 7 + 1, seed % 5 + 1], dtype=np.int32)
    completion = np.array([seed % 3 + 1, seed % 4 + 1], dtype=np.int32)
    return datatypes.TrajectoryResult(
        request_id=req.request_id,
        prompt_id=req.prompt_id,
        status="SUCCEEDED",
        prompt_tokens=prompt_tokens,
        segments=[
            datatypes.TokenSegment(
                source="assistant",
                tokens=completion,
                loss_mask=np.ones_like(completion),
            )
        ],
        env_reward=float(seed % 5),
        policy_version=self._version,
    )
