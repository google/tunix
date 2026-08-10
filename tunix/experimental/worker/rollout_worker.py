# Copyright 2026 Google LLC
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

"""Top-level RolloutWorker abstractions (Service vs Client Driver)."""

import asyncio
import dataclasses
from typing import Any, AsyncIterator, Callable, List, Optional, Sequence, Union
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.worker import abstract_worker
from tunix.rl.rollout import base_rollout


@dataclasses.dataclass
class RolloutConfig(base_rollout.RolloutConfig):
  """Rollout configuration extending base RolloutConfig with sampler choice and registry options.

  Attributes:
    sampler_type: Type of sampler adapter to construct ("vanilla",
      "legacy_vllm", "vllm").
    env_name: Registered name of environment class in ENV_REGISTRY.
    agent_name: Registered name of agent class in AGENT_REGISTRY.
    env_config: Configuration dictionary passed to environment constructor.
    agent_config: Configuration dictionary passed to agent constructor.
  """

  sampler_type: str = "vanilla"
  env_name: str = ""
  agent_name: str = ""
  env_config: dict[str, Any] = dataclasses.field(default_factory=dict)
  agent_config: dict[str, Any] = dataclasses.field(default_factory=dict)


TrajectoryOrError = Union[
    trajectory_lib.Trajectory, trajectory_lib.TrajectoryError
]

WorkerState = datatypes.WorkerState


class RolloutWorker(abstract_worker.Worker):
  """Worker wrapper for rollout collection.

  Encapsulates RolloutManager and executes concurrent episode loops
  locally on its remote CPU host.
  """

  def __init__(
      self,
      worker_id: str,
      config: Optional[RolloutConfig] = None,
      sampler: Optional[sampler_lib.Sampler] = None,
      env_pool: Any = None,
      agent_factory: Optional[Callable[[], Any]] = None,
      max_concurrency: int = 64,
      tokenizer: Any = None,
      chat_parser: Any = None,
  ):
    super().__init__()
    self.worker_id = worker_id
    self.config = config
    if tokenizer is None or chat_parser is None:
      raise ValueError(
          "RolloutWorker requires valid tokenizer and chat_parser arguments"
          " (none can be None)."
      )
    # The AUTHORITATIVE round tracker: the coordinator reconciles failed
    # phase RPCs against THIS, not the sampler's internal sub-state. Its
    # terminal "committed" is composite: publish + admission reopened +
    # worker READY. The phase lock makes admit->manager->complete atomic --
    # a timed-out old post interleaving with abort/retry could otherwise
    # record a published round as aborted.
    self._round_tracker = weight_sync_coordinator.WorkerRoundTracker()
    self._phase_lock = asyncio.Lock()
    self.manager = manager_lib.RolloutManager(
        config=config,
        sampler=sampler,
        env_pool=env_pool,
        agent_factory=agent_factory,
        max_concurrency=max_concurrency,
        tokenizer=tokenizer,
        chat_parser=chat_parser,
    )

  @property
  def sampler(self) -> Any:
    return self.manager.sampler

  def get_worker_id(self) -> str:
    """Returns the unique worker ID."""
    return self.worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  def initialize(self) -> datatypes.Response:
    self.state = WorkerState.INITIALIZING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any) -> datatypes.Response:
    self.state = WorkerState.COMPILING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    self.manager.cancel_all()
    return datatypes.Response()

  def pause(self) -> datatypes.Response:
    self.manager.pause_all()
    return datatypes.Response()

  def resume(self) -> datatypes.Response:
    self.manager.resume_all()
    return datatypes.Response()

  def _infer_shapes(self) -> Any:
    return None

  def _compile_with_shapes(self, abstract_state: Any) -> None:
    pass

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)

  def _to_rollout_response(
      self,
      item: Any,
      request_id: str = "",
      prompt_tokens: np.ndarray | None = None,
      policy_version: int = 0,
  ) -> datatypes.RolloutResponse:
    """Converts internal Trajectory or TrajectoryError to wire-safe RolloutResponse."""
    if isinstance(item, datatypes.RolloutResponse):
      return item
    if isinstance(item, trajectory_lib.TrajectoryError):
      return datatypes.RolloutResponse(
          request_id=request_id
          or getattr(item, "trajectory_id", "")
          or getattr(item, "prompt_id", ""),
          status="ERROR",
          error=item.error_message,  # pyrefly: ignore[bad-argument-type]
          prompt_tokens=(
              prompt_tokens
              if prompt_tokens is not None
              else np.zeros(0, dtype=np.int32)
          ),
          policy_version=policy_version,
      )
    if isinstance(item, trajectory_lib.Trajectory):
      req_id = request_id or getattr(item, "trajectory_id", "default")
      return datatypes.RolloutResponse.from_trajectory(
          request_id=req_id,
          traj=item,  # pyrefly: ignore[bad-argument-type]
          prompt_tokens=(
              prompt_tokens
              if prompt_tokens is not None
              else np.zeros(0, dtype=np.int32)
          ),
          policy_version=policy_version,
      )
    return item

  async def generate(
      self,
      requests: (
          datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest] | Any
      ),
      on_complete: Optional[Callable[[datatypes.RolloutResponse], None]] = None,
  ) -> datatypes.RolloutResponse | List[datatypes.RolloutResponse] | Any:
    """Coroutine method for single or batched generate requests."""
    cb = None
    if on_complete is not None:
      cb = lambda item: on_complete(self._to_rollout_response(item))
    res = await self.manager.generate(requests, on_complete=cb)
    if isinstance(res, (list, tuple)):
      return [self._to_rollout_response(r) for r in res]
    return self._to_rollout_response(res)

  async def pop_next_completed(self) -> datatypes.RolloutResponse | Any:
    """Pull-based stream: yields whichever trajectory finishes first out-of-order."""
    res = await self.manager.pop_next_completed()
    return self._to_rollout_response(res)

  async def as_completed_stream(
      self,
  ) -> AsyncIterator[datatypes.RolloutResponse | Any]:
    """Async stream yielding completed trajectories or errors strictly out-of-order."""
    async for res in self.manager.as_completed_stream():
      yield self._to_rollout_response(res)

  async def bind_weight_sync(self, **kwargs) -> Any:
    """Builds/binds the sampler's transport synchronizer. No downtime."""
    return await self.manager.bind_weight_sync(**kwargs)

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Transport metadata, one entry per physical host. No downtime."""
    return await self.manager.get_weight_sync_metadata(**kwargs)

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Quiesces the worker; it stays SYNCING until post/abort completes.

    Restoring READY in a finally here (the previous behavior) reported a
    drained, cache-less worker as healthy while it could not serve.
    """
    async with self._phase_lock:
      if not self._round_tracker.admit(sync_request, "prepared"):
        return None
      self.state = WorkerState.SYNCING
      res = await self.manager.pre_weight_sync(sync_request, **kwargs)
      self._round_tracker.complete(sync_request, "prepared")
      return res

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Applies received bytes into the STAGING copy only. Still SYNCING."""
    async with self._phase_lock:
      if not self._round_tracker.admit(sync_request, "h2d_done"):
        return None
      if self.state != WorkerState.SYNCING:
        raise RuntimeError(
            f"weight_sync in state {self.state}; pre_weight_sync must run first"
        )
      res = await self.manager.weight_sync(sync_request, **kwargs)
      self._round_tracker.complete(sync_request, "h2d_done")
      return res

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """COMPOSITE terminal: "committed" is recorded only after the sampler

    published, admission reopened, AND this worker is READY.
    """
    async with self._phase_lock:
      if not self._round_tracker.admit(sync_request, "committed"):
        return None
      # Phase ORDER, not just phase identity: publishing a round whose H2D
      # never completed would publish stale staging and then record it as
      # committed.
      report = self._round_tracker.report()
      if report.get("phase") != "h2d_done":
        raise RuntimeError(
            f"post_weight_sync with round phase {report.get('phase')!r};"
            " weight_sync (H2D) must complete first"
        )
      res = await self.manager.post_weight_sync(sync_request, **kwargs)
      if self.state == WorkerState.SYNCING:  # a concurrent stop() wins
        self.state = WorkerState.READY
      if self.state != WorkerState.READY:
        # NOT a normal return: the publish is not serving. The tracker
        # stays at h2d_done, so the coordinator reconciles not-committed.
        raise RuntimeError(
            f"post_weight_sync finished in state {self.state}; publish is"
            " not serving"
        )
      # Inside the composite terminal, not before it: admission reopens only
      # once the publish is known to be serving.
      self.manager.reopen_admission()
      self._round_tracker.complete(sync_request, "committed")
      return res

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Rolls back to the previous weights; idempotent from READY; never

    resurrects a STOPPED worker.
    """
    async with self._phase_lock:
      if not self._round_tracker.admit(sync_request, "aborted"):
        return None
      res = await self.manager.abort_weight_sync(sync_request, **kwargs)
      # FAIL-CLOSED terminal reconciliation: "aborted" is recorded only
      # when the sampler's own sub-state POSITIVELY confirms the rollback
      # for THIS round. An already-published round, an unknown report
      # format, a mismatched round key, or any future phase name all fall
      # through to needs-restart -- an abort that cannot be confirmed must
      # never be recorded as one.
      sampler_state = await self.manager.get_weight_sync_round(**kwargs)
      # No escape hatch for None. A sampler whose status endpoint answers
      # nothing has told us nothing, and "nothing" is the one answer that
      # must never be read as a confirmed rollback.
      extra = getattr(sync_request, "extra_config", None) or {}
      confirmed = (
          isinstance(sampler_state, dict)
          and sampler_state.get("phase")
          in ("aborted", "rollback_complete", "idle")
          and sampler_state.get("req_id") == extra.get("req_id")
          and sampler_state.get("uuid") == extra.get("uuid")
      )
      if not confirmed:
        raise RuntimeError(
            "abort not positively confirmed by the sampler sub-state"
            f" {sampler_state!r}; reconcile as needs-restart, not aborted"
        )
      if self.state == WorkerState.SYNCING:
        self.state = WorkerState.READY
      if self.state != WorkerState.READY:
        raise RuntimeError(
            f"abort_weight_sync finished in state {self.state}; the worker"
            " is not serving and must reconcile as needs-restart"
        )
      # Only here: every path above leaves the sampler's state unconfirmed,
      # and admitting requests over an unconfirmed rollback is exactly what
      # the fail-closed terminal exists to prevent.
      self.manager.reopen_admission()
      self._round_tracker.complete(sync_request, "aborted")
      return res

  async def get_weight_sync_round(self, **kwargs) -> Any:
    """The AUTHORITATIVE round report (this worker's tracker, not the

    sampler's internal sub-state).
    """
    return self._round_tracker.report()
