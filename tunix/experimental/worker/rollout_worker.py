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

import threading
from typing import Any, AsyncIterator, Callable, List, Optional, Sequence, Union

from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import store as trajectory_store_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.worker import abstract_worker

RolloutConfig = sampler_lib.RolloutConfig

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
      config: RolloutConfig,
      sampler: Optional[sampler_lib.Sampler] = None,
      env_pool: Any = None,
      agent_factory: Optional[Callable[[], Any]] = None,
      max_concurrency: int = 64,
      tokenizer: Any = None,
      chat_parser: Any = None,
  ):
    super().__init__()
    if not isinstance(config, RolloutConfig):
      raise TypeError(
          "RolloutWorker requires config to be a RolloutConfig, got"
          f" {type(config).__name__}."
      )
    self.worker_id = worker_id
    self.config: RolloutConfig = config
    self._policy_version = 0
    self._state = datatypes.WorkerState.PENDING
    self._init_lock = threading.Lock()
    self._sync_round = {"req_id": None, "uuid": 0, "phase": "idle"}
    if tokenizer is None or chat_parser is None:
      raise ValueError(
          "RolloutWorker requires valid tokenizer and chat_parser arguments"
          " (none can be None)."
      )
    self.manager = manager_lib.RolloutManager(
        config=self.config,
        sampler=sampler,
        env_pool=env_pool,
        agent_factory=agent_factory,
        max_concurrency=max_concurrency,
        tokenizer=tokenizer,
        chat_parser=chat_parser,
    )
    # Built at most once per process: this __init__ runs exactly once per
    # RolloutWorker instance, so there is no separate guard against
    # constructing the store twice. See store.TrajectoryStore.from_config.
    # TODO(sizhi): Pass self._trajectory_store into RolloutManager / collector
    # to log rollout steps in follow-up CLs.
    self._trajectory_store = trajectory_store_lib.TrajectoryStore.from_config(
        self.config.trajectory_store_config
    )
    if self._trajectory_store is not None:
      # Several workers can share one log stream, and absl log lines carry no
      # process identity, so the worker_id is what attributes a reported
      # config to a process.
      logging.info(
          "[trajectory-store] worker %s built %s",
          worker_id,
          self._trajectory_store.to_config(),
      )

  @property
  def trajectory_store(self) -> trajectory_store_lib.TrajectoryStore | None:
    return self._trajectory_store

  @property
  def sampler(self) -> sampler_lib.Sampler:
    return self.manager.sampler

  def get_worker_id(self) -> str:
    """Returns the unique worker ID."""
    return self.worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=frozenset({"rollout"}),
        resources={
            "sampler": type(self.sampler).__name__,
            "policy_version": self._policy_version,
        },
    )

  def initialize(self) -> datatypes.Response:
    with self._init_lock:
      if self.state == WorkerState.READY:
        return datatypes.Response(
            metadata={
                "worker_id": self.worker_id,
                "state": self.state.value,
                "policy_version": self._policy_version,
                "initialized": True,
                "ready": True,
            }
        )
      self.state = WorkerState.INITIALIZING
      self.sampler.initialize()
      try:
        return datatypes.Response(
            metadata={
                "worker_id": self.worker_id,
                "state": self.state.value,
                "policy_version": self._policy_version,
            }
        )
      finally:
        self.state = WorkerState.READY

  def compile(self, dummy_data: Any) -> datatypes.Response:
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.COMPILING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def start(self) -> datatypes.Response:
    if self.state == WorkerState.PENDING:
      self.initialize()
    return datatypes.Response(
        metadata={
            "worker_id": self.worker_id,
            "state": self.state.value,
            "policy_version": self._policy_version,
        }
    )

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    try:
      self.manager.cancel_all()
    finally:
      # Runs even when cancel_all raises, so a failed stop releases the
      # store's background writer thread instead of leaking it.
      if self._trajectory_store is not None:
        self._trajectory_store.close()
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
    return datatypes.HealthReport(
        state=self.state,
        policy_version=self._policy_version,
        inflight=len(self.manager._active_tasks),  # pylint: disable=protected-access
        queue_depth=self.manager._completed_queue.qsize(),  # pylint: disable=protected-access
    )

  def get_target_state(self) -> Any:
    """Returns rollout-side target-state skeleton for trainer-side conversion."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    return self.manager.get_target_state()

  def _stamp_worker_lineage(self, metadata: dict[str, Any] | None) -> None:
    """Appends worker generation telemetry to the lineage context if present."""
    if metadata is None:
      return
    lineage_ctx = metadata.get("lineage")
    if lineage_ctx is not None and hasattr(lineage_ctx, "add_event"):
      lineage_ctx.add_event(
          component="worker.rollout",
          operation="generate",
          attributes={"worker_id": self.worker_id},
      )

  def _to_rollout_response(
      self,
      item: Any,
      request_id: str = "",
      prompt_tokens: np.ndarray | None = None,
  ) -> datatypes.RolloutResponse:
    """Converts internal Trajectory or TrajectoryError to wire-safe RolloutResponse."""
    if isinstance(item, trajectory_lib.TrajectoryError):
      return datatypes.RolloutResponse(
          request_id=request_id
          or getattr(item, "trajectory_id", "")
          or getattr(item, "prompt_id", ""),
          status="ERROR",
          error=datatypes.ErrorInfo(
              error_type="TrajectoryError",
              message=str(item.error_message),
          ),
          payload=None,
          metadata={"prompt_id": getattr(item, "prompt_id", "")},
      )
    if isinstance(item, datatypes.RolloutResponse):
      return item
    if isinstance(item, datatypes.TrajectoryItem):
      req_id = request_id or getattr(item, "traj_id", "")
      if prompt_tokens is not None and getattr(item, "prompt_tokens", None) is None:
        item.metadata["prompt_tokens"] = prompt_tokens
      self._stamp_worker_lineage(item.metadata)
      return datatypes.RolloutResponse(
          request_id=req_id,
          status="COMPLETED",
          payload=item,
          metadata=item.metadata,
      )
    raise TypeError(
        f"Unsupported item type for RolloutResponse conversion: {type(item)}"
    )

  async def generate(
      self,
      requests: datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest],
      on_complete: Optional[Callable[[datatypes.RolloutResponse], None]] = None,
  ) -> datatypes.RolloutResponse | List[datatypes.RolloutResponse]:
    """Coroutine method for single or batched generate requests."""
    if isinstance(requests, datatypes.RolloutRequest):
      pass
    elif isinstance(requests, Sequence) and not isinstance(
        requests, (str, bytes)
    ):
      if not all(isinstance(req, datatypes.RolloutRequest) for req in requests):
        raise TypeError(
            "generate requires `requests` to be a RolloutRequest or"
            " Sequence[RolloutRequest]."
        )
    else:
      raise TypeError(
          "generate requires `requests` to be a RolloutRequest or"
          " Sequence[RolloutRequest]."
      )

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

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Quiesces the worker; it stays SYNCING until post or abort."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.SYNCING
    self._record_round(sync_request, "idle")
    result = await self.manager.pre_weight_sync(sync_request, **kwargs)
    self._record_round(sync_request, "prepared")
    return result

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Materializes the received weights; the worker stays SYNCING."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.SYNCING
    metadata = kwargs.pop("metadata", None)
    request = sync_request if sync_request is not None else metadata
    result = await self.manager.weight_sync(request, **kwargs)
    if isinstance(result, int):
      self._policy_version = result
    else:
      version = getattr(request, "policy_version", None)
      if version is not None:
        self._policy_version = version
      else:
        self._policy_version += 1
    self._record_round(request, "h2d_done")
    return result

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Publishes the new weights and resumes serving."""
    result = await self.manager.post_weight_sync(sync_request, **kwargs)
    self.state = WorkerState.READY
    self._record_round(sync_request, "committed")
    return result

  async def bind_weight_sync(self, **kwargs) -> Any:
    """Binds the destination-side transport via the manager."""
    return await self.manager.bind_weight_sync(**kwargs)

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns the sampler's transport metadata via the manager."""
    return await self.manager.get_weight_sync_metadata(**kwargs)

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Discards the round and resumes serving the previous weights."""
    res = await self.manager.abort_weight_sync(sync_request, **kwargs)
    self.state = WorkerState.READY
    self._record_round(sync_request, "aborted")
    return res

  async def get_weight_sync_status(self, **kwargs) -> Any:
    """Returns this worker's view of the current weight sync round."""
    return dict(self._sync_round, policy_version=self._policy_version)

  def _record_round(self, sync_request: Any, phase: str) -> None:
    extra = getattr(sync_request, "extra_config", None) or {}
    if extra.get("req_id") is not None:
      self._sync_round["req_id"] = extra.get("req_id")
      self._sync_round["uuid"] = extra.get("uuid", 0)
    self._sync_round["phase"] = phase
