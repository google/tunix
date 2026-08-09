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

import dataclasses
from typing import Any, AsyncIterator, Callable, List, Optional, Sequence, Union
import numpy as np
from tunix.experimental.common import datatypes
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
          error=item.error_message,
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
          traj=item,
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

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Prepares the worker for an upcoming weight synchronization step."""
    self.state = WorkerState.SYNCING
    try:
      return await self.manager.pre_weight_sync(sync_request, **kwargs)
    finally:
      self.state = WorkerState.READY

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Synchronizes the worker's internal model weights."""
    self.state = WorkerState.SYNCING
    try:
      return await self.manager.weight_sync(sync_request, **kwargs)
    finally:
      self.state = WorkerState.READY

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Finalizes policy weight update and resumes workers."""
    return await self.manager.post_weight_sync(sync_request, **kwargs)
