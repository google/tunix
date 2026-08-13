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

"""Rollout Manager concurrency controller and Raiden KV migration orchestrator."""

import asyncio
from typing import Any, AsyncIterator, Callable, Dict, Optional, Sequence, Union
from tunix.experimental.common import datatypes
from tunix.experimental.rl.agentic import registry
from tunix.experimental.rollout import collector as collector_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.rollout import base_rollout

TrajectoryOrError = Union[
    trajectory_lib.Trajectory, trajectory_lib.TrajectoryError
]


class RolloutManager:
  """Internal trajectory and concurrency control core of RolloutWorker.

  Manages active TrajectoryCollectorEngine tasks, out-of-order completion
  streaming, and straggler KV-cache migration across SamplerServer slices.
  """

  def __init__(
      self,
      config: Optional[base_rollout.RolloutConfig] = None,
      sampler: Optional[sampler_lib.Sampler] = None,
      env_pool: Any = None,
      agent_factory: Optional[Callable[[], Any]] = None,
      max_concurrency: int = 64,
      tokenizer: Any = None,
      chat_parser: Any = None,
      trajectory_store: Optional[store.TrajectoryWriter] = None,
  ):
    self.config = config
    if sampler is None:
      sampler_type = getattr(config, "sampler_type", "vanilla")
      if sampler_type == "vllm":
        raise NotImplementedError(
            "vLLM sampler is not implemented yet. Use 'legacy_vllm' or"
            " 'vanilla'."
        )
      elif sampler_type == "legacy_vllm":
        from tunix.experimental.rollout import legacy_vllm_sampler_adapter  # pylint: disable=g-import-not-at-top

        sampler = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(  # pyrefly: ignore[bad-instantiation]
            server_id="legacy_vllm_sampler",
        )
      elif sampler_type == "vanilla":
        sampler = vanilla_sampler_adapter.VanillaSamplerAdapter(  # pyrefly: ignore[bad-instantiation]
            server_id="vanilla_sampler",
        )
      else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
      sampler.initialize()

    if not isinstance(sampler, sampler_lib.Sampler):
      raise TypeError(
          f"Expected object implementing Sampler Protocol, got {type(sampler)}"
      )
    self.sampler = sampler
    self.env_pool = env_pool
    self.agent_factory = agent_factory
    self.max_concurrency = max_concurrency
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self.trajectory_store = trajectory_store
    if self.tokenizer is None or self.chat_parser is None:
      raise ValueError(
          "RolloutManager requires valid tokenizer and chat_parser arguments"
          " (none can be None)."
      )

    self._active_collectors: Dict[
        str, collector_lib.TrajectoryCollectorEngine
    ] = {}
    self._active_tasks: Dict[str, asyncio.Task[Any]] = {}
    self._completed_queue: asyncio.Queue[TrajectoryOrError] = asyncio.Queue()

  async def _generate_one(
      self,
      request: datatypes.RolloutRequest,
      on_complete: Optional[Callable[[TrajectoryOrError], None]] = None,
  ) -> TrajectoryOrError:
    """Spawns an async task running the multi-turn episode loop concurrently."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[TrajectoryOrError] = loop.create_future()

    def _resolve(result: TrajectoryOrError) -> None:
      if on_complete:
        on_complete(result)
      if not future.done():
        future.set_result(result)

    traj_id = request.traj_id

    env_name = getattr(self.config, "env_name", "")
    if env_name and registry.ENV_REGISTRY.contains(env_name):
      env_cls = registry.ENV_REGISTRY.get(env_name)
      env_config = request.metadata.get(
          "env_config", getattr(self.config, "env_config", {})
      )
      env_client = env_cls(**env_config)
    elif self.env_pool and hasattr(self.env_pool, "acquire_env"):
      env_client = self.env_pool.acquire_env(request.metadata.get("env_config"))
    else:
      env_client = None

    agent_name = getattr(self.config, "agent_name", "")
    if agent_name and registry.AGENT_REGISTRY.contains(agent_name):
      agent_cls = registry.AGENT_REGISTRY.get(agent_name)
      agent_config = getattr(self.config, "agent_config", {})
      agent = agent_cls(**agent_config)
    elif self.agent_factory and callable(self.agent_factory):
      agent = self.agent_factory()
    else:
      agent = None

    collector = collector_lib.TrajectoryCollectorEngine(
        traj_id=traj_id,
        request=request,
        sampler=self.sampler,
        env_client=env_client,
        agent=agent,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        trajectory_store=self.trajectory_store,
    )

    self._active_collectors[traj_id] = collector
    task = asyncio.create_task(
        self._run_and_enqueue(collector, request, _resolve)
    )
    self._active_tasks[traj_id] = task

    return await future

  async def generate(
      self,
      requests: (
          datatypes.RolloutRequest
          | Sequence[datatypes.RolloutRequest]
          | Any
          | Sequence[Any]
      ),
      on_complete: Optional[Callable[[TrajectoryOrError], None]] = None,
  ) -> TrajectoryOrError | Sequence[TrajectoryOrError] | Any:
    """Dispatches 1 or N requests concurrently to the internal Collector Engine pool."""
    if isinstance(requests, (list, tuple)):
      tasks = [
          asyncio.create_task(self._generate_one(req, on_complete=on_complete))
          for req in requests
      ]
      return await asyncio.gather(*tasks)
    return await self._generate_one(requests, on_complete=on_complete)  # pyrefly: ignore[bad-argument-type]

  async def _run_and_enqueue(
      self,
      collector: collector_lib.TrajectoryCollectorEngine,
      request: datatypes.RolloutRequest,
      resolve_cb: Callable[[TrajectoryOrError], None],
  ) -> None:
    """Runs episode loop, removes active tracking, and resolves callbacks/streams."""
    try:
      trajectory: TrajectoryOrError = await collector.run_episode()
    except Exception as e:  # pylint: disable=broad-exception-caught
      trajectory = trajectory_lib.TrajectoryError(
          trajectory_id=collector.traj_id,
          prompt_id=request.prompt_id,
          error_message=str(e),
          error_type=type(e).__name__,
      )
    finally:
      self._active_collectors.pop(collector.traj_id, None)
      self._active_tasks.pop(collector.traj_id, None)
      if (
          self.env_pool
          and hasattr(self.env_pool, "release_env")
          and collector.env is not None
      ):
        self.env_pool.release_env(collector.env)

    await self._completed_queue.put(trajectory)
    resolve_cb(trajectory)

  async def pop_next_completed(self) -> TrajectoryOrError:
    """Pull-based stream: yields whichever trajectory finishes first out-of-order."""
    return await self._completed_queue.get()

  async def as_completed_stream(
      self,
  ) -> AsyncIterator[TrajectoryOrError]:
    """Async generator yielding completed trajectories strictly out-of-order."""
    # TODO(lancewang): Add termination condition to prevent hangs when stream
    # is exhausted.
    while True:
      yield await self.pop_next_completed()

  async def migrate_straggler(
      self,
      trajectory_id: str,
      source_server_id: str,
      target_server_id: str,
  ) -> bool:
    """Migrates an active long-tail trajectory using Raiden P2P KV transfer."""
    collector = self._active_collectors.get(trajectory_id)
    if not collector or collector.is_done:
      return False
    if not collector.is_paused:
      raise RuntimeError(
          f"Collector [{trajectory_id}] must be paused before KV migration."
      )

    token_ids = collector.get_accumulated_token_ids()
    return await self.sampler.migrate_kv_cache(
        route_key=trajectory_id,
        source_server_id=source_server_id,
        target_server_id=target_server_id,
        token_ids=token_ids,
    )

  def pause_all(self) -> None:
    for collector in self._active_collectors.values():
      collector.pause()

  def resume_all(self) -> None:
    for collector in self._active_collectors.values():
      collector.resume()

  def cancel_all(self) -> None:
    for collector in self._active_collectors.values():
      collector.cancel()
    for task in self._active_tasks.values():
      task.cancel()

  async def pre_weight_sync(
      self, sync_request: sampler_lib.WeightSyncRequest | Any = None, **kwargs
  ) -> Any:
    """Phase 3 Barrier 1: Pauses active collectors and checks staging."""
    self.pause_all()
    if self.sampler:
      return await self.sampler.pre_weight_sync(sync_request, **kwargs)
    return None

  async def weight_sync(
      self, sync_request: sampler_lib.WeightSyncRequest | Any = None, **kwargs
  ) -> Any:
    """Phase 3 Barrier 2: Executes weight synchronization and resumes collectors."""
    completed_version = getattr(sync_request, "policy_version", 0)
    if self.sampler:
      res = await self.sampler.weight_sync(sync_request, **kwargs)
      if res is not None:
        completed_version = res
    return completed_version

  async def post_weight_sync(
      self, sync_request: sampler_lib.WeightSyncRequest | Any = None, **kwargs
  ) -> Any:
    """Phase 3 Barrier 3: Finalizes policy weight update and resumes collectors."""
    res = None
    if self.sampler:
      res = await self.sampler.post_weight_sync(sync_request, **kwargs)
    self.resume_all()
    return res
