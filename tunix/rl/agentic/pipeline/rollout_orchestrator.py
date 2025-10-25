# Copyright 2025 Google LLC
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

"""Orchestrates parallel rollouts of LLM agents in environments.

This module defines the `RolloutOrchestrator` class, which manages the
concurrent collection of trajectories from multiple agent-environment pairs and
groups them into batches for further processing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Hashable
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from tunix.rl.agentic.queue_manager import group_queue_manager
from tunix.rl.agentic.trajectory import trajectory_collect_engine


Trajectory = trajectory_collect_engine.Trajectory
LLMBaseAgent = trajectory_collect_engine.LLMBaseAgent
BaseEnv = trajectory_collect_engine.BaseEnv
TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine
TrajectoryItem = group_queue_manager.TrajectoryItem
GroupQueueManager = group_queue_manager.GroupQueueManager


class RolloutOrchestrator:
  """Orchestrates parallel rollouts of LLM agents in environments.

  This class manages the concurrent collection of trajectories from multiple
  agent-environment pairs using `TrajectoryCollectEngine` instances. It groups
  the collected trajectories into batches via a `GroupQueueManager` and yields
  these batches for further processing.
  """

  def __init__(
      self,
      *,
      engine_cls: Type[TrajectoryCollectEngine] = TrajectoryCollectEngine,
      engine_defaults: Optional[Dict[str, Any]] = None,
      max_concurrency: Optional[int] = None,
  ):
    self.engine_cls = engine_cls
    self.engine_defaults = engine_defaults or {}
    self.max_concurrency = max_concurrency
    self._semaphore = (
        asyncio.Semaphore(max_concurrency) if max_concurrency else None
    )
    self._tasks: List[asyncio.Task] = []
    self._stop = asyncio.Event()
    self._logger = logging.getLogger(self.__class__.__name__)

  async def _collect_trajectory(
      self, agent: LLMBaseAgent, env: BaseEnv, mode: Optional[str] = None
  ) -> Trajectory:
    """Helper method to collect a single trajectory."""
    engine = self.engine_cls(agent, env, **self.engine_defaults)
    if mode:
      return await engine.collect(mode)
    return await engine.collect()

  async def _runner(
      self,
      i: int,
      agent: LLMBaseAgent,
      env: BaseEnv,
      manager: GroupQueueManager,
      group_key: Callable[[int, BaseEnv, Trajectory], Hashable],
      episodes_per_pair: Optional[int],
      start_step_fn: Optional[Callable[[], int]] = None,
      collect_mode: Optional[str] = None,
  ):
    """Runs the trajectory collection loop for a single agent-environment pair.

    This method continuously collects trajectories using `_collect_trajectory`
    and puts them into the `GroupQueueManager`. It handles potential exceptions
    during trajectory collection and respects the `_stop` event and
    `episodes_per_pair` limit.

    Args:
      i: The index of the agent-environment pair.
      agent: The LLMBaseAgent instance.
      env: The BaseEnv instance.
      manager: The GroupQueueManager to put collected trajectories into.
      group_key: A callable to determine the group ID for a trajectory.
      episodes_per_pair: The maximum number of episodes to collect for this
        pair, or None for unlimited.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.
      collect_mode: An optional string to select the collection mode.
    """
    episode_id = 0
    self._logger.debug("Starting runner for pair %d", i)
    try:
      while not self._stop.is_set() and (
          episodes_per_pair is None or episode_id < episodes_per_pair
      ):
        try:
          if self._semaphore:
            async with self._semaphore:
              traj = await self._collect_trajectory(
                  agent, env, mode=collect_mode
              )
          else:
            traj = await self._collect_trajectory(agent, env, mode=collect_mode)
          gid = group_key(i, env, traj)
          start_step = start_step_fn() if start_step_fn else 0
          item = TrajectoryItem(
              pair_index=i,
              group_id=gid,
              episode_id=episode_id,
              start_step=start_step,
              traj=traj,
              metadata={},
          )
          await manager.put(item)
          episode_id += 1
        except Exception as e:
          self._logger.error(
              "Error collecting trajectory for pair %d, episode %d: %s",
              i,
              episode_id,
              e,
          )
          # Continue with next episode instead of crashing the entire runner
          episode_id += 1
          continue
    except Exception as e:
      self._logger.error("Fatal error in runner for pair %d: %s", i, e)
      raise
    finally:
      self._logger.debug(
          "Runner for pair %d completed with %d episodes", i, episode_id
      )

  async def run_and_yield_batches(
      self,
      pairs: List[Tuple[LLMBaseAgent, BaseEnv]],
      *,
      group_size: int,
      batch_size: int,
      group_key: Callable[[int, BaseEnv, Trajectory], Hashable],
      collect_mode: Optional[str] = None,
      episodes_per_pair: Optional[int] = None,
      max_open_groups: Optional[int] = None,
      start_step_fn: Optional[Callable[[], int]] = None,
  ):
    """Runs multiple agent-environment pairs in parallel and yields batches.

    This method starts `_runner` tasks for each agent-environment pair. It uses
    a `GroupQueueManager` to group collected trajectories and yields batches of
    trajectories as they become available. The orchestrator continues running
    until all `episodes_per_pair` are collected for all pairs or the `_stop`
    event is set.

    Args:
      pairs: A list of tuples, where each tuple contains an LLMBaseAgent and a
        BaseEnv instance.
      group_size: The number of trajectories to collect before forming a group.
      batch_size: The maximum number of items to include in each yielded batch.
      group_key: A callable that takes (pair_index, env, trajectory) and returns
        a hashable group identifier.
      collect_mode: An optional string to select the collection mode for
        `TrajectoryCollectEngine`.
      episodes_per_pair: The maximum number of episodes to collect for each
        agent-environment pair. If None, runs indefinitely until stopped.
      max_open_groups: The maximum number of groups that can be open
        simultaneously in the GroupQueueManager.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.

    Yields:
      A list of `TrajectoryItem` instances, grouped and batched.
    """
    manager = GroupQueueManager(
        group_size=group_size, max_open_buckets=max_open_groups
    )
    expected = len(pairs) * episodes_per_pair if episodes_per_pair else 1
    seen = 0
    try:
      for i, (a, e) in enumerate(pairs):
        self._tasks.append(
            asyncio.create_task(
                self._runner(
                    i,
                    a,
                    e,
                    manager,
                    group_key,
                    episodes_per_pair,
                    start_step_fn,
                    collect_mode,
                )
            )
        )
      while not self._stop.is_set():
        batch = await manager.get_batch(batch_size)
        if batch:
          yield batch
          seen += len(batch)
        all_done = all(t.done() for t in self._tasks)
        if all_done:
          if seen != expected:
            raise ValueError(
                f"Expected {expected} trajectories, but only got {seen}"
            )
          break
    finally:
      self._stop.set()
      self._logger.debug("Stopping orchestrator and cleaning up resources")
      # Cancel all running tasks
      for t in self._tasks:
        if not t.done():
          t.cancel()
      # Wait for all tasks to complete or be cancelled
      if self._tasks:
        await asyncio.gather(*self._tasks, return_exceptions=True)
      # Clean up manager
      await manager.prepare_clear()
      await manager.clear()
      self._tasks.clear()
      self._logger.debug("Cleanup completed")
