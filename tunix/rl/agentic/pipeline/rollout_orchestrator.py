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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

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
    self._tasks: List[asyncio.Task] = []
    self._stop = asyncio.Event()
    self._logger = logging.getLogger(self.__class__.__name__)
    self._manager: Optional[GroupQueueManager] = None

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
    self._logger.debug(
        "Starting generating trajectories(_runner) for pair %d", i
    )
    try:
      while not self._stop.is_set() and (
          episodes_per_pair is None or episode_id < episodes_per_pair
      ):
        traj = await self._collect_trajectory(agent, env, mode=collect_mode)
        # Manually propagate _original_input from env.task to the trajectory
        # because TrajectoryCollectEngine does not do this automatically. This
        # is required for downstream consumers like GRPOLearner.
        if hasattr(env, "task") and isinstance(env.task, dict):
          if "_original_input" in env.task:
            traj["_original_input"] = env.task["_original_input"]
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
      self._logger.error("Fatal error in runner for pair %d: %s", i, e)
      raise
    finally:
      self._logger.debug(
          "Runner for pair %d completed with %d episodes", i, episode_id
      )

  async def run_producers_from_stream(
      self,
      pairs_stream: Iterable[Tuple[LLMBaseAgent, BaseEnv]],
      *,
      group_size: int,
      group_key: Callable[
          [int, BaseEnv, Trajectory], Hashable
      ] = lambda i, _, __: i,
      collect_mode: Optional[str] = None,
      episodes_per_pair: Optional[int] = None,
      max_open_groups: Optional[int] = None,
      start_step_fn: Optional[Callable[[], int]] = None,
  ):
    """Dynamically runs collectors from a stream of agent-env pairs.

    This coroutine manages a pool of producer tasks. It draws pairs from
    `pairs_stream` and starts a `_runner` for each. It maintains up to
    `self.max_concurrency` active runners, starting new ones as they
    finish, until the `pairs_stream` is exhausted. This method is intended to
    be run as a background task. It sets up a shared queue that can be
    consumed from using `yield_batches`.

    Args:
      pairs_stream: An iterable of tuples, where each tuple contains an
        LLMBaseAgent and a BaseEnv instance.
      group_size: The number of trajectories to collect before forming a group.
      group_key: A callable that takes `(pair_index, env, trajectory)` and
        returns a hashable group identifier. Using a callable allows for
        flexible grouping strategies. For example, trajectories can be grouped
        by task properties from the environment (`env`) or by outcomes within
        the collected trajectory (`trajectory`). The default is to group by the
        agent-environment pair index.
      collect_mode: An optional string to select the collection mode for
        `TrajectoryCollectEngine`.
      episodes_per_pair: The maximum number of episodes to collect for each
        agent-environment pair. If None, runs indefinitely until stopped.
      max_open_groups: The maximum number of groups that can be open
        simultaneously in the GroupQueueManager.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.
    """
    self._logger.info(
        "Starting run_producers_from_stream with %d concurrency",
        self.max_concurrency,
    )

    if not self.max_concurrency:
      raise ValueError("max_concurrency must be set to use start_producers.")
    if self._manager:
      raise RuntimeError("Orchestrator is already running.")

    self._manager = GroupQueueManager(
        group_size=group_size, max_open_buckets=max_open_groups
    )
    self._stop.clear()
    self._tasks.clear()

    pairs_iterator = iter(pairs_stream)
    active_tasks: set[asyncio.Task] = set()
    next_pair_index = 0
    stream_exhausted = False

    try:
      self._logger.debug(
          "Orchestrator producer loop starting with %d concurrency",
          self.max_concurrency,
      )
      while not self._stop.is_set():
        while (
            not stream_exhausted
            and len(active_tasks) < self.max_concurrency
            and not self._stop.is_set()
        ):
          try:
            self._logger.debug("Getting one pair: %d", next_pair_index)
            agent, env = next(pairs_iterator)
            task = asyncio.create_task(
                self._runner(
                    i=next_pair_index,
                    agent=agent,
                    env=env,
                    manager=self._manager,
                    group_key=group_key,
                    episodes_per_pair=episodes_per_pair,
                    start_step_fn=start_step_fn,
                    collect_mode=collect_mode,
                )
            )
            active_tasks.add(task)
            self._tasks.append(task)
            next_pair_index += 1
          except StopIteration:
            self._logger.debug("Pairs stream exhausted.")
            stream_exhausted = True
            break
          except Exception as e:
            self._logger.error(
                "Error getting next trajectory pair %d: %s", next_pair_index, e
            )
            raise e
        if not active_tasks:
          break  # All done

        done, pending = await asyncio.wait(
            active_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        # Eagerly check for exceptions in completed tasks. If a runner fails,
        # it could cause a deadlock where the consumer waits for a group that
        # will never be completed. Propagating the exception ensures a clean
        # shutdown.
        for task in done:
          task.result()  # This will re-raise any exception in the task.
          # Remove the completed task from the _tasks list.
          if task in self._tasks:
            self._tasks.remove(task)
        active_tasks = pending

      # Wait for any stragglers if we were stopped prematurely
      if self._tasks:
        await asyncio.gather(*self._tasks, return_exceptions=True)
    except asyncio.CancelledError:
      self._logger.debug("Producer task was cancelled.")
      # The consumer's `finally` block will handle cleanup.
      raise
    except Exception as e:
      self._logger.error("Producer task failed: %s", e)
      if self._manager:
        await self._manager.put_exception(e)
      raise
    finally:
      # Shield the final cleanup step to ensure it runs even if the producer
      # task is being cancelled. This prevents leaving the manager in an
      # inconsistent state.
      if self._manager:
        await asyncio.shield(self._manager.prepare_clear())

  async def yield_batches(self, batch_size: int):
    """Yields batches of trajectories from the internal queue.

    This consumer method should be used in conjunction with
    `run_producers_from_stream`. It will yield batches until the producers have
    finished and the queue is empty. When the consumer is stopped (e.g., the
    async for loop is broken), it will trigger a cleanup of all background
    producer tasks.

    Args:
      batch_size: The maximum number of items to include in each yielded batch.

    Yields:
      A list of `TrajectoryItem` instances.
    """
    if not self._manager:
      raise RuntimeError("Producers have not been started.")
    try:
      while not self._stop.is_set():
        batch = await self._manager.get_batch(batch_size)
        if not batch:
          # If batch is empty, it means producers are done and queue is empty.
          break
        yield batch
    except (GeneratorExit, asyncio.CancelledError):
      # This is the normal shutdown path when the consumer stops listening.
      pass
    except Exception as e:
      self._logger.error("Error yielding batches: %s", e)
      raise
    finally:
      # This block executes when the consumer (the 'async for' loop) stops.
      # The primary responsibility here is to signal all producers to stop.
      # We do not await task completion here as that's fragile in a generator's
      # finally block. Instead, we rely on the parent coroutine
      # (`run_producers_from_stream`) to handle the full cleanup, as it has
      # the correct context to await its child tasks.
      self._stop.set()
      self._logger.debug("Consumer stopped; signaling producers to stop.")
      for t in self._tasks:
        if not t.done():
          t.cancel()

  async def run_and_yield_batches(
      self,
      pairs: List[Tuple[LLMBaseAgent, BaseEnv]],
      *,
      group_size: int,
      batch_size: int,
      group_key: Callable[
          [int, BaseEnv, Trajectory], Hashable
      ] = lambda i, _, __: i,
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
      group_key: A callable that takes `(pair_index, env, trajectory)` and
        returns a hashable group identifier. Using a callable allows for
        flexible grouping strategies. For example, trajectories can be grouped
        by task properties from the environment (`env`) or by outcomes within
        the collected trajectory (`trajectory`). The default is to group by the
        agent-environment pair index.
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
    self._logger.debug("Starting orchestrator with %d pairs", len(pairs))
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
          # After all tasks are done, there might still be items in the
          # manager's queue. We need to drain the queue to get all trajectories.
          while True:
            remaining_batch = await manager.get_batch(batch_size)
            if not remaining_batch:
              break
            yield remaining_batch
            seen += len(remaining_batch)

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
