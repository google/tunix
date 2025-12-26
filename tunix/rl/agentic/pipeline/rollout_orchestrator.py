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
import copy
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.queue_manager import group_queue_manager
from tunix.rl.agentic.trajectory import trajectory_collect_engine


Trajectory = agent_types.Trajectory
ConversationAgentBase = base_agent.ConversationAgentBase
BaseTaskEnv = base_environment.BaseTaskEnv
TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine
TrajectoryItem = agent_types.TrajectoryItem
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
      rollout_sync_lock: utils.RolloutSyncLock,
      engine_cls: Type[TrajectoryCollectEngine] = TrajectoryCollectEngine,
      engine_defaults: Optional[Dict[str, Any]] = None,
      max_concurrency: Optional[int] = None,
  ):
    """Initializes the RolloutOrchestrator.

    The orchestrator manages a pool of trajectory collection engines, each 
    running an agent-environment interaction to collect a trajectory. 
    Each output trajectory is considered an "episode".

    Args:
      rollout_sync_lock: A lock to synchronize the very start of
        multiple parallel rollout operations, ensuring they don't all start at
        the exact same moment, potentially overwhelming resources.
      engine_cls: The class used to instantiate trajectory collection engines.
        Each engine is responsible for running a single episode of interaction
        between an agent and an environment.
      engine_defaults: A dictionary of default keyword arguments to be passed to
        the `engine_cls` constructor when creating new engine instances.
      max_concurrency: The maximum number of agent-environment interaction
        episodes to run in parallel. This limits the number of concurrent calls
        to the underlying language model.
    """
    self.engine_cls = engine_cls
    self.engine_defaults = engine_defaults or {}
    self.max_concurrency = max_concurrency
    self._tasks: List[asyncio.Task] = []
    self._stop = asyncio.Event()
    self._logger = logging.getLogger(self.__class__.__name__)
    self._manager: Optional[GroupQueueManager] = None
    self._rollout_sync_lock = rollout_sync_lock

  async def _collect_trajectory(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      mode: Optional[str] = None,
      model_call_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Trajectory:
    """Helper method to collect a single trajectory."""
    engine_kwargs = self.engine_defaults.copy()
    if model_call_kwargs:
      engine_kwargs["model_call_kwargs"] = model_call_kwargs
    engine = self.engine_cls(agent, env, **engine_kwargs)
    if mode:
      return await engine.collect(mode)
    return await engine.collect()

  async def _run_and_queue_one_episode(
      self,
      pair_idx: int,
      episode_idx: int,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      manager: GroupQueueManager,
      group_key: Callable[[int, BaseTaskEnv, Trajectory], Hashable],
      start_step_fn: Optional[Callable[[], int]],
      collect_mode: Optional[str],
  ):
    """Collects one trajectory and queues it."""
    traj = await self._collect_trajectory(agent, env, mode=collect_mode)
    gid = group_key(pair_idx, env, traj)
    start_step = start_step_fn() if start_step_fn else 0
    item = TrajectoryItem(
        pair_index=pair_idx,
        group_id=gid,
        episode_id=0,
        start_step=start_step,
        traj=traj,
        metadata={"generation_id": episode_idx},
    )
    await manager.put(item)
    return 1

  async def _runner(
      self,
      i: int,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      manager: GroupQueueManager,
      group_key: Callable[[int, BaseTaskEnv, Trajectory], Hashable],
      num_episodes: int = 1,
      start_step_fn: Optional[Callable[[], int]] = None,
      collect_mode: Optional[str] = None,
  ):
    """Runs the trajectory collection loop for a single agent-environment pair.

    This method continuously collects trajectories using `_collect_trajectory`
    and puts them into the `GroupQueueManager`. It handles potential exceptions
    during trajectory collection and respects the `_stop` event and
    `num_episodes` limit.

    Args:
      i: The index of the agent-environment pair.
      agent: The ConversationAgentBase instance.
      env: The BaseTaskEnv instance.
      manager: The GroupQueueManager to put collected trajectories into.
      group_key: A callable to determine the group ID for a trajectory.
      num_episodes: The maximum number of episodes to collect for this pair,
        defaults to 1.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.
      collect_mode: An optional string to select the collection mode.
    """
    episode_count = 0
    if num_episodes <= 0:
      raise ValueError(
          f"num_episodes must be a positive integer, got {num_episodes}"
      )
    self._logger.debug(
        "Starting generating trajectories(_runner) for pair %d", i
    )

    try:
      # Parallel execution for the group
      self._rollout_sync_lock.acquire_rollout()
      try:
        tasks = []
        for ep_id in range(num_episodes):
          # TODO(b/462779884): Replace deepcopy with a factory pattern.
          tasks.append(
              self._run_and_queue_one_episode(
                  pair_idx=i,
                  episode_idx=ep_id,
                  agent=copy.deepcopy(agent),
                  env=copy.deepcopy(env),
                  manager=manager,
                  group_key=group_key,
                  start_step_fn=start_step_fn,
                  collect_mode=collect_mode,
              )
          )
        results = await asyncio.gather(*tasks)
        episode_count = sum(results)
      finally:
        self._rollout_sync_lock.release_rollout()
    except Exception as e:
      self._logger.error("Fatal error in runner for pair %d: %s", i, e)
      raise
    finally:
      self._logger.debug(
          "Runner for pair %d completed with %d episodes", i, episode_count
      )

  async def run_producers_from_stream(
      self,
      pairs_stream: Iterable[Tuple[ConversationAgentBase, BaseTaskEnv]],
      *,
      group_size: int,
      group_key: Callable[
          [int, BaseTaskEnv, Trajectory], Hashable
      ] = lambda i, _, __: i,
      collect_mode: Optional[str] = None,
      num_episodes: int = 1,
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
        ConversationAgentBase and a BaseTaskEnv instance.
      group_size: The number of trajectories to collect before forming a group.
      group_key: A callable that takes `(pair_index, env, trajectory)` and
        returns a hashable group identifier. Using a callable allows for
        flexible grouping strategies. For example, trajectories can be grouped
        by task properties from the environment (`env`) or by outcomes within
        the collected trajectory (`trajectory`). The default is to group by the
        agent-environment pair index.
      collect_mode: An optional string to select the collection mode for
        `TrajectoryCollectEngine`.
      num_episodes: The maximum number of episodes to collect for each
        agent-environment pair. If None, runs indefinitely until stopped.
      max_open_groups: The maximum number of groups that can be open
        simultaneously in the GroupQueueManager.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.

    Raises:
      ValueError: If `max_concurrency` is not set.
      RuntimeError: If the orchestrator is already running.
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
        # Phase 1: Fill worker pool
        # As long as we have concurrency slots available and the input stream
        # is not exhausted, start new runner tasks.
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
                    num_episodes=num_episodes,
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
        # If no tasks are running and stream is exhausted, done.
        if not active_tasks:
          break  # All done

        # Phase 2: Wait for any task to complete
        # This frees up a slot for a new task if the stream is not exhausted.
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

    Raises:
      RuntimeError: If `run_producers_from_stream` has not been called to start
        the producers.
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
