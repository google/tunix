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
import traceback
from typing import Any, AsyncIterable, Callable, Dict, Iterable, List, Optional, Tuple, Type

from absl import logging
import numpy as np
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


# Exception class names that mean "the sandbox/container for this trajectory
# could not be provisioned". These are raised by the external agent_sandbox_rl
# fleet client, which tunix does not depend on directly, so they are matched by
# name over the MRO rather than by isinstance. A failure of this kind is an
# infrastructure problem affecting one trajectory; it is not evidence that the
# training code is wrong, so it must not take down a 256-chip job.
ENV_PROVISIONING_ERROR_NAMES = frozenset({
    "CapacityError",
    "FleetError",
    "FleetOvercommitError",
    "NoClusterAvailableError",
    "PoolNotFoundError",
    "PreflightError",
    "SandboxProvisioningError",
    "SandboxWarmPoolNotFoundError",
})


def _is_env_provisioning_error(exc: BaseException) -> bool:
  """True if `exc` is, or contains, a sandbox provisioning failure."""
  if isinstance(exc, ExceptionGroup):  # pylint: disable=undefined-variable
    return any(_is_env_provisioning_error(sub) for sub in exc.exceptions)
  return any(
      klass.__name__ in ENV_PROVISIONING_ERROR_NAMES
      for klass in type(exc).__mro__
  )


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
      engine_kwargs: Optional[Dict[str, Any]] = None,
      max_concurrency: Optional[int] = None,
      degrade_on_env_failure: bool = True,
  ):
    """Initializes the RolloutOrchestrator.

    The orchestrator manages a pool of trajectory collection engines, each
    running an agent-environment interaction to collect a trajectory.
    Each output trajectory is considered an "episode".

    Args:
      rollout_sync_lock: A lock to synchronize the very start of multiple
        parallel rollout operations, ensuring they don't all start at the exact
        same moment, potentially overwhelming resources.
      engine_cls: The class used to instantiate trajectory collection engines.
        Each engine is responsible for running a single episode of interaction
        between an agent and an environment.
      engine_kwargs: A dictionary of default keyword arguments to be passed to
        the `engine_cls` constructor when creating new engine instances.
      max_concurrency: The maximum number of agent-environment interaction
        episodes to run in parallel. This limits the number of concurrent calls
        to the underlying language model.
      degrade_on_env_failure: If True, a trajectory whose environment could not
        be provisioned (a sandbox/fleet infrastructure failure) is replaced by
        an empty, fully-masked trajectory with zero reward instead of aborting
        the run. The group still closes, so training continues on a degraded
        batch. Any other exception still propagates.
    """
    self.engine_cls = engine_cls
    self.engine_kwargs = engine_kwargs or {}
    self.max_concurrency = max_concurrency
    self.degrade_on_env_failure = degrade_on_env_failure
    self._tasks: List[asyncio.Task] = []
    self._stop = asyncio.Event()
    self._group_queue_manager: Optional[GroupQueueManager] = None
    self._rollout_sync_lock = rollout_sync_lock

  async def _collect_trajectory(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      mode: Optional[str] = None,
      model_call_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Trajectory:
    """Helper method to collect a single trajectory."""
    engine_kwargs = self.engine_kwargs.copy()
    if model_call_kwargs:
      engine_kwargs["model_call_kwargs"] = model_call_kwargs
    engine = self.engine_cls(agent, env, **engine_kwargs)
    if mode:
      return await engine.collect(mode)
    return await engine.collect()

  def _make_failed_trajectory(self, env: BaseTaskEnv) -> Dict[str, Any]:
    """Builds a zero-reward, fully-masked stand-in for an unprovisionable env.

    The payload comes from `trajectory_collect_engine.token_trajectory`, the
    same builder the real collector uses, so it is a `mode="Token"` result by
    construction rather than by resemblance. It describes a trajectory in which
    the agent produced no tokens, which downstream padding already handles: the
    completion mask is empty and therefore pads to all zeros, so the trajectory
    contributes nothing to the loss. The point is that it still enters its
    group, so the group reaches `group_size` and the consumer does not stall
    waiting for a trajectory that will never arrive.

    Args:
      env: The environment whose provisioning failed.

    Returns:
      A Token-mode trajectory payload.
    """
    task = getattr(env, "task", None) or {}
    return trajectory_collect_engine.token_trajectory(
        prompt_tokens=self._placeholder_prompt_tokens(),
        policy_version=task.get("policy_version"),
        original_input=task,
        group_id=env.extra_kwargs.get("group_id"),
    )

  def _placeholder_prompt_tokens(self) -> np.ndarray:
    """A one-token, non-pad prompt for a placeholder trajectory.

    A prompt of length zero left-pads to a row that is entirely padding. Every
    real sequence already contains fully-padded positions (that is what left
    padding is), so this is very likely harmless, but a single genuine token
    makes the placeholder structurally identical to an ordinary sequence and
    removes the question entirely.

    Returns:
      A 1-D int32 array, empty if no tokenizer was configured or if it could
      not produce a usable token.
    """
    tokenizer = self.engine_kwargs.get("tokenizer")
    if tokenizer is None:
      return np.array([], dtype=np.int32)
    try:
      pad_id = tokenizer.pad_id() if hasattr(tokenizer, "pad_id") else None
      token_ids = [t for t in tokenizer.encode(".") if t != pad_id]
      return np.array(token_ids[:1], dtype=np.int32)
    except Exception as e:  # pylint: disable=broad-except
      # Not fatal: an empty prompt still pads to a valid row. But a tokenizer
      # that cannot encode "." is broken in a way that will bite elsewhere, so
      # do not let it pass unrecorded.
      logging.warning(
          "Tokenizer could not produce a placeholder prompt token (%s);"
          " falling back to an empty prompt.",
          e,
      )
      return np.array([], dtype=np.int32)

  async def _run_and_queue_one_episode(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      manager: GroupQueueManager,
      group_key_fn: Callable[[int, BaseTaskEnv, Trajectory], Hashable],
      start_step_fn: Optional[Callable[[], int]],
      collect_mode: Optional[str],
  ):
    """Collects one trajectory and queues it."""
    pair_idx = env.extra_kwargs["pair_index"]
    try:
      traj = await self._collect_trajectory(agent, env, mode=collect_mode)
    except Exception as e:  # pylint: disable=broad-except
      # The placeholder is a Token-mode payload, so it can only stand in for a
      # Token-mode collection; and only an infrastructure failure earns a
      # stand-in, since a bug in the training code should still stop the run.
      degradable = (
          self.degrade_on_env_failure
          and collect_mode == "Token"
          and _is_env_provisioning_error(e)
      )
      if not degradable:
        raise
      logging.error(
          "Sandbox provisioning failed for pair %d (group %s); degrading this"
          " trajectory to a zero-reward, fully-masked placeholder instead of"
          " failing the run: %s",
          pair_idx,
          env.extra_kwargs.get("group_id"),
          e,
      )
      traj = self._make_failed_trajectory(env)
    gid = group_key_fn(pair_idx, env, traj)
    start_step = start_step_fn() if start_step_fn else 0
    item = TrajectoryItem(
        prompt_id=gid,
        group_index=pair_idx,
        start_step=start_step,
        traj=traj,
        metadata={"generation_id": pair_idx},
    )
    await manager.put(item)
    return 1

  async def _runner(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      manager: GroupQueueManager,
      group_key_fn: Callable[[int, BaseTaskEnv, Trajectory], Hashable],
      start_step_fn: Optional[Callable[[], int]] = None,
      collect_mode: Optional[str] = None,
  ):
    """Runs the trajectory collection loop for a single agent-environment pair.

    This method continuously collects trajectories using `_collect_trajectory`
    and puts them into the `GroupQueueManager`. It handles potential exceptions
    during trajectory collection and respects the `_stop` event and
    `num_episodes` limit.

    Args:
      agent: The ConversationAgentBase instance.
      env: The BaseTaskEnv instance.
      manager: The GroupQueueManager to put collected trajectories into.
      group_key_fn: A callable to determine the group ID for a trajectory.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.
      collect_mode: An optional string to select the collection mode.
    """
    episode_count = 0
    logging.debug(
        "Starting generating trajectories(_runner) for pair %d",
        env.extra_kwargs["pair_index"],
    )

    try:
      # Parallel execution for the group
      self._rollout_sync_lock.acquire_rollout()
      try:
        episode_count = await self._run_and_queue_one_episode(
            agent=agent,
            env=env,
            manager=manager,
            group_key_fn=group_key_fn,
            start_step_fn=start_step_fn,
            collect_mode=collect_mode,
        )
      finally:
        self._rollout_sync_lock.release_rollout()
    except Exception as e:
      if isinstance(e, ExceptionGroup):
        for sub_e in e.exceptions:
          logging.error(
              "Fatal error in runner for pair %d: %s",
              env.extra_kwargs["pair_index"],
              sub_e,
          )
      else:
        logging.error(
            "Fatal error in runner for pair %d: %s",
            env.extra_kwargs["pair_index"],
            e,
        )
      traceback.print_exc()
      await manager.put_exception(e)
      raise e
    finally:
      logging.debug(
          "Runner for pair %d completed with %d episodes",
          env.extra_kwargs["pair_index"],
          episode_count,
      )

  async def run_producers_from_stream(
      self,
      pairs_stream: (
          Iterable[Tuple[ConversationAgentBase, BaseTaskEnv]]
          | AsyncIterable[Tuple[ConversationAgentBase, BaseTaskEnv]]
      ),
      *,
      group_size: int,
      group_key_fn: Callable[
          [int, BaseTaskEnv, Trajectory], Hashable
      ] = lambda i, _, __: i,
      collect_mode: Optional[str] = None,
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
      group_key_fn: A callable that takes `(pair_index, env, trajectory)` and
        returns a hashable group identifier. Using a callable allows for
        flexible grouping strategies. For example, trajectories can be grouped
        by task properties from the environment (`env`) or by outcomes within
        the collected trajectory (`trajectory`). The default is to group by the
        agent-environment pair index.
      collect_mode: An optional string to select the collection mode for
        `TrajectoryCollectEngine`.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.

    Raises:
      ValueError: If `max_concurrency` is not set.
      RuntimeError: If the orchestrator is already running.
    """
    logging.info(
        "Starting run_producers_from_stream with %d concurrency",
        self.max_concurrency,
    )

    if not self.max_concurrency:
      raise ValueError("max_concurrency must be set to use start_producers.")
    if self._group_queue_manager:
      raise RuntimeError("Orchestrator is already running.")

    self._group_queue_manager = GroupQueueManager(
        key_fn=lambda x: getattr(x, "prompt_id", id(x)),
        group_size=group_size,
    )
    self._stop.clear()
    self._tasks.clear()

    is_async_stream = hasattr(pairs_stream, "__aiter__")
    if is_async_stream:
      pairs_iterator = aiter(pairs_stream)  # pytype: disable=wrong-arg-types
    else:
      pairs_iterator = iter(pairs_stream)  # pyrefly: ignore[no-matching-overload]
    active_tasks: set[asyncio.Task] = set()
    stream_exhausted = False

    try:
      logging.debug(
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
            if is_async_stream:
              agent, env = await anext(pairs_iterator)  # pytype: disable=name-error
            else:
              agent, env = next(pairs_iterator)  # pyrefly: ignore[bad-argument-type]
            task = asyncio.create_task(
                self._runner(
                    agent=agent,
                    env=env,
                    manager=self._group_queue_manager,
                    group_key_fn=group_key_fn,
                    start_step_fn=start_step_fn,
                    collect_mode=collect_mode,
                )
            )
            active_tasks.add(task)
            self._tasks.append(task)
          except (StopIteration, StopAsyncIteration):
            logging.debug("Pairs stream exhausted.")
            stream_exhausted = True
            break
          except Exception as e:
            logging.error(
                "Error getting next trajectory: %s",
                e,
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
      logging.debug("Producer task was cancelled.")
      # The consumer's `finally` block will handle cleanup.
      raise
    except Exception as e:
      logging.error("Producer task failed: %s", e)
      if self._group_queue_manager:
        await self._group_queue_manager.put_exception(e)
      raise
    finally:
      # Shield the final cleanup step to ensure it runs even if the producer
      # task is being cancelled. This prevents leaving the manager in an
      # inconsistent state.
      if self._group_queue_manager:
        await asyncio.shield(self._group_queue_manager.prepare_clear())

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
    if not self._group_queue_manager:
      raise RuntimeError("Producers have not been started.")
    try:
      while not self._stop.is_set():
        batch = await self._group_queue_manager.get_batch(batch_size)
        if not batch:
          # If batch is empty, it means producers are done and queue is empty.
          break
        yield batch
    except (GeneratorExit, asyncio.CancelledError):
      # This is the normal shutdown path when the consumer stops listening.
      pass
    except Exception as e:
      logging.error("Error yielding batches: %s", e)
      raise
    finally:
      # This block executes when the consumer (the 'async for' loop) stops.
      # The primary responsibility here is to signal all producers to stop.
      # We do not await task completion here as that's fragile in a generator's
      # finally block. Instead, we rely on the parent coroutine
      # (`run_producers_from_stream`) to handle the full cleanup, as it has
      # the correct context to await its child tasks.
      self._stop.set()
      logging.debug("Consumer stopped; signaling producers to stop.")
      for t in self._tasks:
        if not t.done():
          t.cancel()
