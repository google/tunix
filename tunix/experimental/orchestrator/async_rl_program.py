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

"""Asynchronous RL Program (Layer 4) for multi-stage concurrent pipelines.

Separates rollout generation, response long-polling, and training into
concurrent stages supervised by an RLDriver and TrajectoryQueueManager.
"""

import asyncio
from collections.abc import Callable, Iterable
from typing import Any, Optional, Sequence

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_driver
from tunix.experimental.queue_manager import trajectory_queue_manager
from tunix.rl import rl_cluster as rl_engine_lib


class AsyncRLProgram:
  """Asynchronous RL Program separating rollout, polling, and training into concurrent stages.

  Supports:
  1. Asynchronous rollout generation across distributed workers.
  2. Long-polling responses via remote execution handles and load balancers.
  3. Out-of-order prompt grouping and staleness filtering via
  TrajectoryQueueManager.
  4. Non-blocking training step and policy weight synchronization.
  """

  def __init__(
      self,
      driver: rl_driver.RLDriver,
      group_size: int = 1,
      batch_size_groups: int = 1,
      max_staleness: Optional[int] = None,
      filter_fn: Optional[Any] = None,
      queue_manager: Optional[
          trajectory_queue_manager.TrajectoryQueueManager
      ] = None,
      on_step_begin: Optional[Callable[[int], None]] = None,
      on_step_end: Optional[Callable[[int, Any], None]] = None,
  ):
    """Initializes AsyncRLProgram.

    Args:
      driver: The RLDriver (Layer 3) providing compute and algorithm math.
      group_size: Number of rollout trajectories per prompt group (GRPO group
        size).
      batch_size_groups: Number of complete prompt groups per training step.
      max_staleness: Maximum allowed policy lag before discarding stale
        rollouts.
      filter_fn: Pluggable filtering function for candidate trajectory groups.
      queue_manager: Optional custom TrajectoryQueueManager instance.
      on_step_begin: Optional callback invoked before each training step with
        step number.
      on_step_end: Optional callback invoked after each training step with step
        number and result.
    """
    self.driver = driver
    self.group_size = group_size
    self.batch_size_groups = batch_size_groups
    self.max_staleness = max_staleness
    self.filter_fn = filter_fn
    self.on_step_begin = on_step_begin
    self.on_step_end = on_step_end
    self._is_running = False

    self.queue_manager = queue_manager or driver.create_queue_manager(
        group_size=group_size,
        max_staleness=max_staleness,
        filter_fn=filter_fn,
    )

  @property
  def step(self) -> int:
    """Returns the current step (policy version) of the program."""
    return self.driver.policy_version

  def _response_to_trajectory_item(self, resp: Any) -> datatypes.TrajectoryItem:
    """Converts a worker response (RolloutResponse, TrajectoryItem, Trajectory) to a TrajectoryItem."""
    if isinstance(resp, datatypes.TrajectoryItem):
      return resp

    if isinstance(resp, datatypes.RolloutResponse):
      prompt_id = resp.prompt_id or "default_prompt"
      metadata = dict(resp.metadata) if resp.metadata else {}
      group_id = metadata.get("group_id", prompt_id)
      pair_index = metadata.get("pair_index", 0)
      traj = datatypes.Trajectory(
          reward=resp.env_reward,
          status=(
              datatypes.TrajectoryStatus.SUCCEEDED
              if resp.status == "COMPLETED"
              else datatypes.TrajectoryStatus.FAILED
          ),
      )
      item = datatypes.TrajectoryItem(
          pair_index=pair_index,
          group_id=group_id,
          start_step=0,
          traj=traj,
          metadata=metadata,
      )
      item.policy_version = resp.policy_version
      return item

    if isinstance(resp, datatypes.Trajectory):
      item = datatypes.TrajectoryItem(
          pair_index=0,
          group_id=getattr(resp, "task", "default_group"),
          start_step=0,
          traj=resp,
      )
      item.policy_version = getattr(resp, "policy_version", 0)
      return item

    raise TypeError(
        f"Unsupported response type for trajectory conversion: {type(resp)}"
    )

  async def rollout_stage(self, train_dataset: Iterable[Any]) -> None:
    """Stage 1: Dispatches rollout requests across workers asynchronously."""
    for prompt_idx, prompt_item in enumerate(train_dataset):
      if not self._is_running:
        break

      prompt_id = f"prompt_{prompt_idx}"
      group_id = f"group_{prompt_idx}"
      if isinstance(prompt_item, dict):
        prompt_id = prompt_item.get("prompt_id", prompt_id)
        group_id = prompt_item.get("group_id", prompt_id)

      requests = []
      for g_idx in range(self.group_size):
        req_id = f"req_{prompt_idx}_{g_idx}"
        req = datatypes.RolloutRequest(
            request_id=req_id,
            prompt=prompt_item,
            prompt_id=prompt_id,
            group_id=group_id,
            target_policy_version=self.driver.policy_version,
            metadata={"group_id": group_id, "pair_index": g_idx},
        )
        requests.append(req)

      await self.driver.dispatch_rollouts(requests)

  async def polling_stage(self) -> None:
    """Stage 2: Long-polls completed worker rollout responses into the queue."""
    while self._is_running:
      try:
        completed = await self.driver.poll_rollouts(timeout_s=0.1)
        if isinstance(completed, list):
          for resp in completed:
            item = self._response_to_trajectory_item(resp)
            await self.queue_manager.put(item)
        if not completed:
          await asyncio.sleep(0.01)
      except asyncio.CancelledError:
        break
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning("Error in polling_stage: %s", exc)
        await asyncio.sleep(0.01)

  async def train_stage(self, num_steps: int) -> None:
    """Stage 3: Dequeues ready prompt groups 1 at a time for immediate train_step, syncing weights on global step boundary."""
    accumulated_groups = 0
    for _ in range(num_steps * self.batch_size_groups):
      if not self._is_running:
        break

      ready_batches = await self.driver.process_ready_groups(
          self.queue_manager,
          num_groups=1,
          mode=rl_engine_lib.Mode.TRAIN,
          expected_step=self.step,
      )
      if not ready_batches:
        break

      if self.on_step_begin and accumulated_groups == 0:
        self.on_step_begin(self.step)

      step_result = None
      for batch in ready_batches:
        step_result = await self.driver.train_step(batch)

      accumulated_groups += 1

      if accumulated_groups == self.batch_size_groups:
        await self.driver.sync_weights()
        self.driver.policy_version = self.step + 1
        accumulated_groups = 0

        if self.on_step_end:
          self.on_step_end(self.step, step_result)

  async def run_async(
      self,
      train_dataset: Iterable[Any],
      num_steps: Optional[int] = None,
      **kwargs: Any,
  ) -> None:
    """Executes rollout, polling, and training stages concurrently."""
    del kwargs
    target_steps = num_steps or 1000000
    self._is_running = True

    logging.info("Starting AsyncRLProgram concurrent stages...")

    train_task = asyncio.create_task(self.train_stage(target_steps))
    tasks = [
        asyncio.create_task(self.rollout_stage(train_dataset)),
        asyncio.create_task(self.polling_stage()),
        train_task,
    ]

    # Include custom critique stage if defined on subclass
    if hasattr(self, "critique_stage"):
      critique_method: Any = getattr(self, "critique_stage")
      tasks.append(asyncio.create_task(critique_method()))

    try:
      while not train_task.done():
        done, _ = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.05
        )
        for task in done:
          if task.exception():
            raise task.exception()
      if train_task.exception():
        raise train_task.exception()
    except Exception as exc:
      logging.error("Exception in AsyncRLProgram execution: %s", exc)
      if hasattr(self.queue_manager, "abort"):
        await self.queue_manager.abort(exc)
      elif hasattr(self.queue_manager, "put_exception"):
        await self.queue_manager.put_exception(exc)
      raise
    finally:
      self._is_running = False
      for task in tasks:
        if not task.done():
          task.cancel()

  def run(
      self,
      train_dataset: Iterable[Any],
      num_steps: Optional[int] = None,
      **kwargs: Any,
  ) -> None:
    """Synchronous entry point running all stages on an event loop."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      asyncio.create_task(self.run_async(train_dataset, num_steps, **kwargs))
    else:
      asyncio.run(self.run_async(train_dataset, num_steps, **kwargs))


class MultiStageAgenticProgram(AsyncRLProgram):
  """An async 3-stage RLProgram: Rollout -> Critique -> Train.

  Defines a 3-stage pipeline (Rollout -> Critique/Scoring -> Train) where each
  stage runs concurrently. Supervised by RLDriver and TrajectoryQueueManager for
  out-of-order prompt grouping and staleness filtering.
  """

  def __init__(
      self,
      driver: rl_driver.RLDriver,
      group_size: int = 8,
      batch_size_groups: int = 4,
      max_staleness: Optional[int] = None,
  ):
    super().__init__(
        driver=driver,
        group_size=group_size,
        batch_size_groups=batch_size_groups,
        max_staleness=max_staleness,
    )
    self.raw_rollouts_q = self.queue_manager
    self.scored_rollouts_q = driver.create_queue_manager(group_size=group_size)

  async def critique_stage(self) -> None:
    """Stage 2: Dequeue raw rollouts, compute PRM / critique rewards -> put into scored_rollouts_q."""
    async for group in self.raw_rollouts_q:
      if not self._is_running:
        break
      scored_group = await self.driver.score_async(group)
      if isinstance(scored_group, (list, tuple)):
        for item in scored_group:
          traj_item = self._response_to_trajectory_item(item)
          await self.scored_rollouts_q.put(traj_item)
      else:
        await self.scored_rollouts_q.put(
            self._response_to_trajectory_item(scored_group)
        )

  async def train_stage(self, num_steps: int) -> None:
    """Stage 3: Dequeue scored rollouts 1 prompt group at a time -> train_step -> sync weights every batch_size_groups."""
    accumulated_groups = 0
    for _ in range(num_steps * self.batch_size_groups):
      if not self._is_running:
        break
      scored_items = await self.scored_rollouts_q.get_batch(num_groups=1)
      if not scored_items:
        break

      if self.on_step_begin and accumulated_groups == 0:
        self.on_step_begin(self.step)

      step_result = await self.driver.train_step(scored_items)
      accumulated_groups += 1

      if accumulated_groups == self.batch_size_groups:
        await self.driver.sync_weights()
        self.driver.policy_version = self.step + 1
        accumulated_groups = 0

        if self.on_step_end:
          self.on_step_end(self.step, step_result)
