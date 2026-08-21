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

"""Multi-stage reinforcement learning programs.

Provides modular program abstractions for orchestrating  RL training
pipelines.
"""

import abc
import asyncio
from collections.abc import Callable, Iterable, Sequence
import dataclasses
from typing import Any

from absl import logging
import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.queue_manager import trajectory_queue_manager
from tunix.rl import common as rl_common


@dataclasses.dataclass(kw_only=True)
class RLStepResult:
  """Summary of a completed RL training step."""

  step: int
  policy_version: int
  num_rollouts: int
  num_microbatches: int
  reward_mean: float
  reward_std: float
  train_result: Any = None


class RLProgram(abc.ABC):
  """Base class for multi-stage DAG workflows."""

  def __init__(self):
    self._is_running = False
    self.policy_version = 0
    self.last_step_result: RLStepResult | None = None

  @property
  def step(self) -> int:
    return self.policy_version

  @abc.abstractmethod
  def run(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      train_dataset: Iterable[Any] | None = None,
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Entry point running all stages on an event loop."""
    raise NotImplementedError("Subclasses must implement run.")


class StandardRLProgram(RLProgram):
  """Standard RL program handling common multi-stage training workflows asynchronously.

  Runs 4 concurrent stages:
  1. Rollout dispatch stage: Fire-and-forget requests across worker pool.
  2. Polling stage: Long-polls completed rollout responses into grouping queue.
  3. Critique stage: Scores rewards, PRMs, and reference KL logprobs.
  4. Train stage: Streaming gradient accumulation over microbatches.
  """

  def __init__(
      self,
      algo: algorithm_adapter.AlgorithmAdapter,
      dataset: Iterable[Any] | None = None,
      reward_fns: Sequence[Callable[..., Any]] | None = None,
      assembler: batch_assembly.BatchAssembler | None = None,
      group_size: int = 8,
      mini_batch_size: int = 4,
      max_staleness: int | None = None,
      sync_weights: bool = True,
      on_step_begin: Callable[[int], None] | None = None,
      on_step_end: Callable[[int, Any], None] | None = None,
  ):
    super().__init__()
    self.dataset = dataset
    self.algo = algo
    self.reward_fns = list(reward_fns) if reward_fns else []
    self.group_size = getattr(algo, "group_size", group_size)
    self.mini_batch_size = getattr(algo, "mini_batch_size", mini_batch_size)
    self.assembler = assembler or batch_assembly.SequencePackedBatchAssembler(
        max_packed_len=getattr(algo, "max_packed_len", 8192)
    )
    self.sync_weights = sync_weights
    self.on_step_begin = on_step_begin
    self.on_step_end = on_step_end

    self.raw_q = trajectory_queue_manager.TrajectoryQueueManager.create(
        group_size=self.group_size,
        max_staleness=max_staleness,
        current_policy_version=lambda: self.policy_version,
    )
    self.scored_q = trajectory_queue_manager.TrajectoryQueueManager.create(
        group_size=self.group_size
    )

  async def rollout_dispatch_stage(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      train_dataset: Iterable[Any] | None = None,
  ) -> None:
    """Stage 1A: Dispatches rollout requests across workers asynchronously."""
    active_dataset = (
        train_dataset if train_dataset is not None else self.dataset
    )
    if active_dataset is None:
      raise ValueError(
          "StandardRLProgram requires a dataset either at init or in run()."
      )
    for prompt_idx, prompt_item in enumerate(active_dataset):
      # TODO: Extract prompt_id and group_id from standard tunix data
      # structures rather than assuming dictionaries or falling back to index
      # strings.
      # TODO: the logic of creating group id and prompt id is incorrect and
      # should be fixed.
      prompt_id = getattr(prompt_item, "prompt_id", f"prompt_{prompt_idx}")
      group_id = getattr(prompt_item, "group_id", f"group_{prompt_idx}")
      if isinstance(prompt_item, dict):
        prompt_id = prompt_item.get("prompt_id", prompt_id)
        group_id = prompt_item.get("group_id", group_id)

      for g_idx in range(self.group_size):
        await engine.dispatch_rollouts(
            [prompt_item],
            request_id=f"req_{prompt_idx}_{g_idx}",
            policy_version=self.policy_version,
            prompt_ids=[prompt_id],
            metadata={
                "group_id": group_id,
                "pair_index": g_idx,
            },
        )

  async def polling_stage(
      self, engine: rl_engine_interface.AbstractRLEngine
  ) -> None:
    """Stage 1B: Long-polls completed worker rollout responses into the queue."""
    while True:
      try:
        completed = await engine.poll_rollouts()
        if isinstance(completed, list) and completed:
          for item in completed:
            await self.raw_q.put(item)

      except asyncio.CancelledError:
        break
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning("Error in polling_stage: %s", exc)
        await asyncio.sleep(0.01)

  async def critique_stage(
      self, engine: rl_engine_interface.AbstractRLEngine
  ) -> None:
    """Stage 2: Scores rewards, PRMs, and reference KL logprobs."""
    del engine
    while True:
      try:
        group = await self.raw_q.get_group()
      except asyncio.CancelledError:
        break
      except Exception:
        break

      rewards = []
      for item in group:
        if self.reward_fns:
          r = sum(fn(item) for fn in self.reward_fns)
        else:
          r = getattr(item.traj, "reward", 0.0)
        rewards.append(float(r))

      trainer_payloads = self.algo.create_trainer_payloads(
          group, rewards=rewards
      )
      for idx, payload in enumerate(trainer_payloads):
        adv = payload.advantages
        reward_val = (
            float(adv[0])  # pyrefly: ignore[bad-index]
            if hasattr(adv, "__len__") and len(adv) > 0  # pyrefly: ignore[bad-argument-type]
            else float(adv)  # pyrefly: ignore[bad-argument-type]
        )
        item = datatypes.TrajectoryItem(
            pair_index=idx,
            group_id=getattr(group[0], "group_id", "default"),
            start_step=0,
            traj=datatypes.Trajectory(reward=reward_val),
            # TODO: Stream RLTrainerPayload directly instead of
            # re-wrapping in TrajectoryItem.
        )
        item.payload = payload  # pyrefly: ignore[missing-attribute]
        await self.scored_q.put(item)

  async def train_stage(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      num_steps: int | None = None,
  ) -> None:
    """Stage 3: Streaming gradient accumulation with RLTrainerPayloads."""
    step = 0
    while num_steps is None or step < num_steps:
      if self.on_step_begin:
        self.on_step_begin(self.step)

      uncommitted_groups = []
      step_result = None
      step_rewards = []
      num_microbatches = 0
      num_rollouts = 0

      for group_idx in range(self.mini_batch_size):
        scored_items = await self.scored_q.get_batch(num_groups=1)
        if not scored_items:
          break
        uncommitted_groups.append(scored_items)
        num_rollouts += len(scored_items)
        for item in scored_items:
          step_rewards.append(float(getattr(item.traj, "reward", 0.0)))

        payloads = [getattr(item, "payload", None) for item in scored_items]
        # TODO: Implement streaming microbatch assembly to overlap packing
        # with trainer execution.
        microbatches = self.assembler.pack(payloads)  # pyrefly: ignore[bad-argument-type]
        if getattr(self.algo, "requires_reference_kl", False):
          scored_microbatches = []
          for batch in microbatches:
            if not isinstance(batch, rl_common.TrainExample):
              raise TypeError(
                  "Reference KL requires an assembler that returns "
                  "rl_common.TrainExample microbatches; got "
                  f"{type(batch).__name__}."
              )
            ref_logps = await engine.per_token_logps(
                datatypes.Role.REFERENCE, items=batch
            )
            scored_microbatches.append(
                batch_assembly.with_ref_per_token_logps(batch, ref_logps)
            )
          microbatches = scored_microbatches

        num_microbatches += len(microbatches)
        is_final = group_idx == self.mini_batch_size - 1
        for batch in microbatches:
          step_result = await engine.train_step(
              batch,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=is_final,
          )

      if self.sync_weights:
        new_version = await engine.sync_weights(role=datatypes.Role.ACTOR)
        self.policy_version = new_version if new_version else self.step + 1
      else:
        self.policy_version = self.step + 1

      self.scored_q.commit(step, groups=uncommitted_groups)

      self.last_step_result = RLStepResult(
          step=step,
          policy_version=self.policy_version,
          num_rollouts=num_rollouts,
          num_microbatches=num_microbatches,
          reward_mean=float(np.mean(step_rewards)) if step_rewards else 0.0,
          reward_std=float(np.std(step_rewards)) if step_rewards else 0.0,
          train_result=step_result,
      )

      if self.on_step_end:
        self.on_step_end(self.step, step_result)
      step += 1

  async def run_async(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      train_dataset: Iterable[Any] | None = None,
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Launches all stages concurrently on event loop."""
    del kwargs
    logging.info("Starting StandardRLProgram concurrent stages...")

    train_task = asyncio.create_task(self.train_stage(engine, num_steps))
    tasks = [
        asyncio.create_task(
            self.rollout_dispatch_stage(engine, train_dataset=train_dataset)
        ),
        asyncio.create_task(self.polling_stage(engine)),
        asyncio.create_task(self.critique_stage(engine)),
        train_task,
    ]

    try:
      while not train_task.done():
        done, _ = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.05
        )
        for task in done:
          if task.exception():
            raise task.exception()  # pyrefly: ignore[bad-raise]
      if train_task.exception():
        raise train_task.exception()  # pyrefly: ignore[bad-raise]
    except Exception as exc:
      logging.error("Exception in StandardRLProgram execution: %s", exc)
      await self.raw_q.abort(exc)
      await self.scored_q.abort(exc)
      raise
    finally:
      for task in tasks:
        if not task.done():
          task.cancel()

  def run(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      train_dataset: Iterable[Any] | None = None,
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Synchronous entry point running all stages on an event loop."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    def _retrieve_task_exception(t: asyncio.Task[Any]) -> None:
      try:
        t.result()
      except Exception:  # pylint: disable=broad-except
        # Exception is already logged inside run_async, we just need to
        # retrieve it so asyncio doesn't complain about unretrieved exceptions.
        pass

    if loop and loop.is_running():
      self._bg_task = asyncio.create_task(
          self.run_async(
              engine, train_dataset=train_dataset, num_steps=num_steps, **kwargs
          )
      )
      self._bg_task.add_done_callback(_retrieve_task_exception)
    else:
      asyncio.run(
          self.run_async(
              engine, train_dataset=train_dataset, num_steps=num_steps, **kwargs
          )
      )
