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

"""Synchronous RL Program (rl_program.py) coordinating Engine, Algo, and Assembler."""

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import inspect
from typing import Any, Protocol

from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_engine_interface

RewardFn = Callable[[datatypes.TrajectoryItem], float]


class RLProgram(Protocol):
  """Standard contract for RL training programs running on ClusterOrchestrator."""

  def run(
      self,
      engine: rl_engine_interface.AbstractRLEngine | None = None,
      train_dataset: Iterable[Sequence[datatypes.RolloutRequest]] | None = None,
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> Any:
    ...


def _sync_or_async(coro: Any) -> Any:
  """Executes coroutine synchronously if no loop is running, else returns coro."""
  if inspect.iscoroutine(coro):
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      return coro
    return asyncio.run(coro)
  return coro


async def _await_if_needed(value: Any) -> Any:
  """Awaits async engine calls while still tolerating sync test doubles."""
  if inspect.isawaitable(value):
    return await value
  return value


@dataclasses.dataclass(frozen=True)
class RLStepResult:
  """Summary for the most recent synchronous RL step."""

  step: int
  policy_version: int
  num_rollouts: int
  num_microbatches: int
  reward_mean: float
  reward_std: float
  train_result: Any


def _default_reward(item: datatypes.TrajectoryItem) -> float:
  if hasattr(item, "env_reward"):
    return float(getattr(item, "env_reward", 0.0))
  return 0.0


class SyncRLProgram:
  """Synchronous RL Program coordinating an iterative RL training loop."""

  def __init__(
      self,
      algo: algorithm_adapter.AlgorithmAdapter,
      engine: rl_engine_interface.AbstractRLEngine | None = None,
      reward_fns: Sequence[RewardFn] | None = None,
      assembler: batch_assembly.BatchAssembler | None = None,
      on_step_begin: Callable[[int], None] | None = None,
      on_step_end: Callable[[int, Any], None] | None = None,
      sync_weights: bool = True,
      initial_policy_version: int = 0,
  ):
    self.engine = engine
    self.algo = algo
    self.reward_fns = list(reward_fns) if reward_fns else []
    self.assembler = assembler or batch_assembly.SequencePackedBatchAssembler(
        max_packed_len=getattr(algo, "max_packed_len", 8192)
    )
    self.on_step_begin = on_step_begin
    self.on_step_end = on_step_end
    self.sync_weights = sync_weights
    self.policy_version = initial_policy_version
    # Number of train steps already completed by a previous run and recovered
    # from the trainer's checkpoint. 0 fro a fresh run. Used to skip the slice
    # of the dataset that has already been used for training.
    self._resumed_from_step = 0
    self._restored_checkpoint_metadata: dict[str, Any] = {}
    self.mini_batch_size = 1 # hardcoded for the demo
    # Debug/observability hook for examples and tests; not training state.
    self.last_step_result: RLStepResult | None = None

  @property
  def step(self) -> int:
    return self.policy_version

  def _resolve_engine(
      self, engine: rl_engine_interface.AbstractRLEngine | None = None
  ) -> rl_engine_interface.AbstractRLEngine:
    active_engine = engine or self.engine
    if active_engine is None:
      raise ValueError(
          "SyncRLProgram requires an engine either at construction time or via "
          "ClusterOrchestrator.run_program(engine=...)."
      )
    return active_engine

  def step_once(
      self,
      prompts: Sequence[datatypes.RolloutRequest],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> Any:
    """Executes a single end-to-end RL training step."""
    return _sync_or_async(
        self.astep_once(
            prompts=prompts,
            generation_args=generation_args,
            route_metadata=route_metadata,
            **kwargs,
        )
    )

  async def _resume_from_checkpoint(self) -> None:
    """Realigns the program state with a trainer's restored checkpoint."""
    assert self.engine is not None
    metadata = await self.engine.restore_checkpoint(role=datatypes.Role.ACTOR)
    if not isinstance(metadata, Mapping):
      if metadata is not None:
        logging.warning(
            "restore_checkpoint returned %s, not a mapping; starting from fresh"
            " run.",
            type(metadata).__name__,
        )
      return
    metadata = dict(metadata)
    try:
      restored_step = int(metadata.get("step", 0) or 0)
    except (TypeError, ValueError):
      logging.warning(
          "restore_checkpoint returned a non-integer step %r; starting from"
          " fresh run."
      )
      return
    if restored_step <= 0:
      logging.info("No checkpoint to resume from; starting from step 0.")
      return

    self._restored_checkpoint_metadata = metadata
    self._resumed_from_step = restored_step
    self._step = restored_step
    self.policy_version = restored_step
    recorded_version = metadata.get("policy_version")
    if recorded_version is not None and recorded_version != restored_step:
      logging.info(
          "Checkpoint recorded mid-step policy_version=%s; resuming at the"
          " step-boundary value %d",
          recorded_version,
          restored_step,
      )
    logging.info(
        "Resuming from checkpoint: step=%d policy_version=%d (skipping %d)"
        " already-trained dataset items). Metadata: %s",
        self._step,
        self.policy_version,
        self._resumed_from_step * self.mini_batch_size,
        metadata,
    )

  async def astep_once(
      self,
      prompts: Sequence[datatypes.RolloutRequest],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> Any:
    """Async implementation of one end-to-end RL training step."""
    active_engine = self._resolve_engine()
    current_step = self.policy_version
    if self.on_step_begin:
      self.on_step_begin(current_step)

    # 1. Generate rollouts
    engine_call_kwargs = dict(kwargs)
    if generation_args is not None:
      engine_call_kwargs["generation_args"] = generation_args
    if route_metadata is not None:
      engine_call_kwargs["route_metadata"] = route_metadata
    rollouts = await _await_if_needed(
        active_engine.generate(prompts=prompts, **engine_call_kwargs)
    )

    # 2. Evaluate rewards
    rewards = []
    for item in rollouts:
      r = (
          sum(fn(item) for fn in self.reward_fns)
          if self.reward_fns
          else _default_reward(item)
      )
      rewards.append(float(r))

    # 3. Create RLTrainerPayloads via AlgorithmAdapter
    ref_logps = None
    if getattr(self.algo, "requires_reference_kl", False):
      ref_logps = await _await_if_needed(
          active_engine.per_token_logps(datatypes.Role.REFERENCE, items=rollouts)
      )
    trainer_payloads = self.algo.create_trainer_payloads(
        rollouts, rewards=rewards, ref_logps=ref_logps
    )

    # 4. Pack into microbatches
    microbatches = self.assembler.pack(trainer_payloads)
    if not microbatches:
      raise RuntimeError("No trainer microbatches were assembled.")

    # 5. Execute gradient updates
    step_result = None
    for index, batch in enumerate(microbatches):
      is_last = index == len(microbatches) - 1
      step_result = await _await_if_needed(
          active_engine.train_step(
              batch,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=len(microbatches) > 1,
              apply_optimizer=is_last,
          )
      )
      if is_last:
        logging.warning(f"RLProgram.step_once: current_step={current_step} policy_version={self.policy_version}")
        await _await_if_needed(
          active_engine.save_checkpoint(
              role=datatypes.Role.ACTOR,
              metadata={
                  "step": self.policy_version+1,
                  "policy_version": self.policy_version+1,
                  "num_rollouts": len(rollouts),
                  "num_microbatches": len(microbatches),
              },
              **kwargs,
          )
        )


    # 6. Sync weights to rollout replicas
    if self.sync_weights:
      new_version = await _await_if_needed(
          active_engine.sync_weights(role=datatypes.Role.ACTOR)
      )
      if not isinstance(new_version, int) or new_version <= current_step:
        raise RuntimeError(
            "sync_weights must return a monotonically increasing int policy "
            f"version; got {new_version!r} at step {current_step}."
        )
      self.policy_version = new_version
    else:
      self.policy_version = current_step + 1
    self.last_step_result = RLStepResult(
        step=current_step,
        policy_version=self.policy_version,
        num_rollouts=len(rollouts),
        num_microbatches=len(microbatches),
        reward_mean=float(np.mean(rewards)) if rewards else 0.0,
        reward_std=float(np.std(rewards)) if rewards else 0.0,
        train_result=step_result,
    )

    if self.on_step_end:
      self.on_step_end(self.policy_version, step_result)

    return step_result

  def eval_step_once(
      self,
      prompts: Sequence[datatypes.RolloutRequest],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.RLTrainerPayload]:
    """Executes evaluation step without updating weights."""
    return _sync_or_async(
        self.aeval_step_once(
            prompts=prompts,
            generation_args=generation_args,
            route_metadata=route_metadata,
            **kwargs,
        )
    )

  async def aeval_step_once(
      self,
      prompts: Sequence[datatypes.RolloutRequest],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.RLTrainerPayload]:
    """Async implementation of evaluation without updating weights."""
    active_engine = self._resolve_engine()
    engine_call_kwargs = dict(kwargs)
    if generation_args is not None:
      engine_call_kwargs["generation_args"] = generation_args
    if route_metadata is not None:
      engine_call_kwargs["route_metadata"] = route_metadata
    rollouts = await _await_if_needed(
        active_engine.generate(prompts=prompts, **engine_call_kwargs)
    )
    rewards = [
        (
            sum(fn(item) for fn in self.reward_fns)
            if self.reward_fns
            else _default_reward(item)
        )
        for item in rollouts
    ]
    return self.algo.create_trainer_payloads(rollouts, rewards=rewards)

  def run(
      self,
      engine: rl_engine_interface.AbstractRLEngine | None = None,
      train_dataset: Iterable[Sequence[datatypes.RolloutRequest]] | None = None,
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Runs the RL program training loop over the dataset."""
    return _sync_or_async(
        self.arun(
            engine=engine,
            train_dataset=train_dataset,
            num_steps=num_steps,
            **kwargs,
        )
    )

  async def arun(
      self,
      engine: rl_engine_interface.AbstractRLEngine | None = None,
      train_dataset: Iterable[Sequence[datatypes.RolloutRequest]] | None = None,
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Async implementation of the RL program training loop."""
    print(f"SyncRLProgram.arun: num_steps={num_steps}")
    active_engine = self._resolve_engine(engine)
    self.engine = active_engine
    # Must happen before any stage starts: train_stage reas `_step` for its
    # loop bound and rollout_dipatch_stage reads `_resumed_from_step` to skip
    # the dataset prefix the previous run alreayd consumed.
    await self._resume_from_checkpoint()
    if train_dataset is None:
      raise ValueError("SyncRLProgram.run requires a train_dataset.")
    for idx, prompt_batch in enumerate(train_dataset):
      # TODO(tunix-dev): current skip logic assumes mini_batch_size is the same as
      # global batch size. We should support the case that one global batch
      # contains multiple mini-batches.
      print(f"prompt_batch length: {len(prompt_batch)}")
      already_consumed = self._resumed_from_step * self.mini_batch_size
      if idx < already_consumed:
        continue
      if num_steps is not None and idx >= num_steps:
        break
      logging.info("RLProgram starting step %d", self.policy_version)
      await self.astep_once(prompts=prompt_batch, **kwargs)
