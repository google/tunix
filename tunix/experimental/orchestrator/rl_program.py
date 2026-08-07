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
from collections.abc import Callable, Iterable, Sequence
import dataclasses
import inspect
from typing import Any

from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_engine_interface


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


class RLProgram:
  """Synchronous RL Program coordinating an iterative RL training loop."""

  def __init__(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      algo: algorithm_adapter.AlgorithmAdapter,
      reward_fns: Sequence[Callable[..., Any]] | None = None,
      assembler: batch_assembly.BatchAssembler | None = None,
      on_step_begin: Callable[[int], None] | None = None,
      on_step_end: Callable[[int, Any], None] | None = None,
      sync_weights: bool = True,
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
    self.policy_version = 0
    self.last_step_result: RLStepResult | None = None

  @property
  def step(self) -> int:
    return self.policy_version

  def step_once(
      self,
      prompts: Sequence[Any],
      **kwargs: Any,
  ) -> Any:
    """Executes a single end-to-end RL training step."""
    current_step = self.policy_version
    if self.on_step_begin:
      self.on_step_begin(current_step)

    # 1. Generate rollouts
    rollouts = _sync_or_async(self.engine.generate(prompts=prompts, **kwargs))

    # 2. Evaluate rewards
    rewards = []
    for item in rollouts:
      r = sum(fn(item) for fn in self.reward_fns) if self.reward_fns else getattr(item, "env_reward", 0.0)
      rewards.append(float(r))

    # 3. Create RLTrainerPayloads via AlgorithmAdapter
    ref_logps = None
    if getattr(self.algo, "requires_reference_kl", False):
      ref_logps = _sync_or_async(self.engine.per_token_logps(datatypes.Role.REFERENCE, items=rollouts))
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
      step_result = _sync_or_async(
          self.engine.train_step(
              batch,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=len(microbatches) > 1,
              apply_optimizer=is_last,
          )
      )

    # 6. Sync weights to rollout replicas
    if self.sync_weights:
      new_version = _sync_or_async(
          self.engine.sync_weights(role=datatypes.Role.ACTOR)
      )
      if isinstance(new_version, int) and new_version > current_step:
        self.policy_version = new_version
      else:
        self.policy_version = current_step + 1
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
      prompts: Sequence[Any],
      **kwargs: Any,
  ) -> list[datatypes.RLTrainerPayload]:
    """Executes evaluation step without updating weights."""
    rollouts = _sync_or_async(self.engine.generate(prompts=prompts, **kwargs))
    rewards = [
        sum(fn(item) for fn in self.reward_fns) if self.reward_fns else getattr(item, "env_reward", 0.0)
        for item in rollouts
    ]
    return self.algo.create_trainer_payloads(rollouts, rewards=rewards)

  def run(
      self,
      train_dataset: Iterable[Sequence[Any]],
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Runs the RL program training loop over the dataset."""
    for idx, prompt_batch in enumerate(train_dataset):
      if num_steps is not None and idx >= num_steps:
        break
      logging.info("RLProgram starting step %d", self.step)
      self.step_once(prompts=prompt_batch, **kwargs)
