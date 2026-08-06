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
  ):
    self.engine = engine
    self.algo = algo
    self.reward_fns = list(reward_fns) if reward_fns else []
    self.assembler = assembler or batch_assembly.SequencePackedBatchAssembler(
        max_packed_len=getattr(algo, "max_packed_len", 8192)
    )
    self.on_step_begin = on_step_begin
    self.on_step_end = on_step_end
    self.policy_version = 0

  @property
  def step(self) -> int:
    return self.policy_version

  def step_once(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
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

    # 5. Execute gradient updates
    step_result = None
    for batch in microbatches:
      step_result = _sync_or_async(
          self.engine.train_step(
              batch,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=False,
              apply_optimizer=True,
          )
      )

    # 6. Sync weights to rollout replicas
    _sync_or_async(self.engine.sync_weights(role=datatypes.Role.ACTOR))

    # 7. Increment step
    self.policy_version = current_step + 1

    if self.on_step_end:
      self.on_step_end(self.policy_version, step_result)

    return step_result

  def eval_step_once(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
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
      train_dataset: Iterable[list[str] | list[list[dict[str, str]]]],
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Runs the RL program training loop over the dataset."""
    for idx, prompt_batch in enumerate(train_dataset):
      if num_steps is not None and idx >= num_steps:
        break
      logging.info("RLProgram starting step %d", self.step)
      self.step_once(prompts=prompt_batch, **kwargs)
