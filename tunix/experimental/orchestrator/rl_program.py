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

"""RL Program (Layer 4) for iterative reinforcement learning training loops.

Orchestrates iterative rollout generation, reward/advantage computation,
gradient updates, and weight synchronization using an RLDriver.
"""

from collections.abc import Callable, Iterable
from typing import Any

from absl import logging
from tunix.experimental.orchestrator import rl_driver


class RLProgram:
  """RL Program coordinating an RL training loop over an RLDriver.

  Executes the iterative reinforcement learning loop:
  1. Generate rollout completions via RLDriver.
  2. Compute rewards, advantages, and TIS weights via RLDriver.
  3. Perform gradient update steps (train_step) on trainer workers.
  4. Synchronize policy weights across rollout replicas.
  5. Increment policy version and invoke lifecycle callbacks.
  """

  def __init__(
      self,
      driver: rl_driver.RLDriver,
      on_step_begin: Callable[[int], None] | None = None,
      on_step_end: Callable[[int, Any], None] | None = None,
  ):
    """Initializes RLProgram.

    Args:
      driver: The RLDriver (Layer 3) instance providing algorithm math and
        compute.
      on_step_begin: Optional callback invoked before each training step with
        step number.
      on_step_end: Optional callback invoked after each training step with step
        number and result.
    """
    self.driver = driver
    self.on_step_begin = on_step_begin
    self.on_step_end = on_step_end

  @property
  def step(self) -> int:
    """Returns the current step (policy version) of the program."""
    return self.driver.policy_version

  def step_once(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      **kwargs: Any,
  ) -> Any:
    """Executes a single end-to-end RL training step.

    Args:
      prompts: Batch of prompts to generate rollouts for.
      **kwargs: Additional keyword arguments forwarded to driver methods.

    Returns:
      The step result from the trainer worker.
    """
    current_step = self.driver.policy_version
    if self.on_step_begin:
      self.on_step_begin(current_step)

    # 1. Generate rollouts
    rollouts = self.driver.generate(prompts=prompts, **kwargs)

    # 2. Process results (compute rewards and advantages)
    train_examples = self.driver.process_results(rollouts)

    # 3. Execute gradient updates across examples
    step_result = None
    for example in train_examples:
      step_result = self.driver.train_step(example, **kwargs)

    # 4. Sync policy weights to rollout workers
    self.driver.sync_weights()

    # 5. Increment policy version / step counter
    self.driver.policy_version = current_step + 1

    if self.on_step_end:
      self.on_step_end(self.driver.policy_version, step_result)

    return step_result

  def eval_step_once(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      **kwargs: Any,
  ) -> list[Any]:
    """Executes a single evaluation step without weight updates.

    Args:
      prompts: Batch of prompts to evaluate.
      **kwargs: Additional keyword arguments forwarded to driver methods.

    Returns:
      The processed evaluation examples containing rewards and metrics.
    """
    rollouts = self.driver.generate(prompts=prompts, **kwargs)
    return self.driver.process_results(rollouts)

  def run(
      self,
      train_dataset: Iterable[list[str] | list[list[dict[str, str]]]],
      num_steps: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Runs the RL program training loop over the dataset.

    Args:
      train_dataset: An iterable yielding batches of prompts.
      num_steps: Optional maximum number of steps to execute.
      **kwargs: Additional keyword arguments passed to step_once.
    """
    for idx, prompt_batch in enumerate(train_dataset):
      if num_steps is not None and idx >= num_steps:
        break
      logging.info("RLProgram starting step %d", self.step)
      self.step_once(prompts=prompt_batch, **kwargs)

