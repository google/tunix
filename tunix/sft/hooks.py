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

"""Hooks for training and data loading."""

from collections.abc import Iterable
import dataclasses
from typing import Any, Optional

from absl import logging
from flax import nnx
from jax.typing import ArrayLike  # pylint: disable=g-importing-member


@dataclasses.dataclass
class TrainingContext:
  """Context for training."""

  model: nnx.Module


class TrainingHooks:
  """Hooks to be used for training."""

  def on_train_start(self, train_ctx: TrainingContext):
    """Called at the beginning of training."""
    pass

  def on_train_end(self):
    """Called at the end of training."""
    pass

  def on_train_step_start(
      self, step: int, max_steps: int | None = None
  ) -> Optional[int]:
    """Called at the beginning of a training step.

    Args:
      step: The current training step.
      max_steps: The maximum number of training steps.

    Returns:
      -1 if the training should be stopped, None otherwise.
    """
    if max_steps is not None and step >= max_steps:
      return -1

  def on_train_step_end(self, step: int, train_loss: ArrayLike):
    """Called at the end of a training step."""
    pass

  def on_eval_step_start(self, eval_step: int) -> Optional[int]:
    """Called at the beginning of evaluation step."""
    pass

  def on_eval_step_end(self, step: int, eval_loss: ArrayLike):
    """Called at the end of an evaluation step."""
    pass


class DataHooks:
  """Hooks to be used for data related operations."""

  def __init__(
      self, train_ds: Iterable[Any], is_managed_externally: bool = False
  ):
    self.index = 0
    self.is_managed_externally = is_managed_externally
    self.train_data_iterator = iter(train_ds)

  def load_next_train_batch(self, step: int) -> Any | None:
    """Loads the next batch of data for training."""
    while True:
      try:
        # TODO(mridulsahu): Add support to restore the iterator state
        # instead of skipping the already trained examples.
        # Skip the examples that are already trained.
        train_example = next(self.train_data_iterator)
        should_return = self.is_managed_externally or (
            not self.is_managed_externally and self.index >= step
        )
        self.index += 1
        if should_return:
          return train_example
      except StopIteration:
        logging.info("Reached end of training dataset.")
        return None
