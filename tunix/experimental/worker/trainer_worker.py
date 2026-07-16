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

"""TrainerWorker implementation for role-based isolation."""

from typing import Any, Callable, List

from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import abstract_worker


class TrainerWorker(abstract_worker.Worker):
  """Worker wrapper for a Trainer.

  The TrainerWorker owns an AbstractTrainer and orchestrates its initialization,
  compilation, and execution. It exposes the trainer's API so it can be called
  by an orchestrator (either locally or via RPC).
  """

  def __init__(
      self, trainer_factory: Callable[[], abstract_trainer.AbstractTrainer]
  ):
    """Initializes the TrainerWorker.

    Args:
      trainer_factory: A callable that returns an instantiated AbstractTrainer.
    """
    self._trainer = trainer_factory()
    self._is_running = False

  def initialize(self) -> None:
    """Initializes the worker and the underlying trainer."""
    pass

  def compile(self, dummy_data: Any) -> None:
    """Triggers JIT compilation using the provided dummy_data."""
    self._trainer.compile(dummy_data)

  def start(self) -> None:
    """Starts the worker's main loop."""
    self._is_running = True

  def stop(self) -> None:
    """Gracefully stops the worker."""
    self._is_running = False
    self._trainer.close()

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> "TrainerWorker":
    """Sets the loss function used by `fwd_bwd` (and evaluation)."""
    self._trainer.with_loss_fn(loss_fn, has_aux)
    return self

  def fwd_bwd(self, payload: Any, **kwargs) -> None:
    """Executes forward and backward passes."""
    self._trainer.fwd_bwd(payload, **kwargs)

  def update(self, **kwargs) -> int:
    """Applies the accumulated (mean) gradients as one optimizer update."""
    return self._trainer.update(**kwargs)

  def eval_step(self, payload: Any, **kwargs) -> None:
    """Executes one evaluation step on the given payload."""
    self._trainer.eval_step(payload, **kwargs)

  def save_checkpoint(self, metadata: Any, **kwargs) -> None:
    """Force the trainer to serialize its state (model + optimizer)."""
    self._trainer.save_checkpoint(metadata, **kwargs)

  def restore_checkpoint(self, **kwargs) -> Any:
    """Restore state from latest checkpoint and return the metadata pytree."""
    return self._trainer.restore_checkpoint(**kwargs)

  def prepare_weight_sync(self, **kwargs) -> None:
    """Stages weights for transfer and returns coordinates/metadata for Rollouts to pull."""
    self._trainer.prepare_weight_sync(**kwargs)

  def get_metrics(self) -> List[Any]:
    """Returns and clears the recently collected step metric records."""
    return self._trainer.get_metrics()
