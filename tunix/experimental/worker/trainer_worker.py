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

from typing import Any, Callable

from tunix.experimental.common import datatypes
from tunix.experimental.metrics import metrics
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import abstract_worker


class TrainerWorker(abstract_worker.Worker):
  """Worker wrapper for a Trainer.

  The TrainerWorker owns an AbstractTrainer and orchestrates its initialization,
  compilation, and execution. It exposes the trainer's API so it can be called
  by an orchestrator (either locally or via RPC).
  """

  def __init__(
      self,
      trainer_factory: Callable[[], abstract_trainer.AbstractTrainer],
      *,
      worker_id: str = "trainer",
  ):
    """Initializes the TrainerWorker.

    Args:
      trainer_factory: A callable that returns an instantiated AbstractTrainer.
        It is invoked in `initialize()`, not here, so constructing the worker
        stays cheap and free of device/model side effects.
      worker_id: Unique id reported via `info()`.
    """
    self._trainer_factory = trainer_factory
    self._trainer: abstract_trainer.AbstractTrainer | None = None
    self._worker_id = worker_id
    self._is_running = False

  def initialize(self) -> None:
    """Constructs the underlying trainer (deferred from __init__)."""
    if self._trainer is None:
      self._trainer = self._trainer_factory()

  def compile(self, shape_config: datatypes.ShapeConfig) -> None:
    """Triggers JIT compilation using the provided shape hints."""
    self._trainer.compile(shape_config)

  def start(self) -> None:
    """Starts the worker's main loop."""
    self._is_running = True

  def stop(self) -> None:
    """Gracefully stops the worker."""
    self._is_running = False
    if self._trainer is not None:
      self._trainer.close()

  def health(self) -> datatypes.HealthReport:
    """Returns a liveness/status snapshot."""
    return datatypes.HealthReport(
        state="READY" if self._is_running else "STOPPED"
    )

  def info(self) -> datatypes.WorkerInfo:
    """Returns the worker's static description."""
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"trainer"})
    )

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> "TrainerWorker":
    """Sets the loss function used by `fwd_bwd` (and evaluation)."""
    self._trainer.with_loss_fn(loss_fn, has_aux)
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> "TrainerWorker":
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs."""
    self._trainer.with_gen_model_input_fn(gen_model_input_fn)
    return self

  def fwd_bwd(
      self, payload: datatypes.TrainerPayload, **kwargs
  ) -> datatypes.StepReceipt:
    """Executes forward and backward passes."""
    return self._trainer.fwd_bwd(payload, **kwargs)

  def update(self, **kwargs) -> datatypes.UpdateResult:
    """Applies the accumulated gradients as one optimizer update."""
    return self._trainer.update(**kwargs)

  def eval_step(
      self, payload: datatypes.TrainerPayload, **kwargs
  ) -> metrics.MetricsBuffer:
    """Executes one evaluation step on the given payload."""
    return self._trainer.eval_step(payload, **kwargs)

  def save_checkpoint(self, metadata: dict[str, Any], **kwargs) -> None:
    """Force the trainer to serialize its state (model + optimizer)."""
    self._trainer.save_checkpoint(metadata, **kwargs)

  def restore_checkpoint(self, **kwargs) -> dict[str, Any]:
    """Restore state from latest checkpoint and return the metadata pytree."""
    return self._trainer.restore_checkpoint(**kwargs)

  def prepare_weight_sync(
      self, spec: datatypes.WeightSyncSpec
  ) -> datatypes.WeightSyncMetadata:
    """Stages weights for transfer and returns metadata for replicas to pull."""
    return self._trainer.prepare_weight_sync(spec)

  def get_metrics(self) -> metrics.MetricsBuffer:
    """Returns and clears the recently collected step metric records."""
    return self._trainer.get_metrics()
