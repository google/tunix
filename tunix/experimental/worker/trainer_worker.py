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

import contextlib
from typing import Any, Callable, ContextManager, cast

from tunix.experimental.common import datatypes
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import abstract_worker

WorkerState = datatypes.WorkerState


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
      worker_id: str = "trainer_worker",
  ):
    """Initializes the TrainerWorker.

    Args:
      trainer_factory: A callable that returns an instantiated AbstractTrainer.
      worker_id: Unique identifier for this worker.
    """
    self._trainer = trainer_factory()
    self._is_running = False
    self._worker_id = worker_id
    self._state = WorkerState.PENDING
    self._last_error: str | None = None

  def _policy_version(self) -> int:
    return int(getattr(self._trainer, "policy_version", 0))

  def _response(self, **metadata: Any) -> datatypes.Response:
    return datatypes.Response(
        metadata={
            "worker_id": self._worker_id,
            "state": self.state.value,
            "policy_version": self._policy_version(),
            **metadata,
        }
    )

  def _ensure_ready(self) -> None:
    if self.state == WorkerState.PENDING:
      self.initialize()
    if self.state != WorkerState.READY:
      raise RuntimeError(f"TrainerWorker is not ready: {self.state.value}.")

  def initialize(self) -> datatypes.Response:
    """Initializes the worker and the underlying trainer."""
    if self.state == WorkerState.READY:
      return self._response(initialized=True, already_ready=True)
    self.state = WorkerState.INITIALIZING
    try:
      return self._response(initialized=True)
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    """Triggers JIT compilation using the provided dummy_data."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.COMPILING
    try:
      self._trainer.compile(dummy_data)
      return self._response(compiled=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      if self.state == WorkerState.COMPILING:
        self.state = WorkerState.READY

  def start(self) -> datatypes.Response:
    """Starts the worker's main loop."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    if self.state != WorkerState.READY:
      raise RuntimeError(f"Cannot start TrainerWorker from {self.state.value}.")
    self._is_running = True
    return self._response(started=True)

  def stop(self) -> datatypes.Response:
    """Gracefully stops the worker."""
    if self.state == WorkerState.STOPPED:
      return self._response(stopped=True, already_stopped=True)
    self._is_running = False
    if self.state == WorkerState.READY:
      self.state = WorkerState.DRAINING
    self._trainer.close()
    self.state = WorkerState.STOPPED
    return self._response(stopped=True)

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id,
        roles=frozenset({"trainer", "weight_sync"}),
        resources={
            "trainer": type(self._trainer).__name__,
            "policy_version": self._policy_version(),
        },
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state,
        policy_version=self._policy_version(),
        last_error=self._last_error,
    )

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> datatypes.Response:
    """Sets the loss function used by `fwd_bwd` (and evaluation)."""
    self._trainer.with_loss_fn(loss_fn, has_aux)
    return self._response(loss_fn_configured=True)

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> datatypes.Response:
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs."""
    self._trainer.with_gen_model_input_fn(gen_model_input_fn)
    return self._response(gen_model_input_fn_configured=True)

  def fwd_bwd(
      self,
      payload: datatypes.TrainerPayload,
      **kwargs,
  ) -> datatypes.Response:
    """Executes one forward/backward pass."""
    self._ensure_ready()
    try:
      self._trainer.fwd_bwd(payload, **kwargs)
      self._last_error = None
      return self._response(queued=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def update(self, **kwargs) -> int:
    """Applies the accumulated (mean) gradients as one optimizer update."""
    self._ensure_ready()
    try:
      train_step = self._trainer.update(**kwargs)
      self._last_error = None
      return train_step
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def eval_step(
      self, payload: datatypes.TrainerPayload, **kwargs
  ) -> datatypes.Response:
    """Executes one evaluation step on the given payload."""
    self._ensure_ready()
    try:
      self._trainer.eval_step(payload, **kwargs)
      self._last_error = None
      return self._response(evaluated=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def run_eval(self, eval_ds: Any, **kwargs) -> datatypes.Response:
    """Runs an explicit evaluation phase over eval micro-batches."""
    self._ensure_ready()
    if eval_ds is None:
      return self._response(evaluated=True, eval_batches=0)
    try:
      run_eval = getattr(self._trainer, "run_eval", None)
      if callable(run_eval):
        run_eval(eval_ds, **kwargs)
        self._last_error = None
        return self._response(evaluated=True)

      eval_context = getattr(self._trainer, "eval_context", None)
      context = (
          cast(ContextManager[Any], eval_context())
          if callable(eval_context)
          else contextlib.nullcontext()
      )
      eval_batches = 0
      with context:
        for payload in eval_ds:
          self._trainer.eval_step(payload, **kwargs)
          eval_batches += 1
      self._last_error = None
      return self._response(evaluated=True, eval_batches=eval_batches)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def save_checkpoint(self, metadata: Any, **kwargs) -> datatypes.Response:
    """Force the trainer to serialize its state (model + optimizer)."""
    self._ensure_ready()
    try:
      self._trainer.save_checkpoint(metadata, **kwargs)
      self._last_error = None
      return self._response(checkpoint_saved=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def restore_checkpoint(self, **kwargs) -> Any:
    """Restore state from latest checkpoint and return the metadata pytree."""
    return self._trainer.restore_checkpoint(**kwargs)

  def prepare_weight_sync(self, **kwargs) -> Any:
    """Stages weights for transfer and returns coordinates/metadata."""
    self._ensure_ready()
    self.state = WorkerState.SYNCING
    try:
      metadata = self._trainer.prepare_weight_sync(**kwargs)
      self.state = WorkerState.READY
      self._last_error = None
      if metadata is not None:
        return metadata
      return self._response(weight_sync_ready=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def get_metrics(self) -> Any:
    """Returns and clears the recently collected step metric records."""
    return self._trainer.get_metrics()
