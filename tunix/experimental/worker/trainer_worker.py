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

from collections.abc import Mapping
import contextlib
import dataclasses
from typing import Any, Callable, ContextManager, cast

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.train import abstract_trainer
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import abstract_worker
from tunix.rl import common as rl_common

WorkerState = datatypes.WorkerState


def _with_dense_routing(
    payload: datatypes.RLTrainerPayload | None,
) -> datatypes.RLTrainerPayload | None:
  """Expands a compact `routed_experts` on the trainer, right before use."""
  if payload is None:
    return None
  routing = getattr(payload, "routed_experts", None)
  if isinstance(routing, datatypes.CompactRoutedExperts):
    return dataclasses.replace(payload, routed_experts=routing.materialize())
  return payload


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
      logps_chunk_size: int = 0,
      logps_micro_batch_size: int | None = None,
      execution_context: Any = None,
  ):
    """Initializes the TrainerWorker.

    Args:
      trainer_factory: A callable that returns an instantiated AbstractTrainer.
      worker_id: Unique identifier for this worker.
      execution_context: Optional context manager or zero-arg callable returning
        a context manager (e.g., a JAX Mesh) to enter during trainer
        initialization and worker method execution.
      logps_chunk_size: Optionally chunk the vocab (final-logits) computation.
      logps_micro_batch_size: Row chunk size for `per_token_logps`.
        If `None`, the whole request is scored in one forward.
    """
    self._execution_context = execution_context
    self._trainer_factory = trainer_factory
    self._trainer: abstract_trainer.AbstractTrainer | None = None
    self._pending_loss_fn: tuple[Callable[..., Any], bool] | None = None
    self._pending_gen_model_input_fn: (
        Callable[[Any], dict[str, Any]] | None
    ) = None
    self._logps_chunk_size = logps_chunk_size
    self._logps_micro_batch_size = logps_micro_batch_size
    self._is_running = False
    self._worker_id = worker_id
    self._state = WorkerState.PENDING
    self._last_error: str | None = None
    # One micro-batch received ahead of its fwd_bwd (`stage_fwd_bwd`):
    # (request_id, input, fwd_bwd kwargs, request metadata).
    self._staged_fwd_bwd: (
        tuple[str, Any, dict[str, Any], dict[str, Any]] | None
    ) = None
    # The request id of a step's last micro-batch whose update
    # `fwd_bwd(apply_optimizer=True, defer_update=True)` left to
    # `apply_deferred_update`; until then the step is open, so nothing that
    # needs it closed (another micro-batch, metrics, a checkpoint, a weight
    # sync, actor log-probs) is accepted.
    self._deferred_update: str | None = None

  def _refuse_while_update_deferred(self, method: str) -> None:
    if self._deferred_update is not None:
      raise RuntimeError(
          f"TrainerWorker.{method}() refused: the update of the step whose"
          f" last micro-batch was {self._deferred_update!r} is deferred; run"
          " apply_deferred_update() first."
      )

  def _policy_version(self) -> int:
    if self._trainer is None:
      return 0
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
      return self._response(initialized=True, ready=True)
    self.state = WorkerState.INITIALIZING
    try:
      if self._trainer is None:
        with self.execution_context():
          self._trainer = self._trainer_factory()
        if self._pending_loss_fn is not None:
          loss_fn, has_aux = self._pending_loss_fn
          self._trainer.with_loss_fn(loss_fn, has_aux)
          self._pending_loss_fn = None
        if self._pending_gen_model_input_fn is not None:
          self._trainer.with_gen_model_input_fn(
              self._pending_gen_model_input_fn
          )
          self._pending_gen_model_input_fn = None
      return self._response(initialized=True)
    finally:
      self.state = WorkerState.READY

  def _drop_stale_stage(self, by: str) -> None:
    """Drops a micro-batch staged, or an update deferred, before a (re)compile
    or a restore.

    Only `fwd_bwd_staged` and `stop()` release a stage otherwise, and only
    `apply_deferred_update` a deferred update, so either left by an
    orchestrator that died without `stop()` (an OOM kill, a preemption) would
    make its successor's first stage, or first micro-batch, fail, and every
    restart's after it. A successor brings the worker up with `compile` and
    resumes with `restore_checkpoint`; either way the held micro-batch will
    never run and the deferred update will never be applied.

    Args:
      by: The call dropping it, for the log.
    """
    if self._staged_fwd_bwd is not None:
      logging.warning(
          "TrainerWorker %s drops staged micro-batch %r, staged before it and"
          " never run.",
          by,
          self._staged_fwd_bwd[0],
      )
      self._staged_fwd_bwd = None
    if self._deferred_update is not None:
      logging.warning(
          "TrainerWorker %s drops the update deferred after micro-batch %r,"
          " never applied.",
          by,
          self._deferred_update,
      )
      self._deferred_update = None

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    """Triggers JIT compilation using the provided dummy_data."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.COMPILING
    self._drop_stale_stage("compile")
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
    self._staged_fwd_bwd = None
    self._deferred_update = None
    if self.state == WorkerState.READY:
      self.state = WorkerState.DRAINING
    if self._trainer is not None:
      self._trainer.close()
    self.state = WorkerState.STOPPED
    return self._response(stopped=True)

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id,
        roles=frozenset({"trainer", "weight_sync"}),
        resources={
            "trainer": (
                type(self._trainer).__name__
                if self._trainer is not None
                else "LazyTrainer"
            ),
            "policy_version": self._policy_version(),
            # This worker can hold a micro-batch ahead of its fwd_bwd
            # (`stage_fwd_bwd` / `fwd_bwd_staged`) ...
            "stage_fwd_bwd": True,
            # ... and say how long a stage then still waited for the device
            # (`device_wait_s`), which needs the trainer's `wait_for_device`.
            "reports_device_wait": callable(
                getattr(self._trainer, "wait_for_device", None)
            ),
            # ... and take that micro-batch between a step's last fwd_bwd and
            # its update: `fwd_bwd(apply_optimizer=True, defer_update=True)`
            # leaves the update to `apply_deferred_update`.
            "stage_before_update": True,
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
    if self._trainer is not None:
      self._trainer.with_loss_fn(loss_fn, has_aux)
    else:
      self._pending_loss_fn = (loss_fn, has_aux)
    return self._response(loss_fn_configured=True)

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> datatypes.Response:
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs."""
    chunk_size = self._logps_chunk_size
    if chunk_size > 0:
      orig_fn = gen_model_input_fn

      def _wrapped_gen_model_input_fn(payload: Any) -> dict[str, Any]:
        out = dict(orig_fn(payload))
        out.setdefault("compute_logps_chunk_size", chunk_size)
        return out

      gen_model_input_fn = _wrapped_gen_model_input_fn
    if self._trainer is not None:
      self._trainer.with_gen_model_input_fn(gen_model_input_fn)
    else:
      self._pending_gen_model_input_fn = gen_model_input_fn
    return self._response(gen_model_input_fn_configured=True)

  def set_target_state(self, target_state: Any) -> datatypes.Response:
    """Stores rollout target_state so trainer-side weight sync can convert."""
    self._ensure_ready()
    setter = getattr(self._trainer, "set_target_state", None)
    if not callable(setter):
      raise AttributeError(
          f"{type(self._trainer).__name__} does not support set_target_state"
      )
    setter(target_state)
    return self._response(target_state_configured=True)

  def fwd_bwd(
      self,
      request: datatypes.TrainRequest,
      apply_optimizer: bool = False,
      *,
      wait_for_update: bool = True,
      defer_update: bool = False,
      **kwargs: Any,
  ) -> datatypes.Response:
    """Executes one forward/backward pass, optionally applying the optimizer.

    Args:
      request: The micro-batch.
      apply_optimizer: Also apply the accumulated gradients, in this call.
      wait_for_update: With `apply_optimizer`, return only once the update has
        landed when the trainer can tell (`update_ready`). False returns at
        its dispatch, for a caller that sends the next step's first
        micro-batch ahead (`stage_fwd_bwd`) while the device finishes; that
        stage then waits for the device and confirms the update instead.
      defer_update: With `apply_optimizer`, the trainer is still told this is
        the step's last micro-batch, but the update is left to
        `apply_deferred_update`, for a caller that stages the next step's
        first micro-batch in between (`stage_fwd_bwd`). Until then the worker
        accepts nothing that needs the step closed. Never passed to the
        trainer.
      **kwargs: Passed to the trainer's `fwd_bwd`.

    Returns:
      The response, with `updated`, `train_step` and, once confirmed,
      `update_ready` in its metadata when the optimizer ran, or
      `update_deferred` when it was deferred.
    """
    self._ensure_ready()
    if defer_update and not apply_optimizer:
      raise ValueError(
          "fwd_bwd(defer_update=True) needs apply_optimizer=True: only a"
          " step's last micro-batch has an update to defer."
      )
    self._refuse_while_update_deferred("fwd_bwd")
    req_metadata = dict(request.metadata) if request.metadata else {}
    req_metadata.pop("updated", None)
    req_metadata.pop("train_step", None)
    req_metadata.pop("update_ready", None)
    req_metadata.pop("update_deferred", None)
    kwargs.pop("skip_jit", None)
    try:
      # The trainer still sees `apply_optimizer`: it marks the step's last
      # micro-batch (MaxText's retry_dropless_final_micro_batch runs that one
      # with the dropless program). The update itself is run here, below, or
      # by `apply_deferred_update`.
      self._trainer.fwd_bwd(
          _with_dense_routing(request.payload),
          apply_optimizer=apply_optimizer,
          **kwargs,
      )
      if apply_optimizer and defer_update:
        self._deferred_update = str(request.request_id)
        req_metadata["update_deferred"] = True
      elif apply_optimizer:
        try:
          # Through this worker's update(), not the trainer's, so a fused step
          # gets the same bookkeeping (state on failure, step timing) as an
          # update RPC.
          train_step = self.update()
          # update() returns at dispatch. When the trainer exposes an output of
          # the update executable, return only once it is ready and say so: the
          # caller stamps the checkpoint as soon as this call returns, and with
          # `update_ready` it no longer needs get_metrics() to wait for the
          # update before the checkpoint save and weight sync. Not when the
          # caller asked to get control back at dispatch (`wait_for_update`).
          update_token = getattr(self._trainer, "last_update_token", None)
          if wait_for_update and update_token is not None:
            jax.block_until_ready(update_token)
            req_metadata["update_ready"] = True
        except Exception as exc:
          # The caller sees this on a fwd_bwd response; say it was the
          # optimizer step, run after this microbatch's gradients were
          # already accumulated.
          raise RuntimeError(
              "optimizer update() failed inside fwd_bwd(apply_optimizer=True),"
              " after this microbatch's gradients were accumulated:"
              f" {type(exc).__name__}: {exc}"
          ) from exc
        req_metadata["updated"] = True
        req_metadata["train_step"] = train_step
      self._last_error = None
      resp = self._response(queued=True, **req_metadata)
      resp.request_id = request.request_id
      return resp
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def stage_fwd_bwd(
      self,
      request: datatypes.TrainRequest,
      **kwargs: Any,
  ) -> datatypes.Response:
    """Receives a micro-batch ahead of the fwd_bwd that runs it.

    Holds it as is or, when the trainer has `prepare_fwd_bwd_input`, as what
    that returns (e.g. its inputs already placed on the devices). Nothing in
    the model, optimizer or gradient state changes, so the trainer's step
    order is that of the `fwd_bwd` / `fwd_bwd_staged` calls, and this may
    land while a weight sync holds the worker. One at a time: a second stage
    before the first ran is refused rather than replacing it (a `compile` or
    `restore_checkpoint`, as a restarted orchestrator's bring-up, drops it).

    Then, when the trainer has `wait_for_device`, waits for the work it had
    dispatched before this call and reports how long that took
    (`device_wait_s`). A short wait means the device was already idle, i.e.
    the transfer did not hide behind its work; how short depends on the
    runtime (on Pathways an idle device's wait still reads about 2.7-2.9 s).
    The wait is one the next call (`get_metrics`, which needs the update
    landed) would make anyway.

    A stage follows a step's last train call, sent with
    `fwd_bwd(apply_optimizer=True, wait_for_update=False)`, so the update that
    call dispatched is the last one. When the trainer exposes an output of it
    (`last_update_token`), this also returns only once it is ready and says
    so (`update_ready`), as a fused `fwd_bwd` would have.

    Or it follows a last train call sent with `defer_update=True`, before the
    step's update. Then it waits for nothing and confirms nothing: the device
    is meant to be still computing that micro-batch, and the wait (and the
    confirmation) belong to `apply_deferred_update`, which runs the update
    next. `device_wait_s` is then None and `before_update` is set.

    Args:
      request: The micro-batch.
      **kwargs: Passed to the trainer's `fwd_bwd` when it runs.

    Returns:
      A response naming the staged request, with `device_wait_s` (None when
      the trainer cannot tell, or when staged before the update) and, once
      confirmed, `update_ready` in its metadata.
    """
    if self.state == WorkerState.PENDING:
      self.initialize()
    if self.state in (
        WorkerState.ERROR,
        WorkerState.STOPPED,
        WorkerState.DRAINING,
    ):
      raise RuntimeError(
          "TrainerWorker cannot stage a micro-batch in state"
          f" {self.state.value}."
      )
    if self._staged_fwd_bwd is not None:
      raise RuntimeError(
          "TrainerWorker already holds staged micro-batch"
          f" {self._staged_fwd_bwd[0]!r}; run it with fwd_bwd_staged() first."
      )
    req_metadata = dict(request.metadata) if request.metadata else {}
    # Only this worker sets these; a caller-supplied copy is not echoed.
    for key in ("staged", "device_wait_s", "update_ready", "before_update"):
      req_metadata.pop(key, None)
    kwargs.pop("skip_jit", None)
    update_ready = False
    device_wait_s = None
    before_update = self._deferred_update is not None
    try:
      staged_input = _with_dense_routing(request.payload)
      prepare = getattr(self._trainer, "prepare_fwd_bwd_input", None)
      if callable(prepare):
        staged_input = prepare(staged_input)
      if not before_update:
        # Both waits are for the work dispatched before this call, which ends
        # with the step's optimizer update, so a failure in either is that
        # update's and is reported as the fused call reports it.
        waiting_for = "the device"
        try:
          wait_for_device = getattr(self._trainer, "wait_for_device", None)
          device_wait_s = (
              float(wait_for_device()) if callable(wait_for_device) else None
          )
          # After the device wait, which already covers the update when the
          # trainer queues it (MaxText's throttler holds its gradient norm, so
          # a failed update raises there first).
          waiting_for = "the update"
          update_token = getattr(self._trainer, "last_update_token", None)
          if update_token is not None:
            jax.block_until_ready(update_token)
            update_ready = True
        except Exception as exc:
          raise RuntimeError(
              "optimizer update() failed: dispatched by"
              " fwd_bwd(apply_optimizer=True, wait_for_update=False), it did"
              f" not land (seen by stage_fwd_bwd waiting for {waiting_for}):"
              f" {type(exc).__name__}: {exc}"
          ) from exc
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    self._staged_fwd_bwd = (
        request.request_id,
        staged_input,
        kwargs,
        req_metadata,
    )
    response_metadata = {
        **req_metadata,
        "staged": True,
        "device_wait_s": device_wait_s,
    }
    if update_ready:
      response_metadata["update_ready"] = True
    if before_update:
      response_metadata["before_update"] = True
    resp = self._response(**response_metadata)
    resp.request_id = request.request_id
    return resp

  def fwd_bwd_staged(
      self,
      staged_request_id: str,
      **kwargs: Any,
  ) -> datatypes.Response:
    """Runs the forward/backward pass of the micro-batch `stage_fwd_bwd` holds.

    It is `fwd_bwd` on the held input, so it takes the same options
    (`apply_optimizer`, which also runs the update in this call, and
    `wait_for_update`) and gives the same response.

    Args:
      staged_request_id: The id `stage_fwd_bwd` was given; any other id is
        refused, so a staged micro-batch is never trained in another's place.
      **kwargs: Merged over the kwargs given at staging, then passed to
        `fwd_bwd`.

    Returns:
      The response `fwd_bwd` gives for that micro-batch.
    """
    self._ensure_ready()
    staged = self._staged_fwd_bwd
    if staged is None or staged[0] != staged_request_id:
      raise RuntimeError(
          f"TrainerWorker holds no staged micro-batch {staged_request_id!r}"
          f" (holds {staged[0] if staged else None!r})."
      )
    self._staged_fwd_bwd = None
    request_id, staged_input, stage_kwargs, req_metadata = staged
    # The held input may be what `prepare_fwd_bwd_input` made of the payload;
    # the trainer's `fwd_bwd` takes either.
    request = datatypes.TrainRequest(
        request_id=request_id,
        payload=staged_input,
        metadata=req_metadata,
    )
    return self.fwd_bwd(request, **{**stage_kwargs, **kwargs})

  def apply_deferred_update(self) -> datatypes.Response:
    """Runs the update `fwd_bwd(apply_optimizer=True, defer_update=True)` left.

    Between the two, the caller staged the next step's first micro-batch
    (`stage_fwd_bwd`), so its transfer ran while the device computed this
    step's last micro-batch. This first waits for the device when the trainer
    has `wait_for_device` and reports how long (`device_wait_s`): how long that
    micro-batch still ran once the staged one was in place. A short wait
    means the device had gone idle, i.e. the transfer outlasted it and
    delayed the update (on Pathways an idle device's wait reads about
    2.7-2.9 s, not ~0). The wait costs nothing where the trainer's
    `update()` blocks until the step's last micro-batch has finished anyway
    (MaxText with `retry_when_tokens_dropped`, which reads its overflow flag
    on the host first); elsewhere it holds the update's dispatch back by one
    readiness round trip.

    Then runs the update through `update()` and, when the trainer exposes an
    output of it (`last_update_token`), returns only once it has landed and
    says so (`update_ready`), as a fused `fwd_bwd` would have.

    Returns:
      A response with `updated`, `train_step`, `device_wait_s` (None when the
      trainer cannot tell) and, once confirmed, `update_ready` in its
      metadata.
    """
    self._ensure_ready()
    if self._deferred_update is None:
      raise RuntimeError(
          "TrainerWorker holds no deferred update: apply_deferred_update()"
          " follows fwd_bwd(apply_optimizer=True, defer_update=True)."
      )
    metadata: dict[str, Any] = {}
    waiting_for = "the device"
    try:
      try:
        wait_for_device = getattr(self._trainer, "wait_for_device", None)
        device_wait_s = (
            float(wait_for_device()) if callable(wait_for_device) else None
        )
        waiting_for = "the update"
        # Cleared only here: update() refuses to run while an update is
        # deferred, and the device wait above has to come first, because it
        # is what tells the caller whether the stage hid behind the step's
        # last micro-batch.
        self._deferred_update = None
        train_step = self.update()
        update_token = getattr(self._trainer, "last_update_token", None)
        if update_token is not None:
          jax.block_until_ready(update_token)
          metadata["update_ready"] = True
      except Exception as exc:
        # Waiting for the device is waiting for the step's last micro-batch,
        # whose failure surfaces there; the rest is the update's.
        raise RuntimeError(
            "deferred optimizer update failed: left by"
            " fwd_bwd(apply_optimizer=True, defer_update=True), seen by"
            f" apply_deferred_update waiting for {waiting_for}:"
            f" {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    self._last_error = None
    return self._response(
        updated=True,
        train_step=train_step,
        device_wait_s=device_wait_s,
        **metadata,
    )

  def update(self, **kwargs) -> int:
    """Applies the accumulated (mean) gradients as one optimizer update."""
    self._ensure_ready()
    self._refuse_while_update_deferred("update")
    try:
      train_step = self._trainer.update(**kwargs)
      self._last_error = None
      return train_step
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def eval_step(
      self,
      request: datatypes.TrainRequest,
      **kwargs: Any,
  ) -> datatypes.Response:
    """Executes one evaluation step on the given payload."""
    self._ensure_ready()
    self._refuse_while_update_deferred("eval_step")
    req_metadata = dict(request.metadata) if request.metadata else {}
    try:
      self._trainer.eval_step(_with_dense_routing(request.payload), **kwargs)
      self._last_error = None
      resp = self._response(evaluated=True, **req_metadata)
      resp.request_id = request.request_id
      return resp
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def run_eval(self, eval_ds: Any, **kwargs) -> datatypes.Response:
    """Runs an explicit evaluation phase over eval micro-batches."""
    self._ensure_ready()
    self._refuse_while_update_deferred("run_eval")
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
          self._trainer.eval_step(_with_dense_routing(payload), **kwargs)
          eval_batches += 1
      self._last_error = None
      return self._response(evaluated=True, eval_batches=eval_batches)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def per_token_logps(
      self, items: datatypes.LogprobsRequest, **kwargs: Any
  ) -> datatypes.LogprobsResponse:
    """Scores per-token log-probs for a padded request under live weights.

    Read-only: unlike the frozen reference scorer this uses the trainer's
    current (actor) parameters, but it must not mutate trainer state. Accepts a
    ``LogprobsRequest`` composed by the orchestrator; the actor path sets
    ``pad_id``/``eos_id`` (and any packing fields) explicitly.

    Args:
      items: The log-probabilities request payload containing token sequences.
      **kwargs: Unused keyword arguments accepted for forwarding parity.

    Returns:
      A LogprobsResponse containing per-token log-probabilities and model
      version.
    """
    del kwargs  # Accepted for engine-forwarding parity; unused.
    self._ensure_ready()
    self._refuse_while_update_deferred("per_token_logps")
    if items.pad_id is None or items.eos_id is None:
      raise ValueError(
          "TrainerWorker.per_token_logps requires pad_id and eos_id to be set "
          "on the LogprobsRequest; the actor scoring path must compose them."
      )
    try:
      prompt = np.asarray(items.prompt_tokens, dtype=np.int32)
      completion = np.asarray(items.completion_tokens, dtype=np.int32)
      batch_size = prompt.shape[0]
      if batch_size == 0:
        raise ValueError("per_token_logps requires a non-empty batch.")
      temperature = (
          1.0 if items.temperature is None else float(items.temperature)
      )
      seg_ids = (
          None
          if items.segment_ids is None
          else np.asarray(items.segment_ids, dtype=np.int32)
      )
      seg_pos = (
          None
          if items.segment_positions is None
          else np.asarray(items.segment_positions, dtype=np.int32)
      )
      raw_routed = getattr(items, "routed_experts", None)
      if isinstance(raw_routed, datatypes.CompactRoutedExperts):
        routed = raw_routed.materialize()
      elif raw_routed is not None:
        routed = np.asarray(raw_routed, dtype=np.int16)
      else:
        routed = None
      micro_batch_size = self._logps_micro_batch_size or batch_size
      outs = []
      for start in range(0, batch_size, micro_batch_size):
        sl = slice(start, start + micro_batch_size)
        scope_kwargs: dict[str, Any] = dict(
            pad_id=items.pad_id,
            eos_id=items.eos_id,
            temperature=temperature,
            chunk_size=self._logps_chunk_size,
            segment_ids=None if seg_ids is None else seg_ids[sl],
            segment_positions=None if seg_pos is None else seg_pos[sl],
        )
        if routed is not None:
          scope_kwargs["routed_experts"] = routed[sl]
        with self._trainer.model_scope(
            prompt[sl],
            completion[sl],
            **scope_kwargs,
        ) as (model, scoped_args, scoped_kwargs):
          # Split per call, never once in __init__: the optimizer writes new
          # arrays every step, and a cached State would pin the old ones.
          graphdef, state = nnx.split(model)
          outs.append(
              rl_common.compute_per_token_logps(
                  graphdef,
                  state,
                  *scoped_args,
                  stop_gradient=True,
                  **scoped_kwargs,
              )
          )
      result = np.asarray(jnp.concatenate(outs, axis=0), dtype=np.float32)
      self._last_error = None
      return datatypes.LogprobsResponse(
          request_id=items.request_id,
          per_token_logps=result,
          model_version=self._policy_version(),
      )
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def _resolve_checkpoint_path(self, metadata: Mapping[str, Any]) -> str:
    """Resolves the saved Orbax model_params directory from the underlying trainer."""
    ckpt_dir = self._trainer.checkpoint_dir
    step = metadata.get("step")
    if ckpt_dir and step is not None:
      return f"{str(ckpt_dir).rstrip('/')}/{int(step)}/model_params"
    return ""

  def save_checkpoint(self, metadata: Any, **kwargs) -> datatypes.Response:
    """Force the trainer to serialize its state (model + optimizer)."""
    self._ensure_ready()
    self._refuse_while_update_deferred("save_checkpoint")
    try:
      self._trainer.save_checkpoint(metadata, **kwargs)
      ckpt_path = self._resolve_checkpoint_path(metadata)
      self._last_error = None
      return self._response(checkpoint_saved=True, checkpoint_path=ckpt_path)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def restore_checkpoint(self, **kwargs) -> Any:
    """Restore state from latest checkpoint and return the metadata pytree."""
    self._ensure_ready()
    self._drop_stale_stage("restore_checkpoint")
    try:
      result = self._trainer.restore_checkpoint(**kwargs)
      self._last_error = None
      return result
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def prepare_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Stages weights for transfer and returns their metadata."""
    self._ensure_ready()
    self._refuse_while_update_deferred("prepare_weight_sync")
    self.state = WorkerState.SYNCING
    try:
      if sync_request is not None:
        kwargs["sync_request"] = sync_request
      metadata = self._trainer.prepare_weight_sync(**kwargs)
      self._last_error = None
      self._maybe_release_after_stage(sync_request)
      if metadata is not None:
        return metadata
      return self._response(weight_sync_ready=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def _maybe_release_after_stage(self, sync_request: Any) -> None:
    """Goes back to READY once staged if the round runs in the background.

    The next train step then runs during the transfer. That is only safe when
    `d2h` left a host copy, since the step rewrites the device weights. The
    device copy staged for the transfer is dropped here too, so the step does
    not run with a second set of weights in HBM.
    """
    if sync_request is None:
      return
    extra_config = getattr(sync_request, "extra_config", None)
    if not isinstance(extra_config, dict) or not extra_config.get(
        weight_sync.RELEASE_SOURCE_AFTER_STAGE
    ):
      return
    # MaxText's engine keeps its synchronizer in `_weight_sync`, PeftTrainer in
    # `_weight_sync_worker`.
    synchronizer = getattr(self._trainer, "_weight_sync", None) or getattr(
        self._trainer, "_weight_sync_worker", None
    )
    if synchronizer is None:
      raise RuntimeError(
          "Weight sync runs in the background, but trainer"
          f" {type(self._trainer).__name__} has no weight synchronizer."
      )
    if not synchronizer.staged_on_host:
      raise RuntimeError(
          "Weight sync runs in the background, but"
          f" {type(synchronizer).__name__} transfers straight from device"
          " memory, which the next train step would overwrite. Use a"
          " host-staged source (raiden: RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER=0)."
      )
    # The transfer reads the host copy; the synchronizer's references are the
    # last ones to the converted device tree (the trainer deletes its own).
    released = synchronizer.release_buffers()
    logging.info(
        "Background weight sync staged on host; released %d device arrays.",
        released,
    )
    self.state = WorkerState.READY

  def release_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Releases this round's staging and restores READY."""
    release = getattr(self._trainer, "release_weight_sync", None)
    result = release(sync_request=sync_request, **kwargs) if release else None
    if self.state == WorkerState.SYNCING:
      self.state = WorkerState.READY
    return result

  def get_metrics(self) -> Any:
    """Returns and clears the recently collected step metric records."""
    self._refuse_while_update_deferred("get_metrics")
    return self._trainer.get_metrics()
