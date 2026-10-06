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
import threading
import time
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


STEP_TIMING_SCHEMA_VERSION = 1


class StepTimer:
  """Trainer-side timeline of one optimizer step, for the orchestrator log.

  The orchestrator only sees RPC boundaries, and `fwd_bwd` / `update` return
  at JAX dispatch, before the device work completes. This records, per step:

    * per microbatch: when the `fwd_bwd` RPC started and finished dispatching,
      and when its outputs became ready on device;
    * the same three instants for the `update` RPC;
    * when `prepare_weight_sync` was received and finished (it reads the
      updated weights, so it cannot complete before the update has).

  Device readiness is observed off the serving thread: a daemon thread blocks
  on a token array (the loss / grad norm the trainer already produced) and
  stamps the time; the main path never waits.

  All times are `time.monotonic()` on the trainer host. `snapshot()` converts
  them to "seconds ago" relative to the moment it is called, so the consumer
  can anchor them on its own clock without any cross-host clock agreement:
  for a snapshot returned by an RPC that took `rtc` seconds round trip,
  `t_event ~= t_rpc_return - rtt/2 - ago_s`.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._reset_locked()

  def _reset_locked(self) -> None:
    self._mb: list[dict[str, float | None]] = []
    self._update: dict[str, float | None] | None = None
    self._prepare: dict[str, float | None] | None = None
    self._train_step: int | None = None

  def reset(self) -> None:
    with self._lock:
      self._reset_locked()

  # -- microbatches --------------------------------------------------------
  def fwd_bwd_begin(self) -> int:
    with self._lock:
      if self._update is not None:  # first microbatch after an update
        self._reset_locked()
      self._mb.append(
          {"dispatch_begin": time.monotonic(), "dispatch_end": None, "ready": None}
      )
      return len(self._mb) - 1

  def fwd_bwd_dispatched(self, idx: int, token: Any = None) -> None:
    with self._lock:
      self._mb[idx]["dispatch_end"] = time.monotonic()
    self._watch_ready(token, lambda t, i=idx: self._set_mb_ready(i, t))

  def _set_mb_ready(self, idx: int, t: float) -> None:
    with self._lock:
      if idx < len(self._mb) and self._mb[idx]["ready"] is None:
        self._mb[idx]["ready"] = t

  # -- update ----------------------------------------------------------------
  def update_begin(self) -> None:
    with self._lock:
      self._update = {
          "dispatch_begin": time.monotonic(), "dispatch_end": None, "ready": None
      }

  def update_dispatched(self, train_step: int | None, token: Any = None) -> None:
    with self._lock:
      if self._update is not None:
        self._update["dispatch_end"] = time.monotonic()
      self._train_step = train_step
    self._watch_ready(token, self._set_update_ready)

  def _set_update_ready(self, t: float) -> None:
    with self._lock:
      if self._update is not None and self._update["ready"] is None:
        self._update["ready"] = t

  # -- prepare_weight_sync ---------------------------------------------------
  def prepare_begin(self) -> None:
    with self._lock:
      self._prepare = {"recv": time.monotonic(), "done": None}

  def prepare_end(self) -> None:
    with self._lock:
      if self._prepare is not None:
        self._prepare["done"] = time.monotonic()

  # -- readiness watcher -----------------------------------------------------
  @staticmethod
  def _watch_ready(token: Any, on_ready: Callable[[float], None]) -> None:
    """Stamps `on_ready(t)` when `token` (a JAX array) is ready on device."""
    if token is None or not hasattr(token, "block_until_ready"):
      return

    def _wait():
      try:
        jax.block_until_ready(token)
      except Exception:  # pylint: disable=broad-except
        return  # a failed step surfaces through the RPC that ran it
      on_ready(time.monotonic())

    threading.Thread(target=_wait, name="step-timer-ready", daemon=True).start()

  # -- export ------------------------------------------------------------------
  def snapshot(self) -> dict[str, Any]:
    """Wire-safe copy with every instant as seconds before now (`*_ago_s`)."""
    now = time.monotonic()

    def ago(t: float | None) -> float | None:
      return None if t is None else round(now - t, 3)

    with self._lock:
      mb = [
          {
              "i": i,
              "dispatch_begin_ago_s": ago(m["dispatch_begin"]),
              "dispatch_end_ago_s": ago(m["dispatch_end"]),
              "ready_ago_s": ago(m["ready"]),
          }
          for i, m in enumerate(self._mb)
      ]
      upd = None
      if self._update is not None:
        upd = {
            "dispatch_begin_ago_s": ago(self._update["dispatch_begin"]),
            "dispatch_end_ago_s": ago(self._update["dispatch_end"]),
            "ready_ago_s": ago(self._update["ready"]),
        }
      prep = None
      if self._prepare is not None:
        prep = {
            "recv_ago_s": ago(self._prepare["recv"]),
            "done_ago_s": ago(self._prepare["done"]),
        }
      return {
          "v": STEP_TIMING_SCHEMA_VERSION,
          "train_step": self._train_step,
          "mb": mb,
          "update": upd,
          "prepare": prep,
      }


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
    self._step_timer = StepTimer()
    self._missing_timing_tokens: set[str] = set()

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

  def _timing_token(self, name: str) -> Any:
    """Returns the trainer's readiness token `name` for the step timer.

    Trainers set `last_fwd_bwd_token` / `last_update_token` to an array that
    becomes ready when that work finishes on device. A trainer that does not
    set them still trains, but `TRAINER_STEP_TIMING` then carries dispatch
    times only, so warn once per token rather than leave `ready` silently None.
    """
    token = getattr(self._trainer, name, None)
    if token is None and name not in self._missing_timing_tokens:
      self._missing_timing_tokens.add(name)
      logging.warning(
          "TRAINER_STEP_TIMING: %s does not set `%s`; device `ready` times"
          " will be missing from the timeline (dispatch times only).",
          type(self._trainer).__name__,
          name,
      )
    return token

  def fwd_bwd(
      self,
      request: datatypes.TrainRequest,
      **kwargs: Any,
  ) -> datatypes.Response:
    """Executes one forward/backward pass."""
    self._ensure_ready()
    req_metadata = dict(request.metadata) if request.metadata else {}
    kwargs.pop("skip_jit", None)
    mb_idx = self._step_timer.fwd_bwd_begin()
    try:
      self._trainer.fwd_bwd(request.payload, **kwargs)
      self._step_timer.fwd_bwd_dispatched(
          mb_idx, self._timing_token("last_fwd_bwd_token")
      )
      self._last_error = None
      resp = self._response(queued=True, **req_metadata)
      resp.request_id = request.request_id
      return resp
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def update(self, **kwargs) -> int:
    """Applies the accumulated (mean) gradients as one optimizer update."""
    self._ensure_ready()
    self._step_timer.update_begin()
    try:
      train_step = self._trainer.update(**kwargs)
      self._step_timer.update_dispatched(
          train_step, self._timing_token("last_update_token")
      )
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
    req_metadata = dict(request.metadata) if request.metadata else {}
    try:
      self._trainer.eval_step(request.payload, **kwargs)
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
      routed = (
          None
          if getattr(items, "routed_experts", None) is None
          else np.asarray(items.routed_experts, dtype=np.int16)
      )
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
    self.state = WorkerState.SYNCING
    self._step_timer.prepare_begin()
    try:
      if sync_request is not None:
        kwargs["sync_request"] = sync_request
      metadata = self._trainer.prepare_weight_sync(**kwargs)
      self._step_timer.prepare_end()
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
    return self._trainer.get_metrics()

  def get_step_timing(self) -> dict[str, Any]:
    """Returns the trainer-side timeline of the current / last optimizer step.

    See `StepTimer`. Instants are seconds before this call was handled, so the
    caller anchors them on its own clock from the RPC's return time. The record
    is not cleared here; it is replaced by the first `fwd_bwd` after an update.
    """
    return self._step_timer.snapshot()
