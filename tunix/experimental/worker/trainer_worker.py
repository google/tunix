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
import dataclasses
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
from tunix.experimental.worker import abstract_worker
from tunix.rl import common as rl_common

WorkerState = datatypes.WorkerState

# Stack for the warmup compile thread. Tracing a large model's train step
# recurses deeply, and threads get a smaller default stack than the main one.
_WARMUP_THREAD_STACK_BYTES = 64 * 1024 * 1024


def payload_signature(payload: Any) -> tuple[Any, dict[str, Any]]:
  """Returns what a compiled train step is keyed on: structure, shapes, dtypes.

  Metadata is dropped, as the trainer drops it before building its inputs.
  Dtypes are canonicalized the way `jax.jit` sees them.

  Args:
    payload: A trainer payload, of arrays or `jax.ShapeDtypeStruct` leaves.

  Returns:
    `(treedef, {leaf path: (shape, dtype)})`.
  """
  if dataclasses.is_dataclass(payload) and any(
      field.name == "metadata" for field in dataclasses.fields(payload)
  ):
    payload = dataclasses.replace(payload, metadata={})
  leaves, treedef = jax.tree_util.tree_flatten_with_path(payload)
  return treedef, {
      jax.tree_util.keystr(path): (
          tuple(np.shape(leaf)),
          jnp.result_type(leaf).name,
      )
      for path, leaf in leaves
  }


def signature_mismatch(expected: Any, actual: Any) -> str:
  """Describes how two `payload_signature`s differ, or "" if they match."""
  expected_treedef, expected_leaves = expected
  actual_treedef, actual_leaves = actual
  diffs = []
  for path in sorted(expected_leaves.keys() | actual_leaves.keys()):
    want = expected_leaves.get(path)
    got = actual_leaves.get(path)
    if want != got:
      diffs.append(f"{path}: expected {want}, got {got}")
  if not diffs and expected_treedef != actual_treedef:
    diffs.append(
        f"structure: expected {expected_treedef}, got {actual_treedef}"
    )
  return "; ".join(diffs)


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
    with self.execution_context():
      self._trainer = trainer_factory()
    self._logps_chunk_size = logps_chunk_size
    self._logps_micro_batch_size = logps_micro_batch_size
    self._is_running = False
    self._worker_id = worker_id
    self._state = WorkerState.PENDING
    self._last_error: str | None = None
    # Set by `warmup_compile`: done once its background compile has ended, and
    # the signature it compiled for, checked against the first real batch.
    self._warmup_done: threading.Event | None = None
    self._warmup_signature: Any = None

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

  def _wait_for_warmup_compile(self) -> None:
    """Blocks until a background `warmup_compile` has finished, if one runs."""
    done = self._warmup_done
    if done is None or done.is_set():
      return
    logging.info("Waiting for the trainer warmup compile to finish...")
    start = time.monotonic()
    done.wait()
    logging.info(
        "Waited %.1fs for the trainer warmup compile.",
        time.monotonic() - start,
    )

  def _ensure_ready(self) -> None:
    # Everything that runs the trainer comes through here, so nothing sees it
    # mid-compile.
    self._wait_for_warmup_compile()
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
      return self._response(initialized=True)
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    """Triggers JIT compilation using the provided dummy_data."""
    self._wait_for_warmup_compile()
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

  def warmup_compile(self, dummy_data: Any) -> datatypes.Response:
    """Compiles the train step for `dummy_data` in a background thread.

    Returns at once, so neither the caller nor this worker's RPC loop waits on
    the compile; the orchestrator runs rollout meanwhile. Every call that runs
    the trainer waits for it first (`_ensure_ready`). A failure is logged and
    leaves the worker READY, and the first `fwd_bwd` then compiles as it would
    have without the warmup.

    Args:
      dummy_data: A payload shaped like the real micro-batches, e.g. of
        `jax.ShapeDtypeStruct` leaves.

    Returns:
      A response whose `warmup_compile_started` is False if a warmup already
      ran on this worker.
    """
    if self.state == WorkerState.PENDING:
      self.initialize()
    if self._warmup_done is not None:
      return self._response(warmup_compile_started=False)
    signature = payload_signature(dummy_data)
    done = threading.Event()
    stack_size = threading.stack_size(_WARMUP_THREAD_STACK_BYTES)
    try:
      threading.Thread(
          target=self._run_warmup_compile,
          args=(dummy_data, done),
          name="trainer-warmup-compile",
          daemon=True,
      ).start()
    finally:
      threading.stack_size(stack_size)
    # Assigned only once the thread runs: an event nothing would ever set
    # would hang every later call.
    self._warmup_signature = signature
    self._warmup_done = done
    return self._response(warmup_compile_started=True)

  def _run_warmup_compile(
      self, dummy_data: Any, done: threading.Event
  ) -> None:
    start = time.monotonic()
    try:
      # The mesh and sharding contexts are per thread, so enter them here.
      with self.execution_context():
        self._trainer.compile(dummy_data)
      logging.info(
          "Trainer warmup compile finished in %.1fs.", time.monotonic() - start
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception(
          "Trainer warmup compile failed after %.1fs; the first fwd_bwd"
          " compiles instead.",
          time.monotonic() - start,
      )
    finally:
      done.set()

  def routed_experts_shape(self) -> tuple[int, int] | None:
    """Returns the `(num_layers, top_k)` router replay needs, if known.

    Read off a MaxText engine's config; None for trainers without one.
    """
    config = getattr(self._trainer, "_config", None)
    num_layers = getattr(config, "num_decoder_layers", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if not num_layers or not top_k:
      return None
    return int(num_layers), int(top_k)

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
    chunk_size = self._logps_chunk_size
    if chunk_size > 0:
      orig_fn = gen_model_input_fn

      def _wrapped_gen_model_input_fn(payload: Any) -> dict[str, Any]:
        out = dict(orig_fn(payload))
        out.setdefault("compute_logps_chunk_size", chunk_size)
        return out

      gen_model_input_fn = _wrapped_gen_model_input_fn
    self._trainer.with_gen_model_input_fn(gen_model_input_fn)
    return self._response(gen_model_input_fn_configured=True)

  def set_target_state(self, target_state: Any) -> datatypes.Response:
    """Stores rollout target_state so trainer-side weight sync can convert."""
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
      **kwargs: Any,
  ) -> datatypes.Response:
    """Executes one forward/backward pass."""
    self._ensure_ready()
    if self._warmup_signature is not None:
      self._check_warmup_signature(request.payload)
    req_metadata = dict(request.metadata) if request.metadata else {}
    kwargs.pop("skip_jit", None)
    try:
      self._trainer.fwd_bwd(request.payload, **kwargs)
      self._last_error = None
      resp = self._response(queued=True, **req_metadata)
      resp.request_id = request.request_id
      return resp
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def _check_warmup_signature(self, payload: Any) -> None:
    """Logs whether the first real batch hit the warmup-compiled step."""
    expected, self._warmup_signature = self._warmup_signature, None
    try:
      mismatch = signature_mismatch(expected, payload_signature(payload))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning("Could not compare batch to the warmup compile: %s", exc)
      return
    if mismatch:
      logging.warning(
          "First trainer batch differs from the warmup compile's dummy batch,"
          " so this fwd_bwd recompiles: %s",
          mismatch,
      )
    else:
      logging.info("First trainer batch matches the warmup compile.")

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

  def prepare_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Stages weights for transfer and returns their metadata."""
    self._ensure_ready()
    self.state = WorkerState.SYNCING
    try:
      if sync_request is not None:
        kwargs["sync_request"] = sync_request
      metadata = self._trainer.prepare_weight_sync(**kwargs)
      self._last_error = None
      if metadata is not None:
        return metadata
      return self._response(weight_sync_ready=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise

  def release_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Releases this round's staging and restores READY."""
    release = getattr(self._trainer, "release_weight_sync", None)
    result = release(sync_request=sync_request, **kwargs) if release else None
    if self.state == WorkerState.SYNCING:
      self.state = WorkerState.READY
    return result

  def get_metrics(self) -> Any:
    """Returns and clears the recently collected step metric records.

    The records come back host-resident. A trainer may hand back an on-device
    buffer of a few hundred scalars, and pickling that for the RPC reply reads
    them one at a time, each a blocking device-to-host round trip (through the
    Pathways proxy on a remote trainer). `jax.device_get` starts every copy
    before waiting on any of them.
    """
    return jax.device_get(self._trainer.get_metrics())
