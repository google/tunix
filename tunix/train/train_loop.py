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

"""Loop layer driving a step-level trainer.

`TrainLoop` owns loop policy: data iteration, gradient-accumulation call
patterns, eval cadence, checkpoint cadence, hooks, metrics writing, profiling
and the progress bar. The trainer (see `tunix.train.abstract_trainer`) owns
step execution and all model/optimizer state.

Metrics flow pull-based: the trainer records per-step metrics internally and
the driver drains them via `trainer.get_metrics()` at whatever frequency it
chooses, then hands them to `process_metrics`. RPC-based orchestrators can
drain on the worker, move values host-side, and ship them asynchronously.

This is a transitional split extracted from `PeftTrainer.train`: metric
buffers and loop components (throttler, profiler, progress bar) still live on
the trainer so that existing subclass overrides (e.g.
`_post_process_train_step`) keep working.
"""

from collections.abc import Iterable, Sequence
import time
from typing import Any, List, Tuple, TYPE_CHECKING

from absl import logging
import jax
from jax.interpreters import pxla
import numpy as np
from tunix.sft import metrics_logger as sft_metrics_logger
from tunix.sft import progress_bar
from tunix.train import abstract_trainer

if TYPE_CHECKING:
  from tunix.sft import peft_trainer


def process_metrics(
    trainer: abstract_trainer.AbstractTrainer,
    records: List[Tuple[abstract_trainer.StepMetrics, int, bool, bool]],
) -> None:
  """Buffers and writes step metric records drained via `get_metrics`.

  Train records are buffered per step and written whenever a record marks an
  applied optimizer update. Eval records are only buffered: the eval driver
  (e.g. `TrainLoop.run_eval`) decides when an eval run is complete and writes
  the eval buffer itself, so a drain boundary can never split an eval run.

  NOTE: processing is synchronous. Values inside records are device arrays;
  host transfer happens at write time. If drains ever move to a background
  thread (e.g. an RPC worker shipping metrics to an orchestrator), writes
  must be synchronized with `trainer.close()`.

  Args:
    trainer: The trainer the records were drained from.
    records: Records from `trainer.get_metrics()`.
  """
  for step_metrics, step_id, is_eval, applied in records:
    if is_eval:
      trainer._buffered_eval_metrics = trainer._buffer_metrics(  # pylint: disable=protected-access
          trainer._buffered_eval_metrics,  # pylint: disable=protected-access
          loss=step_metrics.loss,
          step=step_id,
      )
      trainer._post_process_eval_step(step_metrics.aux)  # pylint: disable=protected-access
    else:
      trainer._buffered_train_metrics = trainer._buffer_metrics(  # pylint: disable=protected-access
          trainer._buffered_train_metrics,  # pylint: disable=protected-access
          loss=step_metrics.loss,
          step=step_id,
          additional_metrics={
              "grad_norm": (step_metrics.grad_norm, np.mean)
          },
      )
      # NB: put this after _buffer_metrics is important.
      trainer._post_process_train_step(step_metrics.aux)  # pylint: disable=protected-access
      if applied:
        trainer._write_train_metrics()  # pylint: disable=protected-access


def write_pending_eval_metrics(
    trainer: abstract_trainer.AbstractTrainer,
) -> None:
  """Writes and clears the trainer's buffered eval metrics, if any.

  Call after `process_metrics` once an eval run is known to be complete.
  """
  if trainer._buffered_eval_metrics is None:  # pylint: disable=protected-access
    return
  # Re-enter EVAL mode: when called by an outer driver after run() returns,
  # the trainer is back in TRAIN mode, but these are eval records.
  with trainer._switch_mode(sft_metrics_logger.Mode.EVAL):  # pylint: disable=protected-access
    trainer._write_metrics(trainer._buffered_eval_metrics)  # pylint: disable=protected-access
    logging.info(
        "Train step %d eval loss: %f - eval perplexity: %f",
        trainer.train_steps,  # pytype: disable=attribute-error
        trainer.metrics_logger.get_metric(  # pyrefly: ignore[missing-attribute]
            trainer.metrics_prefix, "loss", "eval"
        ),
        trainer.metrics_logger.get_metric(  # pyrefly: ignore[missing-attribute]
            trainer.metrics_prefix, "perplexity", "eval"
        ),
    )
  trainer._buffered_eval_metrics = None  # pylint: disable=protected-access


def train_minibatch(
    trainer: abstract_trainer.AbstractTrainer,
    payloads: Sequence[abstract_trainer.TrainerPayload | Any],
    **kwargs,
) -> None:
  """Trains over a mini-batch of micro-batches by driving `trainer.train`.

  Each micro-batch runs through the full `TrainLoop` (profiling, hooks,
  metrics draining), with the trainer temporarily marked as externally
  managed so that per-call `close()` and the checkpoint-resume skip logic
  are bypassed.

  NOTE: the optimizer-update boundary is governed by
  `config.gradient_accumulation_steps` via the loop's `iter_steps` counter,
  not by `len(payloads)`. Configure them consistently if this call is meant
  to be exactly one optimizer update.

  Args:
    trainer: The trainer to drive.
    payloads: The micro-batches, in order.
    **kwargs: Forwarded to `trainer.train`.

  Raises:
    ValueError: If `payloads` is empty.
  """
  payloads = list(payloads)
  if not payloads:
    raise ValueError("train_minibatch requires at least one payload.")

  # Bypass the loop's session teardown (close()) and resume-skip logic;
  # the caller owns the trainer's lifecycle.
  was_managed = trainer.is_managed_externally  # pytype: disable=attribute-error
  trainer.is_managed_externally = True
  try:
    for micro_batch in payloads:
      # We use a type ignore here since AbstractTrainer does not define
      # train(), but at runtime trainer will be a PeftTrainer or similar.
      # `drain_metrics=False`: the loop leaves step records pending on the
      # trainer; this driver owns metrics readiness and writing.
      trainer.train(train_ds=[micro_batch], drain_metrics=False, **kwargs)  # type: ignore
  finally:
    trainer.is_managed_externally = was_managed

  # This driver decides when metrics are ready: drain and write here, at
  # whatever frequency the caller chooses (it does not need to align with
  # mini-batch boundaries).
  process_metrics(trainer, trainer.get_metrics())
  write_pending_eval_metrics(trainer)

  # Checkpoint cadence is delegated to the manager's save-decision policy.
  trainer.save_checkpoint(force=False)


class TrainLoop:
  """Drives a step-level trainer over train/eval datasets."""

  def __init__(self, trainer: "peft_trainer.PeftTrainer"):
    self.trainer = trainer
    self._drain_metrics = True

  def run(
      self,
      train_ds: Iterable[Any],
      eval_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
      *,
      cache_nnx_graph: bool = True,
      drain_metrics: bool = True,
  ) -> None:
    """Runs the training loop.

    Args:
      train_ds: The training dataset.
      eval_ds: Optional eval dataset, run at `eval_every_n_steps` cadence.
      skip_jit: If True, run the step functions un-jitted.
      cache_nnx_graph: Whether to cache the nnx graph in the jitted fns.
      drain_metrics: If True (default), the loop drains
        `trainer.get_metrics()` after every step, writes metrics itself, and
        triggers policy-driven checkpoint saves per applied update. Set
        False when an outer driver (e.g. `train_minibatch`) owns metrics
        readiness, writing, and checkpoint cadence; pending records are then
        left on the trainer for the driver to pull.
    """
    t = self.trainer
    self._drain_metrics = drain_metrics
    logging.log_first_n(
        logging.INFO,
        f"Training with mesh: {pxla.thread_resources.env.physical_mesh}",
        1,
    )
    t._skip_jit = skip_jit  # pylint: disable=protected-access
    t._cache_nnx_graph = cache_nnx_graph  # pylint: disable=protected-access
    t.jit_train_and_eval_step(skip_jit, cache_nnx_graph)
    if not skip_jit:
      cache_size = t._jitted_train_step_fn.func.jitted_fn._cache_size()  # pylint: disable=protected-access  # pytype: disable=attribute-error
      logging.log_if(
          logging.INFO,
          f"Compiled train_step cache size: {cache_size}",
          condition=cache_size not in t._jit_cache,  # pylint: disable=protected-access
      )
      t._jit_cache.add(cache_size)  # pylint: disable=protected-access

    if eval_ds:
      self.run_eval(eval_ds)

    if t.config.max_steps is not None and t._pbar is None:  # pylint: disable=protected-access
      t._pbar = progress_bar.ProgressBar(  # pylint: disable=protected-access
          metrics_prefix=t.metrics_prefix,
          metrics_logger=t.metrics_logger,  # pyrefly: ignore[bad-argument-type]
          initial_steps=t.train_steps,
          max_steps=t.config.max_steps,
          description=t.config.pbar_description,
      )

    if t.training_hooks:
      t.training_hooks.on_train_start(t)

    grad_accum_steps = t.config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    train_iterator = iter(train_ds)
    index = 0
    last_step_completion_time = time.perf_counter()
    while True:
      t._prof.maybe_activate(t.iter_steps)  # pylint: disable=protected-access
      with jax.profiler.StepTraceAnnotation("train", step_num=t.iter_steps):
        train_example = None
        if t.data_hooks:
          train_example = t.data_hooks.load_next_train_batch(t)
        else:
          try:
            train_example = next(train_iterator)
            if not t.is_managed_externally:
              # TODO(mridulsahu): Add support to restore the iterator state
              # instead of skipping the already trained examples.
              if index < t.iter_steps:
                # Skip the examples that are already trained.
                index += 1
                continue
              index += 1
          except StopIteration:
            pass
            
        if train_example is None:
          break

        # Stop training if max_steps is reached.
        if (
            not t.is_managed_externally
            and t.config.max_steps is not None
            and t.train_steps >= t.config.max_steps
        ):
          break

        if t.training_hooks:
          t.training_hooks.on_train_step_start(t)

        apply_gradients = ((t.iter_steps + 1) % grad_accum_steps == 0)

        t.train_step(train_example, apply_gradients=apply_gradients)
        if self._drain_metrics:
          process_metrics(t, t.get_metrics())

        if apply_gradients:
          if self._drain_metrics:
            # Checkpoint cadence is delegated to the manager's
            # save-decision policy (checkpointing_options). Deferred to the
            # outer driver together with metrics when draining is deferred.
            t.save_checkpoint(force=False)

          if (
              eval_ds
              and t.train_steps % t.config.eval_every_n_steps == 0
          ):
            self.run_eval(eval_ds)

      t._prof.maybe_deactivate(t.iter_steps)  # pylint: disable=protected-access

    t._throttler.wait_for_all()  # pylint: disable=protected-access
    logging.info(
        "Train loop finished in: %.4f seconds",
        time.perf_counter() - last_step_completion_time,
    )
    if t.training_hooks:
      t.training_hooks.on_train_end(t)
    if not t.is_managed_externally:
      t.close()

  def run_eval(self, eval_ds: Iterable[Any]) -> None:
    """Runs the evaluation loop over `eval_ds`."""
    t = self.trainer
    logging.info("Running evaluation on train step %d.", t.train_steps)
    eval_iterator = iter(eval_ds)
    with t._switch_mode(sft_metrics_logger.Mode.EVAL):  # pylint: disable=protected-access
      eval_loss, eval_steps = 0, 0
      while True:
        if t.data_hooks:
          eval_example = t.data_hooks.load_next_eval_batch(t)
        else:
          try:
            eval_example = next(eval_iterator)
          except StopIteration:
            eval_example = None
        if eval_example is None:
          break
        if t.training_hooks:
          t.training_hooks.on_eval_step_start(t)
        step_metrics = t.eval_step(eval_example)

        # Lazy device-side accumulation; no host sync per step.
        eval_loss += step_metrics.loss.compute()
        eval_steps += 1

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        return

      if self._drain_metrics:
        # The eval driver owns the end-of-run boundary: buffer all drained
        # records, then write the eval buffer exactly once. When draining is
        # deferred, the outer driver pulls the records and calls
        # `write_pending_eval_metrics` itself.
        process_metrics(t, t.get_metrics())
        write_pending_eval_metrics(t)
      if t.training_hooks:
        t.training_hooks.on_eval_step_end(t, eval_loss)
