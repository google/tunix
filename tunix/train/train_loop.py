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

This is a transitional split extracted from `PeftTrainer.train`: metric
buffers and loop components (throttler, profiler, progress bar) still live on
the trainer so that existing subclass overrides (e.g.
`_post_process_train_step`) keep working.
"""

from collections.abc import Iterable, Sequence
import concurrent.futures
import time
from typing import Any, List, TYPE_CHECKING

from absl import logging
import jax
from jax.interpreters import pxla
import numpy as np
from tunix.perf.experimental import constants as perf_constants
from tunix.sft import metrics_logger as sft_metrics_logger
from tunix.sft import progress_bar
from tunix.train import abstract_trainer

if TYPE_CHECKING:
  from tunix.sft import peft_trainer


_METRICS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _process_metrics_async(trainer, metrics_data):
  for step_metrics, step_id, apply_gradients in metrics_data:
    trainer._throttler.add_computation(step_metrics.loss.unreduced_sum)  # pylint: disable=protected-access
    trainer._buffered_train_metrics = trainer._buffer_metrics(  # pylint: disable=protected-access
        trainer._buffered_train_metrics,  # pylint: disable=protected-access
        loss=step_metrics.loss,
        step=step_id,
        additional_metrics={
            "grad_norm": (step_metrics.grad_norm, np.mean)
        },
    )
    # NB: put this after t._buffer_metrics is important.
    trainer._post_process_train_step(step_metrics.aux)  # pylint: disable=protected-access

    if apply_gradients:
      trainer._write_train_metrics()  # pylint: disable=protected-access


def train_minibatch(
    trainer: abstract_trainer.AbstractTrainer,
    payloads: Sequence[abstract_trainer.TrainerPayload | Any],
    **kwargs,
) -> None:
  """Executes one optimizer update over a mini-batch.
  !!!Demo purpose!!! Just highlevel idea on the batch driving flow.

  Iterates over the micro-batches in `payloads` and delegates to
  `trainer.train` which will run the `TrainLoop` over each micro-batch.

  Args:
    trainer: The trainer to step.
    payloads: The micro-batches for one optimizer update, in order.
    **kwargs: Forwarded to `trainer.train`.
  """
  payloads = list(payloads)
  if not payloads:
    raise ValueError("train_minibatch requires at least one payload.")

  for micro_batch in payloads:
    # We use a type ignore here since AbstractTrainer does not define train(),
    # but at runtime trainer will be a PeftTrainer or similar.
    trainer.train(train_ds=[micro_batch], **kwargs)  # type: ignore

  metrics_data = trainer.get_metrics()
  if metrics_data:
    # !! Waiting for the metrics ready may live in worker to move metrics to local host
    # before RPC transfer to orchestrator.
    _METRICS_EXECUTOR.submit(_process_metrics_async, trainer, metrics_data)

  trainer.save_checkpoint(force=False)


class TrainLoop:
  """Drives a step-level trainer over train/eval datasets."""

  def __init__(self, trainer: "peft_trainer.PeftTrainer"):
    self.trainer = trainer

  def run(
      self,
      train_ds: Iterable[Any],
      eval_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
      *,
      cache_nnx_graph: bool = True,
  ) -> None:
    """Runs the training loop."""
    t = self.trainer
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

        t._throttler.wait_for_next()  # pylint: disable=protected-access
        if t.training_hooks:
          t.training_hooks.on_train_step_start(t)

        apply_gradients = ((t.iter_steps + 1) % grad_accum_steps == 0)

        t.train_step(train_example, apply_gradients=apply_gradients)

        if apply_gradients:

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
        t._buffered_eval_metrics = t._buffer_metrics(  # pylint: disable=protected-access
            t._buffered_eval_metrics,  # pylint: disable=protected-access
            loss=step_metrics.loss,
            step=t.train_steps,
        )
        t._post_process_eval_step(step_metrics.aux)  # pylint: disable=protected-access
        eval_loss += step_metrics.loss
        eval_steps += 1

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        return

      t._write_metrics(t._buffered_eval_metrics)  # pylint: disable=protected-access
      logging.info(
          "Train step %d eval loss: %f - eval perplexity: %f",
          t.train_steps,
          t.metrics_logger.get_metric(t.metrics_prefix, "loss", "eval"),  # pyrefly: ignore[missing-attribute]
          t.metrics_logger.get_metric(  # pyrefly: ignore[missing-attribute]
              t.metrics_prefix, "perplexity", "eval"
          ),
      )
      t._buffered_eval_metrics = None  # pylint: disable=protected-access
      if t.training_hooks:
        t.training_hooks.on_eval_step_end(t, eval_loss)
