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

"""PEFT trainer."""

from collections.abc import Iterable
import contextlib
import dataclasses
import time
from typing import Any, Callable, Concatenate, Dict, ParamSpec, Tuple
from absl import logging
import flax
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding as shd
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.sft import checkpoint_manager
from tunix.sft import inflight_throttler
from tunix.sft import metrics_logger
from tunix.sft import profiler
from tunix.sft import progress_bar
from tunix.sft import system_metrics_calculator

_ModelInputT = Dict[str, ArrayLike]
P = ParamSpec("P")


@contextlib.contextmanager
def time_measure(context: str = ""):
  start = time.perf_counter()
  try:
    yield
  finally:
    end = time.perf_counter()
    logging.info("%s finished in: %.4f seconds", context, end - start)


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig:
  """Configuration for the trainer."""

  eval_every_n_steps: int
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None

  # If set, the checkpoints will be saved to this path. Checkpoints
  # contains the model params and the train data iterator state.
  checkpoint_root_directory: str | None = None
  # Checkpoint configurations. If None, the default options will be used.
  checkpointing_options: ocp.CheckpointManagerOptions | None = None

  # Configs for the metrics logger.
  metrics_logging_options: metrics_logger.MetricsLoggerOptions | None = None

  # Configs for the profiler.
  profiler_options: profiler.ProfilerOptions | None = None

  data_sharding_axis: Tuple[str, ...] = ("fsdp",)

  # Controls how many train_steps can be scheduled ahead of time.
  max_inflight_computations: int = 2

  def get_with_default(self, key: str, default: Any) -> Any:
    val = getattr(self, key)
    if val is None:
      return default
    return val


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  # Input tokens provided to the model.
  input_tokens: jax.Array | np.ndarray

  # A mask that determines which input tokens are valid.
  input_mask: jax.Array | np.ndarray


def _calculate_global_batch_size(train_example: Any) -> int:
  if dataclasses.is_dataclass(train_example):
    attributes = dataclasses.asdict(train_example)
  else:
    attributes = vars(train_example)

  for field_value in attributes.values():
    if isinstance(field_value, (jax.Array, np.ndarray)):
      # Assume the first array we find has the batch dimension.
      return field_value.shape[0]

  raise TypeError(
      "Could not automatically determine batch size. No JAX or NumPy "
      "array found in the training example."
  )


def is_lora_enabled(model: nnx.Module) -> bool:
  for _, value in nnx.iter_graph(model):
    if isinstance(value, nnx.LoRAParam):
      return True
  return False


class PeftTrainer:
  """PEFT trainer for LoRA. Only LoRA parameters are updated."""

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
  ):
    self.model = model
    self.config = training_config
    self._lora_enabled = is_lora_enabled(self.model)
    if training_config.gradient_accumulation_steps is not None:
      optimizer = optax.MultiSteps(
          optimizer, training_config.gradient_accumulation_steps
      )
    if self._lora_enabled:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.LoRAParam)
    else:
      self.optimizer = nnx.Optimizer(self.model, optimizer)
    self.loss_fn = _default_loss_fn
    self.eval_loss_fn = _default_loss_fn
    self.gen_model_input_fn = lambda x: x
    self.checkpoint_manager = checkpoint_manager.CheckpointManager(
        root_directory=self.config.checkpoint_root_directory,
        options=self.config.checkpointing_options,
    )
    self.metrics_logger = metrics_logger.MetricsLogger(
        self.config.metrics_logging_options
    )
    self.is_managed_externally = False

    self._train_steps = 0
    self._eval_steps = 0
    self._throttler = inflight_throttler.InflightThrottler(
        max_inflight=training_config.max_inflight_computations
    )
    self._mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN
    self._has_aux = False
    self._pbar = None
    self._total_model_params = sum(
        p.size
        for p in jax.tree_util.tree_leaves(
            nnx.state(self.model).filter(nnx.Param, nnx.LoRAParam)
        )
    )

    self._train_steps = self.checkpoint_manager.maybe_restore(
        self.model, restore_only_lora_params=self._lora_enabled
    )

    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    self._prof = profiler.Profiler(
        initial_step=self._train_steps,
        max_step=self.config.max_steps,
        profiler_options=self.config.profiler_options,
    )

  def clear_jit_cache(self):
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overiding the training related states, for example, the loss function.
    """
    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None

  def with_loss_fn(
      self,
      loss_fn: Callable[
          Concatenate[nnx.Module, P], ArrayLike | Tuple[ArrayLike, Any]
      ],
      has_aux: bool = False,
  ):
    self.clear_jit_cache()
    self.loss_fn = loss_fn
    self.eval_loss_fn = loss_fn
    self._has_aux = has_aux
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], _ModelInputT]
  ):
    """Generates model input from training input.

    NB: output of this function will be passed to the loss function, so the args
    should match what loss function expects.

    Args:
      gen_model_input_fn: A function that generates model input from training
        input.

    Returns:
      PeftTrainer.
    """
    self.clear_jit_cache()
    self.gen_model_input_fn = gen_model_input_fn
    return self

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      The loss and auxiliary data if has_aux is True, otherwise the loss.
    """
    inputs = self.gen_model_input_fn(inputs)

    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    out, grads = grad_fn(model, **inputs)
    optimizer.update(grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    inputs = self.gen_model_input_fn(inputs)
    out = self.eval_loss_fn(model, **inputs)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def create_train_step_fn(self) -> Callable[..., ArrayLike]:
    """Creates the train step function."""
    return self._train_step

  def create_eval_step_fn(self) -> Callable[..., ArrayLike]:
    """Creates the eval step function."""
    return self._eval_step

  def _shard_optimizer(self, mesh: shd.Mesh) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the _train_step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    if mesh.empty:
      return
    optimizer_state = nnx.state(
        self.optimizer, nnx.optimizer.OptState
    )  # select only the optimizer state
    optimizer_shardings = nnx.get_named_sharding(optimizer_state, mesh)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_shardings
    )
    nnx.update(self.optimizer, optimizer_sharded_state)

  def jit_train_and_eval_step(self, skip_jit: bool = False):
    """Creates and returns the train and eval step functions.

    This function will return the cached ones if available.

    Args:
      skip_jit: If True, the train and eval step functions will not be JITed.

    Returns:
      A tuple of train and eval step functions.
    """
    train_step = self.create_train_step_fn()
    eval_step = self.create_eval_step_fn()
    if skip_jit:
      return train_step, eval_step
    else:
      if self._jitted_train_step_fn is None:
        mesh = pxla.thread_resources.env.physical_mesh
        self._shard_optimizer(mesh)
        self._jitted_train_step_fn = nnx.jit(
            train_step, donate_argnames=("optimizer",)
        )
        self._jitted_eval_step_fn = nnx.jit(
            eval_step, donate_argnames=("model",)
        )
      return self._jitted_train_step_fn, self._jitted_eval_step_fn

  def _shard_input(self, input_data: TrainingInput) -> TrainingInput:
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty:
      return input_data

    with jax.transfer_guard("allow"):
      return jax.tree.map(
          lambda x: jax.make_array_from_process_local_data(
              shd.NamedSharding(
                  mesh, shd.PartitionSpec(*self.config.data_sharding_axis)
              ),
              x,
          ),
          input_data,
      )

  def _prepare_inputs(self, input_data: Any) -> Any:
    """Override this function for additional input preparation."""
    return input_data

  def _post_process_train_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from train step."""
    pass

  def _post_process_eval_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from eval step."""
    pass

  def _log_metrics(
      self,
      loss: ArrayLike,
      step: int | None = None,
      tflops: float | None = None,
  ):
    self.metrics_logger.log("loss", loss, self._mode, step)
    self.metrics_logger.log("perplexity", jnp.exp(loss), self._mode, step)
    if tflops is not None:
      self.metrics_logger.log("tflops", tflops, self._mode, step)

  @contextlib.contextmanager
  def _switch_mode(self, mode: metrics_logger.Mode):
    original_mode = self._mode
    self._mode = mode
    try:
      yield
    finally:
      self._mode = original_mode

  @property
  def _tqdm_train_metrics(self) -> list[str] | None:
    return ["loss", "perplexity", "tflops"]

  @property
  def _tqdm_eval_metrics(self) -> list[str] | None:
    return ["loss", "perplexity"]

  def _may_update_pbar(self, metrics, increment_steps: bool = False):
    if self._pbar is not None:
      self._pbar.update_metrics(metrics, self._mode, ndigits=3)
      if increment_steps:
        self._pbar.update()

  def train(
      self,
      train_ds: Iterable[Any],
      eval_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Training loop."""
    mesh = pxla.thread_resources.env.physical_mesh
    logging.info("Training with mesh: %s", mesh)

    train_step, eval_step = self.jit_train_and_eval_step(skip_jit)

    if self.config.max_steps is not None and self._pbar is None:
      self._pbar = progress_bar.ProgressBar(
          metrics_logger=self.metrics_logger,
          initial_steps=self._train_steps,
          max_steps=self.config.max_steps,
      )
    with time_measure("Train loop"):
      for index, train_example in enumerate(train_ds):
        if not self.is_managed_externally:
          # TODO(mridulsahu): Add support to restore the iterator state
          # instead of skipping the already trained examples.
          if index < self._train_steps:
            # Skip the examples that are already trained.
            continue
        self._prof.maybe_activate(self._train_steps)
        with jax.profiler.StepTraceAnnotation(
            "train", step_num=self._train_steps
        ):
          if (
              eval_ds
              and self._train_steps % self.config.eval_every_n_steps == 0
          ):
            self._run_eval(eval_ds, eval_step)

          # Stop training if max_steps is reached.
          if (
              self.config.max_steps is not None
              and self._train_steps >= self.config.max_steps
          ):
            break

          train_example = self._prepare_inputs(train_example)
          train_example = self._shard_input(train_example)
          global_batch_size = _calculate_global_batch_size(train_example)

          self._throttler.wait_for_next()
          step_start_time = time.perf_counter()
          train_loss, aux = train_step(
              self.model, self.optimizer, train_example
          )
          step_end_time = time.perf_counter()
          step_time_delta = step_end_time - step_start_time

          tflops = system_metrics_calculator.tflops(
              total_model_params=self._total_model_params,
              global_batch_size=global_batch_size,
              step_time_delta=step_time_delta,
          )

          self._throttler.add_computation(train_loss)
          self._train_steps += 1
          self._post_process_train_step(aux)
          self._log_metrics(
              train_loss,
              self._train_steps,
              tflops,
          )
          self._may_update_pbar(self._tqdm_train_metrics, increment_steps=True)

          logging.info(
              "Train step %d training loss: %f  - training perplexity: %f",
              self._train_steps,
              self.metrics_logger.get_metric("loss", "train"),
              self.metrics_logger.get_metric("perplexity", "train"),
          )

          # Actual checkpoint frequency is configured by checkpointing_options.
          self.checkpoint_manager.save(
              self._train_steps,
              self.model,
              save_only_lora_params=self._lora_enabled,
          )
          self._prof.maybe_deactivate(self._train_steps)

    self._throttler.wait_for_all()
    if not self.is_managed_externally:
      self.close()

  def _save_last_checkpoint(self):
    last_saved_step = self.checkpoint_manager.latest_step()
    if last_saved_step is None or last_saved_step < self._train_steps:
      self.checkpoint_manager.save(
          self._train_steps,
          self.model,
          save_only_lora_params=self._lora_enabled,
          force=True,
      )

  @property
  def train_steps(self) -> int:
    return self._train_steps

  def close(self):
    self._save_last_checkpoint()
    self.checkpoint_manager.close()
    self.metrics_logger.close()
    if self._pbar is not None:
      self._pbar.close()
      self._pbar = None

  def _run_eval(
      self,
      eval_ds: Iterable[Any],
      eval_step: Callable[..., Any],
  ) -> None:
    """Runs evaluation loop."""
    with self._switch_mode(metrics_logger.Mode.EVAL):
      eval_loss, local_eval_steps = 0, 0
      for eval_example in eval_ds:
        eval_example = self._prepare_inputs(eval_example)
        eval_example = self._shard_input(eval_example)
        loss, aux = eval_step(self.model, eval_example)
        loss = jax.lax.stop_gradient(loss)
        self._eval_steps += 1
        self._post_process_eval_step(aux)
        eval_loss += loss
        local_eval_steps += 1
      self._log_metrics(eval_loss / local_eval_steps, self._train_steps)
      self._may_update_pbar(self._tqdm_eval_metrics)

      logging.info(
          "Train step %d eval loss: %f - eval perplexity: %f",
          self._train_steps,
          self.metrics_logger.get_metric("loss", "eval"),
          self.metrics_logger.get_metric("perplexity", "eval"),
      )


def _default_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  """Default loss function for PEFT training."""
  logits, _ = model(input_tokens, positions, None, attention_mask)

  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  # Equivalent to: optax.softmax_cross_entropy(logits, one_hot).mean()
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor
