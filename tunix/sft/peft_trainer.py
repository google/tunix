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
import functools
import os
from typing import Any, Callable, Concatenate, Dict, List, ParamSpec, Tuple

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
from tunix.perf import metrics as perf_metrics
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft import checkpoint_manager
from tunix.sft import hooks
from tunix.sft import inflight_throttler
from tunix.sft import metrics_logger as sft_metrics_logger
from tunix.sft import profiler
from tunix.sft import sharding_utils
from tunix.sft import utils
from tunix.train import abstract_trainer
from tunix.train import train_loop as train_loop_lib

_ModelInputT = Dict[str, ArrayLike]
P = ParamSpec("P")
MetricsLogger = sft_metrics_logger.MetricsLogger
MetricsLoggerOptions = sft_metrics_logger.MetricsLoggerOptions
StepMetrics = abstract_trainer.StepMetrics
TrainerPayload = abstract_trainer.TrainerPayload


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
  metrics_logging_options: MetricsLoggerOptions | None = None

  # Configs for the profiler.
  profiler_options: profiler.ProfilerOptions | None = None

  # Configs for performance metrics.
  perf_metrics_options: perf_metrics.PerfMetricsOptions | None = None

  data_sharding_axis: Tuple[str, ...] = ("fsdp",)

  # Controls how many train_steps can be scheduled ahead of time.
  max_inflight_computations: int = 2

  # Prefix for metric names for logging. Not sticking it in
  # `metrics_logging_options` because the latter is optional.
  metrics_prefix: str = ""

  # Progress bar description.
  pbar_description: str | None = "Training"

  # Sequence packing configuration.
  max_seq_token_per_tpu: int | None = None

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

  # Optional images for vision models.
  images: jax.Array | np.ndarray | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class MetricsBuffer:
  """Metrics collected for a specific step.

  Attributes:
    step: The training step number.
    losses: A list of loss values recorded within this step (e.g., across
      gradient accumulation steps).
    additional_metrics: Dictionary for storing additional metrics. The key is
      the metric name, and the value is a tuple containing a list of metric
      values and a callable to aggregate them.
  """

  step: int
  losses: List[ArrayLike]
  additional_metrics: Dict[
      str, Tuple[List[ArrayLike], Callable[[ArrayLike], ArrayLike]]
  ] = dataclasses.field(default_factory=dict)

  @property
  def loss(self):
    """Returns the mean of the recorded losses for the step."""
    if self.losses and hasattr(self.losses[0], "unreduced_sum"):
      total_sum = sum(np.array(x.unreduced_sum) for x in self.losses)
      total_denom = sum(np.array(x.denominator) for x in self.losses)
      if total_denom == 0:
        return 0.0
      return total_sum / total_denom
    return np.mean(np.array([np.array(x) for x in self.losses]))


def _calculate_global_batch_size(train_example: Any) -> int:
  """Calculates the global batch size from a training example.

  Args:
    train_example: A training example, which can be a dataclass, a dict, or an
      object with attributes.

  Returns:
    The global batch size.

  Raises:
    TypeError: If the batch size cannot be determined from the training example.
  """
  if dataclasses.is_dataclass(train_example):
    attributes = dataclasses.asdict(train_example)
  elif isinstance(train_example, dict):
    attributes = train_example
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


class PeftTrainer(abstract_trainer.AbstractTrainer):
  """PEFT trainer for LoRA. Only LoRA parameters are updated.

  Implements the step-level `AbstractTrainer` API. Loop-level concerns are
  handled by `tunix.train.train_loop.TrainLoop`; `train()` is a convenience
  wrapper that drives one.

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use. To monitor the learning rate at each step,
      use `optax.schedules.inject_hyperparams` to inject learning rate as a
      hyperparameter. For example: ``optimizer =
      optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)``
    loss_fn: The loss function to use.
    eval_loss_fn: The loss function to use for evaluation.
    gen_model_input_fn: The function to generate model input from training
      input.
    checkpoint_manager: The checkpoint manager to use.
    metrics_logger: The metrics logger to use.
    metrics_prefix: The prefix for metric names for logging.
    is_managed_externally: Whether the trainer is managed externally.
    training_hooks: The training hooks to use.
    data_hooks: The data hooks to use.
  """

  supports_sequence_packing = False

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
      metrics_logger: MetricsLogger | None = None,
      perf_tracer: perf_trace.Tracer | None = None,
      perf_tracer_v2: perf_tracer_lib.Tracer | None = None,
  ):
    # TODO(noghabi): Implement sequence packing for SFT and remove this check.
    if (
        training_config.max_seq_token_per_tpu is not None
        and not self.supports_sequence_packing
    ):
      raise ValueError(
          "Sequence packing is not supported in SFT PeftTrainer yet."
      )

    self.model = model
    self.config = training_config
    self._lora_enabled = utils.is_lora_enabled(self.model)
    # Gradient accumulation is caller-driven via
    # `train_step(apply_gradients=...)` (see AbstractTrainer); grads are
    # accumulated internally instead of wrapping with `optax.MultiSteps`.
    if self._lora_enabled:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.LoRAParam)
    else:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.Param)

    self.loss_fn = _default_loss_fn
    self.eval_loss_fn = _default_loss_fn
    self.forward_fn = None
    self.gen_model_input_fn = lambda x: x
    self.checkpoint_manager = checkpoint_manager.CheckpointManager(
        root_directory=self.config.checkpoint_root_directory,
        options=self.config.checkpointing_options,
    )
    self.metrics_logger = metrics_logger
    self.metrics_prefix = self.config.metrics_prefix
    if self.metrics_logger is None:
      self.metrics_logger = MetricsLogger(
          self.config.metrics_logging_options,
      )
    self.is_managed_externally = False
    self._perf_tracer = (
        perf_tracer if perf_tracer is not None else perf_trace.NoopTracer()
    )
    self._perf_tracer_v2 = (
        perf_tracer_v2
        if perf_tracer_v2 is not None
        else perf_tracer_lib.NoopTracer()
    )

    self._train_steps = 0  # represent # of times model has been updated
    self._iter_steps = 0  # represent # of times trainer has looped
    self._throttler = inflight_throttler.InflightThrottler(
        max_inflight=training_config.max_inflight_computations
    )
    self._mode: sft_metrics_logger.Mode = sft_metrics_logger.Mode.TRAIN
    self._has_aux = False
    self._pbar = None

    self._train_steps, self._restored_custom_metadata = (
        self.checkpoint_manager.maybe_restore(
            self.model,
            self.optimizer,
            restore_only_lora_params=self._lora_enabled,
        )
    )
    self._iter_steps = self._train_steps * self.config.get_with_default(
        "gradient_accumulation_steps", 1
    )

    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    self._jitted_grad_step_fn = None
    self._jitted_apply_grads_fn = None
    self._jitted_forward_fn = None
    self._skip_jit = False
    self._cache_nnx_graph = True
    # Internal gradient accumulation buffer (caller-driven accumulation).
    self._accum_grads = None
    self._accum_count = 0
    self._mini_batch_size: int | None = None
    self._pending_metrics: list[Tuple[StepMetrics, int, bool]] = []

    max_step = None
    if self.config.max_steps is not None:
      max_step = self.config.max_steps * self.config.get_with_default(
          "gradient_accumulation_steps", 1
      )
    self._prof = profiler.Profiler(
        initial_step=self._iter_steps,
        max_step=max_step,
        profiler_options=self.config.profiler_options,
    )
    self._buffered_train_metrics: MetricsBuffer | None = None
    self._prev_buffered_train_metrics: MetricsBuffer | None = None
    self._buffered_eval_metrics: MetricsBuffer | None = None
    self.training_hooks = None
    self.data_hooks = None
    self._jit_cache = set()

  def with_training_hooks(self, training_hooks: hooks.TrainingHooks):
    self.training_hooks = training_hooks

  def with_data_hooks(self, data_hooks: hooks.DataHooks):
    self.data_hooks = data_hooks

  def clear_jit_cache(self):
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overriding the training related states, for example, the loss function.
    """
    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    self._jitted_grad_step_fn = None
    self._jitted_apply_grads_fn = None
    self._jitted_forward_fn = None

  def with_loss_fn(
      self,
      loss_fn: Callable[
          Concatenate[nnx.Module, P], ArrayLike | Tuple[ArrayLike, Any]
      ],
      has_aux: bool = False,
  ):
    self.clear_jit_cache()
    self.loss_fn = loss_fn  # pyrefly: ignore[bad-assignment]
    self.eval_loss_fn = loss_fn  # pyrefly: ignore[bad-assignment]
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
    self.gen_model_input_fn = gen_model_input_fn  # pyrefly: ignore[bad-assignment]
    return self

  def with_forward_fn(
      self, forward_fn: Callable[Concatenate[nnx.Module, P], Any]
  ):
    """Sets the function used by `forward_batch`.

    Args:
      forward_fn: Called as `forward_fn(model, **inputs)` inside the jitted
        forward-only step; returns model outputs as a pytree of arrays
        (e.g., per-token log-probs). Inputs are the output of
        `gen_model_input_fn`, same as the loss functions.

    Returns:
      PeftTrainer.
    """
    self._jitted_forward_fn = None
    self.forward_fn = forward_fn
    return self

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> Tuple[ArrayLike, Any | None, ArrayLike]:
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      A tuple containing the loss, auxiliary data (or None if has_aux is False),
      and the gradient norm.
    """
    inputs = self.gen_model_input_fn(inputs)

    def grad_loss_fn(model, *args, **kwargs):
      out = self.loss_fn(model, *args, **kwargs)
      if self._has_aux:
        loss_metrics, aux = out
        return loss_metrics.compute(), (loss_metrics, aux)
      else:
        loss_metrics = out
        return loss_metrics.compute(), loss_metrics

    grad_fn = nnx.value_and_grad(
        grad_loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=True,
    )
    (_, out), grads = grad_fn(model, **inputs)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux, grad_norm
    else:
      return out, None, grad_norm

  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    inputs = self.gen_model_input_fn(inputs)
    out = self.eval_loss_fn(model, **inputs)
    if self._has_aux:
      loss, aux = out  # pyrefly: ignore[not-iterable]
      return loss, aux
    else:
      return out, None

  def create_train_step_fn(
      self,
  ) -> Callable[..., Tuple[ArrayLike, Any | None, ArrayLike]]:
    """Creates the train step function."""
    return self._train_step

  def create_eval_step_fn(self) -> Callable[..., ArrayLike]:
    """Creates the eval step function."""
    return self._eval_step  # pyrefly: ignore[bad-return]

  def _grad_step(
      self, model: nnx.Module, inputs: Any
  ) -> Tuple[ArrayLike, Any | None, ArrayLike, Any]:
    """Computes loss and gradients without applying an optimizer update."""
    inputs = self.gen_model_input_fn(inputs)
    def grad_loss_fn(model, *args, **kwargs):
      out = self.loss_fn(model, *args, **kwargs)
      if self._has_aux:
        loss_metrics, aux = out
        return loss_metrics.compute(), (loss_metrics, aux)
      else:
        loss_metrics = out
        return loss_metrics.compute(), loss_metrics

    grad_fn = nnx.value_and_grad(
        grad_loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=True,
    )
    (_, out), grads = grad_fn(model, **inputs)
    grad_norm = optax.global_norm(grads)
    if self._has_aux:
      loss, aux = out
    else:
      loss, aux = out, None
    return loss, aux, grad_norm, grads

  def _apply_grads_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, grads: Any
  ) -> None:
    """Applies (already averaged) gradients with the optimizer."""
    optimizer.update(model, grads)

  def _forward_step(self, model: nnx.Module, inputs: Any) -> Any:
    """Forward-only step returning model outputs (no loss, no grads)."""
    inputs = self.gen_model_input_fn(inputs)
    return self.forward_fn(model, **inputs)  # pylint: disable=not-callable

  def _get_forward_fn(self) -> Callable[..., Any]:
    """Returns the (jitted) forward-only step function, cached."""
    if self._skip_jit:
      return functools.partial(self._forward_step, self.model)
    if self._jitted_forward_fn is None:
      self._jitted_forward_fn = functools.partial(
          nnx.jit(self._forward_step), self.model
      )
    return self._jitted_forward_fn

  def _accum_step_fns(self) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Returns (grad_step_fn, apply_grads_fn), jitted and cached."""
    if self._skip_jit:
      return (
          functools.partial(self._grad_step, self.model),
          functools.partial(
              self._apply_grads_step, self.model, self.optimizer
          ),
      )
    if self._jitted_grad_step_fn is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      self._jitted_grad_step_fn = functools.partial(
          nnx.jit(self._grad_step), self.model
      )
      self._jitted_apply_grads_fn = functools.partial(
          nnx.jit(self._apply_grads_step, donate_argnames=("optimizer",)),
          self.model,
          self.optimizer,
      )
    return self._jitted_grad_step_fn, self._jitted_apply_grads_fn

  def _shard_optimizer(self, mesh: shd.Mesh) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the _train_step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    if mesh.empty:
      return
    optimizer_state = nnx.state(self.optimizer, nnx.optimizer.OptState)
    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_pspecs
    )
    nnx.update(self.optimizer, optimizer_sharded_state)

  def jit_train_and_eval_step(
      self, skip_jit: bool = False, cache_nnx_graph: bool = False
  ):
    """Creates and returns the train and eval step functions.

    This function will return the cached ones if available.

    Args:
      skip_jit: If True, the train and eval step functions will not be JITed.
      cache_nnx_graph: If True, the nnx graph will be cached.

    Returns:
      A tuple of train and eval step functions.
    """
    train_step = self.create_train_step_fn()
    eval_step = self.create_eval_step_fn()
    if skip_jit:
      return (
          functools.partial(train_step, self.model, self.optimizer),
          functools.partial(eval_step, self.model),
      )

    if self._jitted_train_step_fn is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      self._jitted_train_step_fn = nnx.jit(
          train_step, donate_argnames=("optimizer",)
      )
      self._jitted_eval_step_fn = nnx.jit(eval_step)

      def maybe_cache_and_partial(f, *args):
        if cache_nnx_graph:
          # wrap with partial so we can access jitted_fn in a consistent way.
          return functools.partial(nnx.cached_partial(f, *args))
        else:
          return functools.partial(f, *args)

      self._jitted_train_step_fn = maybe_cache_and_partial(
          self._jitted_train_step_fn, self.model, self.optimizer
      )
      self._jitted_eval_step_fn = maybe_cache_and_partial(
          self._jitted_eval_step_fn, self.model
      )
    return self._jitted_train_step_fn, self._jitted_eval_step_fn

  def _prepare_inputs(self, input_data: Any) -> Any:
    """Override this function for additional input preparation."""
    return input_data

  def _post_process_train_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from train step."""
    pass

  def _post_process_eval_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from eval step."""
    pass

  def _try_get_learning_rate(self) -> float | None:
    """Returns the learning rate from the optimizer state if available."""
    try:
      return self.optimizer.opt_state.hyperparams["learning_rate"].value
    except AttributeError:
      for chainpart in self.optimizer.opt_state:
        if isinstance(chainpart, optax.EmptyState):
          break
        if hasattr(chainpart, "hyperparams"):
          return chainpart.hyperparams["learning_rate"].value
      return None

  def _log_metrics(
      self,
      loss: ArrayLike,
      step: int | None = None,
      additional_metrics: dict[str, ArrayLike] | None = None,
  ):
    """Logs the metrics to the metrics logger and console."""
    perplexity = np.exp(jax.device_get(loss))
    self.metrics_logger.log(self.metrics_prefix, "loss", loss, self._mode, step)  # pyrefly: ignore[missing-attribute]
    self.metrics_logger.log(  # pyrefly: ignore[missing-attribute]
        self.metrics_prefix, "perplexity", perplexity, self._mode, step
    )
    learning_rate = self._try_get_learning_rate()
    if learning_rate is not None:
      self.metrics_logger.log(  # pyrefly: ignore[missing-attribute]
          self.metrics_prefix,
          "learning_rate",
          jax.device_get(learning_rate),
          self._mode,
          step,
      )

    if self._mode == sft_metrics_logger.Mode.TRAIN:
      logging.info(
          "Train step %d training loss: %f  - training perplexity: %f",
          step,
          loss,
          perplexity,
      )
    for k, v in (additional_metrics or {}).items():
      self.metrics_logger.log(self.metrics_prefix, k, v, self._mode, step)  # pyrefly: ignore[missing-attribute]

  def _buffer_metrics(
      self,
      metrics_buffer: MetricsBuffer | None,
      loss: ArrayLike,
      step: int,
      additional_metrics: (
          dict[str, Tuple[ArrayLike, Callable[[ArrayLike], ArrayLike]]] | None
      ) = None,
  ) -> MetricsBuffer:
    """Buffers metrics for the current step."""
    if metrics_buffer is None:
      metrics_buffer = MetricsBuffer(
          step=step,
          losses=[loss],
      )
    else:
      assert metrics_buffer.step == step
      metrics_buffer.losses.append(loss)
    if additional_metrics is not None:
      for k, (v, op) in additional_metrics.items():
        if k not in metrics_buffer.additional_metrics:
          metrics_buffer.additional_metrics[k] = ([v], op)
        else:
          metrics_buffer.additional_metrics[k][0].append(v)
    return metrics_buffer

  def _write_train_metrics(self):
    """Writes previous buffered train metrics."""
    if self._prev_buffered_train_metrics is None:
      # skip the first step so we can overlap I/O with next step.
      self._prev_buffered_train_metrics = self._buffered_train_metrics
      self._buffered_train_metrics = None
      return
    # increment the step by one for logging purpose, because train_step is not
    # incremented until the next model update.
    self._prev_buffered_train_metrics.step += 1
    self._write_metrics(self._prev_buffered_train_metrics)
    self._may_update_pbar(
        self._tqdm_train_metrics,
        step=self._prev_buffered_train_metrics.step,
        loss=self._prev_buffered_train_metrics.loss,
    )
    self._prev_buffered_train_metrics = self._buffered_train_metrics
    self._buffered_train_metrics = None

  def _write_metrics(self, metrics_buffer: MetricsBuffer):
    def _to_np_array(v):
      if isinstance(v, jax.Array):
        return np.asarray(v, dtype=np.float32)
      elif isinstance(v, list):
        return [_to_np_array(x) for x in v]
      return v

    self._log_metrics(
        loss=metrics_buffer.loss,
        step=metrics_buffer.step,
        additional_metrics={
            k: op(_to_np_array(v))
            for k, (
                v,
                op,
            ) in metrics_buffer.additional_metrics.items()
        },
    )

  @contextlib.contextmanager
  def _switch_mode(self, mode: sft_metrics_logger.Mode):
    original_mode = self._mode
    self._mode = mode
    try:
      yield
    finally:
      self._mode = original_mode

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    return ["loss", "perplexity", "learning_rate"]

  def _may_update_pbar(
      self,
      metrics: list[str],
      step: int | None = None,
      loss: ArrayLike | None = None,
  ):
    """Updates the progress bar with the given metrics if available."""
    if self._pbar is not None:
      self._pbar.update_metrics(metrics, self._mode, ndigits=3)
      self._pbar.update()

    if self.training_hooks and self._mode == sft_metrics_logger.Mode.TRAIN:
      self.training_hooks.on_train_step_end(self, step, loss)

  # --- AbstractTrainer step-level API ---

  def init_state(self) -> None:
    """Shards optimizer state and builds the jitted step functions.

    Idempotent. Does not restore checkpoints; use `restore_checkpoint` (note:
    for backward compatibility, the constructor still auto-restores from
    `checkpoint_root_directory` if a checkpoint exists there).
    """
    self.jit_train_and_eval_step(self._skip_jit, self._cache_nnx_graph)

  def train_step(
      self,
      payload: TrainerPayload | Any,
      *,
      apply_gradients: bool = True,
      **kwargs,
  ) -> StepMetrics:
    """Executes one forward/backward pass.

    Gradient accumulation is caller-driven: with `apply_gradients=False`,
    gradients are accumulated internally; with True, the accumulated mean
    gradients (including this step's) are applied and the buffer is reset.
    `global_step` increments only when gradients are applied.

    Args:
      payload: A `TrainerPayload` or a raw training batch.
      apply_gradients: Whether to apply (vs. accumulate) gradients.
      **kwargs: Unused.

    Returns:
      Metrics for this step, as device arrays.
    """
    del kwargs
    inputs = (
        payload.inputs if isinstance(payload, TrainerPayload) else payload
    )
    inputs = self._prepare_inputs(inputs)
    inputs = sharding_utils.shard_input(
        inputs, self.config.data_sharding_axis
    )

    tags = self._perf_tags()
    with self._perf_tracer.span(
        "peft_train_step",
        pxla.thread_resources.env.physical_mesh.devices,
    ) as span, self._perf_tracer_v2.span(
        perf_constants.PEFT_TRAIN,
        pxla.thread_resources.env.physical_mesh.devices,
        tags=tags,
    ) as span_v2:
      if apply_gradients and self._accum_count == 0:
        # Fast path: single fused grad+update step, no pending accumulation.
        train_step_fn, _ = self.jit_train_and_eval_step(
            self._skip_jit, self._cache_nnx_graph
        )
        loss, aux, grad_norm = train_step_fn(inputs)
      else:
        grad_step_fn, apply_grads_fn = self._accum_step_fns()
        loss, aux, grad_norm, grads = grad_step_fn(inputs)
        if self._accum_grads is None:
          self._accum_grads = grads
        else:
          self._accum_grads = jax.tree.map(jnp.add, self._accum_grads, grads)
        self._accum_count += 1
        if apply_gradients:
          count = self._accum_count
          mean_grads = jax.tree.map(lambda g: g / count, self._accum_grads)
          apply_grads_fn(mean_grads)
          self._accum_grads = None
          self._accum_count = 0
          
      span.device_end([loss])
      span_v2.async_end([loss])

    self._iter_steps += 1
    step_id = self.train_steps
    if apply_gradients:
      self._train_steps += 1

    step_metrics = StepMetrics(loss=loss, grad_norm=grad_norm, aux=aux)
    if not hasattr(self, "_pending_metrics"):
      self._pending_metrics = []
    self._pending_metrics.append((step_metrics, step_id, False, apply_gradients))
    return step_metrics

  def eval_step(
      self, payload: TrainerPayload | Any, **kwargs
  ) -> StepMetrics:
    """Executes a forward-only evaluation step. Does not mutate state."""
    del kwargs
    inputs = (
        payload.inputs if isinstance(payload, TrainerPayload) else payload
    )
    inputs = self._prepare_inputs(inputs)
    inputs = sharding_utils.shard_input(
        inputs, self.config.data_sharding_axis
    )
    _, eval_step_fn = self.jit_train_and_eval_step(
        self._skip_jit, self._cache_nnx_graph
    )
    loss, aux = eval_step_fn(inputs)
    loss = jax.lax.stop_gradient(loss)
    step_metrics = StepMetrics(loss=loss, aux=aux)
    
    if not hasattr(self, "_pending_metrics"):
      self._pending_metrics = []
    self._pending_metrics.append((step_metrics, self.train_steps, True, False))
    return step_metrics

  def forward_batch(
      self, payload: TrainerPayload | Any, **kwargs
  ) -> Any:
    """Executes a forward-only pass and returns model outputs.

    Runs the function configured via `with_forward_fn` (e.g., per-token
    log-prob computation for RL recompute) inside a jitted, forward-only
    step. Does not mutate trainer state or counters.

    Args:
      payload: A `TrainerPayload` or a raw batch.
      **kwargs: Unused.

    Returns:
      Model outputs as a pytree of arrays, with gradients stopped.

    Raises:
      ValueError: If no forward function has been configured.
    """
    del kwargs
    if self.forward_fn is None:
      raise ValueError(
          "No forward function configured; call with_forward_fn() first."
      )
    inputs = (
        payload.inputs if isinstance(payload, TrainerPayload) else payload
    )
    inputs = self._prepare_inputs(inputs)
    inputs = sharding_utils.shard_input(
        inputs, self.config.data_sharding_axis
    )
    outputs = self._get_forward_fn()(inputs)
    return jax.lax.stop_gradient(outputs)

  def save_checkpoint(self, path: str | None = None, **kwargs) -> str:
    """Serializes the current model and optimizer state now.

    Args:
      path: Checkpoint root directory. If None, uses the configured
        `checkpoint_root_directory`.
      **kwargs: `force` (bool, default True) to bypass the save decision
        policy.

    Returns:
      The path the checkpoint was written to.
    """
    if self._accum_count > 0:
      logging.warning(
          "Saving checkpoint mid gradient-accumulation window; pending"
          " accumulated gradients are not checkpointed."
      )
    force = kwargs.pop("force", True)
    custom_metadata = self.custom_checkpoint_metadata()
    if path is None:
      if self.config.checkpoint_root_directory is None:
        raise ValueError(
            "No `path` given and `checkpoint_root_directory` is not set."
        )
      self.checkpoint_manager.save(
          self._train_steps,
          self.model,
          self.optimizer,
          save_only_lora_params=self._lora_enabled,
          force=force,
          custom_metadata=custom_metadata,
      )
      return os.path.join(
          self.config.checkpoint_root_directory, str(self._train_steps)
      )
    manager = checkpoint_manager.CheckpointManager(
        root_directory=path, options=self.config.checkpointing_options
    )
    manager.save(
        self._train_steps,
        self.model,
        self.optimizer,
        save_only_lora_params=self._lora_enabled,
        force=True,
        custom_metadata=custom_metadata,
    )
    manager.close()
    return os.path.join(path, str(self._train_steps))

  def restore_checkpoint(self, path: str, **kwargs) -> int:
    """Restores model and optimizer state from a checkpoint root directory.

    Args:
      path: Checkpoint root directory to restore from.
      **kwargs: `step` (int) to restore a specific step; defaults to latest.

    Returns:
      The restored global step (0 if no checkpoint was found).
    """
    step = kwargs.pop("step", None)
    manager = checkpoint_manager.CheckpointManager(
        root_directory=path, options=self.config.checkpointing_options
    )
    restored_step, custom_metadata = manager.maybe_restore(
        self.model,
        self.optimizer,
        step=step,
        restore_only_lora_params=self._lora_enabled,
    )
    manager.close()
    self._train_steps = restored_step
    self._iter_steps = restored_step * self.config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    self._restored_custom_metadata = custom_metadata
    self._accum_grads = None
    self._accum_count = 0
    return restored_step

  def get_weights(
      self,
      *,
      gather: bool = False,
      full_params: bool = False,
      **kwargs,
  ) -> nnx.State:
    """Returns current model weights (e.g., for weight syncing).

    Args:
      gather: If True, fetch weights to host; otherwise keep them sharded.
      full_params: If True, return all params; otherwise only trainable
        params (LoRA-only when LoRA is enabled).
      **kwargs: Unused.

    Returns:
      The model weights as an nnx.State.
    """
    del kwargs
    if self._lora_enabled and not full_params:
      state = nnx.state(self.model, nnx.LoRAParam)
    else:
      state = nnx.state(self.model, nnx.Param)
    if gather:
      state = jax.device_get(state)
    return state

  @property
  def global_step(self) -> int:
    """Number of optimizer updates applied. Alias of `train_steps`."""
    return self._train_steps

  # --- Loop-level convenience API ---

  def train(
      self,
      train_ds: Iterable[Any],
      eval_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
      *,
      cache_nnx_graph: bool = True,
  ) -> None:
    """Training loop. Delegates to `tunix.train.train_loop.TrainLoop`."""
    train_loop_lib.TrainLoop(self).run(
        train_ds, eval_ds, skip_jit, cache_nnx_graph=cache_nnx_graph
    )

  def _save_last_checkpoint(self):
    last_saved_step = self.checkpoint_manager.latest_step()
    if last_saved_step is None or last_saved_step < self._train_steps:
      self.checkpoint_manager.save(
          self._train_steps,
          self.model,
          self.optimizer,
          save_only_lora_params=self._lora_enabled,
          force=True,
      )

  @property
  def train_steps(self) -> int:
    """Returns the number of train steps taken."""
    return self._train_steps

  @property
  def iter_steps(self) -> int:
    """Returns the number of iterator steps taken."""
    return self._iter_steps

  def custom_checkpoint_metadata(self) -> dict[str, Any]:
    """Override this function to return the custom metadata for the checkpoint manager."""
    return {}

  def get_metrics(self) -> list[tuple[abstract_trainer.StepMetrics, int, bool, bool]]:
    """Returns and clears the recently collected step metrics."""
    metrics = getattr(self, "_pending_metrics", [])
    self._pending_metrics = []
    return metrics

  def close(self):
    """Closes the trainer and its associated resources.

    This includes writing any buffered metrics, saving the last checkpoint,
    and closing the checkpoint manager and metrics logger.
    """
    self._write_train_metrics()
    self._save_last_checkpoint()
    self.checkpoint_manager.close()
    self.metrics_logger.close()  # pyrefly: ignore[missing-attribute]
    if self._pbar is not None:
      self._pbar.close()
      self._pbar = None


def _default_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
) -> ArrayLike:
  """Default loss function for PEFT training."""
  # Weird kwargs workaround because not all models support `images` right now.
  kwargs = {} if images is None else {"images": images}
  logits, _ = model(input_tokens, positions, None, attention_mask, **kwargs)

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
