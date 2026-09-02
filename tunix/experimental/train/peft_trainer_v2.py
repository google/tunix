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

from collections.abc import Iterable, Mapping
import contextlib
import dataclasses
import functools
import os
import time
from typing import Any, Callable, Concatenate, Dict, List, ParamSpec, Tuple

from absl import logging
import flax
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding as shd
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from jax.typing import DTypeLike  # pylint: disable=g-importing-member
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.experimental.common import datatypes
from tunix.experimental.metrics import metrics as exp_metrics
from tunix.experimental.train import abstract_trainer
from tunix.perf import metrics as perf_metrics
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft import checkpoint_manager
from tunix.sft import hooks
from tunix.sft import inflight_throttler
from tunix.sft import metrics_logger as sft_metrics_logger
from tunix.sft import profiler
from tunix.sft import progress_bar
from tunix.sft import sharding_utils
from tunix.sft import utils
from typing_extensions import override

_ModelInputT = dict[str, ArrayLike]
P = ParamSpec("P")
MetricsLogger = sft_metrics_logger.MetricsLogger
MetricsLoggerOptions = sft_metrics_logger.MetricsLoggerOptions


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
  # Static upper bound on real segments (sequences) per packed row, used to size
  # the segment-aware loss buckets (num_segments = this + 1 for the padding
  # bucket). ``None`` defaults to ``max_seq_token_per_tpu`` -- provably safe (a
  # pack of ``budget`` tokens holds at most ``budget`` unit-length segments) and
  # needs no tuning. Set a smaller value only to shrink the loss buckets at very
  # large budgets; ``pack_sequences`` raises if a pack exceeds it.
  max_segments_per_packed_row: int | None = None

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


def _opt_state_dtypes(optimizer: nnx.Optimizer) -> Any:
  """Returns the array dtype of every optimizer-state variable."""
  return jax.tree_util.tree_map(
      lambda value: value.get_value().dtype,
      nnx.state(optimizer, nnx.optimizer.OptState),
      is_leaf=lambda value: isinstance(value, nnx.Variable),
  )


def _restore_opt_state_float_dtypes(
    optimizer: nnx.Optimizer, dtypes: Any
) -> None:
  """Restores floating optimizer-state leaves to their pre-update dtypes."""

  def _restore(value, dtype):
    array = value.get_value()
    if jnp.issubdtype(array.dtype, jnp.floating) and array.dtype != dtype:
      value.set_value(array.astype(dtype))

  jax.tree_util.tree_map(
      _restore,
      nnx.state(optimizer, nnx.optimizer.OptState),
      dtypes,
      is_leaf=lambda value: isinstance(value, nnx.Variable),
  )


class GradientAccumulator(nnx.Module):
  """Accumulates gradients over multiple micro-steps.

  Unifies standard (unweighted) micro-batch averaging with sequence packing
  (weighted, denom-aware) accumulation.

  Averaging behavior (optax.MultiSteps semantics):
    When `add(grads)` is called without a denom, each micro-step implicitly
    adds 1.0 to the denominator. `get()` computes `Σ_grads / Σ_1`, which
    is the exact mean of the micro-step gradients. This is mathematically
    equivalent to a single optimization step on a batch of size `B =
    micro_batch_size * grad_acc_steps` when the loss is a mean-reduction
    (e.g., standard cross-entropy).

  Packing-aware behavior (Sum of Grads / Sum of Sizes):
    Under sequence packing, each yielded micro-batch contains a varying
    number of valid target tokens or training examples. The loss is
    computed as an *unreduced sum* over the packed batch. Callers pass the
    true size of the pack via `add(grads, denom=size)`. `get()` computes
    `Σ_grad(sum_loss_i) / Σ_size_i`, recovering the true global mean
    gradient across all items in the accumulated batch, avoiding the bias
    introduced by averaging pre-scaled micro-batch gradients of unequal
    sizes.

  Persistent vs. non-persistent mode (`self.persistent`):
    Controlled by `allocate_grads` at initialization:
    * Persistent mode (`allocate_grads=True`, `self.persistent=True`): Used
      when accumulating across multiple micro-steps (`gradient_accumulation_steps
      > 1`). A parameter-sized buffer is allocated at initialization and zeroed
      in-place when `reset()` is called so the buffer persists across updates.
    * Non-persistent mode (`allocate_grads=False`, `self.persistent=False`):
      Used on single-microstep fast paths (`gradient_accumulation_steps == 1`).
      No buffer is allocated at initialization (`self.grads` starts as `{}`).
      `add()` adopts the reference to incoming backward-pass gradients directly,
      and `reset()` drops the reference (`self.grads = nnx.data({})`) instead of
      writing a full parameter-sized copy of zeros that would never be read.

  Attributes:
    persistent: Whether the accumulator operates in persistent mode
      (`allocate_grads=True`) or non-persistent mode (`allocate_grads=False`).
    grads: The accumulated gradient pytree (`nnx.data`), or an empty dictionary
      `{}` before gradients are added in non-persistent mode.
    denom: The denominator (`nnx.Variable`) used for averaging or packing-aware
      normalization.
  """

  def __init__(
      self,
      model: nnx.Module,
      wrt: type[nnx.Variable],
      *,
      allocate_grads: bool = True,
      accumulator_dtype: DTypeLike = jnp.float32,
  ):
    """Initializes the gradient accumulator.

    Args:
      model: The model whose state to accumulate gradients for.
      wrt: The target variable type (e.g., `nnx.Param` or `nnx.LoRAParam`).
      allocate_grads: Whether to allocate an accumulated gradient buffer
        matching the model's parameter structure. When `False` (used on depth-1
        fast paths where accumulation is skipped), an empty dictionary is
        allocated to save HBM without altering the JIT signature.
      accumulator_dtype: The dtype used for accumulated gradient buffers.
        Defaults to `jnp.float32` to prevent low-precision underflow and
        rounding errors during multi-step accumulation. When returning
        accumulated gradients via `get`, they are cast back to the model's
        native parameter dtypes (e.g. `bfloat16`). Using lower-precision dtypes
        saves HBM but incurs numerical precision trade-offs without upcasting
        for large gradients.
    """
    state = nnx.state(model, wrt)
    self._param_dtypes = nnx.data(
        jax.tree_util.tree_map(
            lambda x: getattr(
                x, "dtype", getattr(getattr(x, "value", None), "dtype", None)
            ),
            state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
    )
    self.persistent = allocate_grads
    if allocate_grads:
      self.grads = nnx.data(
          jax.tree_util.tree_map(
              lambda x: jnp.zeros_like(x, dtype=accumulator_dtype), state
          )
      )
    else:
      # When every update consumes exactly one micro-batch, `set()` overwrites
      # the whole tree before anything reads it, so the initial zeros are dead
      # on arrival. Skipping them avoids writing a full copy of the parameter
      # tree (~3.5 GiB per device for gemma4-e2b at 12 layers in fp32).
      self.grads = nnx.data({})
      self._param_dtypes = nnx.data({})
    self.denom = nnx.Variable(jnp.zeros((), dtype=jnp.float32))

  def add(self, grads: Any, denom: jax.Array | None = None):
    def _add(acc_var, g_var):
      g = g_var[...] if isinstance(g_var, nnx.Variable) else g_var
      # set_value (no index) avoids the indexed __setitem__ "slow" path, whose
      # `.sharding` check on tracers triggers a per-leaf provenance scan that
      # dominates trace time; the stored value is identical.
      acc_var.set_value(acc_var[...] + g)

    if jax.tree_util.tree_leaves(self.grads):
      jax.tree_util.tree_map(
        _add,
        self.grads,
        grads,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
    else:
      # No buffer held: either it was never allocated, or a non-persistent
      # `reset()` released it.
      self.grads = nnx.data(grads)

    if denom is None:
      denom_val = jnp.asarray(1.0, dtype=jnp.float32)
    else:
      denom_val = denom.astype(jnp.float32)
    self.denom.set_value(self.denom[...] + denom_val)

  def get(self):
    scale = 1.0 / jnp.maximum(self.denom[...], jnp.asarray(1.0, jnp.float32))

    def _scale(v):
      return type(v)(v[...] * scale.astype(v[...].dtype))

    def _scale_and_cast(v, target_dtype):
      res = v[...] * scale.astype(v[...].dtype)
      return type(v)(res.astype(target_dtype) if target_dtype else res)

    if not jax.tree_util.tree_leaves(self.grads):
      # Fail here rather than inside optax, where the same problem surfaces as
      # "Mismatch custom node data: ('embedder', ...) != (); value: State({})".
      raise ValueError(
          "The gradient accumulator is empty. Either get() was called without a"
          " preceding add()/set(), or the gradients written by an earlier"
          " executable were discarded on the way out of jit -- nnx.cached_partial"
          " (cache_nnx_graph=True) freezes the bound module's graphdef, so a"
          " step that changes the accumulator's pytree structure cannot hand it"
          " to a later executable."
      )

    if not jax.tree_util.tree_leaves(self._param_dtypes):
      # When `allocate_grads=False` dtype map is empty. Gradients already carry
      # the right parameter dtype.
      return jax.tree_util.tree_map(
          _scale, self.grads, is_leaf=lambda x: isinstance(x, nnx.Variable)
      )

    return jax.tree_util.tree_map(
        _scale_and_cast,
        self.grads,
        self._param_dtypes,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def reset(self):
    """Clears the accumulator, either by zeroing the buffer or by dropping it.

    When self.persistent, the buffer must survive and be zeroed in place. If
    not persistent: zeroing would write a full parameter-sized copy that is
    never read. Drop the reference instead; `add()` re-adopts the incoming
    gradients.
    """
    if self.persistent:
      def _zero_in_place(v):
        # `x * 0` rather than `jnp.zeros_like(x)`, to preserve the buffer's
        # sharding.
        v.set_value(v[...] * 0)
      jax.tree_util.tree_map(
          _zero_in_place,
          self.grads,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
    else:
      self.grads = nnx.data({})
    self.denom.set_value(jnp.zeros_like(self.denom[...]))


def _default_weight_sync_worker() -> Any:
  from tunix.experimental.weight_sync import raiden_synchronizer  # pylint: disable=g-import-not-at-top

  return raiden_synchronizer.RaidenSynchronizer("trainer")


class PeftTrainer(abstract_trainer.AbstractTrainer):
  """PEFT trainer for LoRA. Only LoRA parameters are updated.

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use. To monitor the learning rate at each step,
      use `optax.schedules.inject_hyperparams` to inject learning rate as a
      hyperparameter. For example: ``optimizer =
      optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)``
    grad_accumulator: The gradient accumulator to use for accumulating gradients
      over multiple micro-steps.
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

  supports_sequence_packing = True

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
      metrics_logger: MetricsLogger | None = None,
      perf_tracer: perf_trace.Tracer | None = None,
      perf_tracer_v2: perf_tracer_lib.Tracer | None = None,
      weight_sync_worker_factory: Callable[[], Any] | None = None,
      sampler_type: str = "inprocess_vllm",
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
    wrt_target = nnx.LoRAParam if self._lora_enabled else nnx.Param
    self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=wrt_target)
    self.grad_accumulator = GradientAccumulator(
        self.model, wrt_target, allocate_grads=not self._is_single_microstep()
    )

    self.loss_fn = _default_loss_fn
    self.eval_loss_fn = _default_loss_fn
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
    self._last_update_grad_norm: ArrayLike | None = None

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

    self._jitted_fwd_bwd_step_fn = None
    self._jitted_update_step_fn = None
    self._jitted_eval_step_fn = None
    self._jitted_train_step_fn = None
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
    # TODO(b/532722958): mitigation for current metrics logging implementations.
    # Buffered_train[eval]_metrics will be None after metrics are logged. For
    # get_metrics to retrieve metrics from trainer, we need to write metrics to
    # a buffer. We should delete this once the metrics logging implementations
    # are updated according to the new design.
    self._written_metrics: exp_metrics.MetricsBuffer | None = None
    self.training_hooks = None
    self.data_hooks = None
    self._jit_cache = set()
    self._mini_batch_size = None
    self._target_state = None
    self._sampler_type = sampler_type
    self._weight_sync_worker: Any = None

  def with_training_hooks(self, training_hooks: hooks.TrainingHooks):
    self.training_hooks = training_hooks

  def with_data_hooks(self, data_hooks: hooks.DataHooks):
    self.data_hooks = data_hooks

  def set_target_state(self, target_state: Any) -> None:
    self._target_state = target_state

  def clear_jit_cache(self):
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overriding the training related states, for example, the loss function.
    """
    self._jitted_fwd_bwd_step_fn = None
    self._jitted_update_step_fn = None
    self._jitted_eval_step_fn = None
    self._jitted_train_step_fn = None

  @override
  def with_loss_fn(
      self,
      loss_fn: Callable[
          Concatenate[nnx.Module, P],
          ArrayLike | Tuple[ArrayLike, Any] | utils.LossOutput,
      ],
      has_aux: bool = False,
  ) -> "PeftTrainer":
    self.clear_jit_cache()
    self.loss_fn = loss_fn  # pyrefly: ignore[bad-assignment]
    self.eval_loss_fn = loss_fn  # pyrefly: ignore[bad-assignment]
    self._has_aux = has_aux
    return self

  @override
  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> "PeftTrainer":
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

  def _is_single_microstep(self) -> bool:
    return (
        self.config.get_with_default("gradient_accumulation_steps", 1) == 1
        and self.config.max_seq_token_per_tpu is None
    )

  def _fwd_bwd_step(
      self,
      model: nnx.Module,
      grad_accumulator: GradientAccumulator,
      inputs: Any,
  ) -> Tuple[ArrayLike, Any | None]:
    """Forward and backward passes through grad_fn.

    Args:
      model: The model to train.
      grad_accumulator: The gradient accumulator to use.
      inputs: The training input.

    Returns:
      A tuple containing the loss, and auxiliary data (or None if has_aux is
      False).
    """
    inputs = self.gen_model_input_fn(inputs)

    @functools.wraps(self.loss_fn)
    def diff_fn(model, *args, **kwargs):
      out = self.loss_fn(model, *args, **kwargs)
      if isinstance(out, utils.LossOutput):
        return out.primary_loss.unreduced_sum, out
      elif self._has_aux:
        return out[0], out[1]  # pyrefly: ignore[bad-index]
      else:
        return out, None

    grad_fn = nnx.value_and_grad(
        diff_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=True,
    )
    (loss_val, aux), grads = grad_fn(model, **inputs)

    if isinstance(aux, utils.LossOutput):
      # Compute exactly equivalent legacy loss val
      loss_val = aux.primary_loss.compute()
      # Accumulate the UNREDUCED gradients (d/dparam of the sum) weighted by the
      # loss's real denominator, so the optimizer step sees the GLOBAL weighted
      # mean (Sum grads / Sum denom) across micro-batches rather than a
      # mean-of-means.
      grad_accumulator.add(grads, denom=aux.primary_loss.denominator)
    else:
      grad_accumulator.add(grads, denom=jnp.asarray(1.0, dtype=jnp.float32))

    if isinstance(aux, utils.LossOutput):
      return loss_val, aux.aux_metrics
    elif self._has_aux:
      return loss_val, aux
    else:
      return loss_val, None

  def _update_step(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      grad_accumulator: GradientAccumulator,
  ) -> ArrayLike:
    """Updates the model weights.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      grad_accumulator: The gradient accumulator to use.

    Returns:
      The gradient norm.
    """
    acc_grads = grad_accumulator.get()
    # Compute the norm in float32. For production-size models the sum-of-squares
    # over bf16 grads quickly exhausts bf16, and float32 is needed for numerical
    # stability.
    norm = optax.global_norm(
        jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), acc_grads)
    )
    opt_state_dtypes = _opt_state_dtypes(optimizer)
    optimizer.update(model, acc_grads)
    _restore_opt_state_float_dtypes(optimizer, opt_state_dtypes)
    grad_accumulator.reset()
    return norm

  def _train_step(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      grad_accumulator: GradientAccumulator,
      inputs: Any,
  ) -> Tuple[ArrayLike, Any | None, ArrayLike]:
    """`_fwd_bwd_step` followed by `_update_step`, in one traced function.

    Only valid when `_is_single_microstep()`: there is exactly one micro-batch
    per update, so nothing needs to happen between the two halves. Tracing them
    together lets XLA treat the gradient tree as a module-internal temporary --
    overlappable with backward-pass scratch and dead at the end of the step --
    instead of a program output that has to survive until a second executable
    reads it. That is worth roughly one full copy of the parameter tree.

    The bodies are reused verbatim, so the fused and split paths are the same
    arithmetic in the same order; only XLA's buffer assignment differs.
    """
    loss, aux = self._fwd_bwd_step(model, grad_accumulator, inputs)
    return loss, aux, self._update_step(model, optimizer, grad_accumulator)

  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    inputs = self.gen_model_input_fn(inputs)
    out = self.eval_loss_fn(model, **inputs)
    if isinstance(out, utils.LossOutput):
      return out.primary_loss.compute(), out.aux_metrics
    elif self._has_aux:
      loss, aux = out  # pyrefly: ignore[not-iterable]
      return loss, aux
    else:
      return out, None

  def create_fwd_bwd_step_fn(
      self,
  ) -> Callable[..., Tuple[ArrayLike, Any | None]]:
    """Creates the forward and backward step function."""
    return self._fwd_bwd_step

  def create_update_step_fn(
      self,
  ) -> Callable[..., ArrayLike]:
    """Creates the update step function."""
    return self._update_step

  def create_train_step_fn(
      self,
  ) -> Callable[..., Tuple[ArrayLike, Any | None, ArrayLike]]:
    """Creates the fused forward/backward/update step function."""
    return self._train_step

  def create_eval_step_fn(
      self,
  ) -> Callable[..., ArrayLike | Tuple[ArrayLike, Any]]:
    """Creates the eval step function."""
    return self._eval_step  # pyrefly: ignore[bad-return]

  def _shard_optimizer(self, mesh: shd.Mesh) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the update step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    if mesh.empty:
      return

    def _shard(x, p):
      if not isinstance(x, (jax.Array, np.ndarray)):
        return x
      if p is None:
        p = shd.PartitionSpec()
      sharding = sharding_utils.get_sharding(x, mesh, p)  # pyrefly: ignore[bad-argument-type]
      if hasattr(x, "sharding") and x.sharding == sharding:
        return x
      if getattr(x, "is_fully_addressable", True):
        with jax.transfer_guard("allow"):
          return jax.device_put(x, sharding)
      return x

    optimizer_state = nnx.state(self.optimizer, nnx.optimizer.OptState)
    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)
    optimizer_sharded_state = jax.tree.map(
        _shard, optimizer_state, optimizer_pspecs
    )
    nnx.update(self.optimizer, optimizer_sharded_state)

    # Partition Gradients similar to the model. Skipped when the accumulator was
    # not allocated: there is nothing to shard, and the gradients that flow
    # `fwd_bwd` -> `update` are jit outputs whose sharding XLA derives from the
    # parameters.
    if jax.tree_util.tree_leaves(self.grad_accumulator.grads):
      grad_pspecs = nnx.get_partition_spec(self.grad_accumulator.grads)
      self.grad_accumulator.grads = jax.tree.map(
          _shard, self.grad_accumulator.grads, grad_pspecs
      )

    # Denominator is a scalar — replicate across all devices
    self.grad_accumulator.denom[...] = jax.device_put(
        self.grad_accumulator.denom[...],
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )

  def jit_fwd_bwd_update_and_eval_step(
      self, skip_jit: bool = False, cache_nnx_graph: bool = False
  ):
    """Creates and returns the train and eval step functions.

    This function will return the cached ones if available.

    Args:
      skip_jit: If True, the train and eval step functions will not be JITed.
      cache_nnx_graph: If True, the nnx graph will be cached.

    Returns:
      A tuple of fwd_bwd, update, and eval step functions.
    """
    fwd_bwd_step = self.create_fwd_bwd_step_fn()
    update_step = self.create_update_step_fn()
    eval_step = self.create_eval_step_fn()
    if skip_jit:
      self._jitted_train_step_fn = None
      return (
          functools.partial(fwd_bwd_step, self.model),
          functools.partial(
              update_step, self.model, self.optimizer, self.grad_accumulator
          ),
          functools.partial(eval_step, self.model),
      )

    if getattr(self, "_jitted_fwd_bwd_step_fn", None) is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      if self._is_single_microstep():
        # No grad_accumulator is created in this case.
        donate_argnames = ("model",)
      else:
        donate_argnames = ("model", "grad_accumulator")
      self._jitted_fwd_bwd_step_fn = nnx.jit(
          fwd_bwd_step, donate_argnames=donate_argnames,
      )
      self._jitted_update_step_fn = nnx.jit(
          update_step, donate_argnames=("optimizer", "grad_accumulator")
      )
      self._jitted_eval_step_fn = nnx.jit(eval_step)

      def maybe_cache_and_partial(f, *args):
        if cache_nnx_graph:
          # wrap with partial so we can access jitted_fn in a consistent way.
          return functools.partial(nnx.cached_partial(f, *args))
        else:
          return functools.partial(f, *args)

      self._jitted_fwd_bwd_step_fn = maybe_cache_and_partial(
          self._jitted_fwd_bwd_step_fn,
          self.model,
      )
      self._jitted_update_step_fn = maybe_cache_and_partial(
          self._jitted_update_step_fn,
          self.model,
          self.optimizer,
          self.grad_accumulator,
      )
      self._jitted_eval_step_fn = maybe_cache_and_partial(
          self._jitted_eval_step_fn, self.model
      )

      # Fused single-executable step, used by `train()` when each update
      # consumes one micro-batch. Donation deliberately mirrors the split path
      # (`optimizer` + `grad_accumulator`, model not donated) so the two are
      # structurally comparable and switching between them cannot change
      # numerics. Compilation is lazy, so building the wrapper here costs
      # nothing if the fused path is never called.
      _jitted_train_step_fn = nnx.jit(
          self.create_train_step_fn(),
          donate_argnames=("optimizer", "grad_accumulator"),
      )
      if self._is_single_microstep():
        self._jitted_train_step_fn = maybe_cache_and_partial(
            _jitted_train_step_fn,
            self.model,
            self.optimizer,
            self.grad_accumulator,
        )
      else:
        self._jitted_train_step_fn = None
    return (
        self._jitted_fwd_bwd_step_fn,
        self._jitted_update_step_fn,
        self._jitted_eval_step_fn,
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

    loss = metrics_buffer.loss
    additional_metrics = {
        k: op(_to_np_array(v))
        for k, (v, op) in metrics_buffer.additional_metrics.items()
    }
    self._log_metrics(
        loss=loss,
        step=metrics_buffer.step,
        additional_metrics=additional_metrics,
    )
    weighted_metrics = {}
    scalar_metrics = {"loss": loss}
    for k, val in additional_metrics.items():
      if isinstance(val, (utils.WeightedMetric, exp_metrics.WeightedMetric)):
        weighted_metrics[k] = val
      else:
        scalar_metrics[k] = val
    self._written_metrics = exp_metrics.MetricsBuffer(
        id=metrics_buffer.step,
        weighted_metrics=weighted_metrics,
        scalar_metrics=scalar_metrics,
        mode=self._mode.value,
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

  def _prepare_payload(self, payload: Any) -> Any:
    """Applies input preparation and sharding to one training payload."""
    payload = self._prepare_inputs(payload)
    return sharding_utils.shard_input(payload, self.config.data_sharding_axis)

  def _record_fwd_bwd(self, train_loss: ArrayLike, aux: Any) -> None:
    """Bookkeeping for one forward/backward pass, independent of how it ran."""
    self._buffered_train_metrics = self._buffer_metrics(
        self._buffered_train_metrics,
        loss=train_loss,
        step=self._train_steps,
    )
    self._post_process_train_step(aux)

  def _record_update(self, grad_norm: ArrayLike) -> int:
    """Bookkeeping for one optimizer update, independent of how it ran."""
    self._last_update_grad_norm = grad_norm
    if self._buffered_train_metrics is not None:
      metrics = self._buffered_train_metrics.additional_metrics
      if "grad_norm" not in metrics:
        metrics["grad_norm"] = ([grad_norm], np.mean)
      else:
        metrics["grad_norm"][0].append(grad_norm)
    self._train_steps += 1
    self._write_train_metrics()
    return self._train_steps

  @override
  def fwd_bwd(self, payload: datatypes.TrainerPayload | Any, **kwargs) -> None:
    """Executes forward and backward passes."""
    fwd_bwd_step, _, _ = self.jit_fwd_bwd_update_and_eval_step()
    self._record_fwd_bwd(
        *fwd_bwd_step(
            grad_accumulator=self.grad_accumulator,
            inputs=self._prepare_payload(payload),
        )
    )

  @override
  def update(self, **kwargs) -> int:
    """Applies the accumulated gradients."""
    _, update_step, _ = self.jit_fwd_bwd_update_and_eval_step()
    return self._record_update(update_step())

  def train_step(
      self, payload: datatypes.TrainerPayload | Any, **kwargs
  ) -> int:
    """Runs forward, backward and update as a single executable.

    Equivalent to `fwd_bwd(payload)` followed by `update()` -- same arithmetic
    in the same order -- but traced as one function, which lets XLA keep the
    gradient tree as an internal temporary instead of a program output. Only
    available in the single-microstep regime; when accumulating there is work
    between the two halves, so they must stay separate.
    """
    self.jit_fwd_bwd_update_and_eval_step()
    if self._jitted_train_step_fn is None:
      raise ValueError(
          "train_step() requires exactly one micro-batch per update. Use"
          " fwd_bwd() followed by update() when gradient_accumulation_steps > 1"
          " or sequence packing is enabled."
      )
    train_loss, aux, grad_norm = self._jitted_train_step_fn(
        self._prepare_payload(payload)
    )
    self._record_fwd_bwd(train_loss, aux)
    return self._record_update(grad_norm)

  @override
  def compile(self, dummy_data: Any) -> None:
    pass

  @override
  def eval_step(
      self, payload: datatypes.TrainerPayload | Any, **kwargs
  ) -> None:
    """Executes one eval micro-batch step.

    Prepares inputs, runs the (cached) jitted eval step function, buffers
    eval metrics, and calls _post_process_eval_step.  Callers should bracket
    a sequence of eval_step calls with eval_context() so that the metrics
    mode is set to EVAL and buffered metrics are written on exit.
    """
    _, _, eval_step_fn = self.jit_fwd_bwd_update_and_eval_step()
    payload = self._prepare_inputs(payload)
    payload = sharding_utils.shard_input(
        payload, self.config.data_sharding_axis
    )
    loss, aux = eval_step_fn(payload)
    loss = jax.lax.stop_gradient(loss)
    self._buffered_eval_metrics = self._buffer_metrics(
        self._buffered_eval_metrics,
        loss=loss,
        step=self._train_steps,
    )
    self._post_process_eval_step(aux)

  @contextlib.contextmanager
  def eval_context(self):
    """Context manager for bracketing eval sessions.

    Switches metrics mode to EVAL and writes buffered eval metrics on exit.
    Usage:

        with trainer.eval_context():
            for micro_batch in eval_ds:
                trainer.eval_step(micro_batch)
    """
    logging.info("Running evaluation on train step %d.", self._train_steps)
    with self._switch_mode(sft_metrics_logger.Mode.EVAL):
      try:
        yield
      finally:
        if self._buffered_eval_metrics is not None:
          self._write_metrics(self._buffered_eval_metrics)
          self._buffered_eval_metrics = None

  @override
  def save_checkpoint(self, metadata: Any = None, **kwargs) -> None:
    """Saves a checkpoint of the trainer state (model + optimizer).

    Vanilla implementation of save_checkpoint on a train_steps. Sub-batch
    checkpointing will be added later once it is supported in the checkpoint
    manager.
    """
    if metadata is None:
      metadata = self.custom_checkpoint_metadata()
    step = kwargs.pop("step", None)
    if step is None:
      if isinstance(metadata, (dict, Mapping)):
        step = metadata.get("step", self._train_steps)
      elif hasattr(metadata, "step"):
        step = getattr(metadata, "step")
    if step is None:
      step = self._train_steps
    save_only_lora_params = kwargs.pop(
        "save_only_lora_params", self._lora_enabled
    )
    self.checkpoint_manager.save(
        step,
        self.model,
        self.optimizer,
        save_only_lora_params=save_only_lora_params,
        custom_metadata=metadata,
        **kwargs,
    )

  @override
  def restore_checkpoint(self, **kwargs) -> Any:
    return {}

  @override
  def prepare_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Stages this round's weights on the raiden transport, returns metadata."""
    del sync_request, kwargs
    worker = self._weight_sync_worker
    if worker is None:
      worker = _default_weight_sync_worker()
      self._weight_sync_worker = worker

    backend = (
        "vllm_jax" if "vllm" in self._sampler_type else self._sampler_type
    )
    mapping_config = getattr(self.config, "mapping_config", None)
    if (
        mapping_config is None
        and hasattr(self.model, "to_hf_mappings")
        and backend != "vanilla"
    ):
      try:
        from tunix.generate import mappings as mappings_lib  # pylint: disable=g-import-not-at-top
        mapping_config = mappings_lib.MappingConfig.build(
            model=self.model, backend=backend
        )
      except Exception:  # pylint: disable=broad-exception-caught
        mapping_config = None

    if (
        self._target_state is not None
        and mapping_config is not None
        and mapping_config.to_hf_mappings
    ):
      from tunix.generate import utils as gen_utils  # pylint: disable=g-import-not-at-top
      converted_state = gen_utils.transfer_state_with_mappings(
          src_state=nnx.state(self.model),
          dst_state=self._target_state,
          key_mappings=mapping_config.to_hf_mappings,
          key_mapping_hook_fns=mapping_config.to_hf_hook_fns,
          transpose_keys=mapping_config.to_hf_transpose_keys,
          reshard_fn=None,
          rollout_engine=backend,
      )
      worker.bind(converted_state)
    else:
      # TODO(lancewang): Handle LoRA parameter synchronization.
      worker.bind(nnx.state(self.model))

    worker.d2h()
    if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
      logging.info("source checksums: %s", worker.checksums())
    return [worker.work_unit_metadata()]

  def release_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Ends this round's staging hold."""
    del sync_request, kwargs
    worker = self._weight_sync_worker
    if worker is not None and worker.bound:
      logging.vlog(1, "raiden metrics: %s", worker.metrics())
    return True

  @override
  def get_metrics(self) -> exp_metrics.MetricsBuffer:
    if self._written_metrics is None:
      return exp_metrics.MetricsBuffer(id=-1)
    ret = self._written_metrics
    self._written_metrics = None
    return ret

  def train(
      self,
      train_ds: Iterable[Any],
      eval_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
      *,
      cache_nnx_graph: bool = True,
  ) -> None:
    """Training loop."""
    logging.log_first_n(
        logging.INFO,
        f"Training with mesh: {pxla.thread_resources.env.physical_mesh}",
        1,
    )
    fwd_bwd_step, _, _ = self.jit_fwd_bwd_update_and_eval_step(
        skip_jit, cache_nnx_graph
    )
    if not skip_jit:
      # Report the step function this loop will actually drive: in the fused
      # regime `fwd_bwd_step`'s executable is never compiled, so its cache size
      # would stay at zero and the log would say nothing.
      traced_step = self._jitted_train_step_fn or fwd_bwd_step
      cache_size = traced_step.func.jitted_fn._cache_size()  # pytype: disable=attribute-error
      logging.log_if(
          logging.INFO,
          f"Compiled fwd_bwd_step cache size: {cache_size}",
          condition=cache_size not in self._jit_cache,
      )
      self._jit_cache.add(cache_size)

    if eval_ds:
      self._run_eval(eval_ds)

    if self.config.max_steps is not None and self._pbar is None:
      self._pbar = progress_bar.ProgressBar(
          metrics_prefix=self.metrics_prefix,
          metrics_logger=self.metrics_logger,  # pyrefly: ignore[bad-argument-type]
          initial_steps=self._train_steps,
          max_steps=self.config.max_steps,
          description=self.config.pbar_description,
      )

    if self.training_hooks:
      self.training_hooks.on_train_start(self)

    train_iterator = iter(train_ds)
    index = 0
    last_step_completion_time = time.perf_counter()
    while True:
      self._prof.maybe_activate(self._iter_steps)
      with jax.profiler.StepTraceAnnotation("train", step_num=self._iter_steps):
        train_example = None
        if self.data_hooks:
          train_example = self.data_hooks.load_next_train_batch(self)
        else:
          try:
            train_example = next(train_iterator)
            if not self.is_managed_externally:
              # TODO(mridulsahu): Add support to restore the iterator state
              # instead of skipping the already trained examples.
              if index < self._iter_steps:
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
            not self.is_managed_externally
            and self.config.max_steps is not None
            and self._train_steps >= self.config.max_steps
        ):
          break

        self._throttler.wait_for_next()
        if self.training_hooks:
          self.training_hooks.on_train_step_start(self)

        # Collect tags for the span
        metadata = self.custom_checkpoint_metadata()
        global_step = metadata.get("global_step")

        if global_step is not None:
          # Offset by 1 since global_step is incremented for checkpointing.
          global_step -= 1
          if global_step > 0:
            if self._mini_batch_size is None:
              self._mini_batch_size = max(1, self._train_steps // global_step)
            mini_batch = self._train_steps % self._mini_batch_size
          else:
            mini_batch = self._train_steps
        else:
          mini_batch = None
          global_step = None
        micro_batch = self._iter_steps % self.config.get_with_default(
            "gradient_accumulation_steps", 1
        )
        tags = {
            perf_constants.STEP: global_step,
            perf_constants.ROLE: metadata.get("role"),
            perf_constants.MICRO_BATCH: micro_batch,
            perf_constants.MINI_BATCH: mini_batch,
        }

        self._iter_steps += 1

        is_update_step_val = None
        if (
            isinstance(train_example, dict)
            and "is_update_step" in train_example
        ):
          val = train_example["is_update_step"]
          if val is not None:
            is_update_step_val = bool(np.asarray(val).item())
        elif hasattr(train_example, "is_update_step"):
          val = train_example.is_update_step
          if val is not None:
            is_update_step_val = bool(np.asarray(val).item())

        if is_update_step_val is None:
          is_update_step_val = (
              self._iter_steps
              % self.config.get_with_default("gradient_accumulation_steps", 1)
              == 0
          )

        with self._perf_tracer.span(
            "peft_train_step",
            pxla.thread_resources.env.physical_mesh.devices,
        ) as span, self._perf_tracer_v2.span(
            perf_constants.PEFT_TRAIN,
            pxla.thread_resources.env.physical_mesh.devices,
            tags=tags,
        ) as span_v2:
          if self._jitted_train_step_fn is not None and is_update_step_val:
            self.train_step(train_example)
            computation_to_track = self._last_update_grad_norm
          else:
            self.fwd_bwd(train_example)
            assert self._buffered_train_metrics is not None
            train_loss = self._buffered_train_metrics.losses[-1]
            computation_to_track = train_loss
            if is_update_step_val:
              self.update()
              computation_to_track = getattr(
                  self, "_last_update_grad_norm", train_loss
              )

          span.device_end([computation_to_track])
          span_v2.async_end([computation_to_track])

        self._throttler.add_computation(computation_to_track)  # pyrefly: ignore[bad-argument-type]
        if is_update_step_val:
          self.save_checkpoint()
          if (
              eval_ds
              and self._train_steps % self.config.eval_every_n_steps == 0
          ):
            self._run_eval(eval_ds)

      self._prof.maybe_deactivate(self._iter_steps)

    self._throttler.wait_for_all()
    logging.info(
        "Train loop finished in: %.4f seconds",
        time.perf_counter() - last_step_completion_time,
    )
    if self.training_hooks:
      self.training_hooks.on_train_end(self)
    if not self.is_managed_externally:
      self.close()

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

  def _run_eval(
      self,
      eval_ds: Iterable[Any],
  ) -> None:
    """Runs evaluation loop."""
    eval_iterator = iter(eval_ds)
    with self.eval_context():
      eval_loss, eval_steps = 0, 0
      while True:
        if self.data_hooks:
          eval_example = self.data_hooks.load_next_eval_batch(self)
        else:
          try:
            eval_example = next(eval_iterator)
          except StopIteration:
            eval_example = None
        if eval_example is None:
          break
        if self.training_hooks:
          self.training_hooks.on_eval_step_start(self)
        self.eval_step(eval_example)
        assert self._buffered_eval_metrics is not None
        eval_loss += self._buffered_eval_metrics.losses[-1]
        eval_steps += 1

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        # Clear so eval_context doesn't attempt to write empty metrics.
        self._buffered_eval_metrics = None
        return

      logging.info(
          "Train step %d eval loss: %f",
          self._train_steps,
          eval_loss / eval_steps,
      )
      if self.training_hooks:
        self.training_hooks.on_eval_step_end(self, eval_loss)


def _default_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
) -> utils.LossOutput | ArrayLike:
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
  denominator = jnp.sum(target_mask)

  # Return the negative log likelihood (NLL) loss.
  # Equivalent to: optax.softmax_cross_entropy(logits, one_hot).mean()
  unreduced_loss = -jnp.sum(jax.nn.log_softmax(logits) * one_hot)
  return utils.LossOutput(
      primary_loss=utils.WeightedMetric(unreduced_loss, denominator, eps=1e-8),
      aux_metrics={},
  )
