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
from tunix.sft import progress_bar
from tunix.sft import sharding_utils
from tunix.sft import utils

_ModelInputT = Dict[str, ArrayLike]
P = ParamSpec("P")
MetricsLogger = sft_metrics_logger.MetricsLogger
MetricsLoggerOptions = sft_metrics_logger.MetricsLoggerOptions


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig:
  """Configuration for the trainer."""

  eval_every_n_steps: int
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None
  # Gradient-accumulation strategy. "stream" (default) = the CL's streaming
  # GradientAccumulator; "optax" = optax.MultiSteps (main's path), for ablation.
  grad_accum: str = "stream"

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
  """

  def __init__(self, model: nnx.Module, wrt: type[nnx.Variable]):
    state = nnx.state(model, wrt)
    self.grads = nnx.data(jax.tree_util.tree_map(jnp.zeros_like, state))
    self.denom = nnx.Variable(jnp.zeros((), dtype=jnp.float32))

  def add(self, grads: Any, denom: jax.Array | None = None):
    def _add(acc_var, g_var):
      g = g_var[...] if isinstance(g_var, nnx.Variable) else g_var
      # Use set_value (no index) rather than `acc_var[...] = ...`. The indexed
      # __setitem__ fast path compares the new/old value `.sharding`; during
      # trace those are tracers, and reading `Tracer.sharding` triggers a
      # partial-eval provenance scan (find_progenitors) that is ~super-linear
      # per leaf and dominates compile-time tracing. set_value with no index
      # takes the plain-assignment branch and skips the sharding comparison.
      acc_var.set_value(acc_var[...] + g)

    jax.tree_util.tree_map(
        _add,
        self.grads,
        grads,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    if denom is None:
      denom_val = jnp.asarray(1.0, dtype=jnp.float32)
    else:
      denom_val = denom.astype(jnp.float32)
    self.denom.set_value(self.denom[...] + denom_val)

  def get(self):
    scale = 1.0 / jnp.maximum(self.denom[...], jnp.asarray(1.0, jnp.float32))

    return jax.tree_util.tree_map(
        lambda v: type(v)(v[...] * scale.astype(v[...].dtype)),
        self.grads,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def reset(self):
    def _zero_in_place(v):
      # set_value (no index) instead of `v[...] = ...`; see `add` for why the
      # indexed fast path is expensive during trace.
      v.set_value(jnp.zeros_like(v[...]))

    jax.tree_util.tree_map(
        _zero_in_place,
        self.grads,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    self.denom.set_value(jnp.zeros_like(self.denom[...]))


def _promote_opt_state_floats_to_float32(optimizer: nnx.Optimizer) -> None:
  """Cast the optimizer state's floating-point leaves to float32 in-place.

  Args:
    optimizer: The nnx.Optimizer instance whose state will be modified.
  """

  def _cast(v):
    if isinstance(v, nnx.Variable):
      val = v.value
      if (
          hasattr(val, "dtype")
          and jnp.issubdtype(val.dtype, jnp.floating)
          and val.dtype != jnp.float32
      ):
        v.value = val.astype(jnp.float32)

  opt_state = nnx.state(optimizer, nnx.optimizer.OptState)
  jax.tree_util.tree_map(
      _cast, opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )


class PeftTrainer:
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
    if self.config.grad_accum == "optax":
      # optax path (aligned with main): wrap in MultiSteps, no promote, no
      # streaming accumulator.
      if self.config.gradient_accumulation_steps is not None:
        optimizer = optax.MultiSteps(  # pyrefly: ignore[bad-assignment]
            optimizer, self.config.gradient_accumulation_steps
        )
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=wrt_target)
    else:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=wrt_target)
      # Promote floating-point leaves to float32 in-place to match the dtype of
      # the optimizer update function branch (which is float32 due to
      # `optax.inject_hyperparams`).
      _promote_opt_state_floats_to_float32(self.optimizer)
      self.grad_accumulator = GradientAccumulator(self.model, wrt_target)

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
    self._mini_batch_size = None

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

  def with_loss_fn(
      self,
      loss_fn: Callable[
          Concatenate[nnx.Module, P],
          ArrayLike | Tuple[ArrayLike, Any] | utils.LossOutput,
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

  def _train_step(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      grad_accumulator: GradientAccumulator,
      inputs: Any,
      is_update_step: jax.Array,
  ) -> Tuple[ArrayLike, Any | None, ArrayLike]:
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      grad_accumulator: The gradient accumulator to use.
      inputs: The training input.
      is_update_step: Whether to update the model.

    Returns:
      A tuple containing the loss, auxiliary data (or None if has_aux is False),
      and the gradient norm.
    """
    inputs = self.gen_model_input_fn(inputs)

    @functools.wraps(self.loss_fn)
    def diff_fn(model, *args, **kwargs):
      out = self.loss_fn(model, *args, **kwargs)
      if isinstance(out, utils.LossOutput):
        return out.primary_loss.unreduced_sum, out
      elif self._has_aux:
        return out[0], out[1]
      else:
        return out, None

    grad_fn = nnx.value_and_grad(
        diff_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=True,
    )
    (loss_val, aux), grads = grad_fn(model, **inputs)

    if isinstance(aux, utils.LossOutput):
      # Scale the unreduced gradients using the metric's scale computation
      scale = aux.primary_loss.compute_scale()
      grads = jax.tree.map(lambda g: g * scale, grads)

      # Compute exactly equivalent legacy loss val
      loss_val = aux.primary_loss.compute()

    def apply_updates(model, optimizer, grad_accumulator):
      acc_grads = grad_accumulator.get()
      # Compute the norm in float32 to 1) match `skip_updates()` return type and
      # meet the requirement of `nnx.cond` that both branches return the same
      # dtype, 2) for production-size models the sum-of-squares over bf16 grads
      # quickly exhausts bf16 and float32 is needed for numerical stability.
      norm = optax.global_norm(
          jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), acc_grads)
      )
      optimizer.update(model, acc_grads)
      grad_accumulator.reset()
      return norm

    def skip_updates(model, optimizer, grad_accumulator):
      return jnp.array(0.0, dtype=jnp.float32)

    # At accumulation depth 1 every step is an update step, and the accumulator
    # is a mathematical no-op (add one microbatch with denom=1, divide by 1,
    # reset), so skip both the accumulator and the nnx.cond wrapper and update
    # directly from `grads` — exactly the optax path. This removes the per-leaf
    # set_value provenance scan (add/reset) and the XLA Conditional at depth 1.
    # Unlike the earlier bypass, the accumulator is not touched at all, so there
    # is no reset-to-zeros that lets SPMD re-shard state and force a 2nd compile.
    if self.config.get_with_default("gradient_accumulation_steps", 1) == 1:
      grad_norm = optax.global_norm(
          jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), grads)
      )
      optimizer.update(model, grads)
    else:
      # TODO(b/491970038): update denom for sequence packing.
      grad_accumulator.add(grads, denom=jnp.asarray(1.0, dtype=jnp.float32))
      # If the mesh is not empty, then we need to replicate the is_update_step
      # across all devices to avoid deadlock so that all devices see the same
      # update step.
      mesh = pxla.thread_resources.env.physical_mesh
      if not mesh.empty:
        is_update_step = jax.lax.with_sharding_constraint(
            is_update_step, jax.sharding.PartitionSpec()
        )

      grad_norm = nnx.cond(
          is_update_step,
          apply_updates,
          skip_updates,
          model,
          optimizer,
          grad_accumulator,
      )

    if isinstance(aux, utils.LossOutput):
      # Return the raw aux (WeightedMetric preserved); metric ops reduce them.
      return loss_val, aux.aux_metrics, grad_norm
    elif self._has_aux:
      return loss_val, aux, grad_norm
    else:
      return loss_val, None, grad_norm

  def _train_step_optax(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      inputs: Any,
  ) -> Tuple[ArrayLike, Any | None, ArrayLike]:
    """Train step for the optax.MultiSteps path (grad_accum="optax").

    Shares the loss/gradient front-half with `_train_step` (same unreduced
    LossOutput handling); differs only in accumulation: a direct
    `optimizer.update` (MultiSteps accumulates internally), aligned with main.

    Args:
      model: The model to train.
      optimizer: The optimizer to use (optax.MultiSteps-wrapped).
      inputs: The training input.

    Returns:
      A tuple containing the loss, auxiliary data (or None if has_aux is False),
      and the gradient norm.
    """
    inputs = self.gen_model_input_fn(inputs)

    @functools.wraps(self.loss_fn)
    def diff_fn(model, *args, **kwargs):
      out = self.loss_fn(model, *args, **kwargs)
      if isinstance(out, utils.LossOutput):
        return out.primary_loss.unreduced_sum, out
      elif self._has_aux:
        return out[0], out[1]
      else:
        return out, None

    grad_fn = nnx.value_and_grad(
        diff_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=True,
    )
    (loss_val, aux), grads = grad_fn(model, **inputs)

    if isinstance(aux, utils.LossOutput):
      # Scale the unreduced gradients using the metric's scale computation
      scale = aux.primary_loss.compute_scale()
      grads = jax.tree.map(lambda g: g * scale, grads)

      # Compute exactly equivalent legacy loss val
      loss_val = aux.primary_loss.compute()

    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)

    if isinstance(aux, utils.LossOutput):
      # Return the raw aux (WeightedMetric preserved); metric ops reduce them.
      return loss_val, aux.aux_metrics, grad_norm
    elif self._has_aux:
      return loss_val, aux, grad_norm
    else:
      return loss_val, None, grad_norm

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

  def create_train_step_fn(
      self,
  ) -> Callable[..., Tuple[ArrayLike, Any | None, ArrayLike]]:
    """Creates the train step function."""
    if self.config.grad_accum == "optax":
      return self._train_step_optax
    return self._train_step

  def create_eval_step_fn(
      self,
  ) -> Callable[..., ArrayLike | Tuple[ArrayLike, Any]]:
    """Creates the eval step function."""
    return self._eval_step  # pyrefly: ignore[bad-return]

  def _shard_optimizer(self, mesh: shd.Mesh) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the _train_step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    if mesh.empty:
      return

    # Place concrete state onto its target sharding with jax.device_put instead
    # of eager jax.lax.with_sharding_constraint. The latter is a compiler hint;
    # evaluated eagerly on the full NNX PyTree it forces a full graph trace
    # (minutes on large models). device_put places each addressable leaf
    # directly. (Ported from Tianshu Bao's fix, CL 952815066.)
    def _shard(x, p):
      if not isinstance(x, (jax.Array, np.ndarray)):
        return x
      if p is None:
        p = shd.PartitionSpec()
      sharding = sharding_utils.get_sharding(x, mesh, p)
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

    # optax path uses MultiSteps (no streaming accumulator to shard).
    if self.config.grad_accum == "optax":
      return

    # Partition gradients the same as their target state.
    grad_pspecs = nnx.get_partition_spec(self.grad_accumulator.grads)
    self.grad_accumulator.grads = jax.tree.map(
        _shard, self.grad_accumulator.grads, grad_pspecs
    )

    # Denominator is a scalar — replicate across all devices.
    self.grad_accumulator.denom[...] = jax.device_put(
        self.grad_accumulator.denom[...],
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )

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
      return train_step, eval_step

    if self._jitted_train_step_fn is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      if self.config.grad_accum == "optax":
        self._jitted_train_step_fn = nnx.jit(
            train_step, donate_argnames=("optimizer",)
        )
      else:
        self._jitted_train_step_fn = nnx.jit(
            train_step, donate_argnames=("optimizer", "grad_accumulator")
        )
      self._jitted_eval_step_fn = nnx.jit(eval_step)

      def maybe_cache_and_partial(f, *args):
        if cache_nnx_graph:
          # wrap with partial so we can access jitted_fn in a consistent way.
          return functools.partial(nnx.cached_partial(f, *args))
        else:
          return functools.partial(f, *args)

      if self.config.grad_accum == "optax":
        self._jitted_train_step_fn = maybe_cache_and_partial(
            self._jitted_train_step_fn,
            self.model,
            self.optimizer,
        )
      else:
        self._jitted_train_step_fn = maybe_cache_and_partial(
            self._jitted_train_step_fn,
            self.model,
            self.optimizer,
            self.grad_accumulator,
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
    train_step, eval_step = self.jit_train_and_eval_step(
        skip_jit, cache_nnx_graph
    )
    if not skip_jit:
      cache_size = train_step.func.jitted_fn._cache_size()  # pytype: disable=attribute-error
      logging.log_if(
          logging.INFO,
          f"Compiled train_step cache size: {cache_size}",
          condition=cache_size not in self._jit_cache,
      )
      self._jit_cache.add(cache_size)

    if eval_ds:
      self._run_eval(eval_ds, eval_step)

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

        train_example = self._prepare_inputs(train_example)
        train_example = sharding_utils.shard_input(
            train_example, self.config.data_sharding_axis
        )

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
          if self.config.grad_accum == "optax":
            # optax MultiSteps handles accumulation cadence internally.
            train_loss, aux, grad_norm = train_step(train_example)
          else:
            train_loss, aux, grad_norm = train_step(
                train_example,
                is_update_step=jnp.array(is_update_step_val, dtype=jnp.bool_),
            )
          span.device_end([train_loss])
          span_v2.async_end([train_loss])

        self._throttler.add_computation(train_loss)
        self._buffered_train_metrics = self._buffer_metrics(
            self._buffered_train_metrics,
            loss=train_loss,
            step=self._train_steps,
            additional_metrics={"grad_norm": (grad_norm, np.mean)},
        )
        # NB: put this after self._buffer_metrics is important
        self._post_process_train_step(aux)

        if is_update_step_val:
          self._train_steps += 1
          self._write_train_metrics()

          # Checkpoint frequency is configured by checkpointing_options.
          self.checkpoint_manager.save(
              self._train_steps,
              self.model,
              self.optimizer,
              save_only_lora_params=self._lora_enabled,
              custom_metadata=self.custom_checkpoint_metadata(),
          )

          if (
              eval_ds
              and self._train_steps % self.config.eval_every_n_steps == 0
          ):
            self._run_eval(eval_ds, eval_step)

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
      eval_step_fn: Callable[..., Any],
  ) -> None:
    """Runs evaluation loop."""
    logging.info("Running evaluation on train step %d.", self._train_steps)
    eval_iterator = iter(eval_ds)
    with self._switch_mode(sft_metrics_logger.Mode.EVAL):
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
        eval_example = self._prepare_inputs(eval_example)
        eval_example = sharding_utils.shard_input(
            eval_example, self.config.data_sharding_axis
        )
        if self.training_hooks:
          self.training_hooks.on_eval_step_start(self)
        loss, aux = eval_step_fn(eval_example)
        loss = jax.lax.stop_gradient(loss)
        self._buffered_eval_metrics = self._buffer_metrics(
            self._buffered_eval_metrics,
            loss=loss,
            step=self._train_steps,
        )
        self._post_process_eval_step(aux)
        eval_loss += loss
        eval_steps += 1

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        return

      self._write_metrics(self._buffered_eval_metrics)  # pyrefly: ignore[bad-argument-type]
      logging.info(
          "Train step %d eval loss: %f - eval perplexity: %f",
          self._train_steps,
          self.metrics_logger.get_metric(self.metrics_prefix, "loss", "eval"),  # pyrefly: ignore[missing-attribute]
          self.metrics_logger.get_metric(  # pyrefly: ignore[missing-attribute]
              self.metrics_prefix, "perplexity", "eval"
          ),
      )
      self._buffered_eval_metrics = None
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
