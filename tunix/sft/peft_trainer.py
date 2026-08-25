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

from collections.abc import Iterable, Sequence
import contextlib
import dataclasses
import functools
import gc
import math
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
from tunix.perf import metrics as perf_metrics
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft import checkpoint_manager
from tunix.sft import checkpoint_options
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
_P45_PRECOMPUTED_CHECKPOINT_CONTRACT = "p45-frozenlake-checkpoint-v1"
_P45_CHECKPOINT_ROOT = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake"
)


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
  checkpointing_options: checkpoint_options.CheckpointingOptions | None = None
  # Optional exact checkpoint step to restore. ``None`` retains the historical
  # latest-checkpoint behavior. This is used by isolated evaluators which must
  # compare registered milestones after later checkpoints already exist.
  checkpoint_restore_step: int | None = None
  # Explicit admission token for checkpointing around a precomputed-gradient
  # transaction.  The default remains fail-closed for isolated G6 canaries.
  precomputed_gradient_checkpointing_contract: str | None = None

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

  # Adam moment dtype. None (default) follows the param dtype (optax inits
  # moments as zeros_like(params)); set e.g. jnp.float32 to force fp32.
  optimizer_state_dtype: DTypeLike | None = None

  # Gradient-accumulator dtype (separate from the Adam moments). None (default)
  # uses float32 like optax MultiSteps; set jnp.bfloat16 to trade precision for
  # HBM on the accumulation path.
  gradient_accumulator_dtype: DTypeLike | None = None

  # Accumulate gradients of WeightedMetric.unreduced_sum and divide once by
  # the sum of the corresponding denominators. This is required when valid-row
  # counts can differ across otherwise equal-size micro-batches. Disabled by
  # default so existing unpacked mean-of-means recipes retain their program.
  loss_denominator_weighted_accumulation: bool = False

  # Keep optimizer state in pinned host memory between updates. The state is
  # moved to device only for optimizer.update and returned to pinned host after
  # the update completes. This does not change optimizer-state dtype or update
  # arithmetic. Disabled by default.
  optimizer_offload: bool = False

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

  def __init__(
      self,
      model: nnx.Module,
      wrt: type[nnx.Variable],
      *,
      allocate_grads: bool = True,
      accumulator_dtype: DTypeLike = jnp.float32,
  ):
    state = nnx.state(model, wrt)
    if allocate_grads:
      # Accumulate in float32 by default (like optax MultiSteps): summing bf16
      # grads over microbatches loses small contributions (swamping).
      self.grads = nnx.data(
          jax.tree_util.tree_map(
              lambda x: jnp.zeros_like(x, dtype=accumulator_dtype), state
          )
      )
    else:
      # Fast path never reads the accumulator: skip the model-sized grad-tree
      # allocation. Empty grads keep it a valid tiny jit arg (signature and
      # compilation unchanged).
      self.grads = nnx.data({})
    self.denom = nnx.Variable(jnp.zeros((), dtype=jnp.float32))

  def add(self, grads: Any, denom: jax.Array | None = None):
    def _add(acc_var, g_var):
      g = g_var[...] if isinstance(g_var, nnx.Variable) else g_var
      # set_value (no index) avoids the indexed __setitem__ "slow" path, whose
      # `.sharding` check on tracers triggers a per-leaf provenance scan that
      # dominates trace time; the stored value is identical.
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
      # set_value (no index); see `add` for why.
      v.set_value(jnp.zeros_like(v[...]))

    jax.tree_util.tree_map(
        _zero_in_place,
        self.grads,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    self.denom.set_value(jnp.zeros_like(self.denom[...]))


def _cast_opt_state_floats(
    optimizer: nnx.Optimizer, dtype: jnp.dtype
) -> None:
  """Cast the optimizer state's floating-point leaves to `dtype` in-place."""

  def _cast(v):
    if isinstance(v, nnx.Variable):
      val = v.value
      if (
          hasattr(val, "dtype")
          and jnp.issubdtype(val.dtype, jnp.floating)
          and val.dtype != dtype
      ):
        v.value = val.astype(dtype)

  opt_state = nnx.state(optimizer, nnx.optimizer.OptState)
  jax.tree_util.tree_map(
      _cast, opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )


def _put_state_on_memory_kind(state: Any, memory_kind: str) -> Any:
  """Moves JAX array leaves while preserving their logical shardings."""
  if memory_kind not in ("device", "pinned_host"):
    raise ValueError(
        "optimizer state memory_kind must be device or pinned_host; got "
        f"{memory_kind!r}"
    )

  def _move(value):
    if not isinstance(value, jax.Array):
      return value
    sharding = value.sharding
    if sharding.memory_kind == memory_kind:
      return value
    return jax.device_put(value, sharding.with_memory_kind(memory_kind))

  with jax.transfer_guard("allow"):
    moved = jax.tree.map(_move, state)
  return jax.block_until_ready(moved)


def _state_memory_kinds(state: Any) -> tuple[str, ...]:
  """Returns the distinct memory kinds of all JAX array leaves."""
  kinds = {
      value.sharding.memory_kind
      for value in jax.tree.leaves(state)
      if isinstance(value, jax.Array)
  }
  return tuple(sorted(kinds))


def _state_logical_bytes(state: Any) -> int:
  """Returns logical bytes across JAX array leaves in a state tree."""
  return sum(
      int(value.size * value.dtype.itemsize)
      for value in jax.tree.leaves(state)
      if isinstance(value, jax.Array)
  )


def _precomputed_expected_microbatches(environ) -> int:
  """Returns the fail-closed segmented optimizer transaction length."""
  p41_optimizer_bench = environ.get("CANON_P41_OPTIMIZER_BENCH", "") == "1"
  if p41_optimizer_bench:
    if (
        environ.get("CANON_GSM8K_L3", "") != "1"
        or environ.get("CANON_GSM8K_UPDATE_CANARY", "") != "1"
    ):
      raise ValueError(
          "P41 optimizer transaction requires the bounded GSM8K L3 canary"
      )
    return 1
  if (
      environ.get("CANON_GSM8K_TRAIN", "") == "1"
      and environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") != "1"
  ):
    return 16
  if (
      environ.get("CANON_P31_CONVERGENCE", "") == "1"
  ):
    return 16
  if environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") == "1":
    raw = environ.get("CANON_LOCAL_TRAJECTORIES", "")
    try:
      expected = int(raw)
    except (TypeError, ValueError) as exc:
      raise ValueError(
          "canonical workload requires integer CANON_LOCAL_TRAJECTORIES"
      ) from exc
    if expected <= 0:
      raise ValueError(
          "canonical workload requires positive CANON_LOCAL_TRAJECTORIES"
      )
    return expected
  return 4


def _precomputed_gradient_norm(tree: Any) -> ArrayLike:
  """Returns the stock norm or the exact-profile P63 hybrid norm."""
  float_tree = jax.tree.map(
      lambda value: value.astype(jnp.float32), tree
  )
  if utils.canonical_overflow_safe_clip_max_norm(os.environ) is not None:
    return utils.hybrid_global_norm(float_tree)
  return optax.global_norm(float_tree)


def _requires_precomputed_gradient_accumulator(environ) -> bool:
  """Returns whether the explicit segmented update path needs its accumulator."""
  return (
      environ.get("CANON_P28_SEGMENTED_TRAIN", "") == "1"
      and environ.get("CANON_P28_G6_UPDATE", "") == "1"
  )


def _p45_precomputed_checkpointing_admitted(config, environ) -> bool:
  """Admits checkpoints only for the signed committed P45 transaction."""
  if (
      config.precomputed_gradient_checkpointing_contract
      != _P45_PRECOMPUTED_CHECKPOINT_CONTRACT
  ):
    return False
  tag = environ.get("CANON_FROZENLAKE_CKPT_TAG", "")
  mode = environ.get("CANON_FROZENLAKE_CKPT_MODE", "")
  required = {
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
      "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_FROZENLAKE_CKPT_ROOT": _P45_CHECKPOINT_ROOT,
      "CANON_FROZENLAKE_CKPT_INTERVAL": "10",
      "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP": "1",
      "ENABLE_PATHWAYS_PERSISTENCE": "1",
  }
  if mode not in ("new", "resume") or not tag:
    return False
  if any(environ.get(key, "") != value for key, value in required.items()):
    return False
  expected_directory = f"{_P45_CHECKPOINT_ROOT}/{tag}/actor"
  return config.checkpoint_root_directory == expected_directory


def _deepswe_onehost_no_commit(environ) -> bool:
  """Returns the fail-closed local DeepSWE optimizer-skip selection."""
  smoke = environ.get("CANON_DEEPSWE_ONEHOST_SMOKE", "0")
  no_commit = environ.get("CANON_DEEPSWE_ONEHOST_NO_COMMIT", "0")
  if smoke not in ("0", "1"):
    raise ValueError("CANON_DEEPSWE_ONEHOST_SMOKE must be exactly 0 or 1")
  if no_commit not in ("0", "1"):
    raise ValueError(
        "CANON_DEEPSWE_ONEHOST_NO_COMMIT must be exactly 0 or 1"
    )
  if no_commit == "1" and smoke != "1":
    raise ValueError("one-host no-commit requires one-host smoke mode")
  if no_commit == "1" and any(
      environ.get(key, "0") == "1"
      for key in (
          "CANON_ALIGNMENT_GATE",
          "CANON_ALIGNMENT_GATE_ONLY",
          "CANON_ALIGNMENT_UPDATE_CANARY",
          "CANON_ALIGNMENT_TRAIN",
      )
  ):
    p58_xprof_arm = environ.get("CANON_P58_ONEHOST_XPROF_ARM", "")
    p58_xprof_alignment = (
        p58_xprof_arm in ("native", "zero-hp")
        and environ.get("CANON_DEEPSWE_ONEHOST_STAGE", "")
        == "backward-no-commit"
        and environ.get("CANON_ALIGNMENT_GATE", "0") == "1"
        and environ.get("CANON_ALIGNMENT_GATE_ONLY", "0") == "0"
        and environ.get("CANON_ALIGNMENT_UPDATE_CANARY", "0") == "0"
        and environ.get("CANON_ALIGNMENT_TRAIN", "0") == "1"
    )
    if not p58_xprof_alignment:
      raise ValueError(
          "one-host no-commit cannot overlap a canonical alignment mode"
      )
  return smoke == no_commit == "1"


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
    self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=wrt_target)
    # Adam moments follow the param dtype by default (optax inits them as
    # zeros_like(params)). Set optimizer_state_dtype to override, e.g.
    # jnp.float32.
    if self.config.optimizer_state_dtype is not None:
      _cast_opt_state_floats(self.optimizer, self.config.optimizer_state_dtype)
    # The ordinary depth-1 non-packing path never reads the accumulator.  The
    # explicit segmented update transaction always does, including P41's
    # single-microbatch benchmark, so it must retain the model-shaped tree.
    _uses_cond_path = _requires_precomputed_gradient_accumulator(os.environ) or not (
        self.config.get_with_default("gradient_accumulation_steps", 1) == 1
        and self.config.max_seq_token_per_tpu is None
    )
    accumulator_dtype = self.config.gradient_accumulator_dtype
    if accumulator_dtype is None:
      accumulator_dtype = jnp.float32
    self.grad_accumulator = GradientAccumulator(
        self.model,
        wrt_target,
        allocate_grads=_uses_cond_path,
        accumulator_dtype=accumulator_dtype,
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

    self._train_steps, self._restored_custom_metadata = (
        self.checkpoint_manager.maybe_restore(
            self.model,
            self.optimizer,
            step=self.config.checkpoint_restore_step,
            restore_only_lora_params=self._lora_enabled,
        )
    )
    self._iter_steps = self._train_steps * self.config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    if self.config.optimizer_offload:
      self._put_optimizer_state_on_memory_kind("pinned_host")

    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    self._jitted_precomputed_gradient_step_impl = None
    self._jitted_precomputed_gradient_scaled_step_impl = None
    self._jitted_precomputed_gradient_pair_step_impl = None
    self._jitted_precomputed_gradient_commit_impl = None
    self._jitted_precomputed_gradient_discard_impl = None
    self._jitted_precomputed_gradient_step_fn = None
    self._jitted_precomputed_gradient_scaled_step_fn = None
    self._jitted_precomputed_gradient_pair_step_fn = None
    self._jitted_precomputed_gradient_commit_fn = None
    self._jitted_precomputed_gradient_discard_fn = None
    self._registered_learning_rate_schedule = None
    self._last_precomputed_commit_evidence = None
    self._p28_precomputed_microstep = 0
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

  def register_learning_rate_schedule(
      self, schedule: Callable[[ArrayLike], ArrayLike]
  ) -> None:
    """Registers the schedule used by an externally managed update gate."""
    if not callable(schedule):
      raise TypeError("learning rate schedule must be callable")
    self._registered_learning_rate_schedule = schedule

  def effective_learning_rate(self, step: int | None = None) -> float | None:
    """Returns the learning rate applied by the next optimizer transaction."""
    if step is None:
      step = self._train_steps
    if self._registered_learning_rate_schedule is not None:
      value = self._registered_learning_rate_schedule(
          jnp.asarray(step, dtype=jnp.int32)
      )
    else:
      value = self._try_get_learning_rate()
    if value is None:
      return None
    return float(np.asarray(jax.device_get(value)))

  def consume_precomputed_commit_evidence(self) -> dict[str, Any]:
    """Returns and clears evidence from the latest precomputed commit."""
    evidence = self._last_precomputed_commit_evidence
    if evidence is None:
      raise RuntimeError("precomputed commit evidence is unavailable")
    self._last_precomputed_commit_evidence = None
    return evidence

  def clear_jit_cache(self):
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overriding the training related states, for example, the loss function.
    """
    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    self._jitted_precomputed_gradient_step_impl = None
    self._jitted_precomputed_gradient_scaled_step_impl = None
    self._jitted_precomputed_gradient_pair_step_impl = None
    self._jitted_precomputed_gradient_commit_impl = None
    self._jitted_precomputed_gradient_discard_impl = None
    self._jitted_precomputed_gradient_step_fn = None
    self._jitted_precomputed_gradient_scaled_step_fn = None
    self._jitted_precomputed_gradient_pair_step_fn = None
    self._jitted_precomputed_gradient_commit_fn = None
    self._jitted_precomputed_gradient_discard_fn = None

  def _precomputed_gradient_step(
      self,
      grad_accumulator: GradientAccumulator,
      grads: Any,
  ) -> ArrayLike:
    """Accumulates one externally computed gradient without updating."""
    grad_accumulator.add(grads, denom=jnp.asarray(1.0, jnp.float32))
    return _precomputed_gradient_norm(grads)

  def _precomputed_gradient_pair_step(
      self,
      grad_accumulator: GradientAccumulator,
      left: Any,
      right: Any,
      multiplier: ArrayLike,
  ) -> ArrayLike:
    """Adds one materialization-free `(left + right) * multiplier` pair."""
    paired = jax.tree.map(
        lambda a, b: (a + b) * multiplier.astype(a.dtype), left, right
    )
    grad_accumulator.add(paired, denom=jnp.asarray(1.0, jnp.float32))
    return _precomputed_gradient_norm(paired)

  def _precomputed_gradient_scaled_step(
      self,
      grad_accumulator: GradientAccumulator,
      grads: Any,
      multiplier: ArrayLike,
  ) -> ArrayLike:
    """Adds one materialization-free scaled gradient contribution."""
    scaled = jax.tree.map(
        lambda value: value * multiplier.astype(value.dtype), grads
    )
    grad_accumulator.add(scaled, denom=jnp.asarray(1.0, jnp.float32))
    return _precomputed_gradient_norm(scaled)

  def _precomputed_gradient_commit(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      grad_accumulator: GradientAccumulator,
  ) -> tuple[ArrayLike, dict[str, Any]]:
    """Commits the already accumulated G6 gradient exactly once."""
    acc_grads = grad_accumulator.get()
    norm = _precomputed_gradient_norm(acc_grads)
    acc_leaves, acc_treedef = jax.tree_util.tree_flatten(acc_grads)
    param_dtypes = [
        value.dtype
        for value in jax.tree.leaves(nnx.state(model, nnx.Param))
    ]
    acc_grads = jax.tree_util.tree_unflatten(
        acc_treedef,
        [value.astype(dtype) for value, dtype in zip(
            acc_leaves, param_dtypes, strict=True
        )],
    )
    p63_max_norm = utils.canonical_overflow_safe_clip_max_norm(os.environ)
    p63_stats = (
        utils.hybrid_global_norm_stats(
            acc_grads, max_norm=p63_max_norm
        )
        if p63_max_norm is not None
        else None
    )
    # JAX arrays are immutable.  Holding the input references is sufficient for
    # the post-rounding comparison and avoids asking XLA to materialize an
    # explicit full-model copy before the optimizer transaction.
    params_before = tuple(
        jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    optimizer.update(model, acc_grads)
    params_after = tuple(
        value for value in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    gradient_leaves = tuple(jax.tree.leaves(acc_grads))

    def _nonzero_count(value):
      return jnp.count_nonzero(value).astype(jnp.int32)

    def _max_abs(value):
      return jnp.max(jnp.abs(value.astype(jnp.float32)))

    parameter_deltas = tuple(
        after.astype(jnp.float32) - before.astype(jnp.float32)
        for before, after in zip(params_before, params_after, strict=True)
    )
    commit_evidence = {
        "gradient_nonzero_counts": tuple(
            _nonzero_count(value) for value in gradient_leaves
        ),
        "gradient_max_abs": tuple(
            _max_abs(value) for value in gradient_leaves
        ),
        "gradient_finite": tuple(
            jnp.all(jnp.isfinite(value)) for value in gradient_leaves
        ),
        "parameter_changed_counts": tuple(
            _nonzero_count(delta) for delta in parameter_deltas
        ),
        "parameter_delta_max_abs": tuple(
            _max_abs(delta) for delta in parameter_deltas
        ),
        "parameter_delta_finite": tuple(
            jnp.all(jnp.isfinite(delta)) for delta in parameter_deltas
        ),
    }
    if p63_stats is not None:
      commit_evidence["overflow_safe_clip"] = p63_stats
    grad_accumulator.reset()
    return norm, commit_evidence

  def _precomputed_gradient_discard(
      self, grad_accumulator: GradientAccumulator
  ) -> ArrayLike:
    """Clears one complete streamed transaction without optimizer mutation."""
    denominator = grad_accumulator.denom[...]
    grad_accumulator.reset()
    return denominator

  def _put_optimizer_state_on_memory_kind(self, memory_kind: str) -> None:
    """Moves only OptState and verifies that the requested placement landed."""
    opt_state = nnx.state(self.optimizer, nnx.optimizer.OptState)
    moved_state = _put_state_on_memory_kind(opt_state, memory_kind)
    nnx.update(self.optimizer, moved_state)
    actual = _state_memory_kinds(
        nnx.state(self.optimizer, nnx.optimizer.OptState)
    )
    if actual != (memory_kind,):
      raise RuntimeError(
          "optimizer state memory-kind transfer did not land: "
          f"requested={memory_kind!r} actual={actual!r}"
      )

  def _reshard_grad_accumulator(self, mesh: shd.Mesh) -> dict[str, int]:
    """Restores zeroed accumulator values to their registered shardings."""
    if mesh.empty:
      return {"arrays": 0, "logical_bytes": 0}

    def _shard(value, pspec):
      if not isinstance(value, (jax.Array, np.ndarray)):
        return value
      if pspec is None:
        pspec = shd.PartitionSpec()
      target = sharding_utils.get_sharding(value, mesh, pspec)
      if hasattr(value, "sharding") and value.sharding == target:
        return value
      with jax.transfer_guard("allow"):
        return jax.device_put(value, target)

    grad_pspecs = nnx.get_partition_spec(self.grad_accumulator.grads)
    resharded = jax.tree.map(
        _shard, self.grad_accumulator.grads, grad_pspecs
    )
    resharded = jax.block_until_ready(resharded)
    self.grad_accumulator.grads = resharded
    arrays = [
        value for value in jax.tree.leaves(resharded)
        if isinstance(value, jax.Array)
    ]
    expected = jax.tree.map(
        lambda value, pspec: (
            sharding_utils.get_sharding(
                value,
                mesh,
                pspec if pspec is not None else shd.PartitionSpec(),
            )
            if isinstance(value, jax.Array) else None
        ),
        resharded,
        grad_pspecs,
    )
    mismatches = [
        (value.sharding, target)
        for value, target in zip(
            jax.tree.leaves(resharded),
            jax.tree.leaves(expected),
            strict=True,
        )
        if isinstance(value, jax.Array) and value.sharding != target
    ]
    if mismatches:
      raise RuntimeError(
          "P30 accumulator reshard did not land: "
          f"mismatches={len(mismatches)} first={mismatches[0]!r}"
      )
    return {
        "arrays": len(arrays),
        "logical_bytes": sum(
            int(value.size * value.dtype.itemsize) for value in arrays
        ),
    }

  def optimizer_state_memory_kinds(self) -> tuple[str, ...]:
    """Reports optimizer-state placement for fail-closed capacity gates."""
    return _state_memory_kinds(
        nnx.state(self.optimizer, nnx.optimizer.OptState)
    )

  def apply_precomputed_gradient_microbatches(
      self, gradient_microbatches: Sequence[Any]
  ) -> tuple[ArrayLike, ...]:
    """Applies the default-off segmented gradient transaction.

    This method is deliberately unavailable outside the fully attested G6
    update canary.  It never invokes ``loss_fn`` or ``value_and_grad``.
    """
    expected_microbatches = _precomputed_expected_microbatches(os.environ)
    if len(gradient_microbatches) != expected_microbatches:
      raise ValueError(
          "segmented update gradient count changed: "
          f"{len(gradient_microbatches)} != {expected_microbatches}"
      )
    norms = tuple(
        self.accumulate_precomputed_gradient_microbatch(
            gradients, microbatch_index=index
        )
        for index, gradients in enumerate(gradient_microbatches)
    )
    self.commit_precomputed_gradients()
    return norms

  def _validate_precomputed_gradient_contract(self) -> None:
    """Validates the exclusive, default-off G6 update contract."""
    p31_convergence = os.environ.get("CANON_P31_CONVERGENCE", "") == "1"
    p33_workload = (
        os.environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") == "1"
    )
    required_env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G6_UPDATE": "1",
        (
            "CANON_ALIGNMENT_TRAIN"
            if p31_convergence
            or p33_workload
            or os.environ.get("CANON_GSM8K_TRAIN", "") == "1"
            else "CANON_ALIGNMENT_UPDATE_CANARY"
        ): "1",
    }
    missing = [
        key for key, value in required_env.items()
        if os.environ.get(key, "") != value
    ]
    if missing or os.environ.get("CANON_P28_G5C_ONLY", "") == "1":
      raise ValueError(
          "precomputed gradient update requires the exclusive P28 G6 "
          f"canary contract; invalid keys={missing}"
      )
    steps = self.config.get_with_default("gradient_accumulation_steps", 1)
    expected_steps = _precomputed_expected_microbatches(os.environ)
    if steps != expected_steps:
      raise ValueError(
          "segmented update accumulation changed: "
          f"{steps} != {expected_steps}"
      )
    checkpointing_enabled = self.config.checkpoint_root_directory is not None
    checkpoint_contract = (
        self.config.precomputed_gradient_checkpointing_contract
    )
    if checkpointing_enabled and not _p45_precomputed_checkpointing_admitted(
        self.config, os.environ
    ):
      raise ValueError(
          "P28 G6 canary requires checkpointing disabled unless the committed "
          "P45 checkpoint contract is admitted"
      )
    if not checkpointing_enabled and checkpoint_contract is not None:
      raise ValueError(
          "precomputed-gradient checkpoint contract requires a checkpoint "
          "directory"
      )

  def accumulate_precomputed_gradient_microbatch(
      self, gradients: Any, *, microbatch_index: int
  ) -> ArrayLike:
    """Streams one of the four P28 G6 gradient contributions."""
    self._validate_precomputed_gradient_contract()

    if self._jitted_precomputed_gradient_step_fn is None:
      if self._jitted_precomputed_gradient_step_impl is None:
        self._jitted_precomputed_gradient_step_impl = nnx.jit(
            self._precomputed_gradient_step,
            donate_argnames=("grad_accumulator",),
        )
      self._jitted_precomputed_gradient_step_fn = functools.partial(
          nnx.cached_partial(
              self._jitted_precomputed_gradient_step_impl,
              self.grad_accumulator,
          )
      )

    if microbatch_index != self._p28_precomputed_microstep:
      raise ValueError(
          "P28 G6 gradient microbatch cadence mismatch: "
          f"expected {self._p28_precomputed_microstep}, got {microbatch_index}"
      )
    accumulate_start = time.perf_counter()
    norm = self._jitted_precomputed_gradient_step_fn(
        gradients
    )
    accumulate_call_done = time.perf_counter()
    norm.block_until_ready()
    if os.environ.get("CANON_PERF_LOG", "1") != "0":
      accumulate_done = time.perf_counter()
      print(
          "[PERF] stage=grad_accumulate seconds=%.3f microbatch=%d"
          " variant=plain call=%.3f block=%.3f"
          % (
              accumulate_done - accumulate_start,
              microbatch_index,
              accumulate_call_done - accumulate_start,
              accumulate_done - accumulate_call_done,
          ),
          flush=True,
      )
    self._iter_steps += 1
    self._p28_precomputed_microstep += 1
    return norm

  def accumulate_precomputed_gradient_pair_microbatch(
      self,
      left: Any,
      right: Any,
      multiplier: ArrayLike,
      *,
      microbatch_index: int,
  ) -> ArrayLike:
    """Fuses pair sum/scale into the existing donated accumulator update."""
    self._validate_precomputed_gradient_contract()
    if os.environ.get("CANON_P30_FUSED_PAIR_ACCUMULATION", "") != "1":
      raise ValueError(
          "P30 fused pair accumulation requires its explicit env gate"
      )
    if self._jitted_precomputed_gradient_pair_step_fn is None:
      if self._jitted_precomputed_gradient_pair_step_impl is None:
        self._jitted_precomputed_gradient_pair_step_impl = nnx.jit(
            self._precomputed_gradient_pair_step,
            donate_argnames=("grad_accumulator",),
        )
      self._jitted_precomputed_gradient_pair_step_fn = functools.partial(
          nnx.cached_partial(
              self._jitted_precomputed_gradient_pair_step_impl,
              self.grad_accumulator,
          )
      )
    if microbatch_index != self._p28_precomputed_microstep:
      raise ValueError(
          "P30 pair gradient microbatch cadence mismatch: "
          f"expected {self._p28_precomputed_microstep}, got {microbatch_index}"
      )
    accumulate_start = time.perf_counter()
    norm = self._jitted_precomputed_gradient_pair_step_fn(
        left, right, jnp.asarray(multiplier, jnp.float32)
    )
    accumulate_call_done = time.perf_counter()
    norm.block_until_ready()
    if os.environ.get("CANON_PERF_LOG", "1") != "0":
      accumulate_done = time.perf_counter()
      print(
          "[PERF] stage=grad_accumulate seconds=%.3f microbatch=%d"
          " variant=pair call=%.3f block=%.3f"
          % (
              accumulate_done - accumulate_start,
              microbatch_index,
              accumulate_call_done - accumulate_start,
              accumulate_done - accumulate_call_done,
          ),
          flush=True,
      )
    self._iter_steps += 1
    self._p28_precomputed_microstep += 1
    return norm

  def accumulate_precomputed_scaled_gradient_microbatch(
      self,
      gradients: Any,
      multiplier: ArrayLike,
      *,
      microbatch_index: int,
  ) -> ArrayLike:
    """Streams one scaled P33 rank-reduced gradient contribution."""
    self._validate_precomputed_gradient_contract()
    if os.environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") != "1":
      raise ValueError(
          "scaled gradient accumulation is reserved for an admitted P33 "
          "workload"
      )
    if self._jitted_precomputed_gradient_scaled_step_fn is None:
      if self._jitted_precomputed_gradient_scaled_step_impl is None:
        self._jitted_precomputed_gradient_scaled_step_impl = nnx.jit(
            self._precomputed_gradient_scaled_step,
            donate_argnames=("grad_accumulator",),
        )
      self._jitted_precomputed_gradient_scaled_step_fn = functools.partial(
          nnx.cached_partial(
              self._jitted_precomputed_gradient_scaled_step_impl,
              self.grad_accumulator,
          )
      )
    if microbatch_index != self._p28_precomputed_microstep:
      raise ValueError(
          "P33 scaled gradient microbatch cadence mismatch: "
          f"expected {self._p28_precomputed_microstep}, got {microbatch_index}"
      )
    accumulate_start = time.perf_counter()
    norm = self._jitted_precomputed_gradient_scaled_step_fn(
        gradients, jnp.asarray(multiplier, jnp.float32)
    )
    accumulate_call_done = time.perf_counter()
    norm.block_until_ready()
    if os.environ.get("CANON_PERF_LOG", "1") != "0":
      accumulate_done = time.perf_counter()
      print(
          "[PERF] stage=grad_accumulate seconds=%.3f microbatch=%d"
          " variant=scaled call=%.3f block=%.3f"
          % (
              accumulate_done - accumulate_start,
              microbatch_index,
              accumulate_call_done - accumulate_start,
              accumulate_done - accumulate_call_done,
          ),
          flush=True,
      )
    self._iter_steps += 1
    self._p28_precomputed_microstep += 1
    return norm

  def commit_precomputed_gradients(self) -> ArrayLike:
    """Commits after all streamed microbatches and resets the accumulator."""
    # A failed transaction must never leave evidence from an earlier commit.
    self._last_precomputed_commit_evidence = None
    optimizer_transaction_start = time.perf_counter()
    optimizer_state = nnx.state(
        self.optimizer, nnx.optimizer.OptState
    )
    optimizer_logical_bytes = _state_logical_bytes(optimizer_state)
    optimizer_h2d_seconds = 0.0
    optimizer_d2h_seconds = 0.0
    self._validate_precomputed_gradient_contract()
    expected_microsteps = _precomputed_expected_microbatches(os.environ)
    if self._p28_precomputed_microstep != expected_microsteps:
      raise ValueError(
          "segmented update commit cadence mismatch: "
          f"{self._p28_precomputed_microstep} != {expected_microsteps}"
      )
    effective_learning_rate = self.effective_learning_rate(self._train_steps)
    if self.config.optimizer_offload:
      transfer_start = time.perf_counter()
      self._put_optimizer_state_on_memory_kind("device")
      optimizer_h2d_seconds = time.perf_counter() - transfer_start
      print(
          "[P30.G1] OPT_STATE before_commit memory_kind=device",
          flush=True,
      )
    adam_commit_start = time.perf_counter()
    if self._jitted_precomputed_gradient_commit_impl is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      donate_argnames = ("optimizer", "grad_accumulator")
      if os.environ.get("CANON_P30_DONATE_MODEL", "") == "1":
        donate_argnames = ("model",) + donate_argnames
        print(
            "[P30.G2] DONATE_MODEL on "
            "alias_contract=model,optimizer,grad_accumulator",
            flush=True,
        )
      self._jitted_precomputed_gradient_commit_impl = nnx.jit(
          self._precomputed_gradient_commit,
          donate_argnames=donate_argnames,
      )
    if self._jitted_precomputed_gradient_commit_fn is None:
      self._jitted_precomputed_gradient_commit_fn = functools.partial(
          nnx.cached_partial(
              self._jitted_precomputed_gradient_commit_impl,
              self.model,
              self.optimizer,
              self.grad_accumulator,
          )
      )
    adam_jit_setup_seconds = time.perf_counter() - adam_commit_start
    norm, raw_evidence = self._jitted_precomputed_gradient_commit_fn()
    adam_call_done = time.perf_counter()
    norm.block_until_ready()
    adam_block_done = time.perf_counter()
    host_evidence = jax.device_get(raw_evidence)
    adam_commit_seconds = time.perf_counter() - adam_commit_start
    if os.environ.get("CANON_PERF_LOG", "1") != "0":
      print(
          "[PERF] stage=adam_commit_detail seconds=%.3f jit_setup=%.3f"
          " call=%.3f block=%.3f evidence_get=%.3f"
          % (
              adam_commit_seconds,
              adam_jit_setup_seconds,
              adam_call_done - adam_commit_start - adam_jit_setup_seconds,
              adam_block_done - adam_call_done,
              adam_commit_start + adam_commit_seconds - adam_block_done,
          ),
          flush=True,
      )

    def _sum_counts(name: str) -> int:
      return sum(int(np.asarray(value)) for value in host_evidence[name])

    def _max_value(name: str) -> float:
      values = [float(np.asarray(value)) for value in host_evidence[name]]
      return max(values, default=0.0)

    def _all_true(name: str) -> bool:
      return all(bool(np.asarray(value)) for value in host_evidence[name])

    self._last_precomputed_commit_evidence = {
        "effective_learning_rate": effective_learning_rate,
        "gradient_nonzero_elements": _sum_counts(
            "gradient_nonzero_counts"
        ),
        "gradient_max_abs": _max_value("gradient_max_abs"),
        "gradient_finite": _all_true("gradient_finite"),
        "parameter_changed_elements": _sum_counts(
            "parameter_changed_counts"
        ),
        "parameter_total_elements": sum(
            int(value.size)
            for value in jax.tree.leaves(nnx.state(self.model, nnx.Param))
        ),
        "parameter_delta_max_abs": _max_value(
            "parameter_delta_max_abs"
        ),
        "parameter_delta_finite": _all_true("parameter_delta_finite"),
    }
    p63_max_norm = utils.canonical_overflow_safe_clip_max_norm(os.environ)
    if p63_max_norm is not None:
      clip_stats = host_evidence.get("overflow_safe_clip")
      if not isinstance(clip_stats, dict):
        raise RuntimeError("P63 optimizer clip evidence is unavailable")

      def _p63_float(name: str) -> float:
        return float(np.asarray(clip_stats[name]))

      naive_norm = _p63_float("naive_norm")
      stable_norm = _p63_float("stable_norm")
      selected_norm = _p63_float("selected_norm")
      clip_factor = _p63_float("clip_factor")
      all_finite = bool(np.asarray(clip_stats["all_finite"]))
      naive_norm_finite = bool(
          np.asarray(clip_stats["naive_norm_finite"])
      )
      fallback_used = bool(np.asarray(clip_stats["fallback_used"]))
      max_norm = _p63_float("max_norm")
      valid = (
          all_finite
          and math.isfinite(stable_norm)
          and stable_norm >= 0.0
          and math.isfinite(selected_norm)
          and selected_norm >= 0.0
          and math.isfinite(clip_factor)
          and 0.0 < clip_factor <= 1.0
          and max_norm == p63_max_norm
          and fallback_used == (not naive_norm_finite)
      )
      if not valid:
        raise RuntimeError(
            "P63 optimizer clip evidence is invalid: "
            f"all_finite={all_finite} naive_norm={naive_norm} "
            f"naive_norm_finite={naive_norm_finite} "
            f"stable_norm={stable_norm} selected_norm={selected_norm} "
            f"fallback_used={fallback_used} clip_factor={clip_factor} "
            f"max_norm={max_norm}"
        )
      clip_receipt = {
          "enabled": True,
          "all_finite": all_finite,
          "naive_norm": (
              naive_norm if math.isfinite(naive_norm) else "inf"
          ),
          "naive_norm_finite": naive_norm_finite,
          "stable_norm": stable_norm,
          "selected_norm": selected_norm,
          "fallback_used": fallback_used,
          "clip_factor": clip_factor,
          "max_norm": max_norm,
      }
      self._last_precomputed_commit_evidence[
          "overflow_safe_clip"
      ] = clip_receipt
      print(
          "[P63.STABLE_CLIP] "
          f"update={self._train_steps} all_finite=1 "
          f"naive_norm={clip_receipt['naive_norm']} "
          f"naive_norm_finite={int(naive_norm_finite)} "
          f"stable_norm={stable_norm} selected_norm={selected_norm} "
          f"fallback_used={int(fallback_used)} "
          f"clip_factor={clip_factor} max_norm={max_norm}",
          flush=True,
      )
    if self.config.optimizer_offload:
      transfer_start = time.perf_counter()
      self._put_optimizer_state_on_memory_kind("pinned_host")
      optimizer_d2h_seconds = time.perf_counter() - transfer_start
      print(
          "[P30.G1] OPT_STATE after_commit memory_kind=pinned_host",
          flush=True,
      )
    self._last_precomputed_commit_evidence["optimizer_timing"] = {
        "optimizer_logical_bytes": optimizer_logical_bytes,
        "optimizer_h2d_seconds": optimizer_h2d_seconds,
        "adam_commit_seconds": adam_commit_seconds,
        "optimizer_d2h_seconds": optimizer_d2h_seconds,
        "optimizer_transaction_seconds": (
            time.perf_counter() - optimizer_transaction_start
        ),
    }
    if os.environ.get("CANON_P30_RESHARD_ACCUMULATOR", "") == "1":
      active_mesh = jax.sharding.get_mesh()
      if active_mesh.empty:
        active_mesh = pxla.thread_resources.env.physical_mesh
      summary = self._reshard_grad_accumulator(
          active_mesh
      )
      print(
          "[P30.G2] RESHARD_ACCUMULATOR on "
          f"arrays={summary['arrays']} "
          f"logical_bytes={summary['logical_bytes']} target=metadata_pspecs",
          flush=True,
      )
    # Both cached partials bind mutable NNX objects whose device buffers were
    # donated by the transaction above.  Reusing either binding in the next
    # update can therefore submit an invalid (already donated) buffer on TPU.
    # Rebuild only the NNX bindings; the transformed JIT callables and their
    # compiled executable caches remain intact.
    self._jitted_precomputed_gradient_step_fn = None
    self._jitted_precomputed_gradient_scaled_step_fn = None
    self._jitted_precomputed_gradient_pair_step_fn = None
    self._jitted_precomputed_gradient_commit_fn = None
    if os.environ.get("CANON_P30_POST_COMMIT_GC", "") == "1":
      collected = gc.collect()
      print(
          "[P30.G2] POST_COMMIT_GC on "
          f"collected={collected} cached_bindings=cleared",
          flush=True,
      )
    self._train_steps += 1
    self._p28_precomputed_microstep = 0
    return norm

  def discard_precomputed_gradients(self) -> ArrayLike:
    """Discards one explicitly admitted transaction without committing it."""
    if (
        os.environ.get("CANON_P58_DEEPSWE_TIM", "") != "1"
        and os.environ.get("CANON_P62_BACKWARD_NUMERIC_DEBUG", "") != "1"
    ):
      raise ValueError(
          "precomputed discard is reserved for P58 or P62 diagnostics"
      )
    self._last_precomputed_commit_evidence = None
    self._validate_precomputed_gradient_contract()
    expected_microsteps = _precomputed_expected_microbatches(os.environ)
    if self._p28_precomputed_microstep != expected_microsteps:
      raise ValueError(
          "segmented discard cadence mismatch: "
          f"{self._p28_precomputed_microstep} != {expected_microsteps}"
      )
    if self._jitted_precomputed_gradient_discard_impl is None:
      self._jitted_precomputed_gradient_discard_impl = nnx.jit(
          self._precomputed_gradient_discard,
          donate_argnames=("grad_accumulator",),
      )
    if self._jitted_precomputed_gradient_discard_fn is None:
      self._jitted_precomputed_gradient_discard_fn = functools.partial(
          nnx.cached_partial(
              self._jitted_precomputed_gradient_discard_impl,
              self.grad_accumulator,
          )
      )
    denominator = self._jitted_precomputed_gradient_discard_fn()
    denominator.block_until_ready()
    # The accumulator buffer was donated. Rebind it on the next transaction;
    # the transformed implementation and executable cache remain reusable.
    self._jitted_precomputed_gradient_step_fn = None
    self._jitted_precomputed_gradient_scaled_step_fn = None
    self._jitted_precomputed_gradient_pair_step_fn = None
    self._jitted_precomputed_gradient_commit_fn = None
    self._jitted_precomputed_gradient_discard_fn = None
    self._p28_precomputed_microstep = 0
    return denominator

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

    denominator_weighted = self.config.loss_denominator_weighted_accumulation
    loss_denominator = jnp.asarray(1.0, dtype=jnp.float32)
    if isinstance(aux, utils.LossOutput):
      loss_denominator = aux.primary_loss.denominator.astype(jnp.float32)
      # The default path preserves the historical mean-of-local-means
      # behavior. Denominator-weighted accumulation keeps gradients of the
      # unreduced sum and divides once after all micro-batches have arrived.
      scale = aux.primary_loss.compute_scale()
      if not denominator_weighted:
        grads = jax.tree.map(lambda g: g * scale, grads)

      # Compute exactly equivalent legacy loss val
      loss_val = aux.primary_loss.compute()
    elif denominator_weighted:
      raise ValueError(
          "loss_denominator_weighted_accumulation requires LossOutput"
      )

    def normalized_grads():
      if denominator_weighted:
        return jax.tree.map(
            lambda g: g * aux.primary_loss.compute_scale(), grads
        )
      return grads

    def apply_updates(model, optimizer, grad_accumulator):
      acc_grads = grad_accumulator.get()
      # Compute the norm in float32 to 1) match `skip_updates()` return type and
      # meet the requirement of `nnx.cond` that both branches return the same
      # dtype, 2) for production-size models the sum-of-squares over bf16 grads
      # quickly exhausts bf16 and float32 is needed for numerical stability.
      norm = optax.global_norm(
          jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), acc_grads)
      )
      # The accumulator sums in float32; cast each leaf back to the current
      # grad's dtype (the param dtype) before the update so the (param-dtype)
      # moments aren't promoted, which would break the nnx.cond dtype match.
      # Mirrors optax MultiSteps' cast_like. Flatten to raw arrays so we don't
      # zip the two Variable trees (their sharding metadata differs).
      acc_leaves, acc_treedef = jax.tree_util.tree_flatten(acc_grads)
      grad_dtypes = [g.dtype for g in jax.tree_util.tree_leaves(grads)]
      acc_grads = jax.tree_util.tree_unflatten(
          acc_treedef,
          [a.astype(d) for a, d in zip(acc_leaves, grad_dtypes)],
      )
      optimizer.update(model, acc_grads)
      grad_accumulator.reset()
      return norm

    def skip_updates(model, optimizer, grad_accumulator):
      return jnp.array(0.0, dtype=jnp.float32)

    def discard_empty_update(model, optimizer, grad_accumulator):
      grad_accumulator.reset()
      return jnp.array(0.0, dtype=jnp.float32)

    def finish_denominator_weighted_update(
        model, optimizer, grad_accumulator
    ):
      return nnx.cond(
          grad_accumulator.denom[...] > 0.0,
          apply_updates,
          discard_empty_update,
          model,
          optimizer,
          grad_accumulator,
      )

    def apply_direct_update(model, optimizer, direct_grads):
      norm = optax.global_norm(
          jax.tree_util.tree_map(
              lambda x: x.astype(jnp.float32), direct_grads
          )
      )
      optimizer.update(model, direct_grads)
      return norm

    def skip_direct_update(model, optimizer, direct_grads):
      return jnp.array(0.0, dtype=jnp.float32)

    # P21.3 L3 gate-only mode must return the real value_and_grad primal and a
    # live gradient to the host gate without changing params OR the gradient
    # accumulator.  The host comparison necessarily happens after this
    # compiled call, so skipping only the optimizer would not be sufficient.
    canon_alignment = os.environ.get("CANON_ALIGNMENT_GATE", "") == "1"
    canon_gate_only_requested = (
        os.environ.get("CANON_ALIGNMENT_GATE_ONLY", "") == "1"
    )
    canon_update_canary_requested = (
        os.environ.get("CANON_ALIGNMENT_UPDATE_CANARY", "") == "1"
    )
    canon_train_requested = (
        os.environ.get("CANON_ALIGNMENT_TRAIN", "") == "1"
    )
    if canon_gate_only_requested and not canon_alignment:
      raise ValueError(
          "CANON_ALIGNMENT_GATE_ONLY=1 requires CANON_ALIGNMENT_GATE=1; "
          "refusing to skip an optimizer update without the host gate"
      )
    if canon_update_canary_requested and not canon_alignment:
      raise ValueError(
          "CANON_ALIGNMENT_UPDATE_CANARY=1 requires CANON_ALIGNMENT_GATE=1; "
          "refusing an unattested diagnostic optimizer update"
      )
    if canon_train_requested and not canon_alignment:
      raise ValueError(
          "CANON_ALIGNMENT_TRAIN=1 requires CANON_ALIGNMENT_GATE=1; "
          "refusing unattested training updates"
      )
    if canon_alignment and sum((
        canon_gate_only_requested,
        canon_update_canary_requested,
        canon_train_requested,
    )) != 1:
      raise ValueError(
          "alignment mode requires exactly one of CANON_ALIGNMENT_GATE_ONLY=1, "
          "CANON_ALIGNMENT_UPDATE_CANARY=1, or CANON_ALIGNMENT_TRAIN=1"
      )
    canon_gate_only = canon_alignment and canon_gate_only_requested
    deepswe_onehost_no_commit = _deepswe_onehost_no_commit(os.environ)
    optimizer_committed = jnp.asarray(False, dtype=jnp.bool_)
    accumulated_loss_denominator = loss_denominator
    if canon_gate_only or deepswe_onehost_no_commit:
      update_grads = normalized_grads()
      grad_norm = optax.global_norm(
          jax.tree_util.tree_map(
              lambda x: x.astype(jnp.float32), update_grads
          )
      )
    # At depth 1 the accumulator is a no-op and the cond predicate is always
    # True, so update directly from `grads` (no per-leaf accumulator writes,
    # no XLA Conditional, accumulator shardings untouched); sequence packing
    # keeps the cond path since its cadence comes from `is_update_step`.
    elif (
        self.config.get_with_default("gradient_accumulation_steps", 1) == 1
        and self.config.max_seq_token_per_tpu is None
    ):
      update_grads = normalized_grads()
      if denominator_weighted:
        optimizer_committed = loss_denominator > 0.0
        grad_norm = nnx.cond(
            optimizer_committed,
            apply_direct_update,
            skip_direct_update,
            model,
            optimizer,
            update_grads,
        )
      else:
        grad_norm = apply_direct_update(model, optimizer, update_grads)
    else:
      grad_accumulator.add(
          grads,
          denom=(
              loss_denominator
              if denominator_weighted
              else jnp.asarray(1.0, dtype=jnp.float32)
          ),
      )
      accumulated_loss_denominator = grad_accumulator.denom[...]

      # If the mesh is not empty, then we need to replicate the is_update_step
      # across all devices to avoid deadlock so that all devices see the same
      # update step.
      mesh = pxla.thread_resources.env.physical_mesh
      if not mesh.empty:
        is_update_step = jax.lax.with_sharding_constraint(
            is_update_step, jax.sharding.PartitionSpec()
        )

      if denominator_weighted:
        optimizer_committed = jnp.logical_and(
            is_update_step, accumulated_loss_denominator > 0.0
        )
        grad_norm = nnx.cond(
            is_update_step,
            finish_denominator_weighted_update,
            skip_updates,
            model,
            optimizer,
            grad_accumulator,
        )
      else:
        grad_norm = nnx.cond(
            is_update_step,
            apply_updates,
            skip_updates,
            model,
            optimizer,
            grad_accumulator,
        )

    if isinstance(aux, utils.LossOutput):
      if denominator_weighted:
        aux.aux_metrics["loss/accumulated_denominator"] = (
            accumulated_loss_denominator
        )
        aux.aux_metrics["loss/optimizer_committed"] = optimizer_committed
      if canon_alignment:
        aux.aux_metrics["canon/gradient_norm"] = grad_norm
        aux.aux_metrics["canon/optimizer_skipped"] = jnp.asarray(
            canon_gate_only or deepswe_onehost_no_commit, dtype=jnp.int32
        )
        aux.aux_metrics["canon/is_update_step"] = jnp.asarray(
            is_update_step, dtype=jnp.bool_
        )
      if deepswe_onehost_no_commit:
        aux.aux_metrics["deepswe/optimizer_skipped"] = jnp.asarray(
            1, dtype=jnp.int32
        )
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

    # Partition Gradients same as the model
    grad_pspecs = nnx.get_partition_spec(self.grad_accumulator.grads)
    self.grad_accumulator.grads = jax.tree.map(
        _shard, self.grad_accumulator.grads, grad_pspecs
    )

    # Denominator is a scalar — replicate across all devices
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

    def _apply_op(v, op):
      if isinstance(v, list) and v and isinstance(v[0], utils.WeightedMetric):
        if getattr(op, "__name__", "") in (
            "global_weighted_mean",
            "mean_of_means",
        ):
          return op(v)
        v = [x.compute() for x in v]
      return op(_to_np_array(v))

    self._log_metrics(
        loss=metrics_buffer.loss,
        step=metrics_buffer.step,
        additional_metrics={
            k: _apply_op(v, op)
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
        elif (
            not is_update_step_val
            and self.config.get_with_default("gradient_accumulation_steps", 1)
            == 1
            and self.config.max_seq_token_per_tpu is None
        ):
          # The depth-1 direct-update path in `_train_step` updates on every
          # step; a data-driven skip flag would be silently ignored there.
          raise ValueError(
              "data-driven is_update_step=False conflicts with the depth-1"
              " direct-update path; set gradient_accumulation_steps>1 or"
              " max_seq_token_per_tpu."
          )

        with self._perf_tracer.span(
            "peft_train_step",
            pxla.thread_resources.env.physical_mesh.devices,
        ) as span, self._perf_tracer_v2.span(
            perf_constants.PEFT_TRAIN,
            pxla.thread_resources.env.physical_mesh.devices,
            tags=tags,
        ) as span_v2:
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

        denominator_weighted_commit = True
        if (
            is_update_step_val
            and self.config.loss_denominator_weighted_accumulation
        ):
          try:
            denominator_weighted_commit = bool(
                np.asarray(
                    jax.device_get(aux["loss/optimizer_committed"])
                ).item()
            )
          except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "denominator-weighted optimizer receipt is missing"
            ) from exc

        if is_update_step_val and _deepswe_onehost_no_commit(os.environ):
          print(
              "[DEEPSWE.ONEHOST] optimizer_boundary_skipped commits=0 "
              f"train_steps={self._train_steps}",
              flush=True,
          )
          self._write_train_metrics()
        elif is_update_step_val and not denominator_weighted_commit:
          print(
              "[DEEPSWE.COMPACT_FILTER] optimizer_boundary_skipped "
              f"effective_rows=0 train_steps={self._train_steps}",
              flush=True,
          )
          self._write_train_metrics()
        elif is_update_step_val:
          self._train_steps += 1
          if os.environ.get("CANON_ALIGNMENT_TRAIN", "") == "1":
            print(
                "[CANON_GSM8K_TRAIN] update_step_committed "
                f"train_steps={self._train_steps}",
                flush=True,
            )
          if (
              os.environ.get("CANON_ALIGNMENT_UPDATE_CANARY", "") == "1"
          ):
            print(
                "[CANON_GSM8K_UPDATE] update_step_committed "
                f"train_steps={self._train_steps}",
              flush=True,
            )
          if os.environ.get("CANON_FROZENLAKE_P27", "") == "1":
            p27_marker = (
                "gate_boundary_complete"
                if os.environ.get("CANON_ALIGNMENT_GATE_ONLY", "") == "1"
                else "update_step_committed"
            )
            print(
                f"[CANON_FROZENLAKE_P27] {p27_marker} "
                f"train_steps={self._train_steps}",
                flush=True,
            )
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
    if _deepswe_onehost_no_commit(os.environ):
      logging.info(
          "Skipping final checkpoint for DeepSWE one-host no-commit smoke."
      )
      return
    if not self.checkpoint_manager.save_on_close:
      logging.info(
          "Skipping forced final checkpoint; interval-only policy is active."
      )
      return
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
