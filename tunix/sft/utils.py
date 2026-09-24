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

"""Simple utils used by SFT."""

import collections
from collections.abc import Iterable, Mapping
import contextlib
import functools
import gc
import time
from typing import Any, Callable, List, Optional, Tuple

from absl import logging
from flax import nnx
import flax.struct
import humanize
import jax
import jax.numpy as jnp
import numpy as np
from tunix.oss import utils as google_utils


def make_causal_attn_mask(input_mask: jax.Array) -> jax.Array:
  """Makes a causal attention mask.

  I.e., as in middle diagram of Figure 3 in https://arxiv.org/pdf/1910.10683.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
  """
  if len(input_mask.shape) != 2:
    raise ValueError(
        f"Input mask must be 2D (shape [B, L]), but got {input_mask.shape}."
    )
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  # Prefixes can be attended by all tokens
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def build_positions_from_mask(input_mask: jax.Array) -> jax.Array:
  """Computes `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)


def is_lora_enabled(model: nnx.Module) -> bool:
  for _, value in nnx.iter_graph(model):
    if isinstance(value, nnx.LoRAParam):
      return True
  return False


def is_beft_enabled(model: nnx.Module) -> bool:
  from tunix.sft.peft.beft import BEFTParam  # pylint: disable=g-import-not-at-top
  for _, value in nnx.iter_graph(model):
    if isinstance(value, BEFTParam):
      return True
  return False


def is_peft_enabled(model: nnx.Module) -> bool:
  return is_lora_enabled(model) or is_beft_enabled(model)


def get_peft_param_type(model: nnx.Module) -> type[nnx.Variable] | None:
  """Returns the active PEFT parameter type or None if full fine-tuning."""
  from tunix.sft.peft.beft import BEFTParam  # pylint: disable=g-import-not-at-top
  if is_beft_enabled(model):
    return BEFTParam
  if is_lora_enabled(model):
    return nnx.LoRAParam
  return None


@contextlib.contextmanager
def time_measure(context: str = "", suppress_logging: bool = False):
  start = time.perf_counter()
  try:
    yield lambda: time.perf_counter() - start
  finally:
    if not suppress_logging:
      logging.info(
          "%s finished in: %.4f seconds", context, time.perf_counter() - start
      )


def _pathways_hbm_usage_gb(devices: Any) -> List[Tuple[float, Optional[float]]]:
  """Returns the HBM usage for each device when using Pathways.

  Args:
    devices: The devices to get the HBM usage for.

  Returns:
    A list of tuples, where each tuple contains the HBM usage and limit for a
    device.
  """
  live_arrays = jax.live_arrays()
  hbm_used = collections.defaultdict(int)
  # TODO(lancewang): Find a way to get the accurate hbm limit on Pathways.
  hbm_limit = None
  # Track unique buffers to avoid double-counting when multiple Python
  # variables reference the same underlying JAX array (e.g., a = jnp.ones(10);
  # b = a)
  seen_buffers = set()
  for array in live_arrays:
    assert hasattr(array, "sharding") and hasattr(
        array.sharding, "device_set"
    ), (
        "This function must not be called within jax tracer (e.g. jit, vmap,"
        " grad)"
    )
    # The array could probably be deleted between the time we get the live
    # arrays and now. Skip them if so.
    if array.is_deleted():
      continue

    for buffer in array.addressable_shards:
      # Using id() on the shard data is a good way to get a unique identifier
      # for the underlying buffer. This ensures that even if multiple
      # `DeviceArray` objects point to the same memory, we only count it once.
      buffer_id = id(buffer.data)
      if buffer_id not in seen_buffers:
        seen_buffers.add(buffer_id)
        hbm_used[buffer.data.device] += buffer.data.nbytes
  return [(hbm_used[device], hbm_limit) for device in devices]


def _jax_hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
  """Returns the HBM usage for each device when using JAX."""
  hbm_used = []
  for device in devices:
    if device.platform == "cpu":
      logging.warning(
          "Skipping non-TPU device: %s. You might be missing jax[tpu]"
          " dependency.",
          device.platform,
      )
      return []
    stats = device.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    hbm_used.append((used, limit))
  return hbm_used


def show_hbm_usage(title=""):
  """Prints the current HBM usage.

  Args:
    title: The title to print before the HBM usage.
  """
  fmt_size = functools.partial(humanize.naturalsize, binary=True)
  # Force a GC sweep to catch recently deallocated arrays
  gc.collect()

  if google_utils.pathways_available():
    logging.info("%s - Using Pathways compatible HBM stats collector", title)
    devices = jax.devices()
    hbm_stats = _pathways_hbm_usage_gb(devices)
    for i, (used, _) in enumerate(hbm_stats):
      logging.info("Using %s on %s", fmt_size(used), devices[i])
  else:
    logging.info(
        "%s - Pathways not available. Using default HBM stats collector", title
    )
    devices = jax.local_devices()
    hbm_stats = _jax_hbm_usage_gb(devices)

    for i, (used, limit) in enumerate(hbm_stats):
      logging.info(
          "Using %s / %s (%s) on %s",
          fmt_size(used),
          fmt_size(limit),
          used / limit,
          devices[i],
      )


@flax.struct.dataclass
class WeightedMetric:
  """A metric that requires weighted reduction.

  Attributes:
    unreduced_sum: The sum of the metric values. Should be a scalar ().
    denominator: The weight or count of valid tokens/examples. Should be a
      scalar ().
    eps: Optional epsilon added to denominator for numerical stability.
    min_denom: Optional minimum bound for the denominator.
  """

  unreduced_sum: jax.Array
  denominator: jax.Array
  eps: float | None = flax.struct.field(default=None, pytree_node=False)
  min_denom: float | None = flax.struct.field(default=None, pytree_node=False)

  def compute_scale(self) -> jax.Array:
    """Safely computes the scale factor (1 / denominator) with bounds."""
    denom = self.denominator
    if self.eps is not None:
      denom = denom + self.eps
    if self.min_denom is not None:
      denom = jnp.maximum(denom, self.min_denom)

    # JAX Safe Division: Prevent division-by-zero NaNs from poisoning gradients
    # We replace 0s with 1.0 *before* dividing.
    safe_denom = jnp.where(denom == 0, 1.0, denom)

    # Calculate scale, masking out pure zero denominators to 0.0
    scale = 1.0 / safe_denom
    return jnp.where(denom == 0, 0.0, scale)

  def compute(self) -> jax.Array:
    """Safely computes total / count with optional legacy equivalence bounds."""
    return self.unreduced_sum * self.compute_scale()


# The full field contract `weighted_metric_mean` relies on. Checking all four
# rather than just the two unreduced ones keeps the predicate honest: anything
# it admits can be reduced end to end without tripping over a missing bound
# partway through.
_WEIGHTED_METRIC_FIELDS = ("unreduced_sum", "denominator", "eps", "min_denom")


def is_weighted_metric(value: Any) -> bool:
  """Structurally identifies an unreduced weighted metric.

  `tunix.experimental.metrics.WeightedMetric` mirrors `WeightedMetric` field for
  field without inheriting from it, so reducers have to accept both. Matching on
  the fields rather than the type covers either one without `sft.utils` taking a
  dependency on `experimental`.

  Args:
    value: Candidate metric value to inspect.

  Returns:
    True if `value` exposes all `_WEIGHTED_METRIC_FIELDS`.
  """
  return all(hasattr(value, field) for field in _WEIGHTED_METRIC_FIELDS)


def weighted_metric_mean(values: Iterable[Any]) -> float:
  """Aggregates unreduced metrics without microbatch-mean bias.

  Sums numerators and denominators before dividing, rather than averaging
  per-microbatch means, which would weight microbatches with unequal
  denominators incorrectly.

  Args:
    values: Sequence of unreduced `WeightedMetric` values across microbatches.

  Returns:
    The global weighted mean across all microbatches.
  """
  values = list(values)
  if not values:
    return 0.0
  if not all(is_weighted_metric(value) for value in values):
    raise TypeError("weighted metrics must not include scalar values")

  eps = values[0].eps
  min_denom = values[0].min_denom
  if any(
      value.eps != eps or value.min_denom != min_denom for value in values[1:]
  ):
    raise ValueError("weighted metrics must use consistent denominator bounds")

  numerator = sum(float(np.asarray(value.unreduced_sum)) for value in values)
  denominator = sum(float(np.asarray(value.denominator)) for value in values)
  if eps is not None:
    denominator += eps
  if min_denom is not None:
    denominator = max(denominator, min_denom)
  return numerator / denominator if denominator else 0.0


def metric_reducer(metric: Any) -> Callable[[Any], Any]:
  """Selects the reduction that matches a buffered auxiliary metric."""
  return weighted_metric_mean if is_weighted_metric(metric) else np.mean


@flax.struct.dataclass
class LossOutput:
  """Output of a loss function containing unreduced primary loss and aux metrics.

  Attributes:
    primary_loss: The main loss to be optimized.
    aux_metrics: A dictionary of auxiliary metrics.
  """

  primary_loss: WeightedMetric
  aux_metrics: Mapping[str, WeightedMetric | jax.Array]


_WRAPPER_ATTRS = (
    "inner_opt_state",
    "inner_state",
    "inner_states",
    "fast_state",
)


def try_get_learning_rate(opt_state: Any) -> float | jax.Array | None:
  """Extracts the last injected learning rate from an optax state tree."""
  def _walk(state: Any) -> Iterable[Any]:
    if (
        isinstance(hp := getattr(state, "hyperparams", None), Mapping)
        and "learning_rate" in hp
    ):
      yield getattr(hp["learning_rate"], "value", hp["learning_rate"])
    if isinstance(state, tuple) and not hasattr(state, "_fields"):
      children = state
    elif isinstance(state, Mapping):
      children = state.values()
    else:
      children = (
          getattr(state, a) for a in _WRAPPER_ATTRS if hasattr(state, a)
      )
    for child in children:
      yield from _walk(child)

  lrs = list(_walk(opt_state))
  return lrs[-1] if lrs else None
