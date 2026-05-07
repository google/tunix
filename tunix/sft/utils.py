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
import contextlib
import functools
import gc
import time
from typing import Any, List, Optional, Tuple

from absl import logging
from flax import nnx
import humanize
import jax
import jax.numpy as jnp
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


import flax.struct
from typing import Dict


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


@flax.struct.dataclass
class LossOutput:
  """Output of a loss function containing unreduced primary loss and aux metrics.

  Attributes:
    primary_loss: The main loss to be optimized.
    aux_metrics: A dictionary of auxiliary metrics.
  """

  primary_loss: WeightedMetric
  aux_metrics: Dict[str, WeightedMetric]
