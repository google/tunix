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
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
from flax import nnx
import flax.struct
import humanize
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.oss import utils as google_utils


def _scaled_l2_components(tree: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns overflow-safe ``(max_abs, scaled_sumsq, all_finite)``.

  Squaring a finite float32 value larger than roughly ``1.84e19`` overflows,
  even when the resulting L2 norm is representable.  Scale every leaf by the
  tree-wide maximum before the sum of squares so the reduction itself cannot
  overflow.  Non-finite inputs remain visible through ``all_finite`` and the
  returned maximum; callers must not treat this as a finite-value sanitizer.
  """
  leaves = [
      jnp.asarray(value).astype(jnp.float32)
      for value in jax.tree.leaves(tree)
  ]
  if not leaves:
    return (
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(True),
    )
  max_abs = functools.reduce(
      jnp.maximum,
      (jnp.max(jnp.abs(value)) for value in leaves),
  )
  all_finite = functools.reduce(
      jnp.logical_and,
      (jnp.all(jnp.isfinite(value)) for value in leaves),
  )
  safe_scale = jnp.where(
      jnp.logical_and(all_finite, max_abs > 0.0),
      max_abs,
      jnp.asarray(1.0, jnp.float32),
  )
  scaled_sumsq = sum(
      jnp.sum(jnp.square(value / safe_scale)) for value in leaves
  )
  return max_abs, scaled_sumsq, all_finite


def stable_global_norm(tree: Any) -> jax.Array:
  """Computes a finite-input L2 norm without intermediate FP32 overflow.

  If the mathematical norm itself exceeds the float32 range, the diagnostic
  value saturates at ``float32.max``. For a non-finite tree this returns the
  observed maximum only as context; reduction NaN propagation can be
  backend-dependent, so callers must use the independent ``all_finite`` bit
  from :func:`tree_numeric_stats` as the gate and must never admit a gradient
  from this scalar alone.
  """
  max_abs, scaled_sumsq, all_finite = _scaled_l2_components(tree)
  root = jnp.sqrt(scaled_sumsq)
  raw_norm = max_abs * root
  saturated = jnp.minimum(raw_norm, jnp.finfo(jnp.float32).max)
  finite_norm = jnp.where(max_abs == 0.0, 0.0, saturated)
  return jnp.where(all_finite, finite_norm, max_abs)


def hybrid_global_norm_stats(
    tree: Any, *, max_norm: float | jax.Array | None = None
) -> dict[str, jax.Array]:
  """Returns stock and overflow-safe norm state without sanitizing inputs.

  The selected norm is byte-identical to the stock Optax norm whenever the
  latter is finite.  Max-scaled L2 is selected only for an independently
  all-finite tree whose stock sum of squares overflowed.  A tree containing
  NaN or Inf therefore retains its non-finite stock result.
  """
  naive_norm = optax.tree.norm(tree)
  max_abs, scaled_sumsq, all_finite = _scaled_l2_components(tree)
  root = jnp.sqrt(scaled_sumsq)
  raw_stable_norm = max_abs * root
  stable_norm = jnp.where(
      max_abs == 0.0,
      jnp.asarray(0.0, jnp.float32),
      jnp.minimum(raw_stable_norm, jnp.finfo(jnp.float32).max),
  )
  fallback_used = jnp.logical_and(
      all_finite, jnp.logical_not(jnp.isfinite(naive_norm))
  )
  selected_norm = jnp.where(
      fallback_used, stable_norm, naive_norm.astype(jnp.float32)
  )
  result = {
      "all_finite": all_finite,
      "naive_norm": naive_norm,
      "naive_norm_finite": jnp.isfinite(naive_norm),
      "max_abs": max_abs,
      "scaled_sumsq": scaled_sumsq,
      "stable_norm": stable_norm,
      "fallback_used": fallback_used,
      "selected_norm": selected_norm,
  }
  if max_norm is not None:
    max_norm_array = jnp.asarray(max_norm, jnp.float32)
    safe_norm = jnp.where(
        jnp.logical_and(
            jnp.isfinite(selected_norm), selected_norm > 0.0
        ),
        selected_norm,
        jnp.asarray(1.0, jnp.float32),
    )
    result["max_norm"] = max_norm_array
    result["clip_factor"] = jnp.where(
        selected_norm < max_norm_array,
        jnp.asarray(1.0, jnp.float32),
        max_norm_array / safe_norm,
    )
  return result


def hybrid_global_norm(tree: Any) -> jax.Array:
  """Returns stock norm, falling back only for finite FP32 overflow."""
  return hybrid_global_norm_stats(tree)["selected_norm"]


def overflow_safe_clip_by_global_norm(
    max_norm: float,
) -> optax.GradientTransformation:
  """Preserves stock clipping except for proven finite norm overflow."""
  stock = optax.clip_by_global_norm(max_norm)

  def update_fn(updates, state, params=None):
    stock_updates, _ = stock.update(updates, state, params)
    stats = hybrid_global_norm_stats(updates, max_norm=max_norm)
    stable_norm = stats["stable_norm"]
    stable_trigger = jnp.squeeze(stable_norm < max_norm)

    def stable_clip(value):
      clipped = (
          value / stable_norm.astype(value.dtype)
      ) * jnp.asarray(max_norm, value.dtype)
      return jax.lax.select(stable_trigger, value, clipped)

    stable_updates = jax.tree.map(stable_clip, updates)
    fallback = jnp.squeeze(stats["fallback_used"])
    selected = jax.tree.map(
        lambda stable, original: jax.lax.select(
            fallback, stable, original
        ),
        stable_updates,
        stock_updates,
    )
    return selected, state

  return optax.GradientTransformation(stock.init, update_fn)


_P63_CONTEXTS = {
    (
        "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env",
        "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp",
        "gsm8k",
    ): 1.0,
    (
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
        "qwen3-8b-dp8-tp8-frozenlake-v1-hp",
        "frozenlake-dp8-tp8",
    ): 100.0,
}


def canonical_overflow_safe_clip_max_norm(
    environ: dict[str, str],
) -> float | None:
  """Validates the exact P63 production context and returns its max norm."""
  raw = environ.get("CANON_P63_OVERFLOW_SAFE_CLIP")
  if raw is None or raw == "0":
    return None
  if raw != "1":
    raise ValueError(
        "CANON_P63_OVERFLOW_SAFE_CLIP must be absent, 0, or 1"
    )
  key = (
      environ.get("CANON_PROFILE_FILE", ""),
      environ.get("CANON_PROFILE", ""),
      environ.get("CANON_P32_WORKLOAD", ""),
  )
  if key not in _P63_CONTEXTS:
    raise ValueError(
        "CANON_P63_OVERFLOW_SAFE_CLIP is restricted to the exact Phase4 "
        f"GSM8K/FrozenLake profiles; found {key!r}"
    )
  required = {
      "CANON_V1_HP_FULL": "1",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_P28_SEGMENTED_TRAIN": "1",
      "CANON_P28_G6_UPDATE": "1",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
  }
  wrong = {
      name: environ.get(name)
      for name, expected in required.items()
      if environ.get(name) != expected
  }
  if wrong:
    raise ValueError(
        "CANON_P63_OVERFLOW_SAFE_CLIP production contract changed: "
        f"{wrong}"
    )
  warn_only_name = (
      "CANON_GSM8K_ALIGNMENT_WARN_ONLY"
      if key[2] == "gsm8k"
      else "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"
  )
  if environ.get(warn_only_name) != "0":
    raise ValueError(
        "CANON_P63_OVERFLOW_SAFE_CLIP requires strict alignment: "
        f"{warn_only_name}={environ.get(warn_only_name)!r}"
    )
  return _P63_CONTEXTS[key]


@jax.jit
def tree_numeric_stats(tree: Any) -> dict[str, jax.Array]:
  """Returns compact diagnostics without changing or sanitizing ``tree``.

  The per-leaf vectors let a caller identify the first non-finite leaf and the
  largest-magnitude paths after a single host transfer. ``stable_norm`` is an
  observer only: it must not admit an unexplained finite-huge gradient to an
  optimizer transaction.
  """
  leaves = [
      jnp.asarray(value).astype(jnp.float32)
      for value in jax.tree.leaves(tree)
  ]
  if not leaves:
    return {
        "all_finite": jnp.asarray(True),
        "any_nonzero": jnp.asarray(False),
        "leaf_all_finite": jnp.zeros((0,), dtype=jnp.bool_),
        "leaf_any_nonzero": jnp.zeros((0,), dtype=jnp.bool_),
        "leaf_max_abs": jnp.zeros((0,), dtype=jnp.float32),
        "max_abs": jnp.asarray(0.0, dtype=jnp.float32),
        "naive_norm": jnp.asarray(0.0, dtype=jnp.float32),
        "naive_norm_finite": jnp.asarray(True),
        "scaled_sumsq": jnp.asarray(0.0, dtype=jnp.float32),
        "stable_norm": jnp.asarray(0.0, dtype=jnp.float32),
    }
  leaf_all_finite = jnp.stack(
      tuple(jnp.all(jnp.isfinite(value)) for value in leaves)
  )
  leaf_any_nonzero = jnp.stack(
      tuple(jnp.any(value != 0.0) for value in leaves)
  )
  leaf_max_abs = jnp.stack(
      tuple(jnp.max(jnp.abs(value)) for value in leaves)
  )
  max_abs, scaled_sumsq, all_finite = _scaled_l2_components(tree)
  naive_norm = jnp.sqrt(sum(
      jnp.sum(jnp.square(value)) for value in leaves
  ))
  return {
      "all_finite": all_finite,
      "any_nonzero": jnp.any(leaf_any_nonzero),
      "leaf_all_finite": leaf_all_finite,
      "leaf_any_nonzero": leaf_any_nonzero,
      "leaf_max_abs": leaf_max_abs,
      "max_abs": max_abs,
      "naive_norm": naive_norm,
      "naive_norm_finite": jnp.isfinite(naive_norm),
      "scaled_sumsq": scaled_sumsq,
      "stable_norm": stable_global_norm(tree),
  }


@jax.jit
def scaled_tree_numeric_stats(
    tree: Any, multiplier: jax.Array
) -> dict[str, jax.Array]:
  """Diagnoses ``tree * multiplier`` without returning the scaled tree."""
  multiplier = jnp.asarray(multiplier, dtype=jnp.float32)
  return tree_numeric_stats(jax.tree.map(
      lambda value: jnp.asarray(value).astype(jnp.float32) * multiplier,
      tree,
  ))


@jax.jit
def ranked_tree_numeric_stats(tree: Any) -> dict[str, jax.Array]:
  """Returns one numeric summary per common leading rank axis."""
  return jax.vmap(tree_numeric_stats)(tree)


def _json_float(value: Any) -> float | str:
  value = float(np.asarray(value))
  if np.isnan(value):
    return "nan"
  if np.isposinf(value):
    return "inf"
  if np.isneginf(value):
    return "-inf"
  return value


def tree_numeric_receipt(
    tree: Any,
    *,
    stats: dict[str, Any] | None = None,
    ranked: bool = False,
    top_k: int = 3,
) -> dict[str, Any]:
  """Materializes one compact, JSON-safe diagnostic receipt on the host."""
  flattened = jax.tree_util.tree_flatten_with_path(tree)[0]
  paths = tuple(jax.tree_util.keystr(path) for path, _ in flattened)
  leaves = tuple(value for _, value in flattened)
  if stats is None:
    stats = tree_numeric_stats(tree)
  host = jax.device_get(stats)
  leaf_finite = np.asarray(host["leaf_all_finite"], dtype=np.bool_)
  leaf_nonzero = np.asarray(host["leaf_any_nonzero"], dtype=np.bool_)
  leaf_max_abs = np.asarray(host["leaf_max_abs"], dtype=np.float32)
  expected_shape = (len(paths),)
  if (
      leaf_finite.shape != expected_shape
      or leaf_nonzero.shape != expected_shape
      or leaf_max_abs.shape != expected_shape
  ):
    raise ValueError(
        "numeric diagnostic leaf vectors changed shape: "
        f"finite={leaf_finite.shape} nonzero={leaf_nonzero.shape} "
        f"max_abs={leaf_max_abs.shape} expected={expected_shape}"
    )
  order_values = np.nan_to_num(
      leaf_max_abs, nan=np.inf, posinf=np.inf, neginf=np.inf
  )
  top_indices = np.argsort(order_values)[::-1][:max(0, int(top_k))]
  bad_indices = np.flatnonzero(~leaf_finite)
  receipt = {
      "all_finite": bool(np.asarray(host["all_finite"])),
      "any_nonzero": bool(np.asarray(host["any_nonzero"])),
      "first_nonfinite": (
          None
          if bad_indices.size == 0
          else {
              "leaf": int(bad_indices[0]),
              "path": paths[int(bad_indices[0])],
          }
      ),
      "leaf_count": len(paths),
      "nonzero_leaf_count": int(np.count_nonzero(leaf_nonzero)),
      "total_elements": sum(int(np.prod(value.shape)) for value in leaves),
      "max_abs": _json_float(host["max_abs"]),
      "naive_norm": _json_float(host["naive_norm"]),
      "naive_norm_finite": bool(np.asarray(host["naive_norm_finite"])),
      "scaled_sumsq": _json_float(host["scaled_sumsq"]),
      "stable_norm": _json_float(host["stable_norm"]),
      "top_leaves": [
          {
              "leaf": int(index),
              "path": paths[int(index)],
              "max_abs": _json_float(leaf_max_abs[int(index)]),
          }
          for index in top_indices
      ],
  }
  if ranked:
    ranked_host = jax.device_get(ranked_tree_numeric_stats(tree))
    ranked_finite = np.asarray(
        ranked_host["leaf_all_finite"], dtype=np.bool_
    )
    ranked_max_abs = np.asarray(ranked_host["max_abs"], dtype=np.float32)
    if ranked_finite.ndim != 2 or ranked_finite.shape[1] != len(paths):
      raise ValueError(
          "ranked numeric diagnostic shape changed: "
          f"{ranked_finite.shape} expected=(*,{len(paths)})"
      )
    bad_rank_leaf = np.argwhere(~ranked_finite)
    receipt["rank_count"] = int(ranked_finite.shape[0])
    receipt["rank_max_abs"] = [
        _json_float(value) for value in ranked_max_abs
    ]
    receipt["first_nonfinite_rank"] = (
        None
        if bad_rank_leaf.size == 0
        else {
            "rank": int(bad_rank_leaf[0, 0]),
            "leaf": int(bad_rank_leaf[0, 1]),
            "path": paths[int(bad_rank_leaf[0, 1])],
        }
    )
  return receipt


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
