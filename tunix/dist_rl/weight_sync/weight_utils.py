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

"""Transport-neutral JAX PyTree, manifest, state-install, and checksum helpers."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import math
import re
from typing import Any, Protocol

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.weight_sync import weight_sync

# One bracketed component (`['w']`, `[0]`) or one dotted component (`w`).
_PATH_COMPONENT_RE = re.compile(r"\[([^\]]*)\]|([^.\[\]]+)")
# A trailing ordinal folded into the name, as in `layers_0`.
_FOLDED_INDEX_RE = re.compile(r"^(.+?)_(\d+)$")
_KV_CACHE_SEGMENTS = frozenset(
    {"cache", "cached_prefill_key", "cached_prefill_value"}
)
_CHECKSUM_META_KEYS = frozenset(
    {"__grand_total__", "__tensor_count__", "__element_count__"}
)


class SyncedSamplerTarget(Protocol):
  """Sampler surface required for out-of-place destination weight updates."""

  @property
  def model_runner(self) -> Any:
    ...

  @property
  def transformer_state(self) -> Any:
    ...


def flatten_weights(state: Any) -> tuple[list[str], list[Any]]:
  """Returns `(names, arrays)` for every array leaf, in stable tree order."""
  names: list[str] = []
  arrays: list[Any] = []
  for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    arr = getattr(leaf, "value", leaf)
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
      names.append(jax.tree_util.keystr(path))
      arrays.append(arr)
  return names, arrays


def param_key(name: str) -> str:
  """Canonical dotted key for pairing a bound name with a runner tree path."""
  segments: list[str] = []
  for bracketed, dotted in _PATH_COMPONENT_RE.findall(name):
    component = (bracketed or dotted).strip("'\"")
    if not component:
      continue
    folded = _FOLDED_INDEX_RE.match(component)
    segments.extend(folded.groups() if folded else (component,))
  while segments and segments[0] in ("base", "model"):
    segments.pop(0)
  if segments and segments[-1] == "value":
    segments.pop()
  return ".".join(segments)


def is_kv_cache(name: str) -> bool:
  """True for inference KV-cache buffers, which are not model weights."""
  return any(seg in _KV_CACHE_SEGMENTS for seg in param_key(name).split("."))


def is_bindable(arr: Any, *, allow_proxy: bool = False) -> bool:
  """True if `arr` is a rank>=1 floating-point array on a supported platform."""
  try:
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
      return False
    if arr.ndim < 1:
      return False
    if not jnp.issubdtype(arr.dtype, jnp.floating):
      return False
    devices = arr.devices()
    if not devices:
      return False
    supported_platforms = {"tpu", "cpu"}
    if allow_proxy:
      supported_platforms.update(("proxy", "pathways"))
    return all(
        getattr(d, "platform", "?") in supported_platforms for d in devices
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def filter_bindable(
    names: Sequence[str],
    arrays: Sequence[Any],
    *,
    allow_proxy: bool = False,
    label: str = "raiden",
) -> tuple[list[str], list[Any]]:
  """Drops unbindable leaves and inference-only KV cache buffers."""
  logging.vlog(
      1,
      "%s bind census: %s",
      label,
      collections.Counter(
          (type(a).__name__, str(getattr(a, "dtype", "?"))) for a in arrays
      ),
  )
  keep_names: list[str] = []
  keep_arrays: list[Any] = []
  dropped: list[str] = []
  cache_dropped: list[str] = []
  for name, arr in zip(names, arrays):
    if is_kv_cache(name):
      cache_dropped.append(name)
    elif is_bindable(arr, allow_proxy=allow_proxy):
      arr.block_until_ready()
      keep_names.append(name)
      keep_arrays.append(arr)
    else:
      dropped.append(name)
  if cache_dropped:
    logging.debug(
        "%s bind skipped %d KV-cache leaves: %s",
        label,
        len(cache_dropped),
        cache_dropped[:5],
    )
  if dropped:
    logging.warning(
        "%s bind dropped %d unbindable leaves: %s",
        label,
        len(dropped),
        dropped[:5],
    )
  return keep_names, keep_arrays


def _axis_name(axis: Any) -> str:
  if axis is None:
    return ""
  if isinstance(axis, str):
    return axis
  return ",".join(axis)


def compute_global_shard_index(
    indices: tuple[slice, ...], shape: tuple[int, ...]
) -> int:
  """Flattens a shard's per-dimension slices into its row-major grid index."""
  shard_idx = 0
  stride = 1
  for s, dim in zip(reversed(indices), reversed(shape)):
    start = s.start or 0
    step = s.stop - start if s.stop is not None else dim
    num_shards = dim // step if step > 0 else 1
    idx = start // step if step > 0 else 0
    shard_idx += idx * stride
    stride *= num_shards
  return shard_idx


def global_shard_indices(
    arr: Any,
    sharding: Any,
    devices: Sequence[jax.Device] | None = None,
) -> tuple[int, ...]:
  """Grid index of each shard `arr` is published from, in shard order."""
  shape = tuple(arr.shape)
  index_map = sharding.devices_indices_map(shape)
  if devices is not None:
    missing = [d for d in devices if d not in index_map]
    if missing:
      raise ValueError(
          f"Explicit FFI device list has {len(missing)} device(s) missing"
          f" from array sharding devices_indices_map: {missing[:4]!r}"
      )
    return tuple(
        compute_global_shard_index(index_map[d], shape) for d in devices
    )
  shards = getattr(arr, "addressable_shards", None)
  if shards and not any(
      s.device.platform in ("proxy", "pathways") for s in shards
  ):
    return tuple(compute_global_shard_index(s.index, shape) for s in shards)
  mesh_obj = getattr(sharding, "mesh", None)
  mesh_devices = (
      mesh_obj.devices.flat if hasattr(mesh_obj, "devices") else index_map
  )
  return tuple(
      compute_global_shard_index(index_map[d], shape)
      for d in mesh_devices
      if d in index_map
  )


def tensor_metadata(
    name: str,
    arr: Any,
    layer_idx: int,
    *,
    devices: Sequence[jax.Device] | None = None,
) -> weight_sync.TensorMetadata:
  """Constructs `TensorMetadata` for `arr`."""
  sharding: Any = getattr(arr, "sharding", None)
  spec = tuple(getattr(sharding, "spec", ()) or ())
  spec = (spec + (None,) * arr.ndim)[: arr.ndim]
  if sharding is not None and hasattr(sharding, "shard_shape"):
    try:
      local = sharding.shard_shape(tuple(arr.shape))
      mesh_shape = tuple(g // l for g, l in zip(arr.shape, local))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Could not compute mesh_shape for %s from sharding: %s, falling back"
          " to 1D",
          name,
          e,
      )
      mesh_shape = (1,) * arr.ndim
  else:
    mesh_shape = (1,) * arr.ndim
  return weight_sync.TensorMetadata(
      name=name,
      shape=tuple(arr.shape),
      mesh_shape=mesh_shape,
      layout=tuple(reversed(range(arr.ndim))),
      item_size=arr.dtype.itemsize,
      layer_idx=layer_idx,
      sharding_spec=tuple(_axis_name(a) for a in spec),
      global_shard_indices=(
          global_shard_indices(arr, sharding, devices=devices)
          if hasattr(sharding, "devices_indices_map")
          else ()
      ),
  )


def delete_array_buffers(
    arrays: Sequence[Any], *, keep_ids: set[int] | None = None
) -> int:
  """Deletes live `jax.Array` device buffers not present in `keep_ids`."""
  preserved = keep_ids or set()
  deleted = 0
  seen: set[int] = set()
  for arr in arrays:
    raw = arr.get_value() if isinstance(arr, nnx.Variable) else arr
    obj_id = id(raw)
    if obj_id in preserved or obj_id in seen:
      continue
    seen.add(obj_id)
    if isinstance(raw, jax.Array) and not raw.is_deleted():
      raw.delete()
      deleted += 1
  return deleted


def _claim_key_entry(
    key: str,
    key_to_entries: Mapping[str, Sequence[tuple[int, str, Any]]],
    matched_indices: set[int],
) -> tuple[int, str, Any] | None:
  """Returns the first unmatched `(idx, name, arr)` matching `key` or suffix."""
  direct_entries = key_to_entries[key] if key in key_to_entries else ()
  for entry in direct_entries:
    if entry[0] not in matched_indices:
      return entry
  for candidate, entries in key_to_entries.items():
    if not key.endswith("." + candidate):
      continue
    for entry in entries:
      if entry[0] not in matched_indices:
        return entry
  return None


def _match_tree_leaves(
    job_name: str,
    names: Sequence[str],
    arrays: Sequence[Any],
    leaves_with_path: Sequence[tuple[Any, Any]],
) -> tuple[list[tuple[int, Any, Any]], int]:
  """Validates and pairs `arrays` against `(path, leaf)` entries pre-mutation."""
  key_to_entries: dict[str, list[tuple[int, str, Any]]] = (
      collections.defaultdict(list)
  )
  for idx, (name, arr) in enumerate(zip(names, arrays, strict=True)):
    key_to_entries[param_key(name)].append((idx, name, arr))
  matched_indices: set[int] = set()
  replacements: list[tuple[int, Any, Any]] = []

  for leaf_pos, (path, leaf) in enumerate(leaves_with_path):
    p_str = jax.tree_util.keystr(path)
    entry = _claim_key_entry(param_key(p_str), key_to_entries, matched_indices)
    if entry is None:
      continue
    idx, orig_name, new_arr = entry
    old_arr = leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf
    if hasattr(old_arr, "shape") and old_arr.shape != new_arr.shape:
      raise ValueError(
          f"Shape mismatch for parameter '{orig_name}' (runner path '{p_str}'):"
          f" target shape {old_arr.shape} vs synchronizer shape {new_arr.shape}"
      )
    if (
        hasattr(old_arr, "dtype")
        and hasattr(new_arr, "dtype")
        and old_arr.dtype != new_arr.dtype
    ):
      raise ValueError(
          f"Dtype mismatch for parameter '{orig_name}' (runner path '{p_str}'):"
          f" target dtype {old_arr.dtype} vs synchronizer dtype {new_arr.dtype}"
      )
    replacements.append((leaf_pos, old_arr, new_arr))
    matched_indices.add(idx)

  if len(matched_indices) != len(arrays):
    unmatched = [
        names[j] for j in range(len(arrays)) if j not in matched_indices
    ]
    raise RuntimeError(
        f"{job_name}: Not all synchronizer arrays were matched in target state!"
        f" Matched {len(matched_indices)} of {len(arrays)} arrays. Unmatched"
        f" {len(unmatched)} parameters, e.g.: {unmatched[:10]}"
    )
  return replacements, len(matched_indices)


def _state_kind(state: Any) -> str:
  """Classifies `state` as `'flat_dict'`, `'nnx'`, or `'pytree'`."""
  if (
      isinstance(state, dict)
      and state
      and all(
          not isinstance(v, nnx.Variable)
          and jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(v))
          for v in state.values()
      )
  ):
    return "flat_dict"
  if isinstance(state, nnx.State) or any(
      isinstance(x, nnx.Variable)
      for x in jax.tree_util.tree_leaves(
          state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      )
  ):
    return "nnx"
  return "pytree"


def apply_to_state(
    job_name: str,
    names: Sequence[str],
    arrays: Sequence[Any],
    state: Any,
    *,
    delete_replaced_buffers: bool = False,
) -> None:
  """Applies `arrays` in-place into an `nnx.State` or 1-level parameter `dict`."""
  if state is None or not arrays:
    return
  kind = _state_kind(state)
  if kind == "flat_dict":
    dict_keys = list(state.keys())
    leaves_with_path = [
        ((jax.tree_util.DictKey(k),), state[k]) for k in dict_keys
    ]
    replacements, _ = _match_tree_leaves(
        job_name, names, arrays, leaves_with_path
    )
    old_buffers = [old_arr for _, old_arr, _ in replacements]
    for leaf_pos, _, new_arr in replacements:
      state[dict_keys[leaf_pos]] = new_arr
    if delete_replaced_buffers:
      delete_array_buffers(old_buffers, keep_ids={id(a) for a in arrays})
    return

  if kind != "nnx":
    raise TypeError(
        f"{job_name}: apply_to_state requires an nnx.State, nnx.Variable tree,"
        f" or flat parameter dict; got {type(state).__name__}."
    )

  var_leaves_with_path = list(
      jax.tree_util.tree_leaves_with_path(
          state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      )
  )
  replacements, _ = _match_tree_leaves(
      job_name, names, arrays, var_leaves_with_path
  )
  # Validate every target before mutating any, so a bad leaf cannot leave the
  # state partially updated.
  targets: list[nnx.Variable] = []
  for leaf_pos, _, _ in replacements:
    _, leaf = var_leaves_with_path[leaf_pos]
    if not isinstance(leaf, nnx.Variable):
      raise TypeError(
          f"{job_name}: apply_to_state expected nnx.Variable leaf, got"
          f" {type(leaf).__name__}."
      )
    targets.append(leaf)
  for leaf, (_, _, new_arr) in zip(targets, replacements, strict=True):
    leaf.set_value(new_arr)
  if delete_replaced_buffers:
    delete_array_buffers(
        [old_arr for _, old_arr, _ in replacements],
        keep_ids={id(a) for a in arrays},
    )


def apply_to_runner(
    job_name: str,
    names: Sequence[str],
    arrays: Sequence[Any],
    runner: Any,
    *,
    delete_replaced_buffers: bool = False,
) -> None:
  """Applies `arrays` to `runner.state`, then re-derives `runner.state_leaves`.

  `state_leaves` follows `TPUModelRunner.refresh_state_leaves`: a flat leaf
  tuple when `state` is an `nnx.State`, otherwise an alias of `state` itself.

  Args:
    job_name: Label used in log and error messages.
    names: Bound parameter names, paired with `arrays`.
    arrays: Replacement arrays to install.
    runner: Model runner exposing `state` and `state_leaves`.
    delete_replaced_buffers: Whether to free the replaced device buffers.
  """
  if runner is None or not arrays:
    return
  if runner.state is None:
    raise ValueError(
        f"{job_name}: runner does not have a valid 'state' attribute."
    )

  if _state_kind(runner.state) == "pytree":
    runner_leaves_with_path = list(
        jax.tree_util.tree_leaves_with_path(runner.state)
    )
    new_leaves = [leaf for _, leaf in runner_leaves_with_path]
    replacements, _ = _match_tree_leaves(
        job_name, names, arrays, runner_leaves_with_path
    )
    for leaf_pos, _, new_arr in replacements:
      new_leaves[leaf_pos] = new_arr
    runner.state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(runner.state), new_leaves
    )
    if delete_replaced_buffers:
      delete_array_buffers(
          [old_arr for _, old_arr, _ in replacements],
          keep_ids={id(a) for a in arrays},
      )
  else:
    apply_to_state(
        job_name,
        names,
        arrays,
        runner.state,
        delete_replaced_buffers=delete_replaced_buffers,
    )

  runner.state_leaves = (
      tuple(jax.tree_util.tree_leaves(runner.state))
      if isinstance(runner.state, nnx.State)
      else runner.state
  )
  logging.info(
      "%s apply_to_runner: applied %d arrays to runner state.",
      job_name,
      len(arrays),
  )


def apply_to_sampler(
    job_name: str,
    names: Sequence[str],
    arrays: Sequence[Any],
    sampler: SyncedSamplerTarget,
    *,
    delete_replaced_buffers: bool = False,
) -> None:
  """Installs `arrays` into `sampler.model_runner` or `sampler.transformer_state`."""
  runner = sampler.model_runner
  if runner is not None:
    apply_to_runner(
        job_name,
        names,
        arrays,
        runner,
        delete_replaced_buffers=delete_replaced_buffers,
    )
    return
  apply_to_state(
      job_name,
      names,
      arrays,
      sampler.transformer_state,
      delete_replaced_buffers=delete_replaced_buffers,
  )


@jax.jit
def _fused_l1_sum(arr: jax.Array) -> jax.Array:
  """float32 L1 sum fused into one kernel, so no full-size temporary exists."""
  return jnp.sum(jnp.abs(arr), dtype=jnp.float32)


def compute_checksums(
    names: Sequence[str],
    arrays: Sequence[Any],
    *,
    sample: int | None = 3,
    canonical_keys: bool = False,
) -> dict[str, float | int]:
  """Computes per-tensor float32 abs-sums and exact aggregate counts."""
  if not arrays:
    return {
        "__grand_total__": 0.0,
        "__tensor_count__": 0,
        "__element_count__": 0,
    }
  per_tensor = [0.0] * len(arrays)
  jax_groups: dict[jax.sharding.Sharding, list[tuple[int, jax.Array]]] = (
      collections.defaultdict(list)
  )
  for idx, arr in enumerate(arrays):
    if isinstance(arr, jax.Array):
      scalar = _fused_l1_sum(arr)
      jax_groups[scalar.sharding].append((idx, scalar))
    else:
      per_tensor[idx] = float(np.sum(np.abs(np.asarray(arr, dtype=np.float32))))

  for entries in jax_groups.values():
    stacked = np.asarray(
        jax.device_get(jnp.stack([scalar for _, scalar in entries])),
        dtype=np.float64,
    )
    for (idx, _), value in zip(entries, stacked.tolist(), strict=True):
      per_tensor[idx] = float(value)

  paired = [
      (param_key(name), name, total)
      for name, total in zip(names, per_tensor, strict=True)
  ]
  selected = (
      paired
      if sample is None
      else sorted(paired, key=lambda item: item[0])[:sample]
  )
  out: dict[str, float | int] = {
      (canon if canonical_keys else raw_name): total
      for canon, raw_name, total in selected
  }
  out["__grand_total__"] = float(math.fsum(per_tensor))
  out["__tensor_count__"] = len(arrays)
  out["__element_count__"] = int(sum(int(a.size) for a in arrays))
  return out


def verify_weight_checksums(
    src_checksums: Mapping[str, float | int],
    dst_checksums: Mapping[str, float | int],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
  """Verifies source and destination weight checksum dictionaries match."""
  for count_key in ("__tensor_count__", "__element_count__"):
    if src_checksums[count_key] != dst_checksums[count_key]:
      raise RuntimeError(
          f"Weight checksum {count_key} mismatch: source="
          f"{src_checksums[count_key]} vs destination="
          f"{dst_checksums[count_key]}"
      )

  src_grand = float(src_checksums["__grand_total__"])
  dst_grand = float(dst_checksums["__grand_total__"])
  if not math.isclose(src_grand, dst_grand, rel_tol=rtol, abs_tol=atol):
    raise RuntimeError(
        f"Weight checksum __grand_total__ mismatch: source={src_grand:.6g} vs"
        f" destination={dst_grand:.6g}"
    )

  src_tensors = {
      param_key(k): float(v)
      for k, v in src_checksums.items()
      if k not in _CHECKSUM_META_KEYS
  }
  dst_tensors = {
      param_key(k): float(v)
      for k, v in dst_checksums.items()
      if k not in _CHECKSUM_META_KEYS
  }
  missing = sorted(set(src_tensors) ^ set(dst_tensors))
  if missing:
    raise RuntimeError(
        f"Weight checksum key set mismatch ({len(missing)} keys, e.g."
        f" {missing[:5]!r})"
    )
  mismatches = [
      f"{k} (src={src_tensors[k]:.6g}, dst={dst_tensors[k]:.6g})"
      for k in sorted(src_tensors)
      if not math.isclose(
          src_tensors[k], dst_tensors[k], rel_tol=rtol, abs_tol=atol
      )
  ]
  if mismatches:
    raise RuntimeError(
        f"Weight checksum tensor mismatch on {len(mismatches)} tensor(s):"
        f" {', '.join(mismatches[:5])}"
    )
