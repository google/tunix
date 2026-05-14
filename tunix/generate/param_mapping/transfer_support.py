"""Transfer support helpers shared by parameter mapping APIs."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from absl import logging
from flax import nnx
from flax import traverse_util
import jax
import numpy as np

from tunix.generate.param_mapping.runtime import (
    _get_n_shards,
    _interleave_moe_weights,
)
from tunix.generate.param_mapping.spec import derive_moe_source_keys


def build_flat_dict(
    flat_state: Iterator[tuple[tuple[str, ...], nnx.State]],
    mappings: Dict[str, tuple[str, tuple[int, ...]]],
):
  """Builds a source-key-indexed flat mapping view from a destination flat state.

  Legacy explicit mappings are expressed from source path to destination path.
  During transfer preparation we walk the destination flat state, match each
  destination path against the mapping regexes, and invert that information into
  a dictionary keyed by the effective source path. Layer-sharded destinations are
  grouped so later code can unroll scanned layers into one entry per concrete
  destination layer.

  Example:
    Input:
      mappings = {'encoder.weight': ('decoder.layers.*.weight', ('layer', None))}
      flat_state contains:
        ('decoder', 'layers', '0', 'weight')
        ('decoder', 'layers', '1', 'weight')

    Output:
      {
          'encoder.weight': (
              [param_for_layer_0, param_for_layer_1],
              ['decoder.layers.0.weight', 'decoder.layers.1.weight'],
              ('layer', None),
          )
      }
  """
  new_flat_dict = {}
  compiled_mappings = []

  for src, (tgt, sharding) in mappings.items():
    if any(char in tgt for char in ['|', '(', ')']):
      pattern = '^' + tgt + '$'
    else:
      pattern = '^' + re.escape(tgt).replace('\\.\\*', r'\.(\d+)') + '$'
    compiled_mappings.append((src, re.compile(pattern), sharding))

  for keys, v in flat_state:
    path = '.'.join(str(key) for key in keys)
    mapped = False
    for src, regex, sharding in compiled_mappings:
      matched = regex.match(path)
      if matched:
        wildcards = matched.groups()

        src_parts = []
        wc_index = 0
        for part in src.split('.'):
          if part == '*':
            src_parts.append(wildcards[wc_index])
            wc_index += 1
          else:
            src_parts.append(part)
        actual_src = '.'.join(src_parts)

        if sharding and 'layer' in sharding:
          if actual_src not in new_flat_dict:
            new_flat_dict[actual_src] = ([], [], sharding)

          layer_number = int(wildcards[0]) if wildcards else 0
          new_flat_dict[actual_src][0].append((layer_number, v))
          new_flat_dict[actual_src][1].append((layer_number, path))
        else:
          new_flat_dict[actual_src] = v, path, sharding

        mapped = True
        break
    if not mapped:
      logging.warning('!!! No mapping for flat state: %s', path)

  for key, (layers, paths, sharding) in new_flat_dict.items():
    if isinstance(layers, list):
      layers.sort(key=lambda x: x[0])
      paths.sort(key=lambda x: x[0])
      values = [v for _, v in layers]
      paths = [p for _, p in paths]
      new_flat_dict[key] = (values, paths, sharding)

  return new_flat_dict


def get_layer_axis_from_sharding_spec(sharding_spec) -> Optional[int]:
  """Returns the logical axis marked as `layer` in mapping metadata.

  Legacy explicit mappings use lightweight sharding metadata rather than full
  JAX sharding objects. This helper extracts the scan-layer axis from that
  representation so source tensors can be unrolled correctly.
  """
  if isinstance(sharding_spec, (list, tuple)):
    for i, spec in enumerate(sharding_spec):
      if spec == 'layer':
        return i
  return None


def unroll_scanned_layers(
    src_state: Any,
    src_to_tgt_map: Dict,
) -> Dict[Tuple[str, str], Tuple[Any, Any]]:
  """Expands scanned source tensors into one mapping entry per target layer.

  When the destination mapping metadata says a source tensor is sharded over a
  logical `layer` axis, this helper slices that axis and pairs each slice with
  the corresponding concrete destination leaf. Non-scanned entries pass through
  unchanged.

  Example:
    Input:
      src_state contains 'encoder.weight' with shape (2, 4)
      src_to_tgt_map['encoder.weight'] = (
          [tgt0, tgt1],
          ['decoder.layers.0.weight', 'decoder.layers.1.weight'],
          ('layer', None),
      )

    Output:
      {
          ('encoder.weight', 'decoder.layers.0.weight'): (src_val[0], tgt0),
          ('encoder.weight', 'decoder.layers.1.weight'): (src_val[1], tgt1),
      }
  """
  unscanned_flat = {}

  for src_keys, src_val in src_state.flat_state():
    src_key = '.'.join(str(k) for k in src_keys)

    if 'rng' in src_key:
      logging.debug('Skipping RNG parameter: %s', src_key)
      continue

    if src_key not in src_to_tgt_map:
      logging.error('No mapping for source key: %s', src_key)
      continue

    tgt_param, tgt_path, sharding_spec = src_to_tgt_map[src_key]
    layer_axis = get_layer_axis_from_sharding_spec(sharding_spec)

    if layer_axis is not None:
      num_layers = src_val.value.shape[layer_axis]
      for i in range(num_layers):
        idx = [slice(None)] * src_val.value.ndim
        idx[layer_axis] = i
        layer_val = src_val.value[tuple(idx)]
        layer_key = tgt_path[i]
        unscanned_flat[(src_key, layer_key)] = (layer_val, tgt_param[i])
    else:
      unscanned_flat[(src_key, tgt_path)] = (src_val.value, tgt_param)

  return unscanned_flat


def sync_tied_lm_head_if_needed(
    tgt_flat_list: List[Tuple[Tuple[str, ...], Any]],
    transferred_target_keys: set[str],
) -> None:
  """Restores tied `lm_head` semantics when embeddings were transferred alone.

  Some destination models represent tied embeddings and `lm_head` as separate
  leaves even though they should share values. If `lm_head` was explicitly
  transferred, this helper does nothing. Otherwise it mirrors embedding weights
  into `lm_head` when the shapes indicate the target expects tied behavior.
  """
  if any(key.endswith('lm_head') for key in transferred_target_keys):
    return

  embed_param = None
  lm_head_param = None
  for flat_key, tgt_param in tgt_flat_list:
    if flat_key[-1:] == ('embedding',):
      embed_param = tgt_param
    elif flat_key[-1:] == ('lm_head',):
      lm_head_param = tgt_param

  if embed_param is None or lm_head_param is None:
    return
  if not hasattr(embed_param, 'value') or not hasattr(lm_head_param, 'value'):
    return
  if embed_param.value.shape != lm_head_param.value.shape:
    return

  lm_head_param.value = embed_param.value


def fuse_moe_weights(
    src_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
    tgt_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
) -> Dict[Tuple[str, ...], jax.Array | np.ndarray]:
  """Pre-fuses unscanned MoE `wi_0`/`wi_1` pairs when the target expects `wi`.

  Structural transfer can often simplify planning by rewriting unfused source
  keys into the fused key shape expected by the target before any planner rules
  run. This helper performs that eager rewrite only when the target tree makes
  the fused expectation explicit.

  Example:
    Input:
      src_flat has:
        ('decoder', 'mlp', 'wi_0')
        ('decoder', 'mlp', 'wi_1')
      tgt_flat has:
        ('decoder', 'mlp', 'wi')

    Output:
      A new source dict where `wi_0` and `wi_1` are removed and replaced by
      one fused entry at ('decoder', 'mlp', 'wi').
  """
  new_src_flat = dict(src_flat)
  for tgt_key in tgt_flat.keys():
    if not tgt_key or tgt_key[-1] != 'wi':
      continue
    wi_0_key, wi_1_key = derive_moe_source_keys(tgt_key)
    if wi_0_key not in new_src_flat or wi_1_key not in new_src_flat:
      continue
    wi_0 = new_src_flat.pop(wi_0_key)
    wi_1 = new_src_flat.pop(wi_1_key)
    tgt_val = tgt_flat[tgt_key]
    mismatched_axes = [
        i for i, (s, t) in enumerate(zip(wi_0.shape, tgt_val.shape)) if s != t
    ]
    axis = mismatched_axes[-1] if mismatched_axes else len(tgt_val.shape) - 1
    n_shards = _get_n_shards(tgt_val, axis)
    logging.info(
        'Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d',
        '.'.join(str(k) for k in tgt_key),
        wi_0.shape,
        wi_1.shape,
        tgt_val.shape,
        axis,
    )
    new_src_flat[tgt_key] = _interleave_moe_weights(
        wi_0, wi_1, tgt_val.shape, n_shards, axis=axis
    )
    del wi_0, wi_1
  return new_src_flat


def collect_src_buffer_ids(
    src_flat: Mapping[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
) -> Optional[set[int]]:
  """Collects physical source buffer identities for alias-aware deletion.

  Destination buffers must not be deleted if they alias the same physical data
  as one of the source arrays. This helper gathers backend buffer pointers where
  that information is available. Backends that do not expose compatible buffer
  pointers return `None`, signaling that alias-aware deletion should be skipped.
  """
  ids: set[int] = set()
  for v in src_flat.values():
    arr = v.value if hasattr(v, 'value') else v
    if not hasattr(arr, 'addressable_shards'):
      continue
    for shard in arr.addressable_shards:
      try:
        ids.add(shard.data.unsafe_buffer_pointer())
      except jax.errors.JaxRuntimeError as e:
        if 'PjRt-compatible backend only' in str(e):
          return None
        raise e
      except Exception:
        pass
  return ids


def delete_target_buffers(
    spec_flat: Mapping[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
    src_flat: Mapping[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
) -> None:
  """Deletes destination buffers that are safe to free before resharding.

  This is a peak-memory optimization. When alias information is available, the
  helper preserves any destination buffer that points at the same underlying
  storage as a source shard.
  """
  src_buffer_ids = collect_src_buffer_ids(src_flat)
  for tgt_val in spec_flat.values():
    tgt_arr = tgt_val.value if hasattr(tgt_val, 'value') else tgt_val
    if not hasattr(tgt_arr, 'delete') or getattr(
        tgt_arr, 'is_deleted', lambda: False
    )():
      continue
    if src_buffer_ids is not None and hasattr(tgt_arr, 'addressable_shards'):
      aliases_source = any(
          shard.data.unsafe_buffer_pointer() in src_buffer_ids
          for shard in tgt_arr.addressable_shards
      )
      if aliases_source:
        continue
    tgt_arr.delete()


def snapshot_dst_sharding(
    arr: jax.Array | np.ndarray,
) -> jax.sharding.Sharding:
  """Builds a stable destination sharding leaf for a reshard target tree.

  Some sharding objects can be used directly, while others need to be copied
  into an explicit `NamedSharding` instance so the reshard helper receives a
  pure sharding tree rather than a live array tree.
  """
  s = arr.sharding
  if isinstance(
      s, (jax.sharding.NamedSharding, jax.sharding.SingleDeviceSharding)
  ):
    return s
  return jax.sharding.NamedSharding(s.mesh, s.spec, memory_kind=s.memory_kind)  # type: ignore[attr-defined]


def reshard_in_chunks(
    src_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
    spec_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
    reshard_fn: Callable[..., Mapping[str, Any]],
    chunk_size: int,
    delete_spec_buffers: bool = False,
) -> Dict[Tuple[str, ...], jax.Array | np.ndarray]:
  """Reshards a flat update mapping in sequential chunks to lower peak memory.

  Each chunk is converted into the nested tree shape expected by `reshard_fn`,
  executed independently, blocked until ready, and then flattened back into the
  aggregate result. The source dictionary is consumed incrementally so finished
  chunks can be released promptly.

  Example:
    Input:
      src_flat = {
          ('decoder', 'weight'): ...,
          ('decoder', 'bias'): ...,
      }
      chunk_size = 1

    Output:
      Two sequential `reshard_fn(...)` calls, each handling one key, and one
      merged flat result containing both resharded leaves.
  """
  keys = list(src_flat.keys())
  resharded: Dict[Tuple[str, ...], jax.Array | np.ndarray] = {}
  for start in range(0, len(keys), chunk_size):
    chunk_keys = keys[start : start + chunk_size]
    chunk_src_flat = {}
    chunk_spec_flat = {}
    chunk_dst_shardings_flat = {}
    for k in chunk_keys:
      src_val = src_flat.pop(k)
      tgt_val = spec_flat[k]
      chunk_src_flat[k] = src_val
      chunk_spec_flat[k] = tgt_val
      tgt_arr = tgt_val.value if hasattr(tgt_val, 'value') else tgt_val
      chunk_dst_shardings_flat[k] = snapshot_dst_sharding(tgt_arr)

    if delete_spec_buffers:
      delete_target_buffers(chunk_spec_flat, chunk_src_flat)

    chunk_src = traverse_util.unflatten_dict(chunk_src_flat)
    chunk_dst_shardings = traverse_util.unflatten_dict(chunk_dst_shardings_flat)
    chunk_resharded = reshard_fn(source=chunk_src, target=chunk_dst_shardings)
    jax.block_until_ready(chunk_resharded)
    resharded.update(traverse_util.flatten_dict(chunk_resharded))

    del (
        chunk_src,
        chunk_dst_shardings,
        chunk_resharded,
        chunk_src_flat,
        chunk_spec_flat,
        chunk_dst_shardings_flat,
    )
  return resharded
