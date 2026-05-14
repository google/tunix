"""Runtime helper functions for parameter mapping execution."""

from __future__ import annotations

import functools
import math
import re
from typing import Optional, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np


class ShapeMismatchError(ValueError):
  """Raised when source and target shapes are incompatible."""

  pass


def _resolve_transpose_axes(
    src_key: str,
    transpose_keys: Optional[dict[str, Tuple[int, ...]]],
    rollout_engine: Optional[str],
) -> Optional[Tuple[int, ...]]:
  """Returns the transpose axes configured for a source key, if any.

  Example:
    Input:
      src_key = 'decoder.weight'
      transpose_keys = {'weight': (1, 0)}

    Output:
      (1, 0)
  """
  if not transpose_keys:
    return None

  last_key = src_key.split('.')[-1]
  all_key = src_key
  if last_key in transpose_keys and 'lora' not in last_key:
    return transpose_keys[last_key]
  if all_key in transpose_keys and 'lora' not in all_key:
    return transpose_keys[all_key]

  if rollout_engine == 'sglang_jax' and 'lora' in all_key:
    for r_key in transpose_keys:
      if re.compile(rf'{r_key}').match(all_key):
        return transpose_keys[r_key]

  return None


def _align_shape(
    val: jnp.ndarray,
    tgt_shape: Tuple[int, ...],
    src_key: str,
    rollout_engine: Optional[str] = None,
    **kwargs,
) -> jnp.ndarray:
  """Align source value shape to target shape through padding or repeating."""
  del rollout_engine
  if val.shape == tgt_shape:
    return val

  additional_reshape = False
  new_tgt_shape = tgt_shape
  if len(val.shape) != len(tgt_shape):
    if re.compile(r'layers\..*\.attn\.(q|k|v)_bias').match(src_key):
      if math.prod(tgt_shape) == math.prod(val.shape):
        new_shape = (tgt_shape[0], val.shape[0] // tgt_shape[0])
        logging.debug(
            'Reshaping attention bias on %s: %s -> %s',
            src_key,
            val.shape,
            new_shape,
        )
        return jnp.reshape(val, new_shape)
      assert (
          val.shape[0] == kwargs['num_kv_heads'] * kwargs['head_dim']
          and tgt_shape[0] % kwargs['num_kv_heads'] == 0
          and tgt_shape[1] == kwargs['head_dim']
      ), (
          f'Unexpected attention bias shape: {val.shape} and target shape:'
          f' {tgt_shape}'
      )
      val = jnp.reshape(val, (kwargs['num_kv_heads'], kwargs['head_dim']))
      new_tgt_shape = tgt_shape
    elif re.compile(r'layers\..*\.attn\.(q|k|v|o)_proj').match(src_key):
      if math.prod(tgt_shape) == math.prod(val.shape):
        logging.debug(
            'Reshaping attention proj on %s: %s -> %s',
            src_key,
            val.shape,
            tgt_shape,
        )
        return jnp.reshape(val, tgt_shape)
      additional_reshape = True
      assert len(val.shape) == 3 and len(tgt_shape) == 2, (
          f'Unexpected attention proj shape: {val.shape} and target shape:'
          f' {tgt_shape}'
      )
      if 'o_proj' in src_key:
        padded_dim = (val.shape[-2] + 127) // 128 * 128
        repeated_dim = tgt_shape[-1] // padded_dim
        new_tgt_shape = tgt_shape[:-1] + (padded_dim, repeated_dim)
      else:
        padded_dim = (val.shape[-1] + 127) // 128 * 128
        repeated_dim = tgt_shape[-1] // padded_dim
        new_tgt_shape = tgt_shape[:-1] + (repeated_dim, padded_dim)
    else:
      raise ShapeMismatchError(
          f'Rank mismatch for {src_key}: {val.shape} vs {tgt_shape}'
      )
  elif re.compile(r'layers\..*\.attn\.(k|v)_bias').match(src_key):
    logging.debug('Handling 1-D KV bias for %s in SGLangJAX rollout.', src_key)
    assert tgt_shape[0] > val.shape[0] and tgt_shape[0] % val.shape[0] == 0, (
        f'Unexpected attention bias shape: {val.shape} and target shape:'
        f' {tgt_shape}'
    )
    repeat_factor = tgt_shape[0] // val.shape[0]
    logging.debug(
        'Replicating 1-D KV bias on %s: %s -> %s (repeat x%d per head)',
        src_key,
        val.shape,
        tgt_shape,
        repeat_factor,
    )
    val_2d = jnp.reshape(val, (kwargs['num_kv_heads'], kwargs['head_dim']))
    val_2d = jnp.repeat(val_2d, repeat_factor, axis=0)
    return jnp.reshape(val_2d, tgt_shape)

  attention_patterns = [
      r'.*(q|k|v|o)_proj.*',
      r'.*(q|k|v|o)_bias.*',
      r'.*(key|query|value|output).*',
  ]
  if not any(re.match(pattern, src_key) for pattern in attention_patterns):
    raise ShapeMismatchError(
        f'Shape mismatch for non-attention weight {src_key}: '
        f'{val.shape} vs {tgt_shape}. Padding/repetition only supported '
        'for attention weights.'
    )

  original_shape = val.shape
  pad_width = []
  repeat_ops = []
  for i, (src_dim, tgt_dim) in enumerate(zip(val.shape, new_tgt_shape)):
    if src_dim < tgt_dim:
      if ('o_proj' not in src_key and i == len(val.shape) - 1) or (
          'o_proj' in src_key and i == len(val.shape) - 2
      ):
        pad_width.append((0, tgt_dim - src_dim))
      else:
        repeat_factor = tgt_dim // src_dim
        if tgt_dim % src_dim != 0:
          raise ShapeMismatchError(
              f'Target dimension {tgt_dim} is not divisible by source '
              f'dimension {src_dim} for {src_key}'
          )
        repeat_ops.append((i, repeat_factor))
        pad_width.append((0, 0))
    elif src_dim > tgt_dim:
      raise ShapeMismatchError(
          f'Cannot shrink dimension {i} for {src_key}: {src_dim} -> {tgt_dim}'
      )
    else:
      pad_width.append((0, 0))

  logging.info(
      'Resolved shape mismatch on %s: %s -> %s',
      src_key,
      original_shape,
      tgt_shape,
  )

  for axis, repeat_factor in repeat_ops:
    val = jnp.repeat(val, repeat_factor, axis=axis)
  val = jnp.pad(val, pad_width)

  if additional_reshape:
    assert math.prod(val.shape) == math.prod(
        tgt_shape
    ), f'After align, shape mismatch on {src_key}: {val.shape} vs {tgt_shape}'
    val = jnp.reshape(val, tgt_shape)
  return val


def _apply_dtype_cast(
    val: jax.Array | np.ndarray, tgt_dtype: jnp.dtype, src_key: str
) -> jax.Array | np.ndarray:
  """Casts a value to the target dtype while logging the first mismatch.

  The transfer engine prefers to preserve source values exactly unless a target
  leaf requires a different dtype. Logging only the first mismatch keeps noisy
  model-wide transfers readable while still surfacing unexpected casts.
  """
  if val.dtype != tgt_dtype:
    logging.log_first_n(
        logging.WARNING,
        'Type mismatch on %s: %s -> %s',
        1,
        src_key,
        val.dtype,
        tgt_dtype,
    )
    return val.astype(tgt_dtype)
  return val


def _shapes_are_repeatable(
    candidate_shape: tuple[int, ...],
    tgt_shape: tuple[int, ...],
) -> bool:
  """Returns whether every candidate dimension can tile cleanly to target.

  This is used when inferring which axis of a scanned tensor is the scan axis:
  removing that axis should leave a per-layer shape whose dimensions either
  already match the target or can be repeated to it exactly.
  """
  if len(candidate_shape) != len(tgt_shape):
    return False
  for s, t in zip(candidate_shape, tgt_shape):
    if s > t or t % s != 0:
      return False
  return True


def _unstack_scanned_param(
    src_val: jax.Array | np.ndarray,
    tgt_val: jax.Array | np.ndarray,
    key_path: str,
    scan_axis: Optional[int] = None,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Unstacks a scanned tensor into one per-layer tensor per scan index.

  When the scan axis is known, or can be inferred from the target shape, the
  function moves that axis to the front and returns one slice per layer. If the
  tensor is already per-layer shaped, or if the scan axis cannot be resolved
  safely, the original value is returned as a one-element tuple.

  Example:
    Input:
      src_val.shape = (2, 3, 4)
      tgt_val.shape = (2, 4)
      scan_axis = 1

    Output:
      (
          src_val[:, 0, :],
          src_val[:, 1, :],
          src_val[:, 2, :],
      )
  """
  if not (hasattr(src_val, 'shape') and hasattr(tgt_val, 'shape')):
    return (src_val,)

  src_shape = src_val.shape
  tgt_shape = tgt_val.shape
  if src_shape == tgt_shape:
    return (src_val,)

  if len(src_shape) == len(tgt_shape) + 1:
    if scan_axis is None:
      for i in range(len(src_shape)):
        candidate = src_shape[:i] + src_shape[i + 1 :]
        if _shapes_are_repeatable(candidate, tgt_shape):
          scan_axis = i
          break

    if scan_axis is not None:
      if scan_axis != 0:
        perm = (scan_axis,) + tuple(
            i for i in range(len(src_shape)) if i != scan_axis
        )
        if hasattr(src_val, 'transpose'):
          src_val = src_val.transpose(perm)
        elif isinstance(src_val, np.ndarray):
          src_val = np.transpose(src_val, perm)

      try:
        if hasattr(jax, 'unstack'):
          return jax.unstack(src_val)
        if hasattr(jnp, 'unstack'):
          return jnp.unstack(src_val)
        return [src_val[i] for i in range(src_val.shape[0])]
      except Exception as e:
        logging.debug(
            "Failed to unstack parameter '%s'. Error: %s. Using original.",
            key_path,
            e,
        )
        return (src_val,)

    logging.warning(
        "Shape mismatch in scanned param '%s'. Src: %s, Tgt: %s. Cannot"
        ' determine scan axis.',
        key_path,
        src_shape,
        tgt_shape,
    )

  return (src_val,)


_MOE_MLP_WEIGHTS = frozenset({'wi', 'wi_0', 'wi_1'})


def _partition_size(
    partition: Optional[str | Tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> int:
  """Computes the total mesh size covered by one partition-spec entry."""
  if partition is None:
    return 1
  names = (partition,) if isinstance(partition, str) else tuple(partition)
  size = 1
  for n in names:
    size *= mesh.shape[n]
  return size


def _spec_at_axis(
    sharding: Optional[jax.sharding.Sharding],
    axis: int,
) -> Optional[str | Tuple[str, ...]]:
  """Returns the partition-spec entry for one array axis, if available."""
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return None
  spec = sharding.spec
  return spec[axis] if axis < len(spec) else None


def _get_n_shards(arr: jax.Array | np.ndarray, axis: int) -> int:
  """Returns how many logical shards partition one axis of an array."""
  sharding = getattr(arr, 'sharding', None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    return _partition_size(_spec_at_axis(sharding, axis), sharding.mesh)
  return 1


@functools.partial(jax.jit, static_argnames=('pad_specs',))
def _jit_zero_pad_axes(arr, pad_specs):
  """Pads each requested axis in a shard-aware way under JIT.

  For sharded MoE tensors we pad within each shard chunk instead of appending a
  single tail pad on the full axis. That preserves the per-shard layout the
  target fused kernel expects.
  """
  out = arr
  for axis, n_shards, per_shard_extra in pad_specs:
    if per_shard_extra <= 0:
      continue
    src_dim = out.shape[axis]
    src_chunk_size = src_dim // n_shards
    split_shape = list(out.shape)
    split_shape.insert(axis + 1, src_chunk_size)
    split_shape[axis] = n_shards
    arr_split = out.reshape(split_shape)
    pad_width = [(0, 0)] * arr_split.ndim
    pad_width[axis + 1] = (0, per_shard_extra)
    arr_padded = jnp.pad(arr_split, pad_width)
    final_shape = list(out.shape)
    final_shape[axis] = src_dim + per_shard_extra * n_shards
    out = arr_padded.reshape(final_shape)
  return out


@functools.partial(jax.jit, static_argnames=('repeats',))
def _jit_repeat_axes(arr, repeats):
  """Repeats array axes under JIT using a list of `(axis, count)` specs."""
  out = arr
  for axis, count in repeats:
    out = jnp.repeat(out, count, axis=axis)
  return out


def _align_per_axis(
    arr: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    tgt_sharding: Optional[jax.sharding.Sharding],
    key_path: str,
) -> jax.Array | np.ndarray:
  """Aligns a tensor to a target shape one axis at a time.

  Non-MoE weights are aligned by integer repetition. MoE MLP weights instead
  may require shard-aware padding so that each shard grows uniformly. Any shape
  change that would require shrinking, fractional repetition, or unsupported
  padding semantics raises `ShapeMismatchError`.

  Example:
    Input:
      arr.shape = (2, 4)
      tgt_shape = (8, 4)
      key_path = 'layers.0.attn.q_proj'

    Output:
      A tensor with shape (8, 4) formed by repeating axis 0 four times.
  """
  if not hasattr(arr, 'shape'):
    return arr
  if arr.shape == tgt_shape:
    return arr
  if len(arr.shape) != len(tgt_shape):
    raise ShapeMismatchError(
        f'Rank mismatch for {key_path}: src={arr.shape} vs tgt={tgt_shape}'
    )

  mismatches = []
  for axis, (s, t) in enumerate(zip(arr.shape, tgt_shape)):
    if s == t:
      continue
    if t < s:
      raise ShapeMismatchError(
          f'Cannot shrink axis {axis} for {key_path}: src={s} -> tgt={t}'
      )
    mismatches.append((axis, s, t))
  if not mismatches:
    return arr

  last_key = key_path.split('.')[-1]
  if last_key in _MOE_MLP_WEIGHTS:
    if isinstance(tgt_sharding, jax.sharding.NamedSharding):
      mesh = tgt_sharding.mesh
      pad_specs = []
      for axis, s, t in mismatches:
        n_shards = _partition_size(_spec_at_axis(tgt_sharding, axis), mesh)
        if t % n_shards != 0:
          raise ValueError(
              f'Target dimension {t} on axis {axis} for {key_path} is not '
              f'divisible by n_shards={n_shards}; the target shape itself '
              f'is misconfigured for the requested sharding.'
          )
        if (t - s) % n_shards != 0 or s % n_shards != 0:
          raise ValueError(
              f'Cannot interleave pad axis {axis} for {key_path}: src_dim '
              f'({s}) or extra ({t - s}) is not cleanly divisible by '
              f'n_shards ({n_shards}). Ensure the source tensor is evenly '
              f'partitionable.'
          )
        pad_specs.append((axis, n_shards, (t - s) // n_shards))
    else:
      pad_specs = [(axis, 1, t - s) for axis, s, t in mismatches]
    return _jit_zero_pad_axes(arr, tuple(pad_specs))

  repeats = []
  for axis, s, t in mismatches:
    if t % s != 0:
      raise ShapeMismatchError(
          f'Cannot align axis {axis} for {key_path}: src={s} -> tgt={t} '
          f'is not an integer multiple and the key is not a recognized '
          f'MoE pattern.'
      )
    repeats.append((axis, t // s))
  return _jit_repeat_axes(arr, tuple(repeats))


def _interleave_moe_weights(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    n_shards: int,
    axis: Optional[int] = None,
) -> jax.Array | np.ndarray:
  """Pads and interleaves unfused MoE halves into one fused target tensor.

  Each expert half is chunked by shard, padded to the per-shard target width,
  and then concatenated so the resulting fused tensor matches the layout
  expected by downstream fused-MoE kernels.

  Example:
    Input:
      wi_0.shape = (2, 2)
      wi_1.shape = (2, 2)
      tgt_shape = (2, 4)

    Output:
      A fused tensor with shape (2, 4) whose last axis contains the padded,
      interleaved contributions from `wi_0` and `wi_1`.
  """
  if axis is None:
    axis = len(tgt_shape) - 1

  target_half_dim = tgt_shape[axis] // 2

  def _pad_and_chunk(arr):
    current_total_size = arr.shape[axis]
    chunk_size = current_total_size // n_shards
    target_chunk_size = target_half_dim // n_shards
    new_shape = list(arr.shape)
    new_shape[axis] = n_shards
    new_shape.insert(axis + 1, chunk_size)
    arr_reshaped = arr.reshape(new_shape)

    pad_amount = target_chunk_size - chunk_size
    if pad_amount > 0:
      pad_widths = [(0, 0)] * arr_reshaped.ndim
      pad_widths[axis + 1] = (0, pad_amount)
      arr_reshaped = jnp.pad(arr_reshaped, pad_widths)
    return arr_reshaped

  p_wi_0 = _pad_and_chunk(wi_0)
  p_wi_1 = _pad_and_chunk(wi_1)
  combined = jnp.concatenate([p_wi_0, p_wi_1], axis=axis + 1)
  return combined.reshape(tgt_shape)


def _align_to_model_shape(
    src_val: jax.Array | np.ndarray,
    tgt_val: jax.Array | np.ndarray,
    key_path: str,
) -> jax.Array | np.ndarray:
  """Aligns a source tensor directly against a concrete target leaf value.

  This is the runtime bridge used by `repeat_to_target`-style transforms: it
  reads target shape and sharding from the actual destination leaf and forwards
  the work to `_align_per_axis(...)`.
  """
  if not (hasattr(src_val, 'shape') and hasattr(tgt_val, 'shape')):
    return src_val
  if src_val.shape == tgt_val.shape:
    return src_val

  tgt_sharding = getattr(tgt_val, 'sharding', None)
  return _align_per_axis(src_val, tgt_val.shape, tgt_sharding, key_path)


def _scanned_sharding_from_per_layer(
    per_layer_sharding: Optional[jax.sharding.Sharding],
    scan_axis: int,
) -> Optional[jax.sharding.NamedSharding]:
  """Lifts a per-layer sharding spec to the equivalent scanned tensor spec."""
  if not isinstance(per_layer_sharding, jax.sharding.NamedSharding):
    return None
  spec = list(per_layer_sharding.spec)
  spec.insert(scan_axis, None)
  return jax.sharding.NamedSharding(
      per_layer_sharding.mesh,
      jax.sharding.PartitionSpec(*spec),
      memory_kind=per_layer_sharding.memory_kind,
  )


def _bulk_align_and_unstack(
    arr: jax.Array | np.ndarray,
    scan_axis: int,
    per_layer_tgt_val: jax.Array | np.ndarray,
    key_path: str,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Aligns a full scanned tensor once, then unstacks all per-layer slices.

  When every destination layer would require the same reshape or padding, it is
  cheaper to align the scanned tensor once and then unstack than to repeat the
  same work for every layer independently.
  """
  per_layer_shape = per_layer_tgt_val.shape
  scanned_tgt_shape = (
      per_layer_shape[:scan_axis]
      + (arr.shape[scan_axis],)
      + per_layer_shape[scan_axis:]
  )
  scanned_tgt_sharding = _scanned_sharding_from_per_layer(
      getattr(per_layer_tgt_val, 'sharding', None), scan_axis
  )

  if arr.shape == scanned_tgt_shape:
    return tuple(jnp.unstack(arr, axis=scan_axis))

  aligned = _align_per_axis(
      arr, scanned_tgt_shape, scanned_tgt_sharding, key_path
  )
  return tuple(jnp.unstack(aligned, axis=scan_axis))


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def _jit_fuse_and_unstack_moe(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    scan_axis: int,
    num_layers: int,
    n_shards: int,
    tgt_shape: Tuple[int, ...],
    scan_padded_axis: int,
    tgt_padded_axis: int,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Fuses scanned MoE halves and returns one target-ready slice per layer.

  This combines the expensive MoE fusion and per-layer unstack into one JITted
  TPU-friendly routine so the planner can reuse the materialized results across
  all destination layers for the same scanned source pair.
  """
  del num_layers
  fused_shape = list(wi_0.shape)
  fused_shape[scan_padded_axis] = tgt_shape[tgt_padded_axis]

  fused = _interleave_moe_weights(
      wi_0, wi_1, tuple(fused_shape), n_shards, axis=scan_padded_axis
  )
  return jnp.unstack(fused, axis=scan_axis)
