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

"""Fixed-rank-order collective sum primitives (`RING` / `GATHER` / `SCATTER`).

Replaces XLA's topology- and shape-dependent `lax.psum` tree reduction across a
tensor- or sequence-parallel mesh axis with a deterministic rank-ordered sum:

    ((partial_0 + partial_1) + partial_2) + ... + partial_{count - 1}

Every rank accumulates the per-rank partials in the exact same rank `0 .. count
- 1` association order and rounds each intermediate `bfloat16` addition through
`lax.reduce_precision(..., 8, 7)` (`hard_bf16_add`) so that XLA's
`--xla_allow_excess_precision` pass cannot elide intermediate `bfloat16`
roundings when fusing surrounding operations.

All three collective schedules (`RING`, `GATHER`, `SCATTER`) execute the exact
same sequence of additions and produce bitwise-identical outputs on every rank.
"""

from __future__ import annotations

import operator
import os
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp

# Collective schedule modes for `fixed_order_tp_sum`.
RING = "ring"
GATHER = "gather"
SCATTER = "scatter"
MODES = (RING, GATHER, SCATTER)

GATHER_ENV = "CANON_FIXED_AR_GATHER"
SCATTER_ENV = "CANON_FIXED_AR_SCATTER"

# Per-step binary combiner `add(acc, operand) -> acc`.
AddFn = Callable[[jax.Array, jax.Array], jax.Array]

_BF16_EXPONENT_BITS = 8
_BF16_MANTISSA_BITS = 7


def _round_bf16_in_f32(x: jax.Array) -> jax.Array:
  """Rounds `x` to `bfloat16` precision in `float32` via `reduce_precision`."""
  return lax.reduce_precision(
      x.astype(jnp.float32),
      exponent_bits=_BF16_EXPONENT_BITS,
      mantissa_bits=_BF16_MANTISSA_BITS,
  )


def hard_round(x: jax.Array, dtype: jax.typing.DTypeLike) -> jax.Array:
  """Casts `x` to `dtype`, pinning `bfloat16` RNE rounding against XLA fusion."""
  target = jnp.dtype(dtype)
  if target == jnp.bfloat16:
    return _round_bf16_in_f32(x).astype(jnp.bfloat16)
  return x.astype(target)


def hard_upcast(x: jax.Array) -> jax.Array:
  """Upcasts `x` to `float32`, pinning `bfloat16` precision first if `x` is bf16."""
  if jnp.dtype(x.dtype) == jnp.bfloat16:
    return _round_bf16_in_f32(x)
  return x.astype(jnp.float32)


def hard_bf16_add(acc: jax.Array, operand: jax.Array) -> jax.Array:
  """Adds `acc` and `operand`, pinning `bfloat16` RNE rounding on inputs and sum."""
  out_dtype = jnp.result_type(acc, operand)
  if out_dtype == jnp.bfloat16:
    return _round_bf16_in_f32(hard_upcast(acc) + hard_upcast(operand)).astype(
        jnp.bfloat16
    )
  return acc.astype(out_dtype) + operand.astype(out_dtype)


plain_add: AddFn = operator.add


def _flag_from_env(name: str) -> bool:
  raw = os.environ.get(name, "").strip()
  if raw in ("", "0"):
    return False
  if raw == "1":
    return True
  raise RuntimeError(f"{name} must be unset, '0', or '1'; got {raw!r}")


def fixed_ar_mode_from_env() -> str:
  """Resolves the collective schedule (`RING` / `GATHER` / `SCATTER`) from env."""
  gather = _flag_from_env(GATHER_ENV)
  scatter = _flag_from_env(SCATTER_ENV)
  if scatter and not gather:
    raise RuntimeError(
        f"{SCATTER_ENV}=1 requires {GATHER_ENV}=1 (scatter refines gather)"
    )
  if scatter:
    return SCATTER
  if gather:
    return GATHER
  return RING


def _validate_count(count: int) -> None:
  if count <= 0:
    raise ValueError(f"count must be positive, got {count}")


def _singleton_result(partial: jax.Array, add: AddFn | None) -> jax.Array:
  add_fn = hard_bf16_add if add is None else add
  if add_fn is hard_bf16_add and jnp.dtype(partial.dtype) == jnp.bfloat16:
    return hard_round(partial, jnp.bfloat16)
  return partial


def fixed_order_tp_sum_ring(
    partial: jax.Array,
    axis_name: str,
    count: int,
    *,
    add: AddFn | None = None,
) -> jax.Array:
  """Sums `partial` across `axis_name` in rank `0..count-1` order via `ppermute`."""
  _validate_count(count)
  if count == 1:
    return _singleton_result(partial, add)
  add_fn = hard_bf16_add if add is None else add
  perm = [(i, (i + 1) % count) for i in range(count)]
  me = lax.axis_index(axis_name)
  entries = [partial]
  ranks = [me]
  cur = partial
  for step in range(1, count):
    cur = lax.ppermute(cur, axis_name, perm=perm)
    entries.append(cur)
    ranks.append((me - step) % count)
  stacked = jnp.stack(entries, axis=0)
  order = jnp.argsort(jnp.stack(ranks, axis=0))
  ordered = stacked[order]
  acc = ordered[0]
  for r in range(1, count):
    acc = add_fn(acc, ordered[r])
  return acc


def fixed_order_tp_sum_gather(
    partial: jax.Array,
    axis_name: str,
    count: int,
    *,
    add: AddFn | None = None,
) -> jax.Array:
  """Sums `partial` across `axis_name` in rank `0..count-1` order via `all_gather`."""
  _validate_count(count)
  if count == 1:
    return _singleton_result(partial, add)
  add_fn = hard_bf16_add if add is None else add
  stacked = lax.all_gather(partial, axis_name, axis=0, tiled=False)
  acc = stacked[0]
  for r in range(1, count):
    acc = add_fn(acc, stacked[r])
  return acc


def fixed_order_tp_sum_scatter(
    partial: jax.Array,
    axis_name: str,
    count: int,
    *,
    add: AddFn | None = None,
) -> jax.Array:
  """Sums `partial` across `axis_name` via `all_to_all` + rank-ordered sum + `all_gather`.

  When the leading dimension `M` is divisible by `count`, each rank reduces
  `M // count` rows in rank `0..count-1` order and all-gathers the reduced
  shards; otherwise falls back to `fixed_order_tp_sum_gather` (bitwise
  identical).

  Args:
    partial: Per-rank partial array of shape `[M, ...]`.
    axis_name: Mesh axis name to reduce across.
    count: Size of `axis_name`.
    add: Optional binary combiner (defaults to `hard_bf16_add`).

  Returns:
    The replicated rank-ordered sum of shape `[M, ...]`.
  """
  _validate_count(count)
  if count == 1:
    return _singleton_result(partial, add)
  if partial.ndim < 1 or int(partial.shape[0]) % count != 0:
    return fixed_order_tp_sum_gather(partial, axis_name, count, add=add)
  add_fn = hard_bf16_add if add is None else add
  m = int(partial.shape[0])
  reshaped = partial.reshape((count, m // count) + partial.shape[1:])
  stacked = lax.all_to_all(
      reshaped, axis_name, split_axis=0, concat_axis=0, tiled=False
  )
  acc = stacked[0]
  for r in range(1, count):
    acc = add_fn(acc, stacked[r])
  return lax.all_gather(acc, axis_name, axis=0, tiled=True)


def fixed_order_tp_sum(
    partial: jax.Array,
    axis_name: str,
    count: int,
    *,
    mode: str | None = None,
    add: AddFn | None = None,
) -> jax.Array:
  """Dispatches `fixed_order_tp_sum_{ring,gather,scatter}` by `mode`."""
  resolved = fixed_ar_mode_from_env() if mode is None else mode
  if resolved == RING:
    return fixed_order_tp_sum_ring(partial, axis_name, count, add=add)
  if resolved == GATHER:
    return fixed_order_tp_sum_gather(partial, axis_name, count, add=add)
  if resolved == SCATTER:
    return fixed_order_tp_sum_scatter(partial, axis_name, count, add=add)
  raise ValueError(
      f"unknown fixed-order all-reduce mode {resolved!r}; valid: {MODES}"
  )
