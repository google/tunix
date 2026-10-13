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

"""Bit-packed router-replay routing for trajectories the orchestrator holds.

Router replay returns each rollout's MoE routing as an int16
`[tokens, num_layers, top_k]` array: 2 bytes per expert id. The orchestrator
keeps that array for every finished trajectory until the batch assembler packs
the trajectory into a trainer micro-batch. The array waits in `raw_q` until all
`num_generations` of its group finish, then in `scored_q` until the trainer
reaches its batch. For Qwen3.5-397B (60 MoE layers, top-10 of 512 experts) that
is 1.2 KB per token, about 34 MB per DeepSWE trajectory. It is nearly all of
what the orchestrator holds per trajectory, and at high `max_staleness` it is
what fills the orchestrator's host memory.

An expert id below 512 needs only 9 bits. `BitPackedRoutedExperts` stores the
same routing in `top_k + 2` bytes per token-layer instead of `2 * top_k`: 12
instead of 20 at top_k = 10, i.e. 0.6x.

  lo  uint8  [tokens, L, K]  the low 8 bits of each expert id.
  hi  uint16 [tokens, L]     bit k (k < K) is bit 8 of slot k's id; bit 15
                             (`VALID_BIT`) is 1 iff the token-layer is set.

An unset token-layer (all K ids `UNSET_ROUTED_EXPERT`) is stored as lo = 0,
hi = 0, so zero-filled arrays mean "all unset", the same as the dense path's
`np.full(..., -1)` padding.

The encoding never leaves the orchestrator. `expand` restores the int16 array
before the batch assembler packs the trajectory, so the trainer receives the
same bytes with or without it. `compact` is lossless or nothing: if it cannot
restore an array bit for bit (a token-layer with some ids unset and some not,
an id outside [0, 512), top_k > 15), it returns None and the caller keeps the
dense array.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from absl import logging
import numpy as np
from tunix.experimental.common import datatypes

# Expert ids the encoding holds: [0, MAX_EXPERTS).
NUM_ID_BITS = 9
MAX_EXPERTS = 1 << NUM_ID_BITS
VALID_BIT = 15
# Bits 0..top_k-1 of `hi` carry the 9th id bits, so top_k must stay below
# VALID_BIT.
MAX_TOP_K = VALID_BIT
UNSET = datatypes.UNSET_ROUTED_EXPERT

_LO_DTYPE = np.uint8
_HI_DTYPE = np.uint16
# The dtype `algorithm_adapter._routed_experts_for` and `to_pack_item` give the
# dense array; `expand` returns the same.
_DENSE_DTYPE = np.int16
# Where the routing of a `TrajectoryItem` can live, in the order
# `algorithm_adapter._routed_experts_for` reads it.
_ROUTED_KEY = "routed_experts"
# -1 as a uint16 bit pattern, and the marker for any other id outside
# [0, MAX_EXPERTS) when the input is wider than 16 bits.
_UNSET_U16 = np.uint16(0xFFFF)
_INVALID_U16 = np.uint16(0xFFFE)
# Packs the low bit of each of 8 little-endian bytes (each 0 or 1) into the top
# byte of a uint64 product, byte i -> bit i. The partial products
# 2^(8i) * 2^(56-7j) are distinct bits, so only the i == j terms reach bits
# 56..63 and nothing carries into them.
_GATHER_BITS = np.uint64(0x0102040810204080)
# Token-layers encoded per pass (~1.3 MB of int16 at top_k = 10), so each pass
# runs on cache-resident data.
_BLOCK_ROWS = 1 << 16


@dataclasses.dataclass(frozen=True, eq=False)
class BitPackedRoutedExperts:
  """One trajectory's router-replay routing, bit-packed; see the module doc.

  Deliberately not array-like: `np.asarray` on it raises, because reading the
  bytes as expert ids would replay the wrong experts. Call `expand` instead.

  Attributes:
    lo: `uint8 [tokens, num_layers, top_k]`, the low 8 bits of each id.
    hi: `uint16 [tokens, num_layers]`, the 9th bits and the valid bit.
  """

  lo: np.ndarray
  hi: np.ndarray

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the dense routing: `(tokens, num_layers, top_k)`."""
    return tuple(self.lo.shape)

  @property
  def ndim(self) -> int:
    return self.lo.ndim

  @property
  def nbytes(self) -> int:
    """Bytes held by the encoding."""
    return int(self.lo.nbytes + self.hi.nbytes)

  @property
  def dense_nbytes(self) -> int:
    """Bytes the same routing takes as the dense int16 array."""
    return int(self.lo.size) * np.dtype(_DENSE_DTYPE).itemsize

  def __len__(self) -> int:
    return int(self.lo.shape[0])

  def __array__(self, *args, **kwargs):
    del args, kwargs
    raise TypeError(
        "BitPackedRoutedExperts is a bit-packed routing encoding, not an"
        " array of expert ids. Call held_routing.expand() first."
    )


def is_bitpacked(x: Any) -> bool:
  """True if `x` is routing in the bit-packed encoding."""
  return isinstance(x, BitPackedRoutedExperts)


def _as_uint16_ids(ids: np.ndarray) -> np.ndarray | None:
  """The ids as C-contiguous uint16 bit patterns, or None if not integers.

  16-bit inputs are reinterpreted (an int16 -1 is 0xFFFF). Wider integers are
  mapped explicitly, so no out-of-range id can wrap into [0, 512): -1 becomes
  0xFFFF and any other id outside [0, 512) becomes 0xFFFE, which `compact`
  refuses.

  Args:
    ids: Integer routing array.
  """
  if ids.dtype in (np.int16, np.uint16):
    return np.ascontiguousarray(ids).view(np.uint16)
  if not np.issubdtype(ids.dtype, np.integer):
    return None
  in_range = (ids >= 0) & (ids < MAX_EXPERTS)
  out = np.where(ids == UNSET, _UNSET_U16, _INVALID_U16).astype(np.uint16)
  np.copyto(out, ids, where=in_range, casting="unsafe")
  return out


def _encode_rows(u: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int:
  """Encodes `u` (uint16 `[rows, top_k]`) into `lo` / `hi`.

  Args:
    u: uint16 bit patterns of one block of token-layers.
    lo: `uint8 [rows, top_k]` output.
    hi: `uint16 [rows]` output.

  Returns:
    The number of rows that are neither all in [0, 512) nor all unset: rows the
    encoding cannot restore exactly.
  """
  rows, top_k = u.shape
  # The high byte of every id. An id is in [0, 512) iff its high byte is 0 or
  # 1. A reduction over a short last axis is slow in NumPy, so the max runs as
  # top_k elementwise passes.
  as_bytes = u.view(np.uint8).reshape(rows, 2 * top_k)
  high = as_bytes[:, 1::2] if np.little_endian else as_bytes[:, 0::2]
  max_high = high[:, 0].copy()
  for k in range(1, top_k):
    np.maximum(max_high, high[:, k], out=max_high)
  valid = max_high <= 1

  np.copyto(lo, u, casting="unsafe")  # uint16 -> uint8 keeps the low byte.
  # Bit 8 of a valid id is its whole high byte; gather slot k's into bit k.
  lanes = 1 if top_k <= 8 else 2
  lane_bytes = np.zeros((rows, 8 * lanes), dtype=np.uint8)
  lane_bytes[:, :top_k] = high
  gathered = (lane_bytes.view("<u8") * _GATHER_BITS) >> np.uint64(56)
  np.copyto(hi, gathered[:, 0], casting="unsafe")
  if lanes == 2:
    hi |= gathered[:, 1].astype(_HI_DTYPE) << np.uint16(8)
  hi |= np.uint16(1 << VALID_BIT)

  invalid = np.flatnonzero(~valid)
  if not invalid.size:
    return 0
  # An unset token-layer is all zeros.
  hi[invalid] = 0
  lo[invalid] = 0
  return int(np.count_nonzero((u[invalid] != _UNSET_U16).any(axis=1)))


def compact(ids: Any) -> BitPackedRoutedExperts | None:
  """Bit-packs one trajectory's routing, or returns None.

  Args:
    ids: `[tokens, num_layers, top_k]` integer expert ids, `-1` marking unset
      token-layers.

  Returns:
    The encoding, or None when `expand` could not return exactly
    `np.asarray(ids, dtype=np.int16)`: not a 3-D integer array, top_k outside
    [1, MAX_TOP_K], an id outside [0, MAX_EXPERTS) other than -1, or a
    token-layer with some ids -1 and some not. The caller then keeps `ids`.
  """
  if not isinstance(ids, np.ndarray):
    return None
  if ids.ndim != 3 or not 1 <= ids.shape[-1] <= MAX_TOP_K:
    return None
  u = _as_uint16_ids(ids)
  if u is None:
    return None
  tokens, num_layers, top_k = ids.shape
  u = u.reshape(-1, top_k)
  rows = u.shape[0]
  lo = np.empty((rows, top_k), dtype=_LO_DTYPE)
  hi = np.empty((rows,), dtype=_HI_DTYPE)
  for begin in range(0, rows, _BLOCK_ROWS):
    end = min(begin + _BLOCK_ROWS, rows)
    if _encode_rows(u[begin:end], lo[begin:end], hi[begin:end]):
      return None
  return BitPackedRoutedExperts(
      lo=lo.reshape((tokens, num_layers, top_k)),
      hi=hi.reshape((tokens, num_layers)),
  )


_EXPAND_TABLES: dict[int, np.ndarray] = {}


def _expand_table(top_k: int) -> np.ndarray:
  """`[2**top_k, top_k]` int16: row c holds (bit k of c) << 8 in column k."""
  table = _EXPAND_TABLES.get(top_k)
  if table is None:
    codes = np.arange(1 << top_k, dtype=np.uint32)[:, None]
    bits = (codes >> np.arange(top_k, dtype=np.uint32)) & 1
    table = (bits << 8).astype(_DENSE_DTYPE)
    table.setflags(write=False)
    _EXPAND_TABLES[top_k] = table
  return table


def expand(packed: BitPackedRoutedExperts) -> np.ndarray:
  """Returns the dense int16 `[tokens, num_layers, top_k]` routing, -1 = unset.

  Args:
    packed: The encoding.
  """
  if not is_bitpacked(packed):
    raise TypeError(
        f"expand expects BitPackedRoutedExperts, got {type(packed).__name__}."
    )
  lo, hi = packed.lo, packed.hi
  if hi.shape != lo.shape[:-1]:
    raise ValueError(
        f"BitPackedRoutedExperts hi has shape {hi.shape}; expected"
        f" {lo.shape[:-1]} for lo of shape {lo.shape}."
    )
  top_k = lo.shape[-1]
  # Look the 9th bits of all K slots up at once, then OR in the low bytes.
  out = np.take(_expand_table(top_k), hi & np.uint16((1 << top_k) - 1), axis=0)
  np.bitwise_or(out, lo, out=out)
  unset = hi < np.uint16(1 << VALID_BIT)
  if unset.any():
    out[unset] = UNSET
  return out


def fit_tokens(
    packed: BitPackedRoutedExperts, length: int
) -> BitPackedRoutedExperts:
  """Truncates, or pads with unset token-layers, to `length` tokens.

  The bit-packed counterpart of the dense `routed[:length]` /
  `np.full((length, ...), UNSET)` alignment. Truncation returns views.

  Args:
    packed: The encoding.
    length: Number of tokens wanted.
  """
  n = len(packed)
  if n >= length:
    return BitPackedRoutedExperts(lo=packed.lo[:length], hi=packed.hi[:length])
  lo = np.zeros((length,) + packed.lo.shape[1:], dtype=_LO_DTYPE)
  hi = np.zeros((length,) + packed.hi.shape[1:], dtype=_HI_DTYPE)
  lo[:n] = packed.lo
  hi[:n] = packed.hi
  return BitPackedRoutedExperts(lo=lo, hi=hi)


@dataclasses.dataclass
class CompactionStats:
  """Running totals for `compact_item_routing` calls.

  Attributes:
    items: Trajectory items looked at.
    compacted: Routing arrays replaced by their encoding.
    kept_dense: Routing arrays left dense because `compact` refused them.
    errors: Items skipped because compaction raised; their routing stays dense.
    dense_bytes: Dense bytes of the arrays that were compacted.
    held_bytes: Bytes of their encodings.
    seconds: Time spent compacting.
  """

  items: int = 0
  compacted: int = 0
  kept_dense: int = 0
  errors: int = 0
  dense_bytes: int = 0
  held_bytes: int = 0
  seconds: float = 0.0

  def merge(self, other: "CompactionStats") -> None:
    for field in dataclasses.fields(self):
      setattr(
          self,
          field.name,
          getattr(self, field.name) + getattr(other, field.name),
      )

  def summary(self) -> str:
    ratio = self.held_bytes / self.dense_bytes if self.dense_bytes else 0.0
    return (
        f"items={self.items} compacted={self.compacted}"
        f" kept_dense={self.kept_dense} errors={self.errors}"
        f" dense_gb={self.dense_bytes / 1e9:.2f}"
        f" held_gb={self.held_bytes / 1e9:.2f} ratio={ratio:.3f}"
        f" saved_gb={(self.dense_bytes - self.held_bytes) / 1e9:.2f}"
        f" seconds={self.seconds:.2f}"
    )


def compact_item_routing(item: Any) -> CompactionStats:
  """Bit-packs the routing a finished `TrajectoryItem` carries, in place.

  Looks where `algorithm_adapter._routed_experts_for` looks: the item's own
  `routed_experts` attribute, `item.traj["routed_experts"]` and
  `item.metadata["routed_experts"]`. Each distinct array is encoded once, and
  every place that referenced it gets the same encoding, so the dense array is
  freed once the caller drops its own references. An array `compact` refuses
  stays where it is. The item is only modified after every encoding has
  succeeded, and the call never raises: on an unexpected error the item is
  left exactly as it was and the error is counted.

  Args:
    item: A `TrajectoryItem` as returned by `poll_rollouts`.

  Returns:
    What this call did.
  """
  stats = CompactionStats(items=1)
  start = time.perf_counter()
  try:
    own = getattr(item, "__dict__", {})
    places: list[tuple[Any, bool, Any]] = []  # (holder, is_attr, value)
    if own.get(_ROUTED_KEY) is not None:
      places.append((item, True, own[_ROUTED_KEY]))
    traj = own.get("traj")
    if isinstance(traj, dict) and traj.get(_ROUTED_KEY) is not None:
      places.append((traj, False, traj[_ROUTED_KEY]))
    metadata = own.get("metadata")
    if isinstance(metadata, dict) and metadata.get(_ROUTED_KEY) is not None:
      places.append((metadata, False, metadata[_ROUTED_KEY]))
    replaced: dict[int, Any] = {}
    counted = CompactionStats()
    for _, _, value in places:
      if is_bitpacked(value) or id(value) in replaced:
        continue
      packed = compact(value)
      if packed is None:
        counted.kept_dense += 1
        replaced[id(value)] = value
      else:
        counted.compacted += 1
        counted.dense_bytes += packed.dense_nbytes
        counted.held_bytes += packed.nbytes
        replaced[id(value)] = packed
    for holder, is_attr, value in places:
      new_value = replaced.get(id(value), value)
      if new_value is value:
        continue
      if is_attr:
        setattr(holder, _ROUTED_KEY, new_value)
      else:
        holder[_ROUTED_KEY] = new_value
    stats.merge(counted)
  except Exception:  # pylint: disable=broad-exception-caught
    # Dense routing still trains correctly; losing the trajectory would not.
    stats.errors += 1
    logging.exception("Bit-packing router-replay routing failed; kept dense.")
  stats.seconds = time.perf_counter() - start
  return stats
