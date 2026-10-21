# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type-agnostic core packing utilities.

When the compiled C++ extension (`_packing_ext`) is available, row population
for chunks writing at least `_packing_ext.MIN_PARALLEL_BYTES` (4 MiB) across at
least 2 rows is parallelized across a persistent C++ worker pool of up to
`_packing_ext.MAX_PACKING_THREADS` (16) threads. The pool size is initialized
automatically from the process CPU affinity mask (`sched_getaffinity`) or
hardware concurrency, and each chunk automatically scales its active thread
count to `min(num_rows, total_bytes // MIN_BYTES_PER_THREAD, pool_threads)` and
wakes only the active worker slots (can also be overridden programmatically via
`_packing_ext.set_packing_threads(n)`).
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Literal, Mapping, Sequence, overload

import numpy as np
from tunix.common import datatypes as common_datatypes
from tunix.rl.agentic.agents import agent_types

try:
  from tunix.rl import _packing_ext  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  _packing_ext = None

ROUTED_EXPERTS_ENCODE_MODES = common_datatypes.ROUTED_EXPERTS_ENCODE_MODES
RoutedExpertsEncodeMode = common_datatypes.RoutedExpertsEncodeMode
RoutedExpertsWireEncoding = Literal["int16", "uint8", "uint10"]
validate_routed_experts_encode_config = (
    common_datatypes.validate_routed_experts_encode_config
)

# Optional per-token fields tracked in a PackItem.
PER_TOKEN_FIELDS: tuple[str, ...] = (
    "ref_per_token_logps",
    "old_per_token_logps",
    "returns",
    "old_values",
    "sampler_is_weights",
)

# Default token boundary alignment for packed segments. `1` disables alignment
# (segments are packed back to back). Hybrid recurrent models such as Qwen3.5
# (GatedDeltaNet `gdn_chunk_size = 64`) should use 64 so a packed segment never
# starts mid-chunk.
DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY = 1


def _check_segment_alignment_boundary(segment_alignment_boundary: int) -> None:
  if segment_alignment_boundary <= 0:
    raise ValueError(
        "segment_alignment_boundary must be positive, got"
        f" {segment_alignment_boundary}."
    )


def align_offset(offset: int, segment_alignment_boundary: int) -> int:
  """Rounds `offset` up to the next multiple of `segment_alignment_boundary`."""
  _check_segment_alignment_boundary(segment_alignment_boundary)
  return -(-offset // segment_alignment_boundary) * segment_alignment_boundary


def routed_experts_encoding(
    routed: np.ndarray,
    *,
    batched: bool = False,
) -> RoutedExpertsWireEncoding:
  """Returns the wire encoding (`'int16'`, `'uint8'`, or `'uint10'`) of `routed`.

  Args:
    routed: `routed_experts` numpy array.
    batched: If `True`, expects a batched microbatch layout (`4D` `(B, T, L, K)`
      for `int16`/`uint8` or `5D` `(B, T, L, K // 4, 5)` for `uint10`). If
      `False`, expects an unbatched trajectory layout (`3D` `(T, L, K)` for
      `int16`/`uint8` or `4D` `(T, L, K // 4, 5)` for `uint10`).
  """
  if not isinstance(routed, np.ndarray):
    raise ValueError(
        f"routed_experts must be a numpy array, got {type(routed).__name__}."
    )
  if routed.dtype == np.uint8:
    if batched:
      if routed.ndim == 5 and routed.shape[4] == 5:
        return "uint10"
      if routed.ndim == 4:
        return "uint8"
      raise ValueError(
          "batched uint8 routed_experts must have 4D (B, T, L, K) or 5D"
          f" (B, T, L, K // 4, 5) shape, got shape {routed.shape}."
      )
    if routed.ndim == 4 and routed.shape[3] == 5:
      return "uint10"
    if routed.ndim == 3:
      return "uint8"
    raise ValueError(
        "unbatched uint8 routed_experts must have 3D (T, L, K) or 4D"
        f" (T, L, K // 4, 5) shape, got shape {routed.shape}."
    )
  expected_ndim = 4 if batched else 3
  if routed.ndim == expected_ndim:
    return "int16"
  shape_desc = (
      "[batch, length, num_layers, top_k]"
      if batched
      else "[length, num_layers, top_k]"
  )
  raise ValueError(
      f"routed_experts with batched={batched} must be a {expected_ndim}D"
      f" {shape_desc} array, got dtype={routed.dtype} and shape={routed.shape}."
  )


def routed_experts_logical_shape(
    routed: np.ndarray,
    *,
    batched: bool = False,
) -> tuple[int, ...]:
  """Returns the logical `(..., L, K)` shape of a `routed_experts` array.

  Supports:
  - Raw `int16`: 3D `(T, L, K)` when `batched=False`, or 4D `(B, T, L, K)` when
    `batched=True`.
  - Mode 1 (`+1`-shifted `uint8`): 3D `(T, L, K)` when `batched=False`, or 4D
    `(B, T, L, K)` when `batched=True` (including `K == 5`).
  - Mode 2 (`+1`-shifted `uint10` packed as 4 values per 5 bytes):
    4D `(T, L, K // 4, 5)` when `batched=False`, or 5D `(B, T, L, K // 4, 5)`
    when `batched=True`.

  Args:
    routed: Encoded or raw `routed_experts` array.
    batched: Whether `routed` has a leading batch dimension `B`.

  Returns:
    Logical shape tuple `(T, L, K)` or `(B, T, L, K)`.
  """
  enc = routed_experts_encoding(routed, batched=batched)
  if enc == "uint10":
    return (
        *(int(d) for d in routed.shape[:-2]),
        int(routed.shape[-2]) * 4,
    )
  return tuple(int(d) for d in routed.shape)


@overload
def encode_routed_experts(
    routed: np.ndarray,
    *,
    mode: str = "auto",
    num_experts: int | None = None,
    batched: bool = False,
) -> np.ndarray:
  ...


@overload
def encode_routed_experts(
    routed: None,
    *,
    mode: str = "auto",
    num_experts: int | None = None,
    batched: bool = False,
) -> None:
  ...


def encode_routed_experts(
    routed: np.ndarray | None,
    *,
    mode: str = "auto",
    num_experts: int | None = None,
    batched: bool = False,
) -> np.ndarray | None:
  """Encodes `routed_experts` into compact `+1`-shifted `uint8` or `uint10`.

  - Mode `"uint8"` (E <= 255, max expert ID <= 254):
    v = (e + 1) as `uint8` (`0` = unset `-1`, `1..255` = `0..254`), preserving
    3D `(T, L, K)` or 4D `(B, T, L, K)` shape (1.0 B/expert).
  - Mode `"uint10"` (256 <= E <= 1023, max expert ID <= 1022, K % 4 == 0):
    4 consecutive 10-bit values (v = e + 1 in [0, 1023]) bit-packed into 5
    `uint8` bytes, producing 4D `(T, L, K // 4, 5)` or 5D
    `(B, T, L, K // 4, 5)` shape (1.25 B/expert).
  - Mode `"int16"`: C-contiguous `int16` array (2.0 B/expert).
  - Mode `"auto"`: automatically selects `"uint8"`, `"uint10"`, or `"int16"`
    based on `num_experts` (if provided) and the observed range of expert IDs.

  Args:
    routed: Raw or already-encoded `routed_experts` array, or `None`.
    mode: Target encoding mode (`"auto"`, `"uint8"`, `"uint10"`, or `"int16"`).
    num_experts: Optional total number of MoE experts (`E`) for validation and
      `"auto"` mode selection.
    batched: If `True`, `routed` is a batched microbatch array (`4D` `(B, T, L,
      K)` `uint8`/`int16` or `5D` `(B, T, L, K // 4, 5)` `uint10`).

  Returns:
    Compactly encoded `np.ndarray`, or `None` if `routed` is `None`.
  """
  if routed is None:
    return None
  if _packing_ext is not None:
    return _packing_ext.encode_routed_experts_fast(
        routed, mode, num_experts, batched
    )

  if num_experts is not None:
    if isinstance(num_experts, bool) or not isinstance(
        num_experts, (int, np.integer)
    ):
      raise ValueError(
          f"num_experts must be a positive int or None, got {num_experts!r}."
      )
    if int(num_experts) <= 0:
      raise ValueError(
          f"num_experts must be a positive int or None, got {num_experts}."
      )

  arr = np.asarray(routed)
  if arr.dtype == np.uint8:
    src_enc = routed_experts_encoding(arr, batched=batched)
    if src_enc == "uint8":
      if mode == "uint8" or (
          mode == "auto" and (num_experts is None or int(num_experts) <= 255)
      ):
        if num_experts is not None and arr.size > 0:
          max_val = int(arr.max()) - 1
          if max_val >= int(num_experts):
            raise ValueError(
                f"routed_experts contains expert ID {max_val} >= num_experts"
                f" ({int(num_experts)})."
            )
        return np.ascontiguousarray(arr)
      arr = decode_routed_experts(arr, batched=batched)
    elif src_enc == "uint10":
      if mode == "uint10" or (
          mode == "auto"
          and (num_experts is None or 256 <= int(num_experts) <= 1023)
      ):
        if (
            num_experts is not None
            and int(num_experts) <= 1022
            and arr.size > 0
        ):
          dec = decode_routed_experts(arr, batched=batched)
          max_val = int(dec.max())
          if max_val >= int(num_experts):
            raise ValueError(
                f"routed_experts contains expert ID {max_val} >= num_experts"
                f" ({int(num_experts)})."
            )
        return np.ascontiguousarray(arr)
      arr = decode_routed_experts(arr, batched=batched)
  arr = np.ascontiguousarray(arr, dtype=np.int16)
  if (batched and arr.ndim != 4) or (not batched and arr.ndim not in (3, 4)):
    raise ValueError(
        "encode_routed_experts expects a 3D (T, L, K) or 4D (B, T, L, K)"
        f" array, got shape={arr.shape}."
    )
  top_k = int(arr.shape[-1])
  min_val = int(arr.min()) if arr.size > 0 else 0
  max_val = int(arr.max()) if arr.size > 0 else -1
  if min_val < -1:
    raise ValueError(
        f"routed_experts contains invalid expert ID < -1: {min_val}"
    )
  if num_experts is not None and max_val >= int(num_experts):
    raise ValueError(
        f"routed_experts contains expert ID {max_val} >= num_experts"
        f" ({int(num_experts)})."
    )
  effective_max = (
      max(max_val, int(num_experts) - 1) if num_experts is not None else max_val
  )
  if mode == "auto":
    if effective_max <= 254:
      target = "uint8"
    elif effective_max <= 1022 and top_k % 4 == 0:
      target = "uint10"
    else:
      target = "int16"
  elif mode in ("uint8", "uint10", "int16"):
    target = mode
  else:
    raise ValueError(f"Unsupported routed_experts encoding mode: {mode!r}")

  if target == "int16":
    return arr
  if target == "uint8":
    if max_val > 254:
      raise ValueError(
          f"uint8 encoding requires expert IDs <= 254, got max={max_val}"
      )
    return (arr + np.int16(1)).astype(np.uint8)
  if top_k % 4 != 0:
    raise ValueError(
        f"uint10 encoding requires top_k divisible by 4, got top_k={top_k}"
    )
  if max_val > 1022:
    raise ValueError(
        f"uint10 encoding requires expert IDs <= 1022, got max={max_val}"
    )
  v = (arr.astype(np.uint16) + np.uint16(1)).reshape(
      *arr.shape[:-1], top_k // 4, 4
  )
  v0, v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
  out = np.empty((*arr.shape[:-1], top_k // 4, 5), dtype=np.uint8)
  out[..., 0] = (v0 & 0xFF).astype(np.uint8)
  out[..., 1] = (((v0 >> 8) & 0x03) | ((v1 & 0x3F) << 2)).astype(np.uint8)
  out[..., 2] = (((v1 >> 6) & 0x0F) | ((v2 & 0x0F) << 4)).astype(np.uint8)
  out[..., 3] = (((v2 >> 4) & 0x3F) | ((v3 & 0x03) << 6)).astype(np.uint8)
  out[..., 4] = ((v3 >> 2) & 0xFF).astype(np.uint8)
  return out


@overload
def decode_routed_experts(
    encoded: np.ndarray,
    *,
    batched: bool = False,
) -> np.ndarray:
  ...


@overload
def decode_routed_experts(
    encoded: None,
    *,
    batched: bool = False,
) -> None:
  ...


def decode_routed_experts(
    encoded: np.ndarray | None,
    *,
    batched: bool = False,
) -> np.ndarray | None:
  """Decodes compact `uint8` / `uint10` `routed_experts` back to `int16`.

  Args:
    encoded: Encoded `routed_experts` array (`uint8` or `int16`), or `None`.
    batched: If `True`, `encoded` is a batched microbatch array (`4D` `(B, T, L,
      K)` `uint8`/`int16` or `5D` `(B, T, L, K // 4, 5)` `uint10`). If `False`
      (default), `encoded` is an unbatched trajectory array (`3D` `(T, L, K)`
      `uint8`/`int16` or `4D` `(T, L, K // 4, 5)` `uint10`).

  Returns:
    Decoded `int16` `np.ndarray`, or `None` if `encoded` is `None`.
  """
  if encoded is None:
    return None
  if _packing_ext is not None:
    return _packing_ext.decode_routed_experts_fast(encoded, batched=batched)

  arr = np.ascontiguousarray(encoded)
  enc = routed_experts_encoding(arr, batched=batched)
  if enc == "int16":
    return arr if arr.dtype == np.int16 else arr.astype(np.int16)
  if enc == "uint10":
    b0 = arr[..., 0].astype(np.int16)
    b1 = arr[..., 1].astype(np.int16)
    b2 = arr[..., 2].astype(np.int16)
    b3 = arr[..., 3].astype(np.int16)
    b4 = arr[..., 4].astype(np.int16)
    v0 = (b0 | ((b1 & 0x03) << 8)) - 1
    v1 = ((b1 >> 2) | ((b2 & 0x0F) << 6)) - 1
    v2 = ((b2 >> 4) | ((b3 & 0x3F) << 4)) - 1
    v3 = ((b3 >> 6) | (b4 << 2)) - 1
    stacked = np.stack([v0, v1, v2, v3], axis=-1)
    return stacked.reshape(*arr.shape[:-2], arr.shape[-2] * 4)
  return arr.astype(np.int16) - np.int16(1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackItem:
  """A single unpadded item to be packed into a PackedRow."""

  prompt_ids: np.ndarray
  completion_ids: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  per_token: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
  policy_version: np.ndarray | None = None
  # Router replay: `[p + c - 1 or p + c, num_layers, top_k]` expert ids (int16
  # or `+1`-shifted uint8, or `[p + c - 1 or p + c, num_layers, top_k // 4, 5]`
  # `+1`-shifted uint10), sequence-aligned (prompt then completion), not
  # completion-aligned like the `per_token` fields. Autoregressive rollouts may
  # omit the final sampled token (`p + c - 1`), which remains
  # `UNSET_ROUTED_EXPERT` in the packed output.
  routed_experts: np.ndarray | None = None

  def __post_init__(self):
    for name, expected_dtype in (
        ("prompt_ids", np.int32),
        ("completion_ids", np.int32),
        ("completion_mask", np.float32),
        ("advantages", np.float32),
    ):
      arr = getattr(self, name)
      if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError(
            f"PackItem.{name} must be a 1D numpy array, got"
            f" {type(arr).__name__} with shape {getattr(arr, 'shape', None)}."
            " Unpad and flatten before packing."
        )
      if arr.dtype != expected_dtype or not arr.flags.c_contiguous:
        object.__setattr__(
            self, name, np.ascontiguousarray(arr, dtype=expected_dtype)
        )
    c = self.completion_ids.shape[0]
    for name in ("completion_mask", "advantages"):
      dim = getattr(self, name).shape[0]
      if dim != c:
        raise ValueError(
            f"PackItem.{name} must have length matching completion_ids length;"
            f" got {dim}, expected {c}."
        )
    normalized_pt: dict[str, np.ndarray] | None = None
    for key, arr in self.per_token.items():
      if key not in PER_TOKEN_FIELDS:
        raise ValueError(
            f"Unknown per-token field {key!r}; expected one of"
            f" {PER_TOKEN_FIELDS}."
        )
      if not isinstance(arr, np.ndarray) or arr.ndim != 1 or arr.shape[0] != c:
        raise ValueError(
            f"PackItem.per_token[{key!r}] must be a 1D numpy array or shape"
            f" (c,), got {type(arr).__name__} with shape"
            f" {getattr(arr, 'shape', None)}."
        )
      if arr.dtype != np.float32 or not arr.flags.c_contiguous:
        if normalized_pt is None:
          normalized_pt = dict(self.per_token)
        normalized_pt[key] = np.ascontiguousarray(arr, dtype=np.float32)
    if normalized_pt is not None:
      object.__setattr__(self, "per_token", normalized_pt)
    if self.routed_experts is not None:
      n = self.prompt_ids.shape[0] + c
      min_len = max(n - 1, 0)
      routed = self.routed_experts
      try:
        enc = routed_experts_encoding(routed, batched=False)
      except ValueError as exc:
        raise ValueError(
            "PackItem.routed_experts must be a numpy array of shape"
            " (p + c - 1 or p + c, num_layers, top_k) with length in"
            f" [{min_len}, {n}], got {type(routed).__name__} with shape"
            f" {getattr(routed, 'shape', None)}."
        ) from exc
      if routed.shape[0] < min_len or routed.shape[0] > n:
        raise ValueError(
            "PackItem.routed_experts must be a numpy array of shape"
            " (p + c - 1 or p + c, num_layers, top_k) with length in"
            f" [{min_len}, {n}], got {type(routed).__name__} with shape"
            f" {getattr(routed, 'shape', None)}."
        )
      if enc in ("uint8", "uint10"):
        if not routed.flags.c_contiguous:
          object.__setattr__(
              self,
              "routed_experts",
              np.ascontiguousarray(routed, dtype=np.uint8),
          )
      elif routed.dtype != np.int16 or not routed.flags.c_contiguous:
        object.__setattr__(
            self,
            "routed_experts",
            np.ascontiguousarray(routed, dtype=np.int16),
        )

  @property
  def num_tokens(self) -> int:
    return self.prompt_ids.shape[0] + self.completion_ids.shape[0]


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackedRow:
  """A single row of packed data, corresponding to one RLTrainerPayload."""

  ids: np.ndarray
  prompt_mask: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  segment_ids: np.ndarray
  segment_positions: np.ndarray
  per_token: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
  policy_version: np.ndarray | None = None
  num_real_segments: int = 0
  # `[budget, num_layers, top_k]` int16 or uint8 (or `[budget, num_layers,
  # top_k // 4, 5]` uint8 in uint10 mode), unset wherever no routing was
  # captured (padding, alignment gaps, and routing-less items).
  routed_experts: np.ndarray | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackedChunk(Sequence[PackedRow]):
  """A packed chunk of `n_bins` rows with contiguous `[n_bins, budget]` arrays."""

  ids: np.ndarray
  prompt_mask: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  segment_ids: np.ndarray
  segment_positions: np.ndarray
  per_token: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
  policy_versions: tuple[np.ndarray | None, ...] = ()
  num_real_segments: tuple[int, ...] = ()
  # `[n_bins, budget, num_layers, top_k]` int16 or uint8 (or 5D uint10); see
  # `PackedRow.routed_experts`.
  routed_experts: np.ndarray | None = None

  def __len__(self) -> int:
    return int(self.ids.shape[0])

  @overload
  def __getitem__(self, index: int) -> PackedRow:
    ...

  @overload
  def __getitem__(self, index: slice) -> list[PackedRow]:
    ...

  def __getitem__(self, index: int | slice) -> PackedRow | list[PackedRow]:
    if isinstance(index, slice):
      return [self[i] for i in range(*index.indices(len(self)))]
    if index < 0:
      index += len(self)
    if index < 0 or index >= len(self):
      raise IndexError(f"PackedChunk index {index} out of range.")
    return PackedRow(
        ids=self.ids[index],
        prompt_mask=self.prompt_mask[index],
        completion_mask=self.completion_mask[index],
        advantages=self.advantages[index],
        segment_ids=self.segment_ids[index],
        segment_positions=self.segment_positions[index],
        per_token={name: arr[index] for name, arr in self.per_token.items()},
        policy_version=(
            self.policy_versions[index]
            if index < len(self.policy_versions)
            else None
        ),
        num_real_segments=(
            self.num_real_segments[index]
            if index < len(self.num_real_segments)
            else 0
        ),
        routed_experts=(
            None if self.routed_experts is None else self.routed_experts[index]
        ),
    )


def carried_per_token_fields(items: Sequence[PackItem]) -> tuple[str, ...]:
  """Returns the per-token fields carried by all items in the sequence."""
  if not items:
    return ()
  carried = []
  for name in PER_TOKEN_FIELDS:
    presented = [name in item.per_token for item in items]
    if all(presented):
      carried.append(name)
    elif any(presented):
      missing = [i for i, present in enumerate(presented) if not present]
      raise ValueError(
          f"Some but not all items have per-token field {name!r} (missing at"
          f" indices {missing})."
      )
  return tuple(carried)


def routed_experts_shape(
    items: Sequence[PackItem],
) -> tuple[int, ...] | None:
  """Returns the `(num_layers, top_k)` routing shape if ANY item carries routing.

  An item without routing is packed with all-unset rows, which the MoE layer
  routes with its own gate -- the same as no replay for those tokens. So one
  routing-less trajectory neither disables replay for the rest of the chunk nor
  changes the payload's pytree structure (which would recompile the trainer
  step).

  Args:
    items: Items that will be packed together into one chunk.

  Returns:
    The trailing routing shape, or None if no item has routing.

  Raises:
    ValueError: If the items that carry routing disagree on the trailing shape.
  """
  shapes = {
      routed_experts_logical_shape(item.routed_experts, batched=False)[1:]
      for item in items
      if item.routed_experts is not None
  }
  if not shapes:
    return None
  if len(shapes) != 1:
    raise ValueError(
        f"Items disagree on routed_experts trailing shape: {sorted(shapes)}."
    )
  return shapes.pop()


def fill_one_chunk(
    items: Sequence[PackItem],
    *,
    pack_size: int,
    budget: int,
    max_segments: int,
    segment_alignment_boundary: int = DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY,
) -> tuple[list[list[PackItem]], list[PackItem]]:
  """Fills ONE chunk of `pack_size` fixed-capacity bins, first-fit-decreasing.

  Sorts the items by token length descending and greedily places each into the
  first bin with room, where a bin has room only if it stays within both the
  token `budget` (after aligning the start of every segment but the first to
  `segment_alignment_boundary`) AND `max_segments` sequences (so the loss's
  static `num_segments = max_segments + 1` buckets never overflow). Items that
  fit no bin are returned as `leftover` (in their original order) for a later
  chunk.

  Args:
    items: Sequence of PackItems to pack.
    pack_size: Number of bins in the chunk.
    budget: Token capacity budget per bin.
    max_segments: Maximum number of segments allowed in a single bin.
    segment_alignment_boundary: Token boundary each segment after the first in a
      bin must start on. `1` packs segments back to back.

  Returns:
    A tuple of (bins, leftover), where `bins` is a list of `pack_size` lists of
    PackItems (some may be empty), and `leftover` contains the items that did
    not fit into any bin.
  """
  _check_segment_alignment_boundary(segment_alignment_boundary)
  if _packing_ext is not None and items and pack_size > 0 and max_segments > 0:
    bin_indices, leftover_indices = _packing_ext.fill_one_chunk_fast(
        [item.num_tokens for item in items],
        budget,
        pack_size,
        max_segments,
        segment_alignment_boundary,
    )
    return (
        [[items[i] for i in b_idxs] for b_idxs in bin_indices],
        [items[i] for i in leftover_indices],
    )
  bins: list[list[PackItem]] = [[] for _ in range(pack_size)]
  loads = [0] * pack_size
  order = sorted(
      range(len(items)), key=lambda i: items[i].num_tokens, reverse=True
  )
  placed_flags = [False] * len(items)
  for i in order:
    item = items[i]
    n = item.num_tokens
    for b in range(pack_size):
      # Must mirror the cursor arithmetic in `pack_chunk`.
      start = (
          align_offset(loads[b], segment_alignment_boundary) if bins[b] else 0
      )
      if start + n <= budget and len(bins[b]) < max_segments:
        bins[b].append(item)
        loads[b] = start + n
        placed_flags[i] = True
        break
  leftover = [items[i] for i in range(len(items)) if not placed_flags[i]]
  return bins, leftover


def pack_bin(
    bin_items: Sequence[PackItem],
    *,
    budget: int,
    pad_id: int,
    carried: Sequence[str],
    segment_alignment_boundary: int = DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY,
) -> PackedRow:
  """Packs a single bin of items into a single `[budget]` PackedRow."""
  return pack_chunk(
      [bin_items],
      budget=budget,
      pad_id=pad_id,
      carried=carried,
      segment_alignment_boundary=segment_alignment_boundary,
  )[0]


def pack_chunk(
    bins: Sequence[Sequence[PackItem]],
    *,
    budget: int,
    pad_id: int,
    carried: Sequence[str],
    segment_alignment_boundary: int = DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY,
) -> PackedChunk:
  """Packs a sequence of bins of one chunk into a contiguous `[n_bins, budget]` PackedChunk.

  Args:
    bins: Items per row, in placement order.
    budget: Row length in tokens.
    pad_id: Token id written to padding and alignment-gap positions.
    carried: Per-token fields to pack (see `carried_per_token_fields`).
    segment_alignment_boundary: Token boundary each segment after the first in a
      row must start on. Gaps are padding (`segment_ids == 0`, masks 0,
      `pad_id`, unset routing).

  Returns:
    The packed chunk. If any item carries `routed_experts` (see
    `routed_experts_shape`), the chunk carries a `[n_bins, budget, num_layers,
    top_k]` int16 `routed_experts` buffer, `UNSET_ROUTED_EXPERT` wherever no
    routing was captured, so every row has the same structure.
  """
  _check_segment_alignment_boundary(segment_alignment_boundary)
  n_bins = len(bins)
  routed_shape = routed_experts_shape(
      [item for bin_items in bins for item in bin_items]
  )
  if _packing_ext is not None:
    (
        ids,
        prompt_mask,
        completion_mask,
        advantages,
        segment_ids,
        segment_positions,
        per_token,
        routed,
    ) = _packing_ext.pack_chunk_fast(
        bins,
        list(carried),
        budget,
        pad_id,
        segment_alignment_boundary,
        routed_shape,
    )
    return PackedChunk(
        ids=ids,
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
        advantages=advantages,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        per_token=per_token,
        policy_versions=tuple(
            bin_items[0].policy_version if bin_items else None
            for bin_items in bins
        ),
        num_real_segments=tuple(len(bin_items) for bin_items in bins),
        routed_experts=routed,
    )

  ids = np.full((n_bins, budget), pad_id, dtype=np.int32)
  prompt_mask = np.zeros((n_bins, budget), dtype=np.float32)
  completion_mask = np.zeros((n_bins, budget), dtype=np.float32)
  advantages = np.zeros((n_bins, budget), dtype=np.float32)
  segment_ids = np.zeros((n_bins, budget), dtype=np.int32)
  segment_positions = np.zeros((n_bins, budget), dtype=np.int32)
  per_token = {
      name: np.zeros((n_bins, budget), dtype=np.float32) for name in carried
  }
  all_routed: list[np.ndarray] = []
  for bin_items in bins:
    for item in bin_items:
      if item.routed_experts is not None:
        all_routed.append(item.routed_experts)
  if routed_shape is None or not all_routed:
    routed = None
    routed_mode = "none"
  else:
    encs = {routed_experts_encoding(r, batched=False) for r in all_routed}
    if "int16" in encs:
      routed_mode = "int16"
      routed = np.full(
          (n_bins, budget, *routed_shape),
          agent_types.UNSET_ROUTED_EXPERT,
          dtype=np.int16,
      )
    elif "uint10" in encs:
      routed_mode = "uint10"
      routed = np.zeros(
          (n_bins, budget, routed_shape[0], routed_shape[1] // 4, 5),
          dtype=np.uint8,
      )
    else:
      routed_mode = "uint8"
      routed = np.zeros((n_bins, budget, *routed_shape), dtype=np.uint8)
  policy_versions: list[np.ndarray | None] = []
  num_real_segments: list[int] = []
  for b, bin_items in enumerate(bins):
    cursor = 0
    for seg, item in enumerate(bin_items, start=1):
      p = item.prompt_ids.shape[0]
      c = item.completion_ids.shape[0]
      n = p + c
      if seg > 1:
        cursor = align_offset(cursor, segment_alignment_boundary)
      if cursor + n > budget:
        raise ValueError(
            f"pack_bin: bin size {cursor + n} exceeds budget {budget}."
        )
      seq = slice(cursor, cursor + n)
      comp = slice(cursor + p, cursor + n)

      ids[b, cursor : cursor + p] = item.prompt_ids
      ids[b, comp] = item.completion_ids
      prompt_mask[b, cursor : cursor + p] = 1.0
      segment_ids[b, seq] = seg
      segment_positions[b, seq] = np.arange(n, dtype=np.int32)

      completion_mask[b, comp] = item.completion_mask
      advantages[b, comp] = item.advantages
      for name in carried:
        per_token[name][b, comp] = item.per_token[name]
      if routed is not None and item.routed_experts is not None:
        # Prefix-aligned to `seq`: a token's captured experts land on the
        # position the token itself does, and any uncaptured trailing tokens in
        # the segment remain `UNSET_ROUTED_EXPERT`.
        r_item = item.routed_experts
        m = r_item.shape[0]
        if m > n:
          raise ValueError(
              f"PackItem.routed_experts token length {m} exceeds segment"
              f" token length {n}."
          )
        item_enc = routed_experts_encoding(r_item, batched=False)
        if routed_mode == "int16" and item_enc != "int16":
          r_item = decode_routed_experts(r_item, batched=False)
        elif routed_mode == "uint10" and item_enc == "uint8":
          r_item = encode_routed_experts(
              decode_routed_experts(r_item, batched=False),
              mode="uint10",
              batched=False,
          )
        routed[b, cursor : cursor + m] = r_item
      cursor += n

    policy_versions.append(bin_items[0].policy_version if bin_items else None)
    num_real_segments.append(len(bin_items))
  return PackedChunk(
      ids=ids,
      prompt_mask=prompt_mask,
      completion_mask=completion_mask,
      advantages=advantages,
      segment_ids=segment_ids,
      segment_positions=segment_positions,
      per_token=per_token,
      policy_versions=tuple(policy_versions),
      num_real_segments=tuple(num_real_segments),
      routed_experts=routed,
  )


def effective_max_segments(
    budget: int, max_segments_per_packed_row: int | None
) -> int:
  """Returns the effective max segments per packed row."""
  return (
      max_segments_per_packed_row
      if isinstance(max_segments_per_packed_row, int)
      else budget
  )


def validate_items(items: Iterable[PackItem], budget: int) -> None:
  """Validates that all items are valid and fit within the budget."""
  for i, item in enumerate(items):
    if item.num_tokens > budget:
      raise ValueError(
          f"Item {i} has {item.num_tokens} tokens, exceeding budget {budget}."
      )


def pack_chunks_with_bins_fast(
    items: Sequence[PackItem],
    *,
    carried: Sequence[str],
    budget: int,
    pack_size: int,
    max_segments: int,
    pad_id: int,
    segment_alignment_boundary: int,
    min_buffered_tokens: int = 0,
) -> tuple[list[tuple[PackedChunk, list[list[int]]]], list[int]]:
  """Packs chunks via `_packing_ext`, returning `(chunks_with_bins, leftover)`."""
  assert _packing_ext is not None
  raw_chunks, leftover_indices = _packing_ext.pack_sequence_chunks_fast(
      items,
      list(carried),
      budget,
      pack_size,
      max_segments,
      pad_id,
      segment_alignment_boundary,
      min_buffered_tokens,
  )
  out: list[tuple[PackedChunk, list[list[int]]]] = []
  for (
      ids,
      prompt_mask,
      completion_mask,
      advantages,
      segment_ids,
      segment_positions,
      per_token,
      routed,
      bin_indices,
  ) in raw_chunks:
    chunk = PackedChunk(
        ids=ids,
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
        advantages=advantages,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        per_token=per_token,
        policy_versions=tuple(
            items[b_idxs[0]].policy_version if b_idxs else None
            for b_idxs in bin_indices
        ),
        num_real_segments=tuple(len(b_idxs) for b_idxs in bin_indices),
        routed_experts=routed,
    )
    out.append((chunk, bin_indices))
  return out, leftover_indices


def pack_core(
    items: Sequence[PackItem],
    *,
    budget: int,
    pack_size: int = 1,
    max_segments_per_packed_row: int | None = None,
    pad_id: int = 0,
    segment_alignment_boundary: int = DEFAULT_SEGMENT_ALIGNMENT_BOUNDARY,
) -> list[PackedChunk]:
  """Packs `items` into a sequence of chunks, each containing `pack_size` PackedRows with `budget` tokens."""
  if budget <= 0:
    raise ValueError(f"Budget must be positive, got {budget}.")
  if pack_size <= 0:
    raise ValueError(f"Pack size must be positive, got {pack_size}.")
  if (
      max_segments_per_packed_row is not None
      and max_segments_per_packed_row <= 0
  ):
    raise ValueError(
        "Max segments per packed row must be positive or None, got"
        f" {max_segments_per_packed_row}."
    )
  _check_segment_alignment_boundary(segment_alignment_boundary)
  if not items:
    return []

  validate_items(items, budget)
  max_segments = effective_max_segments(budget, max_segments_per_packed_row)
  carried = carried_per_token_fields(items)

  if _packing_ext is not None:
    chunks_with_bins, _ = pack_chunks_with_bins_fast(
        items,
        carried=carried,
        budget=budget,
        pack_size=pack_size,
        max_segments=max_segments,
        pad_id=pad_id,
        segment_alignment_boundary=segment_alignment_boundary,
    )
    return [chunk for chunk, _ in chunks_with_bins]

  chunks = []
  remaining = list(items)
  while remaining:
    bins, remaining = fill_one_chunk(
        remaining,
        pack_size=pack_size,
        budget=budget,
        max_segments=max_segments,
        segment_alignment_boundary=segment_alignment_boundary,
    )
    if not any(bins):
      raise ValueError("pack_core: no items placed in any bin.")
    chunks.append(
        pack_chunk(
            bins,
            budget=budget,
            pad_id=pad_id,
            carried=carried,
            segment_alignment_boundary=segment_alignment_boundary,
        )
    )
  return chunks
