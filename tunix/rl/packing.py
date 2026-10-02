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

"""Type-agnostic core packing utilities."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping, Sequence

import numpy as np

# Optional per-token fields tracked in a PackItem.
PER_TOKEN_FIELDS: tuple[str, ...] = (
    "ref_per_token_logps",
    "old_per_token_logps",
    "returns",
    "old_values",
    "sampler_is_weights",
    "rollout_per_token_logps",
    "overlong",
)

# Marks a router-replay slot the trainer must leave to the model's own router.
UNSET_ROUTED_EXPERT = -1

# Default token boundary alignment for packed segments (matches NeMo-RL's
# `sequence_length_round: 64` and Qwen3.5's `gdn_chunk_size = 64`).
DEFAULT_SEGMENT_ALIGN_MULTIPLE = 64


def align_offset(offset: int, segment_align_multiple: int) -> int:
  """Rounds `offset` up to the next multiple of `segment_align_multiple`."""
  if segment_align_multiple <= 0:
    raise ValueError(
        "segment_align_multiple must be positive, got"
        f" {segment_align_multiple}."
    )
  if segment_align_multiple == 1:
    return offset
  return (
      (offset + segment_align_multiple - 1) // segment_align_multiple
  ) * segment_align_multiple


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackItem:
  """A single unpadded item to be packed into a PackedRow."""

  prompt_ids: np.ndarray
  completion_ids: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  per_token: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
  policy_version: np.ndarray | None = None
  # Router replay: `[p + c, num_layers, top_k]` expert ids, aligned to the
  # WHOLE sequence (prompt then completion), not just the completion like the
  # `per_token` fields. `UNSET_ROUTED_EXPERT` (-1) marks tokens the trainer
  # should route with its own gate.
  routed_experts: np.ndarray | None = None

  def __post_init__(self):
    for name in (
        "prompt_ids",
        "completion_ids",
        "completion_mask",
        "advantages",
    ):
      arr = getattr(self, name)
      if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError(
            f"PackItem.{name} must be a 1D numpy array, got"
            f" {type(arr).__name__} with shape {getattr(arr, 'shape', None)}."
            " Unpad and flatten before packing."
        )
    c = self.completion_ids.shape[0]
    for name in ("completion_mask", "advantages"):
      dim = getattr(self, name).shape[0]
      if dim != c:
        raise ValueError(
            f"PackItem.{name} must have length matching completion_ids length;"
            f" got {dim}, expected {c}."
        )
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
    if self.routed_experts is not None:
      re_arr = self.routed_experts
      n = self.prompt_ids.shape[0] + c
      if (
          not isinstance(re_arr, np.ndarray)
          or re_arr.ndim != 3
          or re_arr.shape[0] != n
      ):
        raise ValueError(
            "PackItem.routed_experts must be a numpy array of shape"
            f" (p + c, num_layers, top_k) = ({n}, L, K), got"
            f" {type(re_arr).__name__} with shape"
            f" {getattr(re_arr, 'shape', None)}."
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
  # `[budget, num_layers, top_k]` int16, -1 wherever no routing was captured
  # (padding, and any token the rollout did not report).
  routed_experts: np.ndarray | None = None


def carried_per_token_fields(items: Sequence[PackItem]) -> tuple[str, ...]:
  """Returns the per-token fields carried by all items in the sequence."""
  if not items:
    return ()
  non_empty = [item for item in items if item.num_tokens > 0]
  target_items = non_empty if non_empty else items
  carried = []
  for name in PER_TOKEN_FIELDS:
    presented = [name in item.per_token for item in target_items]
    if all(presented):
      carried.append(name)
    elif any(presented):
      missing = [
          i
          for i, item in enumerate(items)
          if (not non_empty or item.num_tokens > 0)
          and name not in item.per_token
      ]
      raise ValueError(
          f"Some but not all items have per-token field {name!r} (missing at"
          f" indices {missing})."
      )
  return tuple(carried)


def routed_experts_shape(
    items: Sequence[PackItem],
) -> tuple[int, ...] | None:
  """Returns the `(num_layers, top_k)` routing shape if ANY item carries routing.

  An item without routing is packed with all-unset (-1) rows, which the MoE
  layer routes with its own gate -- the same as no replay for those tokens. So
  one routing-less trajectory neither disables replay for the rest of the chunk
  nor changes the payload's pytree structure (which would recompile the
  trainer step).

  Args:
    items: Items that will be packed together into one chunk.

  Returns:
    The trailing routing shape, or None if no item has routing.

  Raises:
    ValueError: If the items that carry routing disagree on the trailing shape.
  """
  shapes = {
      item.routed_experts.shape[1:]
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
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
) -> tuple[list[list[PackItem]], list[PackItem]]:
  """Fills ONE chunk of `pack_size` fixed-capacity bins, first-fit-decreasing.

  Sorts the items by token length descending and greedily places each into the
  first bin with room, where a bin has room only if it stays within both the
  token `budget` (after aligning the starting offset of any segment after the
  first to `segment_align_multiple`) AND `max_segments` sequences (so the loss's
  static `num_segments = max_segments + 1` buckets never overflow). Items that
  fit no bin are returned as `leftover` (in their original order) for a later
  chunk.

  Args:
    items: Sequence of PackItems to pack.
    pack_size: Number of bins in the chunk.
    budget: Token capacity budget per bin.
    max_segments: Maximum number of segments allowed in a single bin.
    segment_align_multiple: Token boundary alignment multiple for the start of
      each segment after the first in a bin.

  Returns:
    A tuple of (bins, leftover), where `bins` is a list of `pack_size` lists of
    PackItems (some may be empty), and `leftover` contains the items that did
    not fit into any bin.
  """
  if segment_align_multiple <= 0:
    raise ValueError(
        "segment_align_multiple must be positive, got"
        f" {segment_align_multiple}."
    )
  bins: list[list[PackItem]] = [[] for _ in range(pack_size)]
  loads = [0] * pack_size
  seg_counts = [0] * pack_size
  order = sorted(
      range(len(items)), key=lambda i: items[i].num_tokens, reverse=True
  )
  placed_flags = [False] * len(items)
  for i in order:
    item = items[i]
    n = item.num_tokens
    if n == 0:
      bins[0].append(item)
      placed_flags[i] = True
      continue
    for b in range(pack_size):
      start = (
          align_offset(loads[b], segment_align_multiple)
          if seg_counts[b] > 0
          else 0
      )
      if start + n <= budget and seg_counts[b] < max_segments:
        bins[b].append(item)
        loads[b] = start + n
        seg_counts[b] += 1
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
    routed_shape: tuple[int, ...] | None = None,
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
    routed_out: np.ndarray | None = None,
    with_routing: bool = True,
) -> PackedRow:
  """Packs a single bin of items into a single `[budget]` PackedRow.

  Args:
    bin_items: Items to pack into this row.
    budget: Token capacity of the row.
    pad_id: Token id written to padding positions.
    carried: Per-token fields carried by the chunk.
    routed_shape: Trailing `(num_layers, top_k)` routing shape, or None when no
      item in the chunk carries routing.
    segment_align_multiple: Token boundary alignment for every segment after
      the first.
    routed_out: Optional `[budget, *routed_shape]` int16 buffer, already filled
      with `UNSET_ROUTED_EXPERT`, that this row's routing is written into in
      place and returned as `PackedRow.routed_experts`. `pack_chunk_contiguous`
      passes one row view of a single chunk-wide buffer here so the chunk's
      routing is never allocated per row and then stacked (a second full copy
      of `[B, T, L, K]`). Requires `routed_shape`.
    with_routing: When False the row carries no routing at all
      (`routed_experts=None`) whatever the items hold; `pack_chunk_compact`
      gathers the routing separately, unpadded. Excludes `routed_out`.

  Returns:
    The packed row.
  """
  if segment_align_multiple <= 0:
    raise ValueError(
        "segment_align_multiple must be positive, got"
        f" {segment_align_multiple}."
    )
  zeros_i = lambda: np.zeros(budget, dtype=np.int32)
  zeros_f = lambda: np.zeros(budget, dtype=np.float32)
  if not with_routing:
    if routed_out is not None:
      raise ValueError("pack_bin: routed_out requires with_routing=True.")
    routed_shape = None
  elif routed_shape is None and bin_items:
    routed_shape = routed_experts_shape(bin_items)
  if routed_out is not None:
    if routed_shape is None:
      raise ValueError("pack_bin: routed_out requires routed_shape.")
    expected = (budget,) + tuple(routed_shape)
    if routed_out.shape != expected or routed_out.dtype != np.int16:
      raise ValueError(
          f"pack_bin: routed_out must be int16 with shape {expected}; got"
          f" {routed_out.dtype} with shape {routed_out.shape}."
      )

  def unset_routed() -> np.ndarray | None:
    if routed_out is not None:
      return routed_out
    if routed_shape is None:
      return None
    return np.full(
        (budget,) + tuple(routed_shape), UNSET_ROUTED_EXPERT, dtype=np.int16
    )

  if not bin_items:
    return PackedRow(
        ids=np.full(budget, pad_id, dtype=np.int32),
        prompt_mask=zeros_f(),
        completion_mask=zeros_f(),
        advantages=zeros_f(),
        segment_ids=zeros_i(),
        segment_positions=zeros_i(),
        per_token={name: zeros_f() for name in carried},
        policy_version=None,
        num_real_segments=0,
        routed_experts=unset_routed(),
    )

  ids = np.full(budget, pad_id, dtype=np.int32)
  prompt_mask = zeros_f()
  completion_mask = zeros_f()
  advantages = zeros_f()
  segment_ids = zeros_i()
  segment_positions = zeros_i()
  per_token = {name: zeros_f() for name in carried}
  routed = unset_routed()

  cursor = 0
  num_real_segments = 0
  for item in bin_items:
    p = item.prompt_ids.shape[0]
    c = item.completion_ids.shape[0]
    n = p + c
    if n == 0:
      continue
    num_real_segments += 1
    seg = num_real_segments
    if seg > 1:
      cursor = align_offset(cursor, segment_align_multiple)
    if cursor + n > budget:
      raise ValueError(
          f"pack_bin: bin size {cursor + n} exceeds budget {budget}."
      )
    seq = slice(cursor, cursor + n)
    comp = slice(cursor + p, cursor + n)

    ids[seq] = np.concatenate([item.prompt_ids, item.completion_ids])
    prompt_mask[cursor : cursor + p] = 1.0
    segment_ids[seq] = seg
    segment_positions[seq] = np.arange(n, dtype=np.int32)

    completion_mask[comp] = item.completion_mask
    advantages[comp] = item.advantages
    for name in carried:
      if name in item.per_token:
        per_token[name][comp] = item.per_token[name]
    if routed is not None and item.routed_experts is not None:
      # Same `seq` slice as `ids`: routing is sequence-aligned, so a token's
      # captured experts land on exactly the position the token itself does.
      routed[seq] = item.routed_experts
    cursor += n

  return PackedRow(
      ids=ids,
      prompt_mask=prompt_mask,
      completion_mask=completion_mask,
      advantages=advantages,
      segment_ids=segment_ids,
      segment_positions=segment_positions,
      per_token=per_token,
      policy_version=bin_items[0].policy_version,
      num_real_segments=num_real_segments,
      routed_experts=routed,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackedChunk:
  """The packed rows of one chunk plus their contiguous routing buffer.

  Attributes:
    rows: One PackedRow per bin.
    routed_experts: `[len(rows), budget, num_layers, top_k]` int16 routing for
      the whole chunk, or None when no item carries routing. `rows[i].
      routed_experts` is a view of `routed_experts[i]`, so callers building a
      batched payload use this buffer as-is rather than stacking the rows.
  """

  rows: list[PackedRow]
  routed_experts: np.ndarray | None


def pack_chunk_contiguous(
    bins: Sequence[Sequence[PackItem]],
    *,
    budget: int,
    pad_id: int,
    carried: Sequence[str],
    routed_shape: tuple[int, ...] | None = None,
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
) -> PackedChunk:
  """Packs the bins of one chunk, writing all routing into one buffer.

  Routing dominates a chunk's size (`[B, T, L, K]` int16 is ~4.9 GB at
  B=64, T=65536, L=60, K=10), so it is allocated once for the chunk and each
  row is packed into its own view. That avoids allocating B row arrays and
  then `np.stack`-ing them into a second full-size copy.

  Args:
    bins: The chunk's bins, one per packed row.
    budget: Token capacity of each row.
    pad_id: Token id written to padding positions.
    carried: Per-token fields carried by the chunk.
    routed_shape: Trailing `(num_layers, top_k)` routing shape. Inferred from
      the placed items when None.
    segment_align_multiple: Token boundary alignment for every segment after
      the first.

  Returns:
    The packed rows and the chunk-wide routing buffer they view into.
  """
  if routed_shape is None:
    placed = [item for bin_items in bins for item in bin_items]
    routed_shape = routed_experts_shape(placed)
  routed = (
      None
      if routed_shape is None
      else np.full(
          (len(bins), budget) + tuple(routed_shape),
          UNSET_ROUTED_EXPERT,
          dtype=np.int16,
      )
  )
  rows = [
      pack_bin(
          bin_items,
          budget=budget,
          pad_id=pad_id,
          carried=carried,
          routed_shape=routed_shape,
          segment_align_multiple=segment_align_multiple,
          routed_out=None if routed is None else routed[i],
      )
      for i, bin_items in enumerate(bins)
  ]
  return PackedChunk(rows=rows, routed_experts=routed)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompactPackedChunk:
  """The packed rows of one chunk plus their routing, without the padding.

  Attributes:
    rows: One PackedRow per bin; `rows[i].routed_experts` is None.
    routed_values: `[N, num_layers, top_k]` int16 routing of every real token
      of the chunk, in row-major order: row 0's segments in placement order,
      then row 1's, and so on. So it lines up one-to-one with the positions
      where the stacked `segment_ids > 0`. Tokens of an item that carries no
      routing are `UNSET_ROUTED_EXPERT`. None when no item carries routing.
  """

  rows: list[PackedRow]
  routed_values: np.ndarray | None


def pack_chunk_compact(
    bins: Sequence[Sequence[PackItem]],
    *,
    budget: int,
    pad_id: int,
    carried: Sequence[str],
    routed_shape: tuple[int, ...] | None = None,
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
) -> CompactPackedChunk:
  """Packs the bins of one chunk, gathering routing only for real tokens.

  Unlike `pack_chunk_contiguous`, the `[B, budget, L, K]` routing tensor is
  never built: roughly three quarters of it is padding for long-context packed
  microbatches. The receiver rebuilds it from `routed_values` and the rows'
  `segment_ids`.

  Args:
    bins: The chunk's bins, one per packed row.
    budget: Token capacity of each row.
    pad_id: Token id written to padding positions.
    carried: Per-token fields carried by the chunk.
    routed_shape: Trailing `(num_layers, top_k)` routing shape. Inferred from
      the placed items when None.
    segment_align_multiple: Token boundary alignment for every segment after
      the first.

  Returns:
    The packed rows and the chunk's unpadded routing.
  """
  if routed_shape is None:
    placed = [item for bin_items in bins for item in bin_items]
    routed_shape = routed_experts_shape(placed)
  rows = [
      pack_bin(
          bin_items,
          budget=budget,
          pad_id=pad_id,
          carried=carried,
          segment_align_multiple=segment_align_multiple,
          with_routing=False,
      )
      for bin_items in bins
  ]
  if routed_shape is None:
    return CompactPackedChunk(rows=rows, routed_values=None)
  # Same order `pack_bin` lays the segments out in, skipping empty items just
  # as it does, so part k covers exactly the k-th real segment's positions.
  parts = [
      item.routed_experts
      if item.routed_experts is not None
      else np.full(
          (item.num_tokens,) + tuple(routed_shape),
          UNSET_ROUTED_EXPERT,
          dtype=np.int16,
      )
      for bin_items in bins
      for item in bin_items
      if item.num_tokens > 0
  ]
  routed_values = (
      np.concatenate(parts, axis=0).astype(np.int16, copy=False)
      if parts
      else np.zeros((0,) + tuple(routed_shape), dtype=np.int16)
  )
  return CompactPackedChunk(rows=rows, routed_values=routed_values)


def pack_chunk(
    bins: Sequence[Sequence[PackItem]],
    *,
    budget: int,
    pad_id: int,
    carried: Sequence[str],
    routed_shape: tuple[int, ...] | None = None,
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
) -> list[PackedRow]:
  """Packs a sequence of bins of one chunk into a row."""
  return pack_chunk_contiguous(
      bins,
      budget=budget,
      pad_id=pad_id,
      carried=carried,
      routed_shape=routed_shape,
      segment_align_multiple=segment_align_multiple,
  ).rows


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


def pack_core(
    items: Sequence[PackItem],
    *,
    budget: int,
    pack_size: int = 1,
    max_segments_per_packed_row: int | None = None,
    pad_id: int = 0,
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
) -> list[list[PackedRow]]:
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
  if segment_align_multiple <= 0:
    raise ValueError(
        "segment_align_multiple must be positive, got"
        f" {segment_align_multiple}."
    )
  if not items:
    return []

  validate_items(items, budget)
  max_segments = effective_max_segments(budget, max_segments_per_packed_row)
  carried = carried_per_token_fields(items)

  chunks: list[list[PackedRow]] = []
  remaining = list(items)
  while remaining:
    bins, remaining = fill_one_chunk(
        remaining,
        pack_size=pack_size,
        budget=budget,
        max_segments=max_segments,
        segment_align_multiple=segment_align_multiple,
    )
    if not any(bins):
      raise ValueError("pack_core: no items placed in any bin.")
    placed = [item for bin_items in bins for item in bin_items]
    chunks.append(
        pack_chunk(
            bins,
            budget=budget,
            pad_id=pad_id,
            carried=carried,
            routed_shape=routed_experts_shape(placed),
            segment_align_multiple=segment_align_multiple,
        )
    )
  return chunks
