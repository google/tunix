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
from typing import Iterable, Mapping, Sequence, overload

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
  # Router replay: `[p + c - 1 or p + c, num_layers, top_k]` expert ids,
  # sequence-aligned (prompt then completion), not completion-aligned like the
  # `per_token` fields. Autoregressive rollouts may omit the final sampled token
  # (`p + c - 1`), which remains `UNSET_ROUTED_EXPERT` in the packed output.
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
      n = self.prompt_ids.shape[0] + c
      min_len = max(n - 1, 0)
      routed = self.routed_experts
      if (
          not isinstance(routed, np.ndarray)
          or routed.ndim != 3
          or routed.shape[0] < min_len
          or routed.shape[0] > n
      ):
        raise ValueError(
            "PackItem.routed_experts must be a numpy array of shape"
            " (p + c - 1 or p + c, num_layers, top_k) with length in"
            f" [{min_len}, {n}], got {type(routed).__name__} with shape"
            f" {getattr(routed, 'shape', None)}."
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
  # `[n_bins, budget, num_layers, top_k]` int16; see `PackedRow.routed_experts`.
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
) -> PackedRow:
  """Packs a single bin of items into a single `[budget]` PackedRow."""
  return pack_chunk(
      [bin_items],
      budget=budget,
      pad_id=pad_id,
      carried=carried,
      routed_shape=routed_shape,
      segment_align_multiple=segment_align_multiple,
  )[0]


def pack_chunk(
    bins: Sequence[Sequence[PackItem]],
    *,
    budget: int,
    pad_id: int,
    carried: Sequence[str],
    routed_shape: tuple[int, ...] | None = None,
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
) -> PackedChunk:
  """Packs a sequence of bins of one chunk into a contiguous `[n_bins, budget]` PackedChunk."""
  if segment_align_multiple <= 0:
    raise ValueError(
        "segment_align_multiple must be positive, got"
        f" {segment_align_multiple}."
    )
  n_bins = len(bins)
  ids = np.full((n_bins, budget), pad_id, dtype=np.int32)
  prompt_mask = np.zeros((n_bins, budget), dtype=np.float32)
  completion_mask = np.zeros((n_bins, budget), dtype=np.float32)
  advantages = np.zeros((n_bins, budget), dtype=np.float32)
  segment_ids = np.zeros((n_bins, budget), dtype=np.int32)
  segment_positions = np.zeros((n_bins, budget), dtype=np.int32)
  per_token = {
      name: np.zeros((n_bins, budget), dtype=np.float32) for name in carried
  }
  if routed_shape is None:
    placed = [
        item
        for bin_items in bins
        for item in bin_items
        if item.num_tokens > 0
    ]
    routed_shape = routed_experts_shape(placed)
  routed = (
      None
      if routed_shape is None
      else np.full(
          (n_bins, budget, *routed_shape), UNSET_ROUTED_EXPERT, dtype=np.int16
      )
  )
  policy_versions: list[np.ndarray | None] = []
  num_real_segments_list: list[int] = []
  for b, bin_items in enumerate(bins):
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

      ids[b, cursor : cursor + p] = item.prompt_ids
      ids[b, comp] = item.completion_ids
      prompt_mask[b, cursor : cursor + p] = 1.0
      segment_ids[b, seq] = seg
      segment_positions[b, seq] = np.arange(n, dtype=np.int32)

      completion_mask[b, comp] = item.completion_mask
      advantages[b, comp] = item.advantages
      for name in carried:
        if name in item.per_token:
          per_token[name][b, comp] = item.per_token[name]
      if routed is not None and item.routed_experts is not None:
        # Prefix-aligned to `seq`: a token's captured experts land on the
        # position the token itself does, and any uncaptured trailing tokens in
        # the segment remain `UNSET_ROUTED_EXPERT`.
        m = item.routed_experts.shape[0]
        routed[b, cursor : cursor + m] = item.routed_experts
      cursor += n

    policy_versions.append(bin_items[0].policy_version if bin_items else None)
    num_real_segments_list.append(num_real_segments)
  return PackedChunk(
      ids=ids,
      prompt_mask=prompt_mask,
      completion_mask=completion_mask,
      advantages=advantages,
      segment_ids=segment_ids,
      segment_positions=segment_positions,
      per_token=per_token,
      policy_versions=tuple(policy_versions),
      num_real_segments=tuple(num_real_segments_list),
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


def pack_core(
    items: Sequence[PackItem],
    *,
    budget: int,
    pack_size: int = 1,
    max_segments_per_packed_row: int | None = None,
    pad_id: int = 0,
    segment_align_multiple: int = DEFAULT_SEGMENT_ALIGN_MULTIPLE,
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

  chunks: list[PackedChunk] = []
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
