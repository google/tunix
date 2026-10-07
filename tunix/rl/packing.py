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
from tunix.rl.agentic.agents import agent_types

try:
  from tunix.rl import _packing_ext  # pylint: disable=g-import-not-at-top
except ImportError:
  _packing_ext = None

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
      if routed.dtype != np.int16 or not routed.flags.c_contiguous:
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
  # `[budget, num_layers, top_k]` int16, `UNSET_ROUTED_EXPERT` wherever no
  # routing was captured (padding, alignment gaps, and routing-less items).
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
  routed = (
      None
      if routed_shape is None
      else np.full(
          (n_bins, budget, *routed_shape),
          agent_types.UNSET_ROUTED_EXPERT,
          dtype=np.int16,
      )
  )
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
        m = item.routed_experts.shape[0]
        routed[b, cursor : cursor + m] = item.routed_experts
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
