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
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackItem:
  """A single unpadded item to be packed into a PackedRow."""

  prompt_ids: np.ndarray
  completion_ids: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  per_token: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
  policy_version: np.ndarray | None = None

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


def fill_one_chunk(
    items: Sequence[PackItem],
    *,
    pack_size: int,
    budget: int,
    max_segments: int,
) -> tuple[list[list[PackItem]], list[PackItem]]:
  """Fills ONE chunk of `pack_size` fixed-capacity bins, first-fit-decreasing.

  Sorts the items by token length descending and greedily places each into the
  first bin with room, where a bin has room only if it stays within both the
  token `budget` AND `max_segments` sequences (so the loss's static
  `num_segments = max_segments + 1` buckets never overflow). Items that fit no
  bin are returned as `leftover` (in their original order) for a later chunk.

  Args:
    items: Sequence of PackItems to pack.
    pack_size: Number of bins in the chunk.
    budget: Token capacity budget per bin.
    max_segments: Maximum number of segments allowed in a single bin.

  Returns:
    A tuple of (bins, leftover), where `bins` is a list of `pack_size` lists of
    PackItems (some may be empty), and `leftover` contains the items that did
    not fit into any bin.
  """
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
      if loads[b] + n <= budget and len(bins[b]) < max_segments:
        bins[b].append(item)
        loads[b] += n
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
) -> PackedRow:
  """Packs a single bin of items into a single `[budget]` PackedRow."""
  zeros_i = lambda: np.zeros(budget, dtype=np.int32)
  zeros_f = lambda: np.zeros(budget, dtype=np.float32)

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
    )

  total = sum(item.num_tokens for item in bin_items)
  if total > budget:
    raise ValueError(f"pack_bin: bin size {total} exceeds budget {budget}.")

  ids = np.full(budget, pad_id, dtype=np.int32)
  prompt_mask = zeros_f()
  completion_mask = zeros_f()
  advantages = zeros_f()
  segment_ids = zeros_i()
  segment_positions = zeros_i()
  per_token = {name: zeros_f() for name in carried}

  cursor = 0
  for seg, item in enumerate(bin_items, start=1):
    p = item.prompt_ids.shape[0]
    c = item.completion_ids.shape[0]
    n = p + c
    seq = slice(cursor, cursor + n)
    comp = slice(cursor + p, cursor + n)

    ids[seq] = np.concatenate([item.prompt_ids, item.completion_ids])
    prompt_mask[cursor : cursor + p] = 1.0
    segment_ids[seq] = seg
    segment_positions[seq] = np.arange(n, dtype=np.int32)

    completion_mask[comp] = item.completion_mask
    advantages[comp] = item.advantages
    for name in carried:
      per_token[name][comp] = item.per_token[name]
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
      num_real_segments=len(bin_items),
  )


def pack_chunk(
    bins: Sequence[Sequence[PackItem]],
    *,
    budget: int,
    pad_id: int,
    carried: Sequence[str],
) -> list[PackedRow]:
  """Packs a sequence of bins of one chunk into a row."""
  return [
      pack_bin(bin_items, budget=budget, pad_id=pad_id, carried=carried)
      for bin_items in bins
  ]


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
    )
    if not any(bins):
      raise ValueError("pack_core: no items placed in any bin.")
    chunks.append(
        pack_chunk(bins, budget=budget, pad_id=pad_id, carried=carried)
    )
  return chunks
